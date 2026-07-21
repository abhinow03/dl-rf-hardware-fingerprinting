"""GEN-RFF PHASE 3 — DANN CLOSING CARD (one pre-registered channel-adversarial run).

Vindicated base (FINDINGS F3/F4/F5/F6): WiSig-109 only, residual ON, physics token OFF,
cross-condition positives ON. Add a channel adversary: MLP(512->256->K) on the 512-D
pre-projection via a gradient-reversal layer, predicting the WiSig receiver/session class,
so the encoder is pushed toward channel-invariant (hardware-only) features. lambda ramps
0->0.3 over the first 30% of steps, flat after. Single config, single seed, NO sweeps.

Bands (FINDINGS §5.1): SIGNAL if mav oracle-K@8 km N10>=0.35 OR N120>=0.80; else NULL.

  OMP_NUM_THREADS=1 ... python3 -m gen_rff.train.train_dann

mavicAir2 touched only at final eval; frozen assets read-only; artifacts in runs_gen/results_gen.
"""
import os
import sys
import re
import json
import math
import copy
import time
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DLM not in sys.path:
    sys.path.insert(0, _DLM)

import rffp.training.train_lopo as T
from rffp.data import registry
from rffp.models.gen_encoder import GenRFEncoder, param_count
from rffp.models import SupervisedContrastiveLoss

RUNS_GEN = os.path.join(_DLM, "runs_gen", "dann")
RESULTS_GEN = os.path.join(_DLM, "results_gen")
os.makedirs(RUNS_GEN, exist_ok=True)

SEED = 1234
STEPS = 10000
WARMUP = 1000
VAL_EVERY = 1000
LR = 5e-4
TAU = 0.5
LAMBDA_MAX = 0.3
LAMBDA_RAMP_FRAC = 0.30
SELECTION = "VAL-A oracle-km/0.72 + VAL-A kNN-1 (argmax; no ORACLE -> VAL-A only); frozen"


# ---- gradient reversal ----
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb):
        ctx.lamb = lamb
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lamb * g, None


def grl(x, lamb):
    return _GRL.apply(x, lamb)


def lambda_at(step):
    ramp = int(LAMBDA_RAMP_FRAC * STEPS)
    return LAMBDA_MAX * min(1.0, step / max(1, ramp))


def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * prog))


def build_cond_labels(ds):
    """Per-window WiSig condition labels: rx-only (18-way) and (rx,date) (up to 72-way)."""
    rx_of, rxd_of = {}, {}
    rx_lab = np.full(len(ds.records), -1, int)
    rxd_lab = np.full(len(ds.records), -1, int)
    pat = re.compile(r"rx(\d+)_d(\d+)")
    for i, r in enumerate(ds.records):
        if r["domain"] != "WISIG":
            continue
        m = pat.match(r["condition_key"])
        rx, dt = int(m.group(1)), int(m.group(2))
        rx_lab[i] = rx_of.setdefault(rx, len(rx_of))
        rxd_lab[i] = rxd_of.setdefault((rx, dt), len(rxd_of))
    return rx_lab, len(rx_of), rxd_lab, len(rxd_of)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    print(f"=== DANN Phase-3 (channel-adversarial; WiSig-109 only; residual ON, physics OFF) ===")
    print(f"SELECTION: {SELECTION}")

    ds = T.build_train(domains=("WISIG",))
    dom_idx = T.domain_index(ds)
    rx_lab, n_rx, rxd_lab, n_rxd = build_cond_labels(ds)

    # decide adversary granularity from batch stats BEFORE training
    ndist_rx, ndist_rxd = [], []
    for _ in range(50):
        idxs, _ = T.sample_batch(dom_idx, "WISIG", rng, cross_condition=True)
        ndist_rx.append(len(set(rx_lab[idxs].tolist())))
        ndist_rxd.append(len(set(rxd_lab[idxs].tolist())))
    mean_rx, mean_rxd = float(np.mean(ndist_rx)), float(np.mean(ndist_rxd))
    print(f"  batch-stats: distinct rx/batch={mean_rx:.1f}/{n_rx}, distinct (rx,date)/batch="
          f"{mean_rxd:.1f}/{n_rxd} (batch={T.P_DEV*T.V_VIEW})")
    # 72-way is too sparse if fewer than ~1/3 of classes appear AND per-batch coverage is thin
    use_rx_only = (mean_rxd < 0.5 * (T.P_DEV * T.V_VIEW))
    if use_rx_only:
        cond_lab, n_cond, gran = rx_lab, n_rx, f"rx-only {n_rx}-way"
    else:
        cond_lab, n_cond, gran = rxd_lab, n_rxd, f"(rx,date) {n_rxd}-way"
    print(f"  ADVERSARY TARGET (decided before training): {gran}")

    slice18, cacheA = T.build_val_a()
    nfft, hop = registry.REGISTRY["WISIG"].n_fft, registry.REGISTRY["WISIG"].hop

    model = GenRFEncoder(use_physics=False, use_residual=True, branch_dropout=False).cuda()
    adv = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_cond)).cuda()
    tot_p, _ = param_count(model)
    print(f"  GenRFEncoder params: {tot_p:,} ({tot_p/1e6:.2f}M) + adversary MLP(512->256->{n_cond})")
    opt = torch.optim.AdamW(list(model.parameters()) + list(adv.parameters()), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    supcon = SupervisedContrastiveLoss()
    ce = nn.CrossEntropyLoss()

    curve = []; adv_curve = []; best = dict(comp=-1e9, step=-1, state=None); t0 = time.time()
    for step in range(1, STEPS + 1):
        model.train(); adv.train()
        idxs, y = T.sample_batch(dom_idx, "WISIG", rng, cross_condition=True)
        iq, res, _ = T.fetch_batch(ds, idxs)
        xt = torch.from_numpy(iq).cuda(); rs = torch.from_numpy(res).cuda()
        iq_a, res_a = T.augment_pair(xt, rs, use_residual=True)
        x_time, x_spec = T.make_inputs(iq_a, res_a, nfft, hop, True)
        ycu = torch.from_numpy(y).cuda()
        cbatch = torch.from_numpy(cond_lab[idxs]).cuda()
        lamb = lambda_at(step)
        with torch.amp.autocast('cuda'):
            h512 = model.get_encoder_output(x_time, x_spec, None)
            emb = F.normalize(model.projection_head(h512), dim=1, eps=1e-6)
            loss_sc = supcon(emb, ycu, temperature=TAU)
            adv_logits = adv(grl(h512, lamb))
            loss_adv = ce(adv_logits, cbatch)
            loss = loss_sc + loss_adv
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(adv.parameters()), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % 100 == 0 or step == STEPS:
            acc = float((adv_logits.argmax(1) == cbatch).float().mean())
            adv_curve.append(dict(step=step, lamb=round(lamb, 3), adv_acc=round(acc, 3),
                                  chance=round(1.0 / n_cond, 3), loss_sc=round(float(loss_sc), 3),
                                  loss_adv=round(float(loss_adv), 3)))
            if step % 500 == 0:
                print(f"  step {step:5d}: SC={loss_sc:.3f} ADV={loss_adv:.3f} adv_acc={acc:.3f} "
                      f"(chance {1.0/n_cond:.3f}) lamb={lamb:.3f} ({(time.time()-t0)/60:.1f}m)")
        if step % VAL_EVERY == 0:
            ra = T.val_a(model, slice18, cacheA)
            comp = ra["oracle_km"] / T.SPECIALIST_REF + ra["knn1"]
            curve.append(dict(step=step, VALA_km=round(ra["oracle_km"], 3), VALA_knn1=round(ra["knn1"], 3),
                              composite=round(comp, 4)))
            flag = ""
            if comp > best["comp"]:
                best = dict(comp=comp, step=step, state=copy.deepcopy(model.state_dict())); flag = "  <-best"
            print(f"    [VAL {step}] A:km={ra['oracle_km']:.3f}/knn1={ra['knn1']:.3f} comp={comp:.4f}{flag}")
    model.load_state_dict(best["state"])
    torch.save(best["state"], os.path.join(RUNS_GEN, "best.pt"))
    print(f"  selected step {best['step']} (composite {best['comp']:.4f})")

    # adversary trajectory summary
    accs = [r["adv_acc"] for r in adv_curve] or [1.0 / n_cond]; chance = 1.0 / n_cond
    adv_summary = dict(target=gran, chance=round(chance, 3), first=accs[0], peak=round(max(accs), 3),
                       last=accs[-1], mean_last20pct=round(float(np.mean(accs[int(0.8*len(accs)):])), 3))
    if adv_summary["last"] <= chance * 1.5 and adv_summary["peak"] > chance * 2:
        adv_health = "HEALTHY (adversary above chance then degraded -> invariance pressure applied)"
    elif adv_summary["peak"] <= chance * 1.5:
        adv_health = "REVERSAL-TOO-STRONG (adversary near chance throughout)"
    else:
        adv_health = "WEAK-PRESSURE (adversary stayed high -> little invariance enforced)"
    print(f"  ADVERSARY: {adv_summary}  -> {adv_health}")

    # ---- FINAL EVAL (part-1 harness via train_lopo.final_eval) + self-cell ----
    print("\n=== FINAL EVAL — mavicAir2 (untouched until now) ===")
    fe = T.final_eval(model)
    selfc = T.val_a(model, slice18, cacheA)["oracle_km"]
    print(f"  WiSig-DEV self-cell oracle-km@18 = {selfc:.3f} (specialist ref {T.SPECIALIST_REF})")

    km10, km120 = fe["N10"]["oracle_km"], fe["N120"]["oracle_km"]
    if km10 >= 0.35 or km120 >= 0.80:
        band = f"SIGNAL: mav N10 km={km10:.3f} (>=0.35?) / N120 km={km120:.3f} (>=0.80?) -> a band cleared."
    else:
        band = (f"NULL: mav N10 km={km10:.3f} (<0.35) AND N120 km={km120:.3f} (<0.80) -> generalist "
                f"architecture line CLOSED on this pool.")
    print(f"\nBAND: {band}")

    report = dict(header="GEN-RFF PHASE 3 — DANN channel-adversarial closing card (DEMO-SIDE, R&D)",
                  config=dict(seed=SEED, steps=STEPS, tau=TAU, lambda_max=LAMBDA_MAX,
                              lambda_ramp_frac=LAMBDA_RAMP_FRAC, adversary_target=gran,
                              use_physics=False, use_residual=True, cross_condition=True,
                              domains=["WISIG"]),
                  selection=SELECTION, selected_step=best["step"], selected_composite=round(best["comp"], 4),
                  batch_stats=dict(distinct_rx_per_batch=round(mean_rx, 2), n_rx=n_rx,
                                   distinct_rxdate_per_batch=round(mean_rxd, 2), n_rxdate=n_rxd,
                                   use_rx_only=bool(use_rx_only)),
                  adversary_summary=adv_summary, adversary_health=adv_health,
                  adversary_curve=adv_curve, val_curve=curve, final_eval=fe,
                  wisig_self_cell=round(float(selfc), 3),
                  context_rows=dict(
                      frozen_wisig=dict(N10_locked=0.297, N10_part1cap1500=0.137, N120=0.729),
                      generalist_phase2=dict(N10=0.197, N120=0.625),
                      native_drone=dict(N10=0.276, N120=0.792),
                      A3_no_crosscond=dict(N10=0.272, N120=0.806)),
                  band_verdict=band,
                  guardrails="single pre-registered run; no sweeps/rescues/second-seed; mavicAir2 final-eval "
                             "only; frozen assets read-only; part-1 harness for N10/N120 comparability.")
    json.dump(report, open(os.path.join(RESULTS_GEN, "phase3_dann_report.json"), "w"), indent=2, default=str)
    print(f"\nsaved -> results_gen/phase3_dann_report.json ; ckpt -> runs_gen/dann/best.pt")


if __name__ == "__main__":
    main()
