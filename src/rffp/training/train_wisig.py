"""Step 3 — fresh SupCon metric training on board-disjoint WiSig ManyTx.

Encoder + loss are SHARED from the parent DL_model (via shared.py), trained FRESH
(random init) — we do NOT seed from ORACLE V4 weights.

This file currently exposes the Checkpoint-3 SMOKE run only:
    python3 train/train_wisig.py --smoke
It trains two short fresh models (CFO aug ON vs OFF), reports loss curves and an
embedding-health check (intra- vs inter-device cosine, plus an Rx-leakage probe), so we
can decide whether CFO augmentation helps or erases fingerprint signal before any long run.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)                       # summer_work/
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rffp.models import RFEncoder, SupervisedContrastiveLoss          # parent encoder + loss
from rffp.data import wisig_manytx as W

RUN_DIR    = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT_PATH = os.path.join(RUN_DIR, "splits", "split_manytx.json")

# ── smoke config ───────────────────────────────────────────────
N_DEV, M_SIG = W.DEFAULT_N, W.DEFAULT_M     # 32 x 8 = 256
SMOKE_STEPS  = 300
LR           = 1e-3
WD           = 1e-4
WARMUP       = 30
GRAD_CLIP    = 1.0
TAU          = 0.1            # workhorse temp for smoke (real run: 0.5->0.1->0.07)
SEED         = 42
LOG_EVERY    = 25


def to_cuda(t, s):
    return (torch.from_numpy(t).cuda(non_blocking=True),
            torch.from_numpy(s).cuda(non_blocking=True))


def probe_time_branch(model):
    """Print the time-branch feature length BEFORE global average pooling on [*,2,256]."""
    tb = model.time_branch
    x = torch.randn(2, 2, W.SIG_LEN).cuda()
    with torch.no_grad():
        h = tb.stem(x)
        L_stem = h.shape[-1]
        h = tb.b4(tb.b3(tb.b2(tb.b1(h))))
        L_pre_gap = h.shape[-1]
    print(f"  time-branch: input 256 -> stem {L_stem} -> pre-GAP length {L_pre_gap} "
          f"({'OK (no collapse)' if L_pre_gap >= 8 else 'COLLAPSED — reduce stem kernel'})")
    return L_pre_gap


@torch.no_grad()
def embed_health(model, tx_data, devs, rng, n_dev=16, m=16):
    """Intra/inter-device cosine + an Rx-leakage probe on a no-aug eval batch."""
    model.eval()
    lbl = {tx: i for i, tx in enumerate(devs)}
    t, s, y, info = W.sample_batch(tx_data, devs, lbl, rng, N=n_dev, M=m, train=False)
    xt, xs = to_cuda(t, s)
    with torch.amp.autocast('cuda'):
        emb = model(xt, xs)
    emb = emb.float().cpu().numpy()
    model.train()

    sim = emb @ emb.T
    B = len(y)
    eye = np.eye(B, dtype=bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = (~(y[:, None] == y[None, :])) & ~eye
    intra = float(sim[same].mean()); inter = float(sim[diff].mean())

    rx = info["rx"]
    sd_sr = same & (rx[:, None] == rx[None, :])      # same device, same receiver
    sd_dr = same & (rx[:, None] != rx[None, :])      # same device, diff receiver
    rx_gap = (float(sim[sd_sr].mean()) - float(sim[sd_dr].mean())
              if sd_sr.sum() > 5 and sd_dr.sum() > 5 else float("nan"))
    return {"intra": intra, "inter": inter, "gap": intra - inter,
            "rx_gap": rx_gap, "n_same_rx": int(sd_sr.sum())}


def train_smoke(tx_data, train_tx, use_cfo):
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    label_of = {tx: i for i, tx in enumerate(train_tx)}

    model = RFEncoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = SupervisedContrastiveLoss()
    scaler = torch.amp.GradScaler('cuda')

    health0 = embed_health(model, tx_data, train_tx, np.random.default_rng(7))
    curve = []
    model.train()
    for step in range(SMOKE_STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        t, s, y, _ = W.sample_batch(tx_data, train_tx, label_of, rng,
                                    N=N_DEV, M=M_SIG, train=True, use_cfo=use_cfo)
        xt, xs = to_cuda(t, s)
        yt = torch.from_numpy(y).cuda()
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            emb = model(xt, xs)
        loss = crit(emb.float(), yt, temperature=TAU)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()
        if (step + 1) % LOG_EVERY == 0 or step == 0:
            curve.append((step + 1, round(float(loss), 4)))
    healthN = embed_health(model, tx_data, train_tx, np.random.default_rng(7))
    return curve, health0, healthN


# ── full-run config (LOCKED) ───────────────────────────────────
TOTAL_STEPS = 13000
WARMUP_FULL = 1300                      # ~10% of total
LR_FULL     = 5e-4                      # SupCon unstable at 1e-3 early
TAU_HI, TAU_MID, TAU_LO = 0.5, 0.1, 0.07
TAU_F1, TAU_F2 = 0.15, 0.60            # step fractions: 0.5 until 15%, ->0.1 by 60%, 0.07 tail
LOG_STEP    = 250
CKPT_STEP   = 2000


def lr_at(step):
    if step < WARMUP_FULL:
        return LR_FULL * (step + 1) / WARMUP_FULL
    prog = (step - WARMUP_FULL) / (TOTAL_STEPS - WARMUP_FULL)
    return 0.5 * LR_FULL * (1 + np.cos(np.pi * prog))


def tau_at(step):
    f = step / TOTAL_STEPS
    if f < TAU_F1:
        return TAU_HI
    if f < TAU_F2:
        a = (f - TAU_F1) / (TAU_F2 - TAU_F1)
        return TAU_HI + a * (TAU_MID - TAU_HI)     # linear 0.5 -> 0.1
    return TAU_LO                                   # 0.07 tail


def make_fixed_eval(tx_data, devs, n_dev=16, m=16, seed=7):
    """One fixed no-aug eval batch so the health trajectory is comparable across steps."""
    rng = np.random.default_rng(seed)
    lbl = {tx: i for i, tx in enumerate(devs)}
    t, s, y, info = W.sample_batch(tx_data, devs, lbl, rng, N=n_dev, M=m, train=False)
    return to_cuda(t, s), y, info


@torch.no_grad()
def eval_fixed(model, fixed):
    (xt, xs), y, info = fixed
    model.eval()
    with torch.amp.autocast('cuda'):
        emb = model(xt, xs)
    emb = emb.float().cpu().numpy()
    model.train()
    sim = emb @ emb.T
    B = len(y); eye = np.eye(B, dtype=bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = (~(y[:, None] == y[None, :])) & ~eye
    rx = info["rx"]
    sd_sr = same & (rx[:, None] == rx[None, :])
    sd_dr = same & (rx[:, None] != rx[None, :])
    rx_gap = (float(sim[sd_sr].mean()) - float(sim[sd_dr].mean())
              if sd_sr.sum() > 5 and sd_dr.sum() > 5 else float("nan"))
    intra = float(sim[same].mean()); inter = float(sim[diff].mean())
    return intra, inter, intra - inter, rx_gap


def run_full(tx_data, train_tx):
    import csv
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    label_of = {tx: i for i, tx in enumerate(train_tx)}

    model = RFEncoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_FULL, weight_decay=WD)
    crit = SupervisedContrastiveLoss()
    scaler = torch.amp.GradScaler('cuda')
    fixed = make_fixed_eval(tx_data, train_tx)

    n_train_sig = sum(tx_data[t]["iq"].shape[0] for t in train_tx)
    print(f"train signals (109 Tx) = {n_train_sig}")
    print(f"TOTAL_STEPS={TOTAL_STEPS} warmup={WARMUP_FULL} | tau 0.5<{int(TAU_F1*TOTAL_STEPS)} "
          f"-> 0.1@{int(TAU_F2*TOTAL_STEPS)} -> 0.07 tail | LR={LR_FULL} batch={N_DEV}x{M_SIG}")
    print(f"signal-draws/run = {TOTAL_STEPS*N_DEV*M_SIG} (~{TOTAL_STEPS*N_DEV*M_SIG/n_train_sig:.1f}x over train sigs)")

    csv_path = os.path.join(RUN_DIR, "train_log.csv")
    cf = open(csv_path, "w", newline="")
    wr = csv.writer(cf); wr.writerow(["step", "loss", "intra", "inter", "gap", "rx_gap", "lr", "tau"])

    best_gap, best_step = -1.0, 0
    model.train()
    for step in range(TOTAL_STEPS):
        lr = lr_at(step); tau = tau_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        t, s, y, _ = W.sample_batch(tx_data, train_tx, label_of, rng,
                                    N=N_DEV, M=M_SIG, train=True, use_cfo=False)
        xt, xs = to_cuda(t, s)
        yt = torch.from_numpy(y).cuda()
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            emb = model(xt, xs)
        loss = crit(emb.float(), yt, temperature=tau)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()

        if (step + 1) % LOG_STEP == 0 or step == 0:
            intra, inter, gap, rx_gap = eval_fixed(model, fixed)
            wr.writerow([step + 1, round(float(loss), 4), round(intra, 4), round(inter, 4),
                         round(gap, 4), round(rx_gap, 4), round(lr, 7), round(tau, 4)])
            cf.flush()
            print(f"step {step+1:5d}/{TOTAL_STEPS} loss={float(loss):.4f} "
                  f"intra={intra:.3f} inter={inter:.3f} gap={gap:.3f} rx_gap={rx_gap:.3f} "
                  f"lr={lr:.2e} tau={tau:.3f}", flush=True)
            if gap > best_gap:
                best_gap, best_step = gap, step + 1
                torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pt"))

        if (step + 1) % CKPT_STEP == 0:
            torch.save(model.state_dict(), os.path.join(RUN_DIR, f"ckpt_step{step+1}.pt"))

    torch.save(model.state_dict(), os.path.join(RUN_DIR, "last_model.pt"))
    cf.close()
    print(f"\nDONE. best gap={best_gap:.4f} @ step {best_step} -> best_model.pt | last_model.pt")
    with open(os.path.join(RUN_DIR, "full_summary.json"), "w") as f:
        json.dump({"total_steps": TOTAL_STEPS, "best_gap": best_gap, "best_step": best_step,
                   "lr": LR_FULL, "warmup": WARMUP_FULL, "csv": csv_path}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tx_data, meta = W.load_manytx()
    sp = json.load(open(SPLIT_PATH))
    assert sp["mode"] == "board", "split must be board-disjoint"
    train_tx = sp["train_tx"]
    print(f"Split: board-disjoint  train={sp['n_train']}  discover={sp['n_discover']}")

    print("\nTime-branch collapse check (256-sample input):")
    probe_time_branch(RFEncoder().cuda())

    if args.full:
        print(f"\n{'='*60}\nFULL RUN — board-disjoint WiSig ManyTx, CFO OFF\n{'='*60}")
        run_full(tx_data, train_tx)
        return

    if not args.smoke:
        print("\n(Pass --smoke or --full.)")
        return

    results = {}
    for use_cfo in (True, False):
        tag = "CFO_on" if use_cfo else "CFO_off"
        print(f"\n{'='*60}\nSMOKE [{tag}]  {SMOKE_STEPS} steps, batch {N_DEV}x{M_SIG}, tau={TAU}\n{'='*60}")
        curve, h0, hN = train_smoke(tx_data, train_tx, use_cfo)
        results[tag] = {"curve": curve, "health_init": h0, "health_final": hN}
        print(f"  loss: {curve[0][1]} (step1) -> {curve[-1][1]} (step{curve[-1][0]})")
        print(f"  embedding health  init : intra={h0['intra']:.3f} inter={h0['inter']:.3f} gap={h0['gap']:.3f}")
        print(f"  embedding health  final: intra={hN['intra']:.3f} inter={hN['inter']:.3f} gap={hN['gap']:.3f} "
              f"| rx_gap={hN['rx_gap']:.3f} (n_same_rx={hN['n_same_rx']})")

    print(f"\n{'='*60}\nCFO ABLATION SUMMARY (higher gap = better device separation; rx_gap near 0 = no receiver leakage)\n{'='*60}")
    print(f"{'config':10s} {'final_loss':>10s} {'intra':>7s} {'inter':>7s} {'gap':>7s} {'rx_gap':>8s}")
    for tag, r in results.items():
        h = r["health_final"]
        print(f"{tag:10s} {r['curve'][-1][1]:>10.4f} {h['intra']:>7.3f} {h['inter']:>7.3f} {h['gap']:>7.3f} {h['rx_gap']:>8.3f}")

    out = os.path.join(RUN_DIR, "checkpoint3_smoke.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
