"""SupCon + additive SEMI-HARD MARGIN trainer — SMOKE SWEEP. No full run, no drones/drive.

WHY: reweighted-SupCon is exhausted (validated random-slice ARI: vanilla 0.44 vs v2 0.418 — no
gain; structural — SupCon's log-partition couples negatives to positives). KEY EVIDENCE: vanilla
gets purity 0.457 at K_est 19 / K_true 18 on the random slice — the ARI cap is DIFFUSE MODERATE
confusion across ORDINARY devices, NOT just the cos-0.999 worst pairs. => SEMI-HARD mining (the
moderate-confusion bulk), not hardest-negative.

LOSS (additive — replace only the broken half):
    total = SupCon(unchanged, multi-positive, tau=0.5)  +  lambda * SemiHardMargin
SemiHardMargin: hinge max(0, m - s_ap + s_an*) on (anchor,positive,negative) triplets, where the
negative n* is SEMI-HARD = the negative LESS similar than the positive (s_an < s_ap) but closest
to the anchor within that band (FaceNet semi-hard). Triplets with no semi-hard negative contribute
0 — the hinge zeroes out once the negative clears the margin (the decoupling reweighting lacked).
Mined within-batch; confuser-aware batches give it material.

SWEEP: margin m in {0.1,0.2,0.3} x lambda in {0.5,1.0}. KEEP eq=0, tau=0.5 const, RFEncoder,
LR 5e-4, fresh init, confuser-aware batching. SELECT by RANDOM grid-scattered-slice ARI
(18-dev seed-123 + HDBSCAN mcs15 — the validated ruler). OLD split for A/B.

    python3 train/train_wisig_triplet.py --smoke
STOP at the checkpoint.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder, SupervisedContrastiveLoss
import wisig_manytx as W
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# reuse validated helpers (same module that produced v2 + the ruler)
from train_wisig_hardneg import (build_batch, confuser_devices, centroid_graph, cooccur_rate,
                                 build_eval, embed_pack, centroids_of, nearest_confuser,
                                 worst_pairs_from, make_health, health, load_baseline)

RUN_DIR    = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR    = os.path.join(RUN_DIR, "discover")
SPLIT_OLD  = os.path.join(RUN_DIR, "splits", "split_manytx.json")
SMOKE_DIR  = os.path.join(OUT_DIR, "triplet_smoke")
REPORT     = os.path.join(OUT_DIR, "triplet_smoke_report.json")

N_DEV, M_SIG = 32, 8
LR, WD, WARMUP, GRAD_CLIP, TAU = 5e-4, 1e-4, 350, 1.0, 0.5
SEED = 42
CFRAC, REFRESH = 0.5, 500
STEPS, LOG = 4000, 250
RAND_N, RAND_SEED, VAL_CAP, VAL_MCS = 18, 123, 160, 15     # locked ruler
MARGINS, LAMBDAS = [0.1, 0.2, 0.3], [0.5, 1.0]


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def purity(true, pred):
    if len(true) == 0:
        return float("nan")
    return sum(np.unique(true[pred == c], return_counts=True)[1].max()
               for c in np.unique(pred)) / len(true)


def discovery(embs, dids, n_true, mcs=VAL_MCS):
    pred = HDBSCAN(min_cluster_size=mcs, metric="euclidean", n_jobs=-1, copy=True).fit_predict(embs)
    keep = pred != -1
    k = len(np.unique(pred[keep]))
    noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return {"ARI": 0.0, "NMI": 0.0, "purity": float(purity(dids, pred)),
                "K_est": k, "K_true": n_true, "noise": noise}
    return {"ARI": float(adjusted_rand_score(dids[keep], pred[keep])),
            "NMI": float(normalized_mutual_info_score(dids[keep], pred[keep])),
            "purity": float(purity(dids, pred)), "K_est": k, "K_true": n_true, "noise": noise}


def scatter_slice(held_tx, n=RAND_N, seed=RAND_SEED):
    rng = np.random.default_rng(seed)
    rows = {}
    for t in held_tx:
        rows.setdefault(int(t.split("-")[0]), []).append(t)
    for r in rows:
        rng.shuffle(rows[r])
    keys = sorted(rows); ptr = {r: 0 for r in keys}; out = []
    while len(out) < n:
        prog = False
        for r in keys:
            if len(out) >= n:
                break
            if ptr[r] < len(rows[r]):
                out.append(rows[r][ptr[r]]); ptr[r] += 1; prog = True
        if not prog:
            break
    return sorted(out)


# ════════════════════ SEMI-HARD MARGIN TERM ════════════════════
def semihard_margin(emb, labels, m):
    """Hinge max(0, m - s_ap + s_an*) with s_an* = max negative sim BELOW the positive sim
    (FaceNet semi-hard). Triplets with no semi-hard negative contribute 0."""
    emb = emb.float()
    B = emb.shape[0]
    dev = emb.device
    S = emb @ emb.t()
    self_mask = torch.eye(B, dtype=torch.bool, device=dev)
    lc = labels.view(-1, 1)
    pos = (lc == lc.t()) & ~self_mask                       # [a,p]
    neg = (lc != lc.t())                                    # [a,n]
    Sap = S.unsqueeze(2)                                    # [a,p,1]
    San = S.unsqueeze(1).expand(B, B, B).clone()           # [a,1,n]->[a,p,n]
    cond = pos.unsqueeze(2) & neg.unsqueeze(1) & (San < Sap)   # semi-hard band
    San = San.masked_fill(~cond, float("-inf"))
    shn, _ = San.max(dim=2)                                 # [a,p] hardest semi-hard neg sim
    valid = torch.isfinite(shn) & pos
    if valid.sum() == 0:
        return emb.sum() * 0.0
    hinge = torch.relu(m - S + shn)                         # [a,p]
    return hinge[valid].mean()


def to_cuda(t, s):
    return torch.from_numpy(t).cuda(non_blocking=True), torch.from_numpy(s).cuda(non_blocking=True)


def run_config(tx_data, train_tx, lbl_all, graph0, ref_top1, rand_pack, full_pack,
               health_pack, n_held, base_worst, m, lam, tag):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = RFEncoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sup = SupervisedContrastiveLoss()
    scaler = torch.amp.GradScaler('cuda')
    rng = np.random.default_rng(SEED)
    graph = dict(graph0)
    ckpt = os.path.join(SMOKE_DIR, f"{tag}.pt")

    best = {"ARI": -1.0}; best_step = 0; traj = []
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        if (step + 1) % REFRESH == 0:
            graph, _ = centroid_graph(model, tx_data, train_tx)
        if rng.random() < CFRAC:
            dev_idx = confuser_devices(graph, N_DEV, rng)
        else:
            dev_idx = list(rng.choice(len(train_tx), N_DEV, replace=False))
        t, s, y = build_batch(tx_data, dev_idx, train_tx, lbl_all, rng, train=True, use_cfo=False)
        xt, xs = to_cuda(t, s); yt = torch.from_numpy(y).cuda()
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            emb = model(xt, xs)
        emb = emb.float()
        l_sup = sup(emb, yt, temperature=TAU)
        l_mar = semihard_margin(emb, yt, m)
        loss = l_sup + lam * l_mar
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()

        if (step + 1) % LOG == 0 or step == 0:
            re, rd = embed_pack(model, rand_pack)
            disc = discovery(re, rd, RAND_N)
            fe, fd = embed_pack(model, full_pack)
            cents = centroids_of(fe, fd, n_held)
            med, mean, C = nearest_confuser(cents)
            h = health(model, health_pack)
            traj.append({"step": step + 1, "loss": round(float(loss), 3),
                         "l_sup": round(float(l_sup), 3), "l_mar": round(float(l_mar), 4),
                         "rand_ARI": disc["ARI"], "purity": round(disc["purity"], 3),
                         "K_est": disc["K_est"], "nn_med": round(med, 4),
                         "intra": round(h["intra"], 3), "inter": round(h["inter"], 3),
                         "rx_gap": round(h["rx_gap"], 3)})
            print(f"  [{tag}] step {step+1:5d} loss={float(loss):.3f}(sup{float(l_sup):.2f}+"
                  f"{lam}*mar{float(l_mar):.3f}) | randARI={disc['ARI']:.3f} pur={disc['purity']:.3f} "
                  f"K={disc['K_est']}/{RAND_N} | intra={h['intra']:.3f} nn={med:.4f} "
                  f"rxg={h['rx_gap']:.3f}", flush=True)
            if disc["ARI"] > best["ARI"]:
                best = {**disc, "nn_med": med, "intra": h["intra"], "inter": h["inter"],
                        "rx_gap": h["rx_gap"], "C": C.copy()}
                best_step = step + 1
                torch.save(model.state_dict(), ckpt)
    dt = time.time() - t0
    wp = [{"pair": f"{base_worst[k][3]}", "baseline": round(base_worst[k][2], 4),
           "smoke": round(float(best["C"][base_worst[k][0], base_worst[k][1]]), 4)}
          for k in range(len(base_worst))]
    print(f"  [{tag}] BEST randARI={best['ARI']:.3f} @step{best_step} "
          f"pur={best['purity']:.3f} K={best['K_est']} intra={best['intra']:.3f} "
          f"nn={best['nn_med']:.4f} ({dt/60:.1f}min)")
    return {"m": m, "lambda": lam, "best_step": best_step, "minutes": round(dt / 60, 1),
            "best": {k: v for k, v in best.items() if k != "C"}, "worst_pairs": wp,
            "trajectory": traj}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.parse_args()
    os.makedirs(SMOKE_DIR, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); assert sp["mode"] == "board"
    train_tx, held_tx = sp["train_tx"], sp["discover_tx"]
    n_held = len(held_tx)
    lbl_all = {tx: i for i, tx in enumerate(train_tx)}

    rand_tx = scatter_slice(held_tx)
    print(f"random ruler slice (seed {RAND_SEED}, n={len(rand_tx)}, rows "
          f"{sorted(set(int(t.split('-')[0]) for t in rand_tx))})")
    rand_pack = build_eval(tx_data, rand_tx, cap=VAL_CAP, seed0=7000)
    full_pack = build_eval(tx_data, held_tx, cap=256, seed0=1000)
    health_pack = make_health(tx_data, train_tx)

    # baseline bar + worst pairs
    base = load_baseline()
    re, rd = embed_pack(base, rand_pack)
    base_disc = discovery(re, rd, RAND_N)
    fe, fd = embed_pack(base, full_pack)
    bcents = centroids_of(fe, fd, n_held)
    b_med, b_mean, bC = nearest_confuser(bcents)
    bw = worst_pairs_from(bC, 10)
    base_worst = [(i, j, c, f"{held_tx[i]}/{held_tx[j]}") for i, j, c in bw]
    base_h = health(base, health_pack)
    del base; torch.cuda.empty_cache()
    print(f"\n=== BAR (vanilla 0.859) on random slice: ARI={base_disc['ARI']:.3f} "
          f"purity={base_disc['purity']:.3f} K_est={base_disc['K_est']}/{RAND_N} | "
          f"intra={base_h['intra']:.3f} nn_med={b_med:.4f} ===")

    # confuser graph + co-occurrence
    bg = load_baseline()
    graph0, ref_top1 = centroid_graph(bg, tx_data, train_tx)
    del bg; torch.cuda.empty_cache()
    rng = np.random.default_rng(SEED)
    ca = np.mean([cooccur_rate(confuser_devices(graph0, N_DEV, rng), ref_top1) for _ in range(20)])
    print(f"confuser-aware co-occurrence={ca:.3f} (frac={CFRAC}) — semi-hard mining has material\n")

    results = []
    for m in MARGINS:
        for lam in LAMBDAS:
            tag = f"m{m}_l{lam}"
            print(f"=== CONFIG {tag} (margin={m}, lambda={lam}) — {STEPS} steps ===")
            results.append(run_config(tx_data, train_tx, lbl_all, graph0, ref_top1, rand_pack,
                                      full_pack, health_pack, n_held, base_worst, m, lam, tag))

    # sweep table
    print(f"\n{'='*72}\nSWEEP — random-slice ARI (BAR vanilla {base_disc['ARI']:.3f})\n{'='*72}")
    print(f"{'config':>12} {'ARI':>7} {'dARI':>7} {'purity':>7} {'K_est':>6} {'intra':>7} "
          f"{'nn_med':>7} {'rx_gap':>7} {'intra>=.83':>10}")
    bar = base_disc["ARI"]
    best_cfg = None
    for r in results:
        b = r["best"]; d = b["ARI"] - bar; gate = b["intra"] >= 0.83
        print(f"{('m'+str(r['m'])+' l'+str(r['lambda'])):>12} {b['ARI']:>7.3f} {d:>+7.3f} "
              f"{b['purity']:>7.3f} {b['K_est']:>6} {b['intra']:>7.3f} {b['nn_med']:>7.4f} "
              f"{b['rx_gap']:>7.3f} {'YES' if gate else 'no':>10}")
        if gate and (best_cfg is None or b["ARI"] > best_cfg["best"]["ARI"]):
            best_cfg = r

    print(f"\n=== DECISION ===")
    print(f"  bar (vanilla random-slice ARI) = {bar:.3f}")
    if best_cfg and best_cfg["best"]["ARI"] > bar + 0.02:
        b = best_cfg["best"]
        verdict = (f"WIN: config m{best_cfg['m']} lambda{best_cfg['lambda']} -> ARI {b['ARI']:.3f} "
                   f"(+{b['ARI']-bar:.3f}) with intra {b['intra']:.3f}>=0.83 -> FULL RUN on scattered split next.")
    else:
        top = max(results, key=lambda r: r["best"]["ARI"])["best"]["ARI"]
        verdict = (f"NO GAIN: best ARI {top:.3f} vs bar {bar:.3f} across the whole sweep "
                   f"(or intra<0.83) -> additive margin also doesn't move discovery. STOP & rethink "
                   f"(window length / is ~0.44 near the data ceiling) — do NOT draft a third loss.")
    print(f"  {verdict}")

    out = {"bar_vanilla": base_disc, "baseline_health": base_h, "baseline_nn_med": b_med,
           "cooccur": float(ca), "margins": MARGINS, "lambdas": LAMBDAS, "steps": STEPS,
           "results": results, "verdict": verdict}
    with open(REPORT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {REPORT}\nCHECKPOINT — smoke sweep only. best_model.pt UNTOUCHED. No full run, no drones.")


if __name__ == "__main__":
    main()
