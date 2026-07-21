"""Validate the discovery val-ARI RULER before the triplet pivot — checkpoints only, NO training.

v2's val-ARI was flat (0.182->0.176) but flat EVEN in the pure-vanilla-SupCon opening phase —
the same loss that produced the working 0.859 encoder. So the metric itself is suspect. The v2
val slice was WORST-PAIR-ONLY (16 adversarial cos~0.999 devices) and may be SATURATING.

This re-measures discovery on a RANDOM, grid-scattered held-out slice (NOT confuser-selected)
for BOTH encoders, and re-checks the worst-pair slice for contrast:
  - vanilla baseline  : retrain_best/best_model.pt   (gap 0.859)
  - v2 hard-neg smoke : discover/hardneg_smoke_best.pt
Reports ARI / NMI / purity / K_est-vs-true + median nn-confuser per slice per encoder.

    python3 discover/validate_ruler.py
Cache/checkpoints only. IDs scoring-only. Slices seed-fixed. No drones/drive.
"""
import os, sys, json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rffp.models import RFEncoder
from rffp.data import wisig_manytx as W
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN_DIR    = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR    = os.path.join(RUN_DIR, "discover")
SPLIT_OLD  = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT  = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
V2_CKPT    = os.path.join(OUT_DIR, "hardneg_smoke_best.pt")
REPORT     = os.path.join(OUT_DIR, "validate_ruler_report.json")

CAP, MCS = 160, 15        # SAME config as the v2 val ruler under test
RAND_N, RAND_SEED = 18, 123


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def purity(true, pred):
    if len(true) == 0:
        return float("nan")
    tot = 0
    for c in np.unique(pred):
        tot += np.unique(true[pred == c], return_counts=True)[1].max()
    return tot / len(true)


def load_enc(path):
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(path, map_location="cuda", weights_only=True), strict=True)
    m.eval()
    return m


def build_pack(tx_data, devs, cap=CAP, seed0=0):
    times, dids = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        idx = np.random.default_rng(seed0 + di).integers(0, iq.shape[0], size=min(cap, iq.shape[0]))
        for k in idx:
            times.append(W.standardize(iq[k].T.copy())); dids.append(di)
    t = np.stack(times).astype(np.float32)
    return (torch.from_numpy(t).cuda(), torch.from_numpy(W.compute_stft_batch(t)).cuda()), np.array(dids)


@torch.no_grad()
def embed(model, pack, batch=1024):
    (xt, xs), dids = pack
    e = np.empty((xt.shape[0], 128), dtype=np.float32)
    for s in range(0, xt.shape[0], batch):
        with torch.amp.autocast('cuda'):
            e[s:s + batch] = model(xt[s:s + batch], xs[s:s + batch]).float().cpu().numpy()
    return e, dids


def discovery(embs, dids, n_true):
    pred = HDBSCAN(min_cluster_size=MCS, metric="euclidean", n_jobs=-1, copy=True).fit_predict(embs)
    keep = pred != -1
    k = len(np.unique(pred[keep]))
    noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return {"ARI": 0.0, "NMI": 0.0, "purity": float(purity(dids, pred)),
                "K_est": k, "K_true": n_true, "noise": noise}
    return {"ARI": float(adjusted_rand_score(dids[keep], pred[keep])),
            "NMI": float(normalized_mutual_info_score(dids[keep], pred[keep])),
            "purity": float(purity(dids, pred)), "K_est": k, "K_true": n_true, "noise": noise}


def med_nn_confuser(embs, dids, n_dev):
    cents = np.stack([unit(embs[dids == d].mean(0)) for d in range(n_dev)])
    C = cents @ cents.T; np.fill_diagonal(C, -2.0)
    return float(np.median(C.max(1)))


def scatter_slice(held_tx, n, seed):
    """Round-robin across grid rows -> grid-scattered, NOT confuser-selected."""
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


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); held_tx = sp["discover_tx"]
    print(f"held-out devices: {len(held_tx)}")

    # ── worst-pair slice (reproduce v2's adversarial slice exactly) ──
    base = load_enc(BASE_CKPT)
    full_pack = build_pack(tx_data, held_tx, cap=256, seed0=1000)
    be, bd = embed(base, full_pack)
    cents = np.stack([unit(be[bd == d].mean(0)) for d in range(len(held_tx))])
    C = cents @ cents.T; np.fill_diagonal(C, -2.0)
    iu, ju = np.triu_indices(len(held_tx), 1)
    order = np.argsort(C[iu, ju])[::-1][:10]
    wp_dev = sorted(set([int(iu[o]) for o in order] + [int(ju[o]) for o in order]))[:16]
    wp_tx = [held_tx[i] for i in wp_dev]
    del base; torch.cuda.empty_cache()

    # ── random grid-scattered slice ──
    rand_tx = scatter_slice(held_tx, RAND_N, RAND_SEED)
    rows = sorted(set(int(t.split("-")[0]) for t in rand_tx))
    print(f"\nRANDOM slice (seed={RAND_SEED}, n={len(rand_tx)}, rows {rows}):\n  {rand_tx}")
    print(f"WORST-PAIR slice (n={len(wp_tx)}):\n  {wp_tx}")

    slices = {"random": (rand_tx, build_pack(tx_data, rand_tx, seed0=7000)),
              "worst_pair": (wp_tx, build_pack(tx_data, wp_tx, seed0=5000))}
    encs = {"vanilla_0.859": BASE_CKPT, "v2_hardneg": V2_CKPT}

    results = {}
    print(f"\n=== DISCOVERY on each slice (HDBSCAN mcs={MCS}, cap={CAP}/dev) ===")
    print(f"{'encoder':>15} {'slice':>11} {'ARI':>7} {'NMI':>7} {'purity':>7} "
          f"{'K_est':>6} {'K_true':>6} {'noise':>6} {'nn_med':>7}")
    for ename, epath in encs.items():
        m = load_enc(epath); results[ename] = {}
        for sname, (stx, spack) in slices.items():
            e, d = embed(m, spack)
            disc = discovery(e, d, len(stx))
            nnm = med_nn_confuser(e, d, len(stx))
            disc["nn_med"] = nnm
            results[ename][sname] = disc
            print(f"{ename:>15} {sname:>11} {disc['ARI']:>7.3f} {disc['NMI']:>7.3f} "
                  f"{disc['purity']:>7.3f} {disc['K_est']:>6} {disc['K_true']:>6} "
                  f"{disc['noise']*100:>5.0f}% {nnm:>7.4f}")
        del m; torch.cuda.empty_cache()

    # ── interpretation ──
    v_rand = results["vanilla_0.859"]["random"]["ARI"]
    v_wp   = results["vanilla_0.859"]["worst_pair"]["ARI"]
    h_rand = results["v2_hardneg"]["random"]["ARI"]
    h_wp   = results["v2_hardneg"]["worst_pair"]["ARI"]
    print(f"\n=== INTERPRETATION ===")
    print(f"  vanilla ARI: random={v_rand:.3f}  worst-pair={v_wp:.3f}")
    print(f"  v2      ARI: random={h_rand:.3f}  worst-pair={h_wp:.3f}")
    saturating = v_rand > v_wp + 0.10
    if saturating:
        print(f"  -> worst-pair slice WAS saturating (vanilla {v_wp:.3f} << {v_rand:.3f} on random).")
    if h_rand > v_rand + 0.02:
        verdict = "v2 BEATS vanilla on the trustworthy random slice -> reweighting DID help discovery; pivot calculus changes."
    elif (v_rand < 0.10) and (h_rand < 0.10):
        verdict = "BOTH encoders ~flat on the RANDOM slice too -> the discovery metric/pipeline can't register improvement; FIX THE METRIC before any pivot."
    elif v_rand >= 0.10 and abs(h_rand - v_rand) <= 0.02:
        verdict = ("vanilla healthy on random, v2 no better -> reweighting genuinely didn't move discovery; "
                   "PIVOT TO TRIPLET with a now-validated ruler"
                   + (" (and worst-pair val metric was saturating/untrustworthy)." if saturating else "."))
    else:
        verdict = "MIXED — inspect the table; no clean interpretation."
    print(f"  VERDICT: {verdict}")

    with open(REPORT, "w") as f:
        json.dump({"config": {"cap": CAP, "mcs": MCS, "rand_n": RAND_N, "rand_seed": RAND_SEED},
                   "random_slice": rand_tx, "worst_pair_slice": wp_tx,
                   "results": results, "saturating_worstpair": bool(saturating),
                   "verdict": verdict}, f, indent=2)
    print(f"\nsaved -> {REPORT}")
    print("CHECKPOINT — measurement only, no training, no drones.")


if __name__ == "__main__":
    main()
