"""Open-world emitter discovery — IN-DOMAIN validation on WiSig's 41 held-out Tx.

The encoder (runs/.../retrain_best/best_model.pt, gap 0.859) is FROZEN. The 41 held-out
transmitters (5 held-out boards, never seen in training) are clustered WITHOUT their labels;
labels are used for SCORING ONLY. No augmentation at inference.

Stages:
  1 EMBED    — frozen encoder -> per-signal 128-D L2-normalized embeddings.
  2 CLUSTER  — HDBSCAN on RAW embeddings, sweep min_cluster_size {10,15,20,25,30}.
  3 METRICS  — cluster diagnostics + metrics under BOTH noise modes (exclude / nearest-centroid).
  4 BASELINE — parent greedy EmitterRegistry on the SAME embeddings, scored identically.

    python3 discover/discover_wisig.py

Sealed: do NOT touch M100 or ORACLE here — those are the cross-domain transfer step.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W

from sklearn.cluster import HDBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             v_measure_score)

RUN_DIR    = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT_PATH = os.path.join(RUN_DIR, "splits", "split_manytx.json")
CKPT_PATH  = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
OUT_DIR    = os.path.join(RUN_DIR, "discover")
EMB_CACHE  = os.path.join(OUT_DIR, "discover_embeddings.npz")

MCS_SWEEP  = [10, 15, 20, 25, 30]
SEED       = 42


# ───────────────────────── STAGE 1 — EMBED ─────────────────────────
def load_encoder():
    model = RFEncoder().cuda()
    state = torch.load(CKPT_PATH, map_location="cuda", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, f"NON-STRICT LOAD missing={missing} unexpected={unexpected}"
    model.eval()
    print(f"encoder loaded STRICT from {CKPT_PATH} (no missing/unexpected keys)")
    return model


@torch.no_grad()
def embed_discover(model, tx_data, discover_tx, batch=512):
    """All signals from the 41 held-out Tx -> (emb[N,128] f32, labels[N] int, id_map)."""
    id_map = {tx: i for i, tx in enumerate(discover_tx)}      # scoring labels only
    times, labels = [], []
    for tx in discover_tx:
        iq = tx_data[tx]["iq"]                                # [n,256,2]
        for k in range(iq.shape[0]):
            times.append(W.standardize(iq[k].T.copy()))      # [2,256], NO augmentation
            labels.append(id_map[tx])
    time_arr = np.stack(times).astype(np.float32)            # [N,2,256]
    labels = np.array(labels, dtype=np.int64)

    embs = np.empty((time_arr.shape[0], 128), dtype=np.float32)
    for s in range(0, time_arr.shape[0], batch):
        tb = time_arr[s:s + batch]
        sb = W.compute_stft_batch(tb)                        # [b,2,33,13]
        xt = torch.from_numpy(tb).cuda()
        xs = torch.from_numpy(sb).cuda()
        with torch.amp.autocast('cuda'):
            e = model(xt, xs)
        embs[s:s + batch] = e.float().cpu().numpy()
    return embs, labels, id_map


# ───────────────────────── metric helpers ─────────────────────────
def purity(true, pred):
    """Cluster purity over the given points: sum_c max_class(c) / N."""
    if len(true) == 0:
        return float("nan")
    tot = 0
    for c in np.unique(pred):
        m = pred == c
        vals, cnts = np.unique(true[m], return_counts=True)
        tot += cnts.max()
    return tot / len(true)


def score(true, pred):
    return {
        "ARI": adjusted_rand_score(true, pred),
        "NMI": normalized_mutual_info_score(true, pred),
        "Vmeasure": v_measure_score(true, pred),
        "purity": purity(true, pred),
    }


def assign_noise_to_centroids(emb, pred):
    """Assign every -1 point to the nearest (cosine) cluster centroid. Returns new labels."""
    out = pred.copy()
    clusters = [c for c in np.unique(pred) if c != -1]
    if not clusters:
        return out
    cents = np.stack([_unit(emb[pred == c].mean(0)) for c in clusters])   # [K,128]
    noise = np.where(pred == -1)[0]
    if len(noise):
        sims = emb[noise] @ cents.T                          # cosine (unit vectors)
        out[noise] = np.array(clusters)[sims.argmax(1)]
    return out


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


# ───────────────────── STAGE 3 — diagnostics ─────────────────────
def diagnostics(true, pred, n_true):
    """Cluster size distribution, largest-cluster share, and boards silently lost."""
    clustered = pred != -1
    sizes = sorted((int((pred == c).sum()) for c in np.unique(pred) if c != -1), reverse=True)
    n_clustered = int(clustered.sum())
    largest_share = (sizes[0] / n_clustered) if sizes and n_clustered else float("nan")

    # which true boards "own" (are the majority class of) at least one cluster?
    owners = set()
    for c in np.unique(pred):
        if c == -1:
            continue
        m = pred == c
        vals, cnts = np.unique(true[m], return_counts=True)
        owners.add(int(vals[cnts.argmax()]))

    # boards whose own plurality assignment is noise
    lost_to_noise = []
    for d in range(n_true):
        md = true == d
        labs, cnts = np.unique(pred[md], return_counts=True)
        if labs[cnts.argmax()] == -1:
            lost_to_noise.append(d)

    boards_no_cluster = [d for d in range(n_true) if d not in owners]
    return {
        "sizes": sizes, "n_clusters": len(sizes), "largest_share": largest_share,
        "boards_owning_a_cluster": len(owners),
        "boards_no_majority_cluster": len(boards_no_cluster),
        "boards_plurality_is_noise": len(lost_to_noise),
    }


# ───────────────────── STAGE 4 — greedy baseline ─────────────────────
def greedy_registry(emb, rng, threshold, ema_alpha=0.1):
    """Parent EmitterRegistry logic: streaming single-linkage-ish threshold matcher.
    Returns per-point emitter labels in original order."""
    order = rng.permutation(len(emb))
    refs, counts = [], []          # list of unit centroids
    assign = np.full(len(emb), -1, dtype=np.int64)
    for idx in order:
        e = emb[idx]
        if not refs:
            refs.append(e.copy()); counts.append(1); assign[idx] = 0
            continue
        sims = np.stack(refs) @ e
        b = int(sims.argmax())
        if sims[b] >= threshold:
            refs[b] = _unit((1 - ema_alpha) * refs[b] + ema_alpha * e)
            counts[b] += 1; assign[idx] = b
        else:
            refs.append(e.copy()); counts.append(1); assign[idx] = len(refs) - 1
    return assign, len(refs)


def calibrate_threshold(emb, true, rng, n_pairs=20000):
    """Midpoint of same-Tx vs diff-Tx cosine — same scheme as the parent pipeline."""
    N = len(emb)
    a = rng.integers(0, N, n_pairs); b = rng.integers(0, N, n_pairs)
    m = a != b
    a, b = a[m], b[m]
    sims = np.einsum("ij,ij->i", emb[a], emb[b])
    same = true[a] == true[b]
    pos, neg = float(sims[same].mean()), float(sims[~same].mean())
    return 0.5 * (pos + neg), pos, neg


# ─────────────────────────────── main ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-cache", action="store_true",
                    help="load STAGE-1 embeddings from cache, skip re-embedding")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    sp = json.load(open(SPLIT_PATH))
    assert sp["mode"] == "board"
    discover_tx = sp["discover_tx"]
    n_true = len(discover_tx)
    print(f"discover set: {n_true} held-out Tx from boards {sp['held_out_boards']}")

    # STAGE 1
    print("\n=== STAGE 1 — EMBED ===")
    if args.from_cache and os.path.exists(EMB_CACHE):
        z = np.load(EMB_CACHE)
        emb, true = z["emb"].astype(np.float32), z["true"]
        print(f"  loaded cached embeddings from {EMB_CACHE}")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        model = load_encoder()
        tx_data, _ = W.load_manytx()
        emb, true, id_map = embed_discover(model, tx_data, discover_tx)
        np.savez(EMB_CACHE, emb=emb, true=true, tx=np.array(discover_tx))
    norms = np.linalg.norm(emb, axis=1)
    print(f"  #signals={len(emb)}  #true devices={len(np.unique(true))} (expect {n_true})  "
          f"emb shape={emb.shape}")
    print(f"  embedding norm: mean={norms.mean():.4f} std={norms.std():.2e} "
          f"(L2-normalized sanity {'OK' if abs(norms.mean()-1) < 1e-3 else 'FAIL'})")
    sig_per_tx = [int((true == i).sum()) for i in range(n_true)]
    print(f"  signals/Tx: min={min(sig_per_tx)} median={int(np.median(sig_per_tx))} "
          f"max={max(sig_per_tx)}")

    # STAGE 2 + 3
    print("\n=== STAGE 2 — HDBSCAN sweep (raw 128-D, euclidean on unit vectors) ===")
    print(f"{'min_cs':>6} {'K_est':>6} {'|K-41|':>7} {'noise%':>7} | "
          f"{'ARI':>6} {'NMI':>6} {'V':>6} {'purity':>7}  (noise EXCLUDED)")
    rows = {}
    for mcs in MCS_SWEEP:
        t0 = time.time()
        pred = HDBSCAN(min_cluster_size=mcs, metric="euclidean",
                       n_jobs=-1, copy=True).fit_predict(emb)
        dt = time.time() - t0
        k = len(set(pred)) - (1 if -1 in pred else 0)
        noise = float((pred == -1).mean())
        cl = pred != -1
        s_excl = score(true[cl], pred[cl]) if cl.any() else {k: float('nan') for k in ['ARI','NMI','Vmeasure','purity']}
        pred_assign = assign_noise_to_centroids(emb, pred)
        s_asgn = score(true, pred_assign)
        diag = diagnostics(true, pred, n_true)
        rows[mcs] = {"pred": pred, "K": k, "noise": noise, "excl": s_excl,
                     "asgn": s_asgn, "diag": diag}
        print(f"{mcs:>6} {k:>6} {abs(k-n_true):>7} {noise*100:>6.1f}% | "
              f"{s_excl['ARI']:>6.3f} {s_excl['NMI']:>6.3f} {s_excl['Vmeasure']:>6.3f} "
              f"{s_excl['purity']:>7.3f}   ({dt:.0f}s)", flush=True)

    print("\n=== STAGE 3 — noise-assigned metrics (every point -> nearest centroid) ===")
    print(f"{'min_cs':>6} {'noise%':>7} | {'ARI':>6} {'NMI':>6} {'V':>6} {'purity':>7}  (ALL points)")
    for mcs in MCS_SWEEP:
        r = rows[mcs]
        print(f"{mcs:>6} {r['noise']*100:>6.1f}% | {r['asgn']['ARI']:>6.3f} "
              f"{r['asgn']['NMI']:>6.3f} {r['asgn']['Vmeasure']:>6.3f} {r['asgn']['purity']:>7.3f}")

    print("\n=== STAGE 3 — cluster diagnostics (anti-self-deception) ===")
    print(f"{'min_cs':>6} {'K_est':>6} {'largest%':>9} {'own_cluster':>12} "
          f"{'lost(no cluster)':>17} {'plurality=noise':>16}")
    for mcs in MCS_SWEEP:
        d = rows[mcs]["diag"]
        ls = d["largest_share"] * 100 if d["largest_share"] == d["largest_share"] else float('nan')
        print(f"{mcs:>6} {d['n_clusters']:>6} {ls:>8.1f}% {d['boards_owning_a_cluster']:>9}/{n_true} "
              f"{d['boards_no_majority_cluster']:>14}/{n_true} {d['boards_plurality_is_noise']:>13}/{n_true}")
    # size distribution for the most-recovering config (closest K to 41)
    best_k = min(MCS_SWEEP, key=lambda m: abs(rows[m]["K"] - n_true))
    print(f"\n  cluster-size distribution @ min_cs={best_k} (closest K to {n_true}): "
          f"{rows[best_k]['diag']['sizes']}")

    # STAGE 4
    print("\n=== STAGE 4 — greedy EmitterRegistry baseline (same embeddings) ===")
    rng = np.random.default_rng(SEED)
    thr, pos, neg = calibrate_threshold(emb, true, rng)
    print(f"  calibrated threshold={thr:.4f} (same-Tx cos={pos:.4f}, diff-Tx cos={neg:.4f})")
    g_assign, g_k = greedy_registry(emb, np.random.default_rng(SEED), thr)
    g_score = score(true, g_assign)
    print(f"  emitters discovered={g_k} (true {n_true}, |K-{n_true}|={abs(g_k-n_true)})  "
          f"{'UNDER-MERGE' if g_k < n_true else 'OVER-SPLIT' if g_k > n_true else 'exact'}")
    print(f"  ARI={g_score['ARI']:.3f} NMI={g_score['NMI']:.3f} V={g_score['Vmeasure']:.3f} "
          f"purity={g_score['purity']:.3f}  (all points, no noise concept)")

    # persist
    out = {
        "n_true": n_true, "n_signals": int(len(emb)), "emb_dim": 128,
        "ckpt": CKPT_PATH,
        "sweep": {str(m): {"K": rows[m]["K"], "noise_frac": rows[m]["noise"],
                           "metrics_noise_excluded": rows[m]["excl"],
                           "metrics_noise_assigned": rows[m]["asgn"],
                           "diagnostics": {k: v for k, v in rows[m]["diag"].items() if k != "sizes"},
                           "sizes": rows[m]["diag"]["sizes"]}
                  for m in MCS_SWEEP},
        "greedy_baseline": {"threshold": thr, "same_cos": pos, "diff_cos": neg,
                            "K": g_k, "metrics": g_score},
    }
    with open(os.path.join(OUT_DIR, "discover_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {os.path.join(OUT_DIR, 'discover_report.json')}")
    print("\nCHECKPOINT — operating point NOT auto-picked. Review tables above and choose together.")


if __name__ == "__main__":
    main()
