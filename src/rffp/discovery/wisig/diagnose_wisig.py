"""Three diagnostics on the WiSig 41-held-out discovery failure.

ALL off the cached 128-D embeddings (runs/.../discover/discover_embeddings.npz).
NO retraining, NO new embedding. Board labels are SCORING-ONLY. M100/ORACLE/DRFF untouched.

  D1  within-board spread  — is each board a tight cone or a diffuse cloud?
  D2  confusion structure  — most-confusable board pairs + source-board correlation.
  D3  burst-level test     — device-BLIND consecutive bursts, does mean-pool tighten + cluster?

    python3 discover/diagnose_wisig.py
"""
import os, sys, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
EMB_CACHE = os.path.join(RUN_DIR, "discover", "discover_embeddings.npz")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
OUT_JSON  = os.path.join(RUN_DIR, "discover", "diagnose_report.json")
SEED      = 42

from sklearn.cluster import HDBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             v_measure_score)


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def purity(true, pred):
    if len(true) == 0:
        return float("nan")
    tot = 0
    for c in np.unique(pred):
        m = pred == c
        _, cnts = np.unique(true[m], return_counts=True)
        tot += cnts.max()
    return tot / len(true)


def score(true, pred):
    return {"ARI": adjusted_rand_score(true, pred),
            "NMI": normalized_mutual_info_score(true, pred),
            "Vmeasure": v_measure_score(true, pred),
            "purity": purity(true, pred)}


def hist_line(vals, edges):
    h, _ = np.histogram(vals, bins=edges)
    return h


def main():
    z = np.load(EMB_CACHE)
    emb, true, tx = z["emb"].astype(np.float32), z["true"], z["tx"]
    sp = json.load(open(SPLIT))
    n_true = len(tx)
    src_board = np.array([t.split("-")[0] for t in tx])       # per-tx source board
    boards = sp["held_out_boards"]
    print(f"loaded {len(emb)} signals, {n_true} boards, source boards {boards}")
    print(f"embeddings already unit-norm: mean={np.linalg.norm(emb,axis=1).mean():.4f}\n")

    # per-board centroids (unit direction of the mean)
    cents = np.stack([unit(emb[true == i].mean(0)) for i in range(n_true)])   # [41,128]

    # ───────────── DIAGNOSTIC 1 — within-board spread ─────────────
    print("=== DIAGNOSTIC 1 — within-board spread (cosine to own centroid) ===")
    rows = []
    for i in range(n_true):
        m = true == i
        cos = emb[m] @ cents[i]                                # cosine to own centroid
        rows.append((tx[i], src_board[i], int(m.sum()),
                     float(cos.mean()), float(cos.std()), float(np.percentile(cos, 10))))
    rows.sort(key=lambda r: r[3])                              # loosest first
    print(f"{'tx':>8} {'src':>4} {'n':>5} {'mean':>7} {'std':>6} {'p10':>7}")
    for r in rows[:8]:
        print(f"{r[0]:>8} {r[1]:>4} {r[2]:>5} {r[3]:>7.3f} {r[4]:>6.3f} {r[5]:>7.3f}   <- loosest")
    print("   ...")
    for r in rows[-5:]:
        print(f"{r[0]:>8} {r[1]:>4} {r[2]:>5} {r[3]:>7.3f} {r[4]:>6.3f} {r[5]:>7.3f}   <- tightest")

    means = np.array([r[3] for r in rows])
    edges = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
    h = hist_line(means, edges)
    print("\n  histogram of per-board MEAN intra-cosine:")
    for lo, hi, c in zip(edges[:-1], edges[1:], h):
        print(f"    [{lo:.2f},{hi:.2f}): {'#'*int(c)} {c}")
    print(f"\n  across 41 boards: mean-of-means={means.mean():.3f}  "
          f"median={np.median(means):.3f}  min={means.min():.3f}  max={means.max():.3f}")
    n_tight = int((means >= 0.90).sum()); n_loose = int((means < 0.80).sum())
    print(f"  boards >=0.90 (tight): {n_tight}/{n_true}   boards <0.80 (loose): {n_loose}/{n_true}")
    verdict1 = ("UNIFORMLY LOOSE -> training-objective problem"
                if means.mean() < 0.82 and means.std() < 0.06 else
                "FEW BAD BOARDS dragging a tight majority -> data/leakage problem"
                if n_tight >= n_true * 0.6 else "MIXED")
    print(f"  VERDICT D1: {verdict1}")

    # ───────────── DIAGNOSTIC 2 — confusion + source-board ─────────────
    print("\n=== DIAGNOSTIC 2 — between-board confusion (nearest other centroid) ===")
    C = cents @ cents.T
    np.fill_diagonal(C, -1.0)
    pairs = []
    for i in range(n_true):
        j = int(C[i].argmax())
        pairs.append((tx[i], src_board[i], tx[j], src_board[j], float(C[i, j])))
    # unique unordered pairs ranked by cosine
    seen, uniq = set(), []
    for a, sa, b, sb, c in sorted(pairs, key=lambda p: -p[4]):
        key = tuple(sorted((a, b)))
        if key in seen:
            continue
        seen.add(key); uniq.append((a, sa, b, sb, c))
    print(f"  top-12 most-confusable board pairs (centroid cosine):")
    print(f"{'board A':>8} {'src':>4}   {'board B':>8} {'src':>4}   {'cos':>6}  {'sameSrc':>7}")
    for a, sa, b, sb, c in uniq[:12]:
        print(f"{a:>8} {sa:>4}   {b:>8} {sb:>4}   {c:>6.3f}  {'YES' if sa==sb else '':>7}")
    nn_cos = np.array([p[4] for p in pairs])
    print(f"\n  nearest-other-centroid cosine: mean={nn_cos.mean():.3f} "
          f"median={np.median(nn_cos):.3f} max={nn_cos.max():.3f} min={nn_cos.min():.3f}")
    n_high = int((nn_cos > 0.5).sum())
    print(f"  boards whose nearest other centroid cos>0.5 (mergeable): {n_high}/{n_true}")
    same_src_frac = np.mean([1.0 if sa == sb else 0.0 for _, sa, _, sb, _ in pairs])
    print(f"  fraction of nearest-neighbour pairs sharing a SOURCE board: {same_src_frac:.2f}")
    # per-source-board: mean intra-cosine + mean nearest-other-cos of its members
    print(f"\n  per SOURCE board ({len(boards)} held-out boards):")
    print(f"{'src':>4} {'n_tx':>5} {'mean_intra':>11} {'mean_nn_cos':>12}")
    src_rows = []
    for sb in boards:
        idx = [i for i in range(n_true) if src_board[i] == sb]
        mi = float(np.mean([means[i] for i in idx]))
        mn = float(np.mean([nn_cos[i] for i in idx]))
        src_rows.append((sb, len(idx), mi, mn))
        print(f"{sb:>4} {len(idx):>5} {mi:>11.3f} {mn:>12.3f}")
    confusion_concentrated = uniq[0][4] - uniq[11][4] > 0.15
    print(f"  confusion {'CONCENTRATED in few pairs' if confusion_concentrated else 'DIFFUSE across boards'}")

    # ───────────── DIAGNOSTIC 3 — burst-level (device-blind) ─────────────
    print("\n=== DIAGNOSTIC 3 — burst-level clustering (device-BLIND consecutive) ===")
    burst_results = {}
    for B in (10, 20, 50):
        n_b = len(emb) // B
        # consecutive, NOT by label
        be = np.stack([unit(emb[k*B:(k+1)*B].mean(0)) for k in range(n_b)])      # [n_b,128]
        bt = np.array([np.bincount(true[k*B:(k+1)*B]).argmax() for k in range(n_b)])  # majority label (scoring)
        # within-board spread at burst level
        bcents = np.stack([unit(be[bt == i].mean(0)) if (bt == i).any() else np.zeros(128)
                           for i in range(n_true)])
        intra_b = np.array([float((be[bt == i] @ bcents[i]).mean()) for i in range(n_true) if (bt == i).any()])
        # purity of bursts themselves (how device-pure is consecutive grouping?)
        bpure = np.mean([np.bincount(true[k*B:(k+1)*B]).max() / B for k in range(n_b)])
        print(f"\n  B={B}: {n_b} bursts | burst self-purity={bpure:.3f} | "
              f"burst-level mean intra-cosine={intra_b.mean():.3f} (was {means.mean():.3f} @ signal)")
        print(f"  {'min_cs':>6} {'K_est':>6} {'|K-41|':>7} {'noise%':>7} | "
              f"{'ARI':>6} {'NMI':>6} {'V':>6} {'purity':>7}")
        sweep = {}
        for mcs in (5, 10, 15):
            pred = HDBSCAN(min_cluster_size=mcs, metric="euclidean", n_jobs=-1, copy=True).fit_predict(be)
            k = len(set(pred)) - (1 if -1 in pred else 0)
            noise = float((pred == -1).mean())
            cl = pred != -1
            s = score(bt[cl], pred[cl]) if cl.any() else {x: float('nan') for x in ['ARI','NMI','Vmeasure','purity']}
            sweep[mcs] = {"K": k, "noise": noise, **s}
            print(f"  {mcs:>6} {k:>6} {abs(k-n_true):>7} {noise*100:>6.1f}% | "
                  f"{s['ARI']:>6.3f} {s['NMI']:>6.3f} {s['Vmeasure']:>6.3f} {s['purity']:>7.3f}")
        burst_results[B] = {"n_bursts": n_b, "burst_self_purity": float(bpure),
                            "burst_intra_mean": float(intra_b.mean()), "sweep": sweep}

    # ───────────── persist ─────────────
    out = {
        "n_true": n_true, "n_signals": int(len(emb)),
        "D1": {"per_board_mean_intra": means.tolist(),
               "mean_of_means": float(means.mean()), "median": float(np.median(means)),
               "min": float(means.min()), "max": float(means.max()),
               "n_tight_ge0.90": n_tight, "n_loose_lt0.80": n_loose, "verdict": verdict1},
        "D2": {"top_pairs": [{"a": a, "src_a": sa, "b": b, "src_b": sb, "cos": c}
                             for a, sa, b, sb, c in uniq[:12]],
               "nn_cos_mean": float(nn_cos.mean()), "nn_cos_max": float(nn_cos.max()),
               "n_mergeable_gt0.5": n_high, "same_src_frac": float(same_src_frac),
               "per_source": [{"src": s, "n_tx": n, "mean_intra": mi, "mean_nn_cos": mn}
                              for s, n, mi, mn in src_rows],
               "concentrated": bool(confusion_concentrated)},
        "D3": burst_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")
    print("\nCHECKPOINT — diagnostics only. No fix implemented.")


if __name__ == "__main__":
    main()
