"""Re-score the SAME cached WiSig 41-held-out embeddings against COARSER ground truths
to find the physically-achievable discovery ceiling. NO retraining, off cached embeddings.

  1 board-level   — relabel 41 tx by physical board (5/18/4/16/20), re-score same clustering.
  2 merged-id     — merge tx whose centroids collapse (cos>0.95 / >0.99) via connected
                    components -> "separable identity" GT; re-score against it.
  3 per-board     — how many distinct identities each source board actually resolves into.

Clustering (HDBSCAN sweep + greedy) depends only on embeddings, so predictions are
computed ONCE and cached; only the SCORING ground truth changes.

    python3 discover/rescore_wisig.py
"""
import os, sys, json, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
RUN_DIR    = os.path.join(_SW, "runs", "wisig_supcon_fft64")
EMB_CACHE  = os.path.join(RUN_DIR, "discover", "discover_embeddings.npz")
PRED_CACHE = os.path.join(RUN_DIR, "discover", "rescore_preds.npz")
OUT_JSON   = os.path.join(RUN_DIR, "discover", "rescore_report.json")
MCS_SWEEP  = [10, 15, 20, 25, 30]
SEED       = 42

from sklearn.cluster import HDBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             v_measure_score)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def purity(true, pred):
    if len(true) == 0:
        return float("nan")
    tot = 0
    for c in np.unique(pred):
        _, cnts = np.unique(true[pred == c], return_counts=True)
        tot += cnts.max()
    return tot / len(true)


def score(true, pred):
    return {"ARI": float(adjusted_rand_score(true, pred)),
            "NMI": float(normalized_mutual_info_score(true, pred)),
            "Vmeasure": float(v_measure_score(true, pred)),
            "purity": float(purity(true, pred))}


def greedy_registry(emb, rng, threshold, ema_alpha=0.1):
    order = rng.permutation(len(emb))
    refs = []
    assign = np.full(len(emb), -1, dtype=np.int64)
    for idx in order:
        e = emb[idx]
        if not refs:
            refs.append(e.copy()); assign[idx] = 0; continue
        sims = np.stack(refs) @ e
        b = int(sims.argmax())
        if sims[b] >= threshold:
            refs[b] = unit((1 - ema_alpha) * refs[b] + ema_alpha * e); assign[idx] = b
        else:
            refs.append(e.copy()); assign[idx] = len(refs) - 1
    return assign, len(refs)


def calibrate_threshold(emb, true, rng, n_pairs=20000):
    N = len(emb)
    a = rng.integers(0, N, n_pairs); b = rng.integers(0, N, n_pairs)
    m = a != b; a, b = a[m], b[m]
    sims = np.einsum("ij,ij->i", emb[a], emb[b])
    same = true[a] == true[b]
    return 0.5 * (float(sims[same].mean()) + float(sims[~same].mean()))


def merge_components(cents, thr):
    """Connected components of the centroid-cosine>thr graph among the tx."""
    C = cents @ cents.T
    A = (C > thr).astype(np.int8)
    n_comp, comp = connected_components(csr_matrix(A), directed=False)
    return n_comp, comp        # comp[tx] -> component id


def report_block(title, true_lbl, n_classes, preds, emb, true_tx):
    print(f"\n=== {title} (true K={n_classes}) ===")
    print(f"{'min_cs':>6} {'K_est':>6} {'|K-true|':>9} {'noise%':>7} | "
          f"{'ARI':>6} {'NMI':>6} {'V':>6} {'purity':>7}  (noise EXCLUDED)")
    out = {}
    for mcs in MCS_SWEEP:
        pred = preds[mcs]
        k = len(set(pred)) - (1 if -1 in pred else 0)
        noise = float((pred == -1).mean())
        cl = pred != -1
        s = score(true_lbl[cl], pred[cl])
        out[mcs] = {"K": k, "noise": noise, **s}
        print(f"{mcs:>6} {k:>6} {abs(k-n_classes):>9} {noise*100:>6.1f}% | "
              f"{s['ARI']:>6.3f} {s['NMI']:>6.3f} {s['Vmeasure']:>6.3f} {s['purity']:>7.3f}")
    # greedy
    rng = np.random.default_rng(SEED)
    thr = calibrate_threshold(emb, true_tx, rng)     # threshold from tx pairs (unchanged)
    g_assign, g_k = greedy_registry(emb, np.random.default_rng(SEED), thr)
    g = score(true_lbl, g_assign)
    print(f"  greedy: K={g_k} (true {n_classes}) ARI={g['ARI']:.3f} NMI={g['NMI']:.3f} "
          f"V={g['Vmeasure']:.3f} purity={g['purity']:.3f}")
    return {"sweep": out, "greedy": {"K": g_k, **g}}


def main():
    z = np.load(EMB_CACHE)
    emb, true, tx = z["emb"].astype(np.float32), z["true"], z["tx"]
    n_tx = len(tx)
    src = np.array([t.split("-")[0] for t in tx])
    boards = sorted(set(src), key=lambda b: int(b))
    board_id = {b: i for i, b in enumerate(boards)}
    print(f"loaded {len(emb)} signals, {n_tx} tx, {len(boards)} boards {boards}")

    cents = np.stack([unit(emb[true == i].mean(0)) for i in range(n_tx)])

    # ---- compute/load HDBSCAN predictions ONCE (label-independent) ----
    if os.path.exists(PRED_CACHE):
        zp = np.load(PRED_CACHE)
        preds = {m: zp[f"mcs{m}"] for m in MCS_SWEEP}
        print(f"loaded cached HDBSCAN predictions from {PRED_CACHE}")
    else:
        print("running HDBSCAN sweep (label-independent, cached after)...")
        preds = {}
        for mcs in MCS_SWEEP:
            t0 = time.time()
            preds[mcs] = HDBSCAN(min_cluster_size=mcs, metric="euclidean",
                                 n_jobs=-1, copy=True).fit_predict(emb)
            print(f"  min_cs={mcs} done ({time.time()-t0:.0f}s)", flush=True)
        np.savez(PRED_CACHE, **{f"mcs{m}": preds[m] for m in MCS_SWEEP})

    # ---------- PART 1: board-level ----------
    true_board = np.array([board_id[src[t]] for t in true])
    r_board = report_block("PART 1 — BOARD-LEVEL", true_board, len(boards), preds, emb, true)

    # ---------- PART 2: merged-identity GT ----------
    part2 = {}
    for thr in (0.95, 0.99):
        n_comp, comp = merge_components(cents, thr)
        true_merged = comp[true]
        print(f"\n  merge @ cos>{thr}: {n_tx} tx -> {n_comp} separable identities")
        r = report_block(f"PART 2 — MERGED IDENTITY (cos>{thr})", true_merged, n_comp,
                          preds, emb, true)
        part2[str(thr)] = {"n_identities": int(n_comp), "comp": comp.tolist(), **r}

    # ---------- PART 3: per-source-board resolution ----------
    print("\n=== PART 3 — per-source-board: distinct identities after merge ===")
    print(f"{'board':>5} {'n_tx':>5} {'ids@0.95':>9} {'ids@0.99':>9}")
    per_board = {}
    comp95 = merge_components(cents, 0.95)[1]
    comp99 = merge_components(cents, 0.99)[1]
    for b in boards:
        idx = [i for i in range(n_tx) if src[i] == b]
        ids95 = len(set(comp95[i] for i in idx))
        ids99 = len(set(comp99[i] for i in idx))
        per_board[b] = {"n_tx": len(idx), "ids_0.95": ids95, "ids_0.99": ids99}
        print(f"{b:>5} {len(idx):>5} {ids95:>9} {ids99:>9}")
    tot95 = len(set(comp95)); tot99 = len(set(comp99))
    print(f"  TOTAL distinct identities: {tot95} @0.95, {tot99} @0.99 (from {n_tx} tx)")

    out = {"n_tx": n_tx, "n_signals": int(len(emb)), "boards": boards,
           "part1_board": r_board,
           "part2_merged": part2,
           "part3_per_board": per_board,
           "totals": {"ids_0.95": tot95, "ids_0.99": tot99}}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")
    print("\nCHECKPOINT — re-scoring only, no retrain.")


if __name__ == "__main__":
    main()
