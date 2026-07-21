"""Locked open-world discovery scorer (the frozen evaluation protocol).

This is the single source of truth for how a clustering prediction is graded against
ground truth. The **locked rule** (`score_locked`) treats every HDBSCAN noise point
(label ``-1``) as its own singleton cluster, then computes ARI / NMI / purity over ALL
points — so a method cannot inflate its score by dumping hard emitters into "noise".
See docs/EVAL_PROTOCOL.md §2 for the flattery example that motivated locking this rule.

Vendored from the Phase-2 integrity harness (``summer_work/results/step2_integrity``);
only the pure numpy/sklearn primitives are kept here (no checkpoint/data dependencies),
so it is import-safe and reusable across every benchmark.

Public API:
    unitrows(M)                        L2-normalize rows
    purity_std(true, pred)             cluster purity over the given support
    relabel_noise_as_singletons(pred)  -1 -> distinct singleton labels
    score_current(pred, true)          legacy rule (noise excluded) — for comparison only
    score_locked(pred, true)           THE locked rule (noise = singletons)
    hdbscan_pred(bp, mcs=15)           HDBSCAN cluster prediction
    bursts_coherent(emb, N)            consecutive-window burst pooling (N-window mean)
    bursts_scattered(emb, rx, date, N) decorrelated burst pooling (distinct rx/date cells)
    build_slice_bursts(cache, slice_tx, kind, N)
"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BURST_N = 10  # default windows pooled per burst (see docs/EVAL_PROTOCOL.md §3)


def unitrows(M):
    """L2-normalize each row of M."""
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def purity_std(true, pred):
    """Standard purity over the support of (true, pred); each distinct pred label is a cluster."""
    N = len(true)
    if N == 0:
        return float("nan")
    return sum(np.unique(true[pred == c], return_counts=True)[1].max()
               for c in np.unique(pred)) / N


def relabel_noise_as_singletons(pred):
    """Map every -1 (HDBSCAN noise) point to its own fresh singleton cluster label."""
    p = pred.copy()
    nm = p == -1
    if nm.any():
        start = (p.max() + 1) if (p >= 0).any() else 0
        p[nm] = np.arange(start, start + nm.sum())
    return p


def score_current(pred, true):
    """Legacy behavior (noise excluded from ARI/NMI). Kept only for side-by-side comparison."""
    keep = pred != -1
    k = len(np.unique(pred[keep]))
    noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return dict(ARI=0.0, NMI=0.0, purity=float(purity_std(true, pred)), K_est=int(k), noise=noise)
    return dict(ARI=float(adjusted_rand_score(true[keep], pred[keep])),
                NMI=float(normalized_mutual_info_score(true[keep], pred[keep])),
                purity=float(purity_std(true, pred)), K_est=int(k), noise=noise)


def score_locked(pred, true):
    """LOCKED rule: each -1 point becomes its own singleton; ARI/NMI/purity over ALL points."""
    p = relabel_noise_as_singletons(pred)
    k = len(np.unique(pred[pred != -1]))          # real (non-noise) cluster count, reported as-is
    noise = float((pred == -1).mean())
    return dict(ARI=float(adjusted_rand_score(true, p)),
                NMI=float(normalized_mutual_info_score(true, p)),
                purity=float(purity_std(true, p)), K_est=int(k), noise=noise)


def hdbscan_pred(bp, mcs=15):
    """HDBSCAN prediction (euclidean) at a given min_cluster_size."""
    return HDBSCAN(min_cluster_size=mcs, metric="euclidean", copy=True).fit_predict(bp)


def bursts_coherent(emb, N=BURST_N):
    """Coherent burst pooling: mean of N consecutive windows, then L2-normalized."""
    nb = emb.shape[0] // N
    return unitrows(emb[:nb * N].reshape(nb, N, -1).mean(1))


def bursts_scattered(emb, rx, date, N=BURST_N, n_bursts=None, seed=0):
    """Decorrelated burst pooling: each burst = N windows from N DISTINCT (rx, date) cells."""
    cells = {}
    for i, (r, d) in enumerate(zip(rx.tolist(), date.tolist())):
        cells.setdefault((int(r), int(d)), []).append(i)
    keys = list(cells.keys())
    rng = np.random.default_rng(seed)
    if n_bursts is None:
        n_bursts = emb.shape[0] // N
    if len(keys) < N:
        return None, len(keys)
    out = []
    for _ in range(n_bursts):
        ck = rng.choice(len(keys), size=N, replace=False)
        idx = [cells[keys[c]][rng.integers(0, len(cells[keys[c]]))] for c in ck]
        out.append(emb[idx].mean(0))
    return unitrows(np.stack(out)), len(keys)


def bursts_p2_multidate(emb, rx, date, N=BURST_N, n_bursts=240, seed=0):
    """P2 burst: same receiver, N windows spread across all available dates (max date decoherence).

    Vendored from the Phase-2b decoherence study. Returns (bursts, count, mean_dates_per_burst).
    """
    rng = np.random.default_rng(seed)
    by_rx = {}
    for i, (r, d) in enumerate(zip(rx.tolist(), date.tolist())):
        by_rx.setdefault(int(r), {}).setdefault(int(d), []).append(i)
    usable = {r: dd for r, dd in by_rx.items()
              if len(dd) >= 2 and sum(len(v) for v in dd.values()) >= N}
    if not usable:
        return None, 0, 0
    rxs = list(usable.keys())
    out = []; dates_per_burst = []
    for _ in range(n_bursts):
        r = rxs[rng.integers(0, len(rxs))]
        dd = usable[r]; dts = list(dd.keys())
        alloc = np.zeros(len(dts), dtype=int)
        for k in range(N):
            alloc[k % len(dts)] += 1           # balanced across dates: e.g. 3,3,2,2
        idx = []
        for j, dt in enumerate(dts):
            pool = dd[dt]
            sel = rng.choice(len(pool), size=min(alloc[j], len(pool)),
                             replace=(len(pool) < alloc[j]))
            idx += [pool[s] for s in sel]
        allpool = [i for v in dd.values() for i in v]
        while len(idx) < N:
            idx.append(allpool[rng.integers(0, len(allpool))])
        out.append(emb[idx[:N]].mean(0)); dates_per_burst.append(len(dts))
    return unitrows(np.stack(out)), len(out), float(np.mean(dates_per_burst))


def build_slice_bursts(cache, slice_tx, kind, N=BURST_N, seed=0):
    """Build (bursts, labels) for a set of tx from a per-tx embedding cache."""
    bp, bl = [], []
    for di, tx in enumerate(slice_tx):
        c = cache[tx]
        if kind == "coherent":
            b = bursts_coherent(c["emb"], N)
        else:
            b, _ = bursts_scattered(c["emb"], c["rx"], c["date"], N, seed=seed)
        bp.append(b); bl.append(np.full(len(b), di))
    return np.concatenate(bp), np.concatenate(bl)
