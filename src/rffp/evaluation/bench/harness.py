"""rffp.evaluation.bench.harness — THE locked scoring harness (docs/EVAL_PROTOCOL §4.3).

Locked rules (unchanged): oracle-K = k-means AND spectral reported SEPARATELY (never max);
HDBSCAN = locked noise rule, MEAN ARI over a fixed mcs grid (no argmax); bursts = mean of
N=10 windows L2-renormalized, balanced to min per-class before scoring.

The scoring primitives (`score_locked`, `hdbscan_pred`, `unitrows`, `bursts_coherent`) are
provided by the vendored, import-safe `rffp.evaluation.scoring` module.
"""
import numpy as np

from rffp.evaluation.scoring import score_locked, hdbscan_pred, unitrows, bursts_coherent   # noqa: F401
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

MCS_WISIG = [13, 15, 17]
MCS_DRFF = [5, 7]


def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def knn_purity(X, y, ks=(1, 5)):
    Xn = unit(X); k = min(max(ks) + 1, len(Xn))
    nn = NearestNeighbors(n_neighbors=k).fit(Xn); _, ind = nn.kneighbors(Xn)
    yi = np.unique(y, return_inverse=True)[1]; out = {}
    for kk in ks:
        nb = ind[:, 1:kk + 1]; out[kk] = float((yi[nb] == yi[:, None]).mean())
    return out


def cos_gap(X, y):
    Xn = unit(X); yi = np.unique(y, return_inverse=True)[1]; intra, inter = [], []
    for a in np.unique(yi):
        mm = yi == a
        if mm.sum() < 2:
            continue
        C = Xn[mm] @ Xn[mm].T; iu = np.triu_indices(mm.sum(), 1)
        intra.append(C[iu].mean()); inter.append((Xn[mm] @ Xn[~mm].T).mean())
    return float(np.mean(intra)), float(np.mean(inter))


def oracle_km_sp(bp, bl, K):
    """oracle-K CEILING: k-means AND spectral, reported SEPARATELY (never max)."""
    yi = np.unique(bl, return_inverse=True)[1]
    km = float(adjusted_rand_score(yi, KMeans(K, n_init=10, random_state=0).fit_predict(bp)))
    try:
        sp = SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                n_neighbors=15).fit_predict(bp)
        sp = float(adjusted_rand_score(yi, sp))
    except Exception:
        sp = float("nan")
    return km, sp


def hdbscan_grid(bp, bl, mcs_grid):
    """locked noise rule per mcs; per-mcs dicts + MEAN ARI (no argmax selection)."""
    per = {mcs: score_locked(hdbscan_pred(bp, mcs), bl) for mcs in mcs_grid}
    mean_ari = float(np.mean([per[m]["ARI"] for m in mcs_grid]))
    return per, mean_ari


def full_score(bp, bl, K, mcs_grid, with_oracle=True):
    bpu = unit(bp)
    per, hdb_mean = hdbscan_grid(bpu, bl, mcs_grid)
    kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl)
    out = dict(hdb_per_mcs={m: per[m]["ARI"] for m in mcs_grid}, hdb_mean_ARI=hdb_mean,
               knn1=kp[1], knn5=kp[5], intra=intra, inter=inter, gap=intra - inter,
               n_bursts=int(len(bl)))
    if with_oracle:
        km, sp = oracle_km_sp(bpu, bl, K)
        out["oracleK_km"] = km; out["oracleK_sp"] = sp
    return out


def balance_arr(bp, bl, seed=0):
    r = np.random.default_rng(seed)
    per = min(int((bl == a).sum()) for a in np.unique(bl)); keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    k = np.array(sorted(keep)); return bp[k], bl[k]
