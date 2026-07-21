"""BASE STATION — per-protocol-group discovery. Pool track embeddings PER ROUTED PROTOCOL
GROUP (never cluster across protocols); eigengap K-estimate (or assisted-K) + spectral
partition; assign namespaced global fingerprint_id d<k>.<protocol>.

LABEL-FREE: clusters ONLY on embeddings (deep-512 for wifi/drone; block-scaled 50/50
[deep-512 | classical-19] hybrid for the UNKNOWN fallback group, per FINDINGS F1/F4). Never
sees ground truth, never touches rssi/aoa/tdoa. DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from collections import defaultdict
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian

_SHORT = {"wifi": "wifi", "drone_ocusync": "drone", "unknown": "unk", "ble": "ble"}


def _unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def _zscore(M):
    return (M - M.mean(0)) / (M.std(0) + 1e-8)


def _nn(n):
    return int(min(15, max(3, n // 4)))


def eigengap_k(X, kmax=10):
    n = len(X); K = min(kmax, n - 1)
    if K < 2:
        return 1
    A = kneighbors_graph(X, n_neighbors=min(_nn(n), n - 1), mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    vals = np.sort(np.linalg.eigvalsh(laplacian(A, normed=True).toarray()))
    gaps = {k: vals[k] - vals[k - 1] for k in range(2, K + 1)}
    return int(max(gaps, key=gaps.get))


def _partition(X, K):
    if K >= len(X):
        return np.arange(len(X))
    try:
        return SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                  n_neighbors=min(_nn(len(X)), len(X) - 1)).fit_predict(X)
    except Exception:
        return KMeans(K, n_init=10, random_state=0).fit_predict(X)


def _feature_matrix(protocol, embs, auxes):
    """wifi/drone -> unit(deep512); unknown -> block-scaled 50/50 [deep | classical-19]."""
    D = _unit(np.stack(embs))
    if protocol != "unknown" or auxes[0] is None:
        return D
    A = _zscore(np.stack(auxes)); Dz = _zscore(np.stack(embs))
    return _unit(np.concatenate([0.5 * Dz, 0.5 * A], axis=1))


def associate(messages, aux_by_tid, assisted_K=None):
    """messages: routed contract messages (with 'protocol','embedding','track_id').
    assisted_K: {protocol: K} or None (unassisted eigengap per group).
    Returns (associated_messages, info_by_protocol)."""
    by_proto = defaultdict(list)
    for m in messages:
        by_proto[m["protocol"]].append(m)
    out = []; info = {}
    for proto, grp in by_proto.items():
        embs = [np.asarray(m["embedding"], np.float32) for m in grp]
        auxes = [aux_by_tid.get(m["track_id"]) for m in grp]
        X = _feature_matrix(proto, embs, auxes)
        if assisted_K is not None and proto in assisted_K:
            K = int(assisted_K[proto]); assisted = True
        else:
            K = eigengap_k(X); assisted = False
        labels = _partition(X, K)
        remap = {}; order = []
        short = _SHORT.get(proto, proto)
        for lb in labels:
            if lb not in remap:
                remap[lb] = f"d{len(order)+1}.{short}"; order.append(lb)
        for m, lb in zip(grp, labels):
            mm = dict(m); mm["fingerprint_id"] = remap[lb]; out.append(mm)
        info[proto] = dict(K_est=int(K), n_clusters=int(len(order)), assisted=assisted,
                           n_tracks=len(grp))
    return out, info
