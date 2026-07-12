"""Base-station association (the DISCOVERY path).

Pools track embeddings from all receivers, estimates the number of distinct emitters K
(eigengap; or assisted = operator-supplied K), partitions with the eigengap's spectral
algorithm, and assigns a global fingerprint_id d1..dK to each track.

LABEL-FREE BY CONSTRUCTION: `associate` reads ONLY the `embedding` field of each message.
It never receives ground-truth identity and never touches rssi/aoa/tdoa. This is the audit
surface for G3 -- grep this file: no import of replay ground truth, no label argument.

Estimator + partition are copied verbatim from the locked operating point (Play-1b
est_eigengap): kNN connectivity graph -> normalized Laplacian -> largest eigen-gap ->
SpectralClustering(nearest_neighbors, n_neighbors=15).

DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian


def _unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def _nn(n):
    """kNN-graph neighbor count. The locked recipe uses 15 at operating-point scale
    (n >= 60 tracks); with the demo's small graphs (12-16 tracks) 15 neighbors makes the
    graph nearly complete and washes out cluster structure, so scale down to n//4 while
    keeping 15 whenever the graph is large enough. Preserves the operating point exactly
    for n >= 60; only adapts the tiny demo graphs."""
    return int(min(15, max(3, n // 4)))


def eigengap_k(X, kmax_cap=10):
    """Estimate K by the largest eigen-gap of the normalized Laplacian (locked recipe)."""
    n = len(X)
    Kmax = min(kmax_cap, n - 1)
    A = kneighbors_graph(X, n_neighbors=min(_nn(n), n - 1), mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    L = laplacian(A, normed=True).toarray()
    vals = np.sort(np.linalg.eigvalsh(L))
    gaps = {k: vals[k] - vals[k - 1] for k in range(2, Kmax + 1)}
    return int(max(gaps, key=gaps.get))


def _partition(X, K):
    try:
        return SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                  n_neighbors=min(_nn(len(X)), len(X) - 1)).fit_predict(X)
    except Exception:
        return KMeans(K, n_init=10, random_state=0).fit_predict(X)


def associate(messages, assisted_K=None):
    """Group track messages by fingerprint. Reads ONLY message['embedding'].

    Returns (associated_messages, info):
      associated_messages: deep-ish copies with 'fingerprint_id' (d1..dK) added.
      info: { 'K_est': int, 'assisted': bool }.
    """
    X = _unit(np.array([m["embedding"] for m in messages], dtype=np.float32))
    if assisted_K is not None:
        K = int(assisted_K)
        assisted = True
    else:
        K = eigengap_k(X)
        assisted = False
    labels = _partition(X, K)
    n_clusters = len(np.unique(labels))
    # map raw cluster ids -> stable d1..dK by first appearance
    order, remap = [], {}
    for lb in labels:
        if lb not in remap:
            remap[lb] = f"d{len(order) + 1}"
            order.append(lb)
    out = []
    for m, lb in zip(messages, labels):
        mm = dict(m)
        mm["fingerprint_id"] = remap[lb]
        out.append(mm)
    return out, dict(K_est=int(K), n_clusters=int(n_clusters), assisted=assisted)
