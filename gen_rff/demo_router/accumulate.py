"""TRACK ACCUMULATOR: per track, mean-pool + L2 at N* windows -> one track embedding.

Emits a `partial` flag if the track ended before N* (discovery uses only full-N tracks;
partials are logged). Also mean-pools the fallback tier's classical-19 aux (used only by the
UNKNOWN group at the base station). DEMO-SIDE — NOT PAPER RESULTS.
"""
import numpy as np

NSTAR = 120


def accumulate(deep512, aux19=None, nstar=NSTAR):
    """deep512 [n,512] (+ aux19 [n,19]) -> (track_emb[512] L2-normed, aux_mean[19] or None,
    n_used, partial)."""
    n = deep512.shape[0]
    use = min(n, nstar)
    emb = deep512[:use].mean(0)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    aux = aux19[:use].mean(0) if aux19 is not None else None
    return emb.astype(np.float32), aux, int(use), bool(n < nstar)
