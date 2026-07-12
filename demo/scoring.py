"""Scoring (evaluation only).

THE ONLY module that consumes ground-truth emitter identity. It joins the base station's
predicted fingerprint_id (per track_id) with the replay world's ground_truth (per track_id)
to compute ARI, correct-K, a confusion matrix, and a human-readable association table.

Nothing here feeds back into the discovery path.

DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.metrics import adjusted_rand_score


def score_run(associated_messages, ground_truth, K_est, K_true):
    """Return dict: ari, K_est, K_true, correct_k, confusion, association_table."""
    tids = [m["track_id"] for m in associated_messages]
    pred = [m["fingerprint_id"] for m in associated_messages]
    true = [ground_truth[t] for t in tids]

    pi = np.unique(pred, return_inverse=True)[1]
    ti = np.unique(true, return_inverse=True)[1]
    ari = float(adjusted_rand_score(ti, pi))

    # confusion: fingerprint_id -> {true_emitter: count}
    confusion = defaultdict(lambda: defaultdict(int))
    for p, t in zip(pred, true):
        confusion[p][t] += 1

    # association table: per fingerprint, which receiver-tracks (and true emitters) grouped
    table = OrderedDict()
    for m in associated_messages:
        fid = m["fingerprint_id"]
        table.setdefault(fid, []).append(dict(
            track_id=m["track_id"], receiver_id=m["receiver_id"],
            true_emitter=ground_truth[m["track_id"]], n_windows=m["n_windows"]))

    return dict(
        ari=round(ari, 4),
        K_est=int(K_est), K_true=int(K_true),
        correct_k=bool(K_est == K_true),
        confusion={p: dict(v) for p, v in confusion.items()},
        association_table=table,
    )


def format_association_table(table):
    lines = []
    for fid, rows in table.items():
        emitters = sorted({r["true_emitter"] for r in rows})
        lines.append(f"  {fid}: {len(rows)} tracks | true emitters grouped = {emitters}")
        for r in sorted(rows, key=lambda x: (x["true_emitter"], str(x["receiver_id"]))):
            lines.append(f"      {r['track_id']:<40s} {r['receiver_id']} "
                         f"[{r['true_emitter']}] n_win={r['n_windows']}")
    return "\n".join(lines)
