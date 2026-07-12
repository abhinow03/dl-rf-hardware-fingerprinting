"""Scoring (evaluation only) — THE ONLY module that consumes ground truth.

Joins router `protocol` + base-station `fingerprint_id` (per track_id) with the replay
world's ground_truth (per track_id) to compute routing accuracy, per-protocol-group ARI,
correct-K, and association tables. Nothing here feeds back into router / tiers / base station.
Ground-truth = {track_id: (protocol, device_id)}.

DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from collections import defaultdict, OrderedDict
import numpy as np
from sklearn.metrics import adjusted_rand_score

# map ground-truth protocol -> router label
GT2ROUTE = {"drone": "drone_ocusync", "wifi": "wifi"}


def routing_accuracy(messages, ground_truth, forced_unknown=False):
    """Fraction of tracks routed to the correct protocol. If forced_unknown, correct == 'unknown'."""
    ok = 0
    for m in messages:
        gt_proto = ground_truth[m["track_id"]][0]
        target = "unknown" if forced_unknown else GT2ROUTE[gt_proto]
        ok += int(m["protocol"] == target)
    return ok / max(1, len(messages))


def score_group(assoc_messages, ground_truth, K_est, K_true):
    """Discovery quality within one protocol group (device-identity level)."""
    tids = [m["track_id"] for m in assoc_messages]
    pred = [m["fingerprint_id"] for m in assoc_messages]
    true = [ground_truth[t][1] for t in tids]          # device_id
    ti = np.unique(true, return_inverse=True)[1]
    pi = np.unique(pred, return_inverse=True)[1]
    ari = float(adjusted_rand_score(ti, pi))
    table = OrderedDict()
    for m in assoc_messages:
        table.setdefault(m["fingerprint_id"], []).append(
            dict(track_id=m["track_id"], receiver_id=m["receiver_id"],
                 true_emitter=ground_truth[m["track_id"]][1], n_windows=m["n_windows"]))
    return dict(ari=round(ari, 4), K_est=int(K_est), K_true=int(K_true),
                correct_k=bool(K_est == K_true), n_tracks=len(tids), association_table=table)


def format_association_table(table):
    lines = []
    for fid, rows in table.items():
        ems = sorted({r["true_emitter"] for r in rows})
        lines.append(f"  {fid}: {len(rows)} tracks | true = {ems}")
        for r in sorted(rows, key=lambda x: (x["true_emitter"], str(x["receiver_id"]))):
            lines.append(f"      {r['track_id']:<44s} {r['receiver_id']} [{r['true_emitter']}] n={r['n_windows']}")
    return "\n".join(lines)
