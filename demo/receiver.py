"""Drone-side receiver stage.

Consumes a track record (windows + opaque ids) and produces ONE contract JSON message:
  embed each window (frozen encoder, head-free 512-D) -> mean-pool -> L2-normalize =
  track fingerprint; attach MOCKED rssi/aoa/tdoa from the fixed synthetic geometry;
  class = "unknown". The receiver never assigns fingerprint_id and never sees the airframe
  label (only the emitter's scene index, a physical position, feeds the geometry mock).

DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import geometry
from encoder_frontend import embed_windows, l2


def process_tracks(model, track_records):
    """Embed every track's windows in one batched pass, then emit contract messages.

    Returns list of pre-association messages (contract.RECEIVER_FIELDS)."""
    if not track_records:
        return []
    # batch all windows across tracks through the encoder once
    counts = [t["n_windows"] for t in track_records]
    Xt = np.concatenate([t["Xt"] for t in track_records], axis=0)
    feats = embed_windows(model, Xt)             # [sum(counts), 512]
    msgs, off = [], 0
    for t, c in zip(track_records, counts):
        w = feats[off:off + c]
        off += c
        track_emb = l2(w.mean(axis=0)).astype(np.float32)     # mean-pool + L2
        r, s = t["receiver_id"], t["scene_idx"]
        msgs.append({
            "receiver_id": f"rx{r}",
            "track_id": t["track_id"],
            "timestamp": float(t["timestamp"]),
            "embedding": track_emb.tolist(),
            "rssi": geometry.rssi_dbm(r, s),
            "aoa_deg": geometry.aoa_deg(r, s),
            "tdoa_ns": geometry.tdoa_ns(r, s),
            "n_windows": int(c),
            "class": "unknown",
        })
    return msgs
