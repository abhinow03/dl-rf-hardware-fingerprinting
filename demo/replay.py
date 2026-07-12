"""REPLAY FEED (simulated RF world).

For each emitter, stream its raw 50 MS/s IQ as native 4096 windows in segment (capture)
order; deal consecutive windows round-robin to R receiver-drones. Each receiver-emitter
pair accumulates up to N* windows -> one track. Grouping of windows into a track comes
ONLY from track continuity (which receiver dealt which windows), never from labels.

This module IS the physical world, so it legitimately knows ground truth (which airframe
produced a track). That truth is returned on a SIDE CHANNEL (`ground_truth`) consumed only
by scoring.py. The track records handed to the receiver carry NO airframe label — only an
opaque track_id, the receiver index, the emitter's SCENE index (a physical position for the
geometry mock), and the window IQ.

DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import re
from collections import defaultdict
import numpy as np

from encoder_frontend import WIN, unit_power

DRFF_DIR = os.path.expanduser("~/Desktop/processed/drff_r2")
WPSEG = 8                    # max windows per segment (spread), matches Play-1/1b
POOL_CAP = 2000             # windows cached per emitter (enough for R*N* draws)

# ---- airframe registry (same conventions as Play-1/1b) ----
_manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
_pat = re.compile(r'(.+?)_(\d+)_hover')
_af_files = defaultdict(list)
for _r in _manifest["clean"]:
    _af_files[_r["TD"]].append(_r["file"].replace(".mat", ".npz"))
ALL_AF = sorted(_af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
MODEL_OF = {a: _pat.match(a + "_hover").group(1) for a in ALL_AF}

_POOL_CACHE = {}


def _build_pool(af, cap=POOL_CAP, seed=779):
    """Cache native 4096 unit-power windows for one airframe, in segment order, with the
    per-window segment id (a segment = one continuous capture chunk)."""
    rng = np.random.default_rng(seed)
    files = _af_files[af][:]
    rng.shuffle(files)
    units = []
    for fn in files:
        z = np.load(os.path.join(DRFF_DIR, fn))
        for si in range(z["seg_bounds"].shape[0]):
            units.append((fn, si))
    Xt, segid = [], []
    got, gseg, zc = 0, 0, {}
    for fn, si in units:
        if got >= cap:
            break
        if fn not in zc:
            z = np.load(os.path.join(DRFF_DIR, fn))
            zc[fn] = dict(iq=z["iq"].astype(np.float32), sb=z["seg_bounds"])
        zz = zc[fn]
        off, ln = zz["sb"][si]
        nw = int(ln) // WIN
        if nw < 1:
            continue
        take = min(nw, cap - got, WPSEG)
        base = int(off)
        for k in range(take):
            Xt.append(unit_power(zz["iq"][:, base + k * WIN: base + (k + 1) * WIN]))
            segid.append(gseg)
        got += take
        gseg += 1
    return dict(Xt=np.stack(Xt).astype(np.float32), seg=np.array(segid))


def get_pool(af):
    if af not in _POOL_CACHE:
        _POOL_CACHE[af] = _build_pool(af)
    return _POOL_CACHE[af]


def stream_tracks(emitters, R=4, N=120, repeat_seed=0, timestamp0=0.0):
    """Produce R receiver-tracks per emitter (round-robin deal of segment-ordered windows).

    Returns (track_records, ground_truth):
      track_records: list of dicts, each = one receiver-emitter track:
        { track_id, receiver_id(int), scene_idx(int), Xt(np[n,2,4096]), n_windows(int),
          timestamp(float) }   -- NO airframe label.
      ground_truth: { track_id -> airframe } (side channel; scoring only).
    """
    tracks, ground_truth = [], {}
    for scene_idx, af in enumerate(emitters):
        pool = get_pool(af)
        segs = np.unique(pool["seg"])
        rng = np.random.default_rng(repeat_seed * 1000 + scene_idx)
        rng.shuffle(segs)                       # different segment draw per repeat
        # segment-ordered window stream (windows within a segment stay contiguous)
        order = []
        for s in segs:
            order.extend(np.where(pool["seg"] == s)[0].tolist())
        order = np.array(order)
        # deal consecutive windows round-robin to R receivers, cap N per receiver
        buckets = [[] for _ in range(R)]
        for pos, wi in enumerate(order):
            r = pos % R
            if len(buckets[r]) < N:
                buckets[r].append(wi)
            if all(len(b) >= N for b in buckets):
                break
        for r in range(R):
            idx = np.array(buckets[r], dtype=int)
            tid = f"{af}__rx{r}__s{scene_idx}__rep{repeat_seed}"
            tracks.append(dict(
                track_id=tid, receiver_id=r, scene_idx=scene_idx,
                Xt=pool["Xt"][idx], n_windows=int(len(idx)),
                timestamp=float(timestamp0 + scene_idx * 0.01 + r * 0.001),
            ))
            ground_truth[tid] = af
    return tracks, ground_truth
