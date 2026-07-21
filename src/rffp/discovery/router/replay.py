"""REPLAY / TRACKER (simulated multi-protocol RF world).

Each emitter = (kind, device_id): kind in {"drone","wifi"}. Its window pool is built at the
kind's native rate/length (drone = 4096@50 MS/s raw IQ; wifi = 256@25 MS/s WiSig packets).
Windows are dealt round-robin to R receivers in capture/session order -> one track per
receiver-emitter (grouping from continuity ONLY). Each track carries its window IQ, the
sample-rate metadata a receiver would physically have, opaque ids, and a scene index for the
geometry mock. NO protocol/device label on the track — the true (protocol, device_id) goes on
the `ground_truth` side channel (scoring only).

mavicAir2 airframes appear here as the demo's unknown-emitter content (standing exception,
see README). DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from rffp.data import loaders, registry
from rffp.config import BLE_DIR as BLE_ROOT

# BLE D2 (Seeed XIAO ESP32-C3): native (2,1850) I/Q @ 6 MS/s, one dir per collection.
# BLE_ROOT resolves via rffp.config (RFFP_BLE-overridable).
RATE = {"drone": 50e6, "wifi": 25e6, "ble": 6e6}
_POOL = {}
_BLE_POOL = {}


def _ble_pool(unit, collections, cap_per_coll=2000):
    """Windows for one BLE unit pooled across `collections`; seg = collection index (a physical
    condition/location). Single-collection variant => 1 seg (canonical-like); pooled variant =>
    4 outdoor segs mixed by the round-robin dealer (demo-honest, receiver-shaped, EF7). Read-only
    on X_train rows of eval/held-out units — the frozen B3 encoder never saw these units."""
    Xt, seg = [], []
    for si, c in enumerate(collections):
        d = os.path.join(BLE_ROOT, c)
        X = np.load(f"{d}/X_train.npy", mmap_mode="r")
        y = np.asarray(np.load(f"{d}/Y_train.npy", mmap_mode="r")).argmax(1)
        idx = np.where(y == unit)[0][:cap_per_coll]
        if len(idx):
            Xt.append(np.asarray(X[idx]).astype(np.float32)); seg.append(np.full(len(idx), si))
    return dict(Xt=np.concatenate(Xt), seg=np.concatenate(seg))


def get_ble_pool(unit, collections):
    key = (int(unit), tuple(collections))
    if key not in _BLE_POOL:
        _BLE_POOL[key] = _ble_pool(unit, list(collections))
    return _BLE_POOL[key]


def _drone_pool(af, cap=2000):
    """native 4096 unit-power windows for one airframe, session-ordered (seg = capture chunk)."""
    units = loaders.load_drff_native([af], cap=cap)          # per-(D,C,U) units
    Xt, seg = [], []
    for si, u in enumerate(units):
        for j in range(u["X"].shape[0]):
            Xt.append(u["X"][j]); seg.append(si)
    return dict(Xt=np.stack(Xt).astype(np.float32), seg=np.array(seg))


def _wifi_pool(tx, cap=700):
    """256 zero-mean windows for one WiSig device, session-ordered (seg = (rx,date) cell)."""
    units = loaders.load_wisig([tx], cap=cap)                # per-(rx,date) units
    Xt, seg = [], []
    for si, u in enumerate(units):
        for j in range(u["X"].shape[0]):
            Xt.append(u["X"][j]); seg.append(si)
    return dict(Xt=np.stack(Xt).astype(np.float32), seg=np.array(seg))


def get_pool(kind, device_id):
    key = (kind, device_id)
    if key not in _POOL:
        _POOL[key] = _drone_pool(device_id) if kind == "drone" else _wifi_pool(device_id)
    return _POOL[key]


def stream_tracks(emitters, R=4, N=120, repeat_seed=0, ble_collections=None):
    """emitters: list of (kind, device_id). Returns (tracks, ground_truth).
      track: {track_id, receiver_id, scene_idx, Xt[n,2,L], n_windows, partial, sample_rate, timestamp}
      ground_truth: {track_id: (kind, device_id)}.
    ble_collections: which BLE collections a 'ble' emitter draws from (single vs pooled variant)."""
    tracks, gt = [], {}
    for scene_idx, (kind, dev) in enumerate(emitters):
        pool = get_ble_pool(dev, ble_collections) if kind == "ble" else get_pool(kind, dev)
        segs = np.unique(pool["seg"]); rng = np.random.default_rng(repeat_seed * 1000 + scene_idx)
        rng.shuffle(segs)
        order = [w for s in segs for w in np.where(pool["seg"] == s)[0].tolist()]
        order = np.array(order)
        buckets = [[] for _ in range(R)]
        for pos, wi in enumerate(order):
            r = pos % R
            if len(buckets[r]) < N:
                buckets[r].append(wi)
            if all(len(b) >= N for b in buckets):
                break
        for r in range(R):
            idx = np.array(buckets[r], dtype=int)
            tid = f"{kind}:{dev}__rx{r}__s{scene_idx}__rep{repeat_seed}"
            tracks.append(dict(track_id=tid, receiver_id=r, scene_idx=scene_idx,
                               Xt=pool["Xt"][idx], n_windows=int(len(idx)),
                               partial=bool(len(idx) < N), sample_rate=RATE[kind],
                               timestamp=float(scene_idx * 0.01 + r * 0.001)))
            gt[tid] = (kind, dev)
    return tracks, gt
