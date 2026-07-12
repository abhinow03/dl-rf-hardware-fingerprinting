"""gen_rff.data.loaders — per-domain window loaders (domain-native shapes).

Each loader returns a list of "unit" dicts:
  {device_gid: str, condition_key: str, meta: dict, X: np.ndarray[n,2,L] float32 (standardized)}
where all windows in a unit share one device+condition (the atom the sampler needs for
cross-condition positive pairs).

BRANCH-SAFE: WiSig uses the import-safe `wisig_manytx.load_manytx` (COPYing its call pattern,
no edits). DRFF native window-build is COPIED from results/demo_play1/drone_native.py (raw
50 MS/s, NOT OPT-B). ORACLE builds a minimal windowed cache under results_gen/ read-only on
the raw SigMF. Frozen artifacts are opened read-only.
"""
import os
import sys
import json
import re
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))          # ~/CAPSTONE/DL_model
_SW = os.path.join(_DLM, "summer_work")
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_GEN = os.path.join(_DLM, "results_gen")
DRFF_DIR = os.path.expanduser("~/Desktop/processed/drff_r2")
ORACLE_DIR = os.path.expanduser("~/Desktop/neu_m044q5210./KRI-16Devices-RawData")
os.makedirs(RESULTS_GEN, exist_ok=True)


def _std_zeromean(x):
    """per-window zero-mean unit-std over the whole [2,L] window (WiSig/ORACLE convention)."""
    x = x.astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-8)


def _unit_power(w):
    """per-window unit-power (DRFF-native convention, matches drone_native.py)."""
    return (w / (np.sqrt((w.astype(np.float32) ** 2).mean()) + 1e-8)).astype(np.float32)


# ============================================================================
# WiSig ManyTx (256-sample packets, native 25 MS/s)
# ============================================================================
_TXD = None


def _wisig_txd():
    global _TXD
    if _TXD is None:
        import wisig_manytx as W
        _TXD, _ = W.load_manytx(eq=0, verbose=False)
    return _TXD


def wisig_devices():
    """Return (train_tx, dev_tx, test_tx) from the locked splits (read-only)."""
    rd = os.path.join(_SW, "runs", "wisig_supcon_fft64", "splits")
    sp = json.load(open(os.path.join(rd, "split_manytx.json")))
    part = json.load(open(os.path.join(rd, "discovery_partition.json")))
    return sp["train_tx"], part["dev_tx"], part["test_tx"]


def load_wisig(devices, cap=600):
    """WiSig: [n,2,256] zero-mean windows per device, grouped by (rx,date) condition cells."""
    import wisig_manytx as W
    TXD = _wisig_txd()
    units = []
    for tx in devices:
        d = TXD[tx]
        iq = d["iq"]; rx = d["rx"]; date = d["date"]
        nw = min(cap, iq.shape[0])
        cells = defaultdict(list)
        for k in range(nw):
            cells[(int(rx[k]), int(date[k]))].append(k)
        for (r, dt), idxs in cells.items():
            X = np.stack([W.standardize(iq[k].T.copy()) for k in idxs]).astype(np.float32)
            units.append(dict(device_gid=f"WISIG:{tx}", condition_key=f"rx{r}_d{dt}",
                              meta=dict(tx=tx, rx=r, date=dt, board=tx.split("-")[0]), X=X))
    return units


# ============================================================================
# DRFF-R2 native (4096-sample windows @ native 50 MS/s, raw iq — NOT OPT-B)
# ============================================================================
_DRFF_REG = None


def _drff_registry():
    global _DRFF_REG
    if _DRFF_REG is None:
        manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
        pat = re.compile(r'(.+?)_(\d+)_hover')
        af_files = defaultdict(list)
        for r in manifest["clean"]:
            af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
        all_af = sorted(af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
        model_of = {a: pat.match(a + "_hover").group(1) for a in all_af}
        _DRFF_REG = dict(af_files=af_files, all_af=all_af, model_of=model_of)
    return _DRFF_REG


def drff_airframes():
    """(pool_airframes(non-mavicAir2), eval_airframes(mavicAir2))."""
    reg = _drff_registry()
    pool = [a for a in reg["all_af"] if reg["model_of"][a] != "mavicAir2"]
    ev = [a for a in reg["all_af"] if reg["model_of"][a] == "mavicAir2"]
    return pool, ev


def load_drff_native(airframes, cap=400, win=4096, wpseg=8, seed=779):
    """DRFF native 4096 windows (raw 50 MS/s, unit-power), grouped by (D,C) condition."""
    reg = _drff_registry(); af_files = reg["af_files"]; model_of = reg["model_of"]
    rng = np.random.default_rng(seed)
    units = []
    for a in airframes:
        files = af_files[a][:]; rng.shuffle(files)
        segunits = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]):
                segunits.append((fn, si))
        rng.shuffle(segunits)
        got = 0; zc = {}
        cells = defaultdict(list)   # (D,C) -> list of windows
        for fn, si in segunits:
            if got >= cap:
                break
            if fn not in zc:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zc[fn] = dict(iq=z["iq"].astype(np.float32), sb=z["seg_bounds"],
                              D=str(z["D"]), C=int(z["C"]), U=str(z["U"]))
            zz = zc[fn]; off, ln = zz["sb"][si]
            nw = int(ln) // win
            if nw < 1:
                continue
            take = min(nw, cap - got, wpseg); base = int(off)
            for k in range(take):
                w = zz["iq"][:, base + k * win: base + (k + 1) * win]
                cells[(zz["D"], zz["C"], zz["U"])].append(_unit_power(w))
            got += take
        for (D, C, U), ws in cells.items():
            units.append(dict(device_gid=f"DRFF:{a}", condition_key=f"D{D}_C{C}_U{U}",
                              meta=dict(airframe=a, model=model_of[a], D=D, C=C, U=U,
                                        eval_only=(model_of[a] == "mavicAir2")),
                              X=np.stack(ws).astype(np.float32)))
    return units


# ============================================================================
# ORACLE (16 USRP X310, native 5 MS/s, 4096 windows) — minimal cache from raw SigMF
# ============================================================================
def oracle_devices():
    """16 device serials, deterministic order."""
    d8 = os.path.join(ORACLE_DIR, "8ft")
    ser = sorted({re.match(r"WiFi_air_X310_([0-9A-F]+)_", f).group(1)
                  for f in os.listdir(d8) if f.endswith(".sigmf-data")})
    return ser


def _read_sigmf_cf32(path, max_complex, skip=1000):
    """read up to max_complex complex samples (skip startup) from a cf32 interleaved file."""
    off = skip * 2 * 4
    cnt = max_complex * 2
    a = np.fromfile(path, dtype="<f4", count=cnt, offset=off)
    n = (a.shape[0] // 2) * 2
    a = a[:n]
    return a[0::2] + 1j * a[1::2]


def build_oracle_cache(devices, distances=("8ft", "14ft"), run="run1", cap=150,
                       win=4096, clamp=5.0, force=False):
    """Build (or load) a MINIMAL windowed ORACLE cache under results_gen/ (read-only on raw).

    NOTE (documented deviation): the raw ORACLE cf32 stream carries sparse out-of-range
    saturation spikes (~2-3% of samples at ~2^65). Clean samples are in ~[-2, 2]. At
    4096-sample windows almost every window contains a spike, so the historical per-window
    reject-clip empties the set (and the spikes overflow float32 variance). This minimal
    sandbox cache instead CLAMPS samples to +/-`clamp` before standardizing, and drops only
    windows that are >50% clamped (fully corrupt). Adequate for a domain-registration /
    forward-pass source; it is not the V4 ORACLE benchmark pipeline.

    Returns list of units [{device_gid, condition_key(=distance), meta, X[n,2,4096]}].
    """
    cache_path = os.path.join(RESULTS_GEN, "oracle_min_cache.npz")
    tag = f"{','.join(devices)}|{','.join(distances)}|{run}|cap{cap}|win{win}|clamp{clamp}"
    if os.path.exists(cache_path) and not force:
        z = np.load(cache_path, allow_pickle=True)
        if str(z["tag"]) == tag:
            return list(z["units"])
    units = []
    for dev in devices:
        for dist in distances:
            path = os.path.join(ORACLE_DIR, dist, f"WiFi_air_X310_{dev}_{dist}_{run}.sigmf-data")
            if not os.path.exists(path):
                continue
            x = _read_sigmf_cf32(path, (cap + 4) * win)
            nw = min(cap, x.shape[0] // win)
            if nw < 1:
                continue
            Xs = []
            for k in range(nw):
                seg = x[k * win:(k + 1) * win]
                w = np.stack([seg.real, seg.imag]).astype(np.float32)
                corrupt = float((np.abs(w) > clamp).mean())
                if corrupt > 0.5:            # drop fully-corrupt windows only
                    continue
                w = np.clip(w, -clamp, clamp)   # clamp sparse saturation spikes
                Xs.append(_std_zeromean(w))
            if Xs:
                units.append(dict(device_gid=f"ORACLE:{dev}", condition_key=dist,
                                  meta=dict(device_id=dev, distance=dist, run=run,
                                            n_clamped_windows_kept=len(Xs)),
                                  X=np.stack(Xs).astype(np.float32)))
    np.savez(cache_path, units=np.array(units, dtype=object), tag=tag)
    return units


def load_oracle(devices, cap=150):
    return build_oracle_cache(devices, cap=cap)
