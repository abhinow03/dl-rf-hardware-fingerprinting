"""PROTOCOL ROUTER — deliberately simple per-track classifier on cheap spectral statistics.

Phase-5: RATE-AWARE. Frequency-dimensioned features (centroid / spread / 85%-rolloff / hop-rate
proxy) and the occupied-bandwidth fraction are expressed in ABSOLUTE frequency, referenced to the
legacy WiSig rate (REF_RATE = 25 MS/s): value_abs = value_norm * (sample_rate / REF_RATE). At
25 MS/s the factor is exactly 1.0, so every legacy WiSig feature vector is numerically unchanged
(regression evidence); at 6 MS/s (BLE) and 50 MS/s (drone) the same physical bandwidth now maps
to the same feature value regardless of rate — a 1 MHz GFSK BLE burst reads as narrowband in
absolute terms instead of "1/6 of the band". Dimensionless shape ratios (flatness, band-energy
thirds) and log10(sample_rate) are left as-is.

Classifier = StandardScaler + LogisticRegression on WiSig (wifi) / DRFF non-mavicAir2 (drone) /
BLE train_units×train_collections (ble) tracks. UNKNOWN = per-class Mahalanobis novelty gate:
distance to the nearest class mean above the 99th percentile of training distances -> "unknown".

Route on PROTOCOL only; mavicAir2 and BLE eval_units are NEVER used to fit the router (reserved
as demo eval content). DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from rffp.discovery.router import replay
from rffp.data import loaders

LABELS = ["wifi", "drone_ocusync", "ble"]
REF_RATE = 25e6                                  # legacy WiSig rate = absolute-frequency reference

_HERE = os.path.dirname(os.path.abspath(__file__))
# BLE train/eval unit split lives beside the ext_ble study code
_SP = json.load(open(os.path.join(_HERE, "..", "ext_ble", "splits_ext_ble.json")))
BLE_TRAIN_UNITS = _SP["train_units"]
BLE_TRAIN_COLLECTIONS = _SP["train_collections"]
BLE_EVAL_UNITS = _SP["eval_units"]
BLE_EVAL_COLLECTIONS = _SP["eval_collections"]


def _win_psd(w):
    x = w[0].astype(np.float64) + 1j * w[1].astype(np.float64)
    P = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    return P / (P.sum() + 1e-12)


def track_features(Xt, sample_rate, max_win=80):
    """Xt [n,2,L] -> ~10-D rate-AWARE spectral feature vector (absolute frequency, REF_RATE unit)."""
    n = min(len(Xt), max_win)
    cents = []
    Pacc = None
    for k in range(n):
        P = _win_psd(Xt[k]); L = len(P)
        f = np.linspace(-0.5, 0.5, L)
        cents.append(float((f * P).sum()))
        # accumulate on a fixed 128-bin normalized grid so lengths are comparable
        g = np.interp(np.linspace(-0.5, 0.5, 128), f, P)
        Pacc = g if Pacc is None else Pacc + g
    Pm = Pacc / n; Pm = Pm / (Pm.sum() + 1e-12)
    fg = np.linspace(-0.5, 0.5, 128)
    cent = float((fg * Pm).sum())
    spread = float(np.sqrt(((fg - cent) ** 2 * Pm).sum()))
    flat = float(np.exp(np.log(Pm + 1e-12).mean()) / (Pm.mean() + 1e-12))
    cum = np.cumsum(Pm); roll = float(fg[np.searchsorted(cum, 0.85)])
    occ = float((Pm > 0.1 * Pm.max()).mean())               # occupied bandwidth fraction
    t3 = 128 // 3
    blo, bmid, bhi = float(Pm[:t3].sum()), float(Pm[t3:2 * t3].sum()), float(Pm[2 * t3:].sum())
    hop = float(np.std(cents))                              # spectrogram/centroid variance
    rf = sample_rate / REF_RATE                            # ==1.0 at 25 MS/s -> legacy unchanged
    return np.array([cent * rf, spread * rf, flat, roll * rf, occ * rf, blo, bmid, bhi, hop * rf,
                     np.log10(sample_rate / 1e6)], dtype=np.float32)


def _chunk_tracks(pool, sample_rate, n_win=60, max_tracks=40):
    """Split a device pool into disjoint n_win-window tracks -> feature rows."""
    X = pool["Xt"]; feats = []
    for t in range(min(max_tracks, len(X) // n_win)):
        feats.append(track_features(X[t * n_win:(t + 1) * n_win], sample_rate))
    return feats


def build_training_tracks(n_wifi=24, n_drone=15, n_ble=21, n_win=60, ble_max_tracks=12, seed=0):
    """WiSig train devices (wifi) + DRFF non-mavicAir2 airframes (drone) + BLE train_units pooled
    over train_collections (ble) -> (X, y). Zero eval contact (no mavicAir2, no BLE eval_units)."""
    rng = np.random.default_rng(seed)
    train_tx, _, _ = loaders.wisig_devices()                # (train, dev, test)
    wifi_ids = list(train_tx); rng.shuffle(wifi_ids); wifi_ids = wifi_ids[:n_wifi]
    drone_pool, _ = loaders.drff_airframes()                # non-mavicAir2 pool
    drone_ids = list(drone_pool)[:n_drone]
    ble_ids = list(BLE_TRAIN_UNITS)[:n_ble]
    X = []
    for tx in wifi_ids:
        X += _chunk_tracks(replay.get_pool("wifi", tx), replay.RATE["wifi"], n_win)
    n_wifi_rows = len(X)
    for af in drone_ids:
        X += _chunk_tracks(replay.get_pool("drone", af), replay.RATE["drone"], n_win)
    n_drone_rows = len(X) - n_wifi_rows
    for u in ble_ids:
        pool = replay.get_ble_pool(u, BLE_TRAIN_COLLECTIONS)
        X += _chunk_tracks(pool, replay.RATE["ble"], n_win, max_tracks=ble_max_tracks)
    n_ble_rows = len(X) - n_wifi_rows - n_drone_rows
    y = [0] * n_wifi_rows + [1] * n_drone_rows + [2] * n_ble_rows
    return np.stack(X), np.array(y)


class ProtocolRouter:
    def __init__(self, ood_pct=99.0):
        self.ood_pct = ood_pct

    def fit(self, X, y):
        self.classes_ = sorted(np.unique(y).tolist())
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.clf = LogisticRegression(max_iter=1000).fit(Z, y)
        # per-class Gaussian for a Mahalanobis novelty (OOD) gate — weights the low-variance
        # discriminative features (occupied-BW / spread / flatness) that an out-of-family tone breaks
        self.means, self.inv_cov = [], []
        d = X.shape[1]
        for c in self.classes_:
            Zc = Z[y == c]
            cov = np.cov(Zc, rowvar=False) + 1e-3 * np.eye(d)
            self.means.append(Zc.mean(0)); self.inv_cov.append(np.linalg.inv(cov))
        dtr = np.array([self._maha(z) for z in Z])
        self.ood_thresh = float(np.percentile(dtr, self.ood_pct))
        return self

    def _maha_all(self, z):
        out = []
        for i in range(len(self.classes_)):
            v = z - self.means[i]
            out.append(float(np.sqrt(max(0.0, v @ self.inv_cov[i] @ v))))
        return out

    def _maha(self, z):
        return min(self._maha_all(z))

    def distances(self, feat):
        """Per-class Mahalanobis distances (label->dist) + threshold, for the FM/OOD audit table."""
        z = self.scaler.transform(feat[None, :])[0]
        return {LABELS[self.classes_[i]]: round(d, 3) for i, d in enumerate(self._maha_all(z))}

    def route(self, feat):
        z = self.scaler.transform(feat[None, :])[0]
        dmin = self._maha(z)
        proba = self.clf.predict_proba(z[None, :])[0]
        if dmin > self.ood_thresh:
            return "unknown", round(float(self.ood_thresh / (dmin + 1e-9)), 3), dmin
        return LABELS[self.classes_[int(np.argmax(proba))]], float(proba.max()), dmin


def make_fm_sweep(sample_rate, L, n_win=60, seed=1):
    """Synthetic narrowband FM tone sweep (out-of-family OOD smoke test, NOT a claim)."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n_win):
        t = np.arange(L)
        f0 = rng.uniform(-0.02, 0.02); drift = rng.uniform(-1e-5, 1e-5)
        ph = 2 * np.pi * np.cumsum(f0 + drift * t)
        x = np.exp(1j * ph) + 0.02 * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
        out.append(np.stack([x.real, x.imag]).astype(np.float32))
    return np.stack(out)
