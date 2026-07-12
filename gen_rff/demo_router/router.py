"""PROTOCOL ROUTER — deliberately simple per-track classifier on cheap spectral statistics.

Features (rate-normalized so WiSig-256 and DRFF-4096 windows are comparable): PSD centroid /
spread / flatness / 85%-rolloff, occupied bandwidth, low/mid/high band energy, a hop-rate
proxy (temporal std of per-window centroid = spectrogram variance), and log10(sample_rate).
Classifier = StandardScaler + LogisticRegression on WiSig (wifi) vs DRFF non-mavicAir2 (drone)
tracks. UNKNOWN = out-of-family gate: standardized distance to the nearest class mean above the
99th percentile of training distances -> "unknown" (below-threshold confidence).

Route on PROTOCOL only; mavicAir2 is NOT used to train the router (it is reserved as demo eval
content). DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from gen_rff.demo_router import replay
from gen_rff.data import loaders

LABELS = ["wifi", "drone_ocusync"]


def _win_psd(w):
    x = w[0].astype(np.float64) + 1j * w[1].astype(np.float64)
    P = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    return P / (P.sum() + 1e-12)


def track_features(Xt, sample_rate, max_win=80):
    """Xt [n,2,L] -> ~10-D rate-normalized spectral feature vector."""
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
    hop = float(np.std(cents))                               # spectrogram/centroid variance
    return np.array([cent, spread, flat, roll, occ, blo, bmid, bhi, hop,
                     np.log10(sample_rate / 1e6)], dtype=np.float32)


def _chunk_tracks(pool, sample_rate, n_win=60, max_tracks=40):
    """Split a device pool into disjoint n_win-window tracks -> feature rows."""
    X = pool["Xt"]; feats = []
    for t in range(min(max_tracks, len(X) // n_win)):
        feats.append(track_features(X[t * n_win:(t + 1) * n_win], sample_rate))
    return feats


def build_training_tracks(n_wifi=24, n_drone=15, n_win=60, seed=0):
    """WiSig train devices (wifi) + DRFF non-mavicAir2 airframes (drone) -> (X, y)."""
    rng = np.random.default_rng(seed)
    train_tx, _, _ = loaders.wisig_devices()                # (train, dev, test)
    wifi_ids = list(train_tx); rng.shuffle(wifi_ids); wifi_ids = wifi_ids[:n_wifi]
    drone_pool, _ = loaders.drff_airframes()                # non-mavicAir2 pool
    drone_ids = list(drone_pool)[:n_drone]
    X = []
    for tx in wifi_ids:
        X += _chunk_tracks(replay.get_pool("wifi", tx), replay.RATE["wifi"], n_win)
    n_wifi_rows = len(X)
    for af in drone_ids:
        X += _chunk_tracks(replay.get_pool("drone", af), replay.RATE["drone"], n_win)
    y = [0] * n_wifi_rows + [1] * (len(X) - n_wifi_rows)
    return np.stack(X), np.array(y)


class ProtocolRouter:
    def __init__(self, ood_pct=99.0):
        self.ood_pct = ood_pct

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.clf = LogisticRegression(max_iter=1000).fit(Z, y)
        # per-class Gaussian for a Mahalanobis novelty (OOD) gate — weights the low-variance
        # discriminative features (occupied-BW / spread / flatness) that a narrowband tone breaks
        self.means, self.inv_cov = [], []
        d = X.shape[1]
        for c in (0, 1):
            Zc = Z[y == c]
            cov = np.cov(Zc, rowvar=False) + 1e-3 * np.eye(d)
            self.means.append(Zc.mean(0)); self.inv_cov.append(np.linalg.inv(cov))
        dtr = np.array([self._maha(z) for z in Z])
        self.ood_thresh = float(np.percentile(dtr, self.ood_pct))
        return self

    def _maha(self, z):
        out = []
        for c in (0, 1):
            v = z - self.means[c]
            out.append(float(np.sqrt(max(0.0, v @ self.inv_cov[c] @ v))))
        return min(out)

    def route(self, feat):
        z = self.scaler.transform(feat[None, :])[0]
        dmin = self._maha(z)
        proba = self.clf.predict_proba(z[None, :])[0]
        if dmin > self.ood_thresh:
            return "unknown", round(float(self.ood_thresh / (dmin + 1e-9)), 3), dmin
        return LABELS[int(np.argmax(proba))], float(proba.max()), dmin


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
