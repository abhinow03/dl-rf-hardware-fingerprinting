"""TIER ENCODERS behind one interface: encode(protocol, windows) -> per-window 512-D deep
features (+ classical-19 aux for the fallback tier).

  T-A ENROLLED-DRONE : runs/drone_native/seed2024/best.pt   (native 4096@50 windows)
  T-B ENROLLED-WIFI  : frozen WiSig retrain_best/best_model.pt (256@25 windows)
  T-C FALLBACK (UNKNOWN): frozen WiSig encoder on best-effort windowing + classical-19 aux,
       block-scaled 50/50 at the base station. Evidence line (FINDINGS F1): frozen@N120 0.729,
       classical-19 0.713, frozen+eigengap K7/ARI 0.68 -> the validated untrained fallback.

Both checkpoints are RFEncoder (plain dual-branch); loaded READ-ONLY. GAP-based encoder is
length-agnostic, so STFT sizing is chosen from the window length. DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SW = os.path.join(_DLM, "summer_work")
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
from gen_rff.data import registry
from gen_rff.physics.features import classical_matrix

DRONE_CKPT = os.path.join(_SW, "runs", "drone_native", "seed2024", "best.pt")
WIFI_CKPT = os.path.join(_SW, "runs", "wisig_supcon_fft64", "retrain_best", "best_model.pt")


def stft_for(L):
    return (256, 64) if L >= 1024 else (64, 16)


class _Enc:
    def __init__(self, ckpt):
        self.m = RFEncoder().cuda()
        self.m.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True), strict=True)
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, windows, batch=256):
        L = windows.shape[-1]; nfft, hop = stft_for(L)
        f = np.empty((windows.shape[0], 512), np.float32)
        xt = torch.from_numpy(np.ascontiguousarray(windows.astype(np.float32)))
        for i in range(0, windows.shape[0], batch):
            xb = xt[i:i + batch].cuda(); xs = registry.stft_mag(xb, nfft, hop)
            with torch.amp.autocast('cuda'):
                f[i:i + batch] = self.m.get_encoder_output(xb, xs).float().cpu().numpy()
        return f


class Tiers:
    """Lazy-loaded tier encoders + protocol->tier routing."""
    def __init__(self):
        self._drone = None; self._wifi = None

    def _A(self):
        if self._drone is None:
            self._drone = _Enc(DRONE_CKPT)
        return self._drone

    def _B(self):
        if self._wifi is None:
            self._wifi = _Enc(WIFI_CKPT)
        return self._wifi

    def encode(self, protocol, windows):
        """Return (deep512[n,512], aux19[n,19] or None) for a track's windows."""
        if protocol == "drone_ocusync":
            return self._A().encode(windows), None            # T-A
        if protocol == "wifi":
            return self._B().encode(windows), None            # T-B
        # T-C fallback (unknown): frozen WiFi encoder (best-effort) + classical-19 aux
        return self._B().encode(windows), classical_matrix(windows).astype(np.float32)
