"""TIER ENCODERS behind one config-driven registry: encode(protocol, windows) -> per-window deep
features (+ classical-19 aux for the fallback tier).

  TIER_REGISTRY {protocol: backend spec}. Each protocol routes to one frozen encoder:

  T-A ENROLLED-DRONE : runs/drone_native/seed2024/best.pt   (native 4096@50 windows) -> deep512
  T-B ENROLLED-WIFI  : frozen WiSig retrain_best/best_model.pt (256@25 windows)      -> deep512
  T-C FALLBACK (UNKNOWN): frozen WiSig encoder on best-effort windowing + classical-19 aux,
       block-scaled 50/50 at the base station. Evidence line (FINDINGS F1): frozen@N120 0.729,
       classical-19 0.713, frozen+eigengap K7/ARI 0.68 -> the validated untrained fallback.
  T-D ENROLLED-BLE   : frozen B3 native from-scratch runs/ble_native_s2024/best.pt (native
       1850@6 windows) -> dual-branch forward() head -> mean-pool -> L2 -> 128-D. RAW embeddings,
       NO per-collection calibration (EXT_FINDINGS EF5 / W3: B3 is calibration-indifferent —
       collection eta^2 ~= 0, so robust standardization is a no-op; raw is the shipping path).

  T-A/T-B/T-C use `get_encoder_output` (512-D LayerNorm fusion, magnitude STFT sized by window
  length). T-D uses `forward()` (128-D L2-normed metric head, native real/imag STFT nfft=128/
  hop=64/center=False) — the native-arm discovery space from EXT-PROTOCOLS Phase 3c.

All checkpoints are RFEncoder (plain dual-branch), loaded READ-ONLY. DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

from rffp.models import RFEncoder
from rffp.data import registry
from rffp.physics.features import classical_matrix
from rffp.config import DRONE_CKPT, WIFI_CKPT, BLE_CKPT
# checkpoint paths (DRONE_CKPT / WIFI_CKPT / BLE_CKPT) resolve via rffp.config (RFFP_RUNS-overridable).
# Phase-5 frozen BLE tier = EXT-PROTOCOLS Phase 3c B3 full_s2024, sha256 f1d7745eaa72...

# ---- config-driven tier map: existing tiers are entries with UNCHANGED behavior -------------
TIER_REGISTRY = {
    "drone_ocusync": dict(tier="T-A", ckpt=DRONE_CKPT, backend="deep512",   dim=512, aux=False),
    "wifi":          dict(tier="T-B", ckpt=WIFI_CKPT,  backend="deep512",   dim=512, aux=False),
    "unknown":       dict(tier="T-C", ckpt=WIFI_CKPT,  backend="deep512",   dim=512, aux=True),
    "ble":           dict(tier="T-D", ckpt=BLE_CKPT,   backend="native128", dim=128, aux=False),
}


def stft_for(L):
    return (256, 64) if L >= 1024 else (64, 16)


class _EncDeep:
    """T-A/T-B/T-C backend: 512-D get_encoder_output on magnitude STFT (window-length-sized).
    Behaviourally identical to the pre-Phase-5 _Enc (regression gate proves it)."""
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


class _EncNative:
    """T-D BLE backend: 128-D forward() metric head on native real/imag STFT (nfft=128/hop=64/
    center=False). RAW output — no per-collection calibration (EF5/W3 calibration-indifference).
    Mirrors ext_protocols/embed_b3_ble.py exactly (same STFT + head + AMP)."""
    def __init__(self, ckpt):
        self.m = RFEncoder().cuda()
        self.m.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True), strict=True)
        self.m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self._hann = torch.hann_window(128, device="cuda")

    def _native_stft(self, x):
        z = torch.complex(x[:, 0].float(), x[:, 1].float())
        Z = torch.stft(z, n_fft=128, hop_length=64, win_length=128, window=self._hann,
                       center=False, return_complex=True)
        return torch.stack([Z.real, Z.imag], dim=1)

    @torch.no_grad()
    def encode(self, windows, batch=512):
        f = np.empty((windows.shape[0], 128), np.float32)
        xt = torch.from_numpy(np.ascontiguousarray(windows.astype(np.float32)))
        for i in range(0, windows.shape[0], batch):
            xb = xt[i:i + batch].cuda()
            with torch.amp.autocast('cuda'):
                f[i:i + batch] = self.m(xb, self._native_stft(xb)).float().cpu().numpy()  # 128-D L2-normed head
        return f


_BACKENDS = {"deep512": _EncDeep, "native128": _EncNative}


class Tiers:
    """Lazy-loaded tier encoders + protocol->tier routing (config-driven TIER_REGISTRY).
    Encoders are cached by (backend, ckpt) so T-B and T-C share ONE frozen WiFi encoder instance
    exactly as the pre-Phase-5 implementation did."""
    def __init__(self):
        self._enc = {}

    def _backend_for(self, protocol):
        spec = TIER_REGISTRY[protocol]
        key = (spec["backend"], spec["ckpt"])
        if key not in self._enc:
            self._enc[key] = _BACKENDS[spec["backend"]](spec["ckpt"])
        return self._enc[key], spec

    def encode(self, protocol, windows):
        """Return (deep[n,dim], aux19[n,19] or None) for a track's windows.
        dim = TIER_REGISTRY[protocol]['dim'] (512 for A/B/C, 128 for BLE T-D)."""
        enc, spec = self._backend_for(protocol)
        deep = enc.encode(windows)
        aux = classical_matrix(windows).astype(np.float32) if spec["aux"] else None
        return deep, aux
