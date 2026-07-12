"""Drone-side encoder front-end (DEMO build).

Implements the LOCKED operating point feature extractor exactly:
  native 4096-sample windows @ 50 MS/s -> per-window unit-power standardize ->
  STFT n_fft=256 hop=64 Hann -> RFEncoder.get_encoder_output -> 512-D (head-free).

Encoder checkpoint = Play-1 native-from-scratch runs/drone_native/seed{SEED}/best.pt
(frozen; never trained here). best_model.pt / TEST / M100 are NOT touched.

DEMO-SIDE — NOT PAPER RESULTS.
"""
import os
# --- single-threaded BLAS: known native segfault otherwise (multithreaded BLAS on
#     many tiny linear-algebra calls). Must be set BEFORE numpy import. ---
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import numpy as np
import torch

# summer_work is where shared.RFEncoder + the trained checkpoints live
_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", "summer_work"))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "results", "step2_integrity")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder  # noqa: E402

WIN, NFFT, HOP = 4096, 256, 64          # locked window/STFT spec
EMB_DIM = 512
CKPT_ROOT = os.path.join(_SW, "runs", "drone_native")
_HANN = torch.hann_window(NFFT)


def unit_power(w):
    """Per-window unit-power standardize (matches Play-1/1b exactly)."""
    return (w / (np.sqrt((w.astype(np.float32) ** 2).mean()) + 1e-8)).astype(np.float32)


def l2(v):
    return v / (np.linalg.norm(v) + 1e-8)


def _stft_gpu(xt):                       # xt [B,2,4096] cuda f32 -> [B,2,129,61]
    B = xt.shape[0]
    x = xt.reshape(B * 2, WIN)
    sp = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                    window=_HANN.to(xt.device), center=False, return_complex=True)
    return sp.abs().reshape(B, 2, NFFT // 2 + 1, -1)


def load_encoder(seed=2024):
    """Load a frozen Play-1 native encoder checkpoint (eval mode)."""
    path = os.path.join(CKPT_ROOT, f"seed{seed}", "best.pt")
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(path, map_location="cuda", weights_only=True), strict=True)
    m.eval()
    return m


@torch.no_grad()
def embed_windows(model, Xt, batch=384):
    """Xt [n,2,4096] f32 (already unit-power standardized) -> [n,512] head-free features."""
    f = np.empty((Xt.shape[0], EMB_DIM), dtype=np.float32)
    xt = torch.from_numpy(np.ascontiguousarray(Xt))
    for i in range(0, Xt.shape[0], batch):
        xb = xt[i:i + batch].cuda()
        xs = _stft_gpu(xb)
        with torch.amp.autocast('cuda'):
            f[i:i + batch] = model.get_encoder_output(xb, xs).float().cpu().numpy()
    return f
