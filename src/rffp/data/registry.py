"""rffp.data.registry — the DOMAIN REGISTRY (single source of truth).

Each domain declares its native shape/rate, STFT params, loader, and metadata fields.
Windows stay DOMAIN-NATIVE (no commensurability forcing — that constraint died with the
frozen-encoder era; the GenRFEncoder is GAP-based / length-agnostic). `stft_mag` builds the
spectral input at each domain's native (n_fft, hop) so shapes differ across domains by design.

M100 is registered but quarantined=True (resampled provenance) — excluded from default pools.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import numpy as np
import torch

from rffp.data import loaders
from rffp.config import WISIG_PKL, DRFF_DIR, ORACLE_DIR

# protocol families -> stable integer id
PROTOCOL_IDS = {"wifi_80211a": 0, "ocusync_drone": 1}


@dataclass
class Domain:
    name: str
    protocol_family: str
    native_rate: float                 # Sps
    window_len: int
    n_fft: int
    hop: int
    path: str
    loader_fn: Callable                # (devices, cap) -> list[unit]
    metadata_fields: Tuple[str, ...]
    quarantined: bool = False
    standardize: str = "zeromean"      # provenance of per-window standardization

    @property
    def protocol_id(self) -> int:
        return PROTOCOL_IDS[self.protocol_family]

    @property
    def stft_shape(self) -> Tuple[int, int]:
        F = self.n_fft // 2 + 1
        T = (self.window_len - self.n_fft) // self.hop + 1
        return (F, T)


REGISTRY = {
    "WISIG": Domain(
        name="WISIG", protocol_family="wifi_80211a", native_rate=25e6,
        window_len=256, n_fft=64, hop=16,
        path=WISIG_PKL, loader_fn=loaders.load_wisig,
        metadata_fields=("tx", "board", "rx", "date"), standardize="zeromean"),
    "DRFF": Domain(
        name="DRFF", protocol_family="ocusync_drone", native_rate=50e6,
        window_len=4096, n_fft=256, hop=64,
        path=DRFF_DIR, loader_fn=loaders.load_drff_native,
        metadata_fields=("airframe", "model", "U", "D", "C", "eval_only"),
        standardize="unit_power"),
    "ORACLE": Domain(
        name="ORACLE", protocol_family="wifi_80211a", native_rate=5e6,
        window_len=4096, n_fft=256, hop=64,
        path=ORACLE_DIR, loader_fn=loaders.load_oracle,
        metadata_fields=("device_id", "distance", "run"), standardize="zeromean"),
    "M100": Domain(
        name="M100", protocol_family="ocusync_drone", native_rate=25e6,
        window_len=4096, n_fft=256, hop=64,
        path="(resampled cache)", loader_fn=None,
        metadata_fields=("airframe", "condition"), quarantined=True,
        standardize="unit_power"),
}

DEFAULT_DOMAINS = [k for k, d in REGISTRY.items() if not d.quarantined]   # WISIG, DRFF, ORACLE


def domain_device_pool(name):
    """Return the default (train_pool, eval_group) device_gid lists for a domain."""
    if name == "WISIG":
        train, dev, test = loaders.wisig_devices()
        # eval group = DEV discover devices; TEST (board 18) stays CLOSED (never returned here)
        return [f"WISIG:{t}" for t in train], [f"WISIG:{t}" for t in dev]
    if name == "DRFF":
        pool, ev = loaders.drff_airframes()
        return [f"DRFF:{a}" for a in pool], [f"DRFF:{a}" for a in ev]   # mavicAir2 = eval-only
    if name == "ORACLE":
        devs = loaders.oracle_devices()
        # historical 12/4: hold out last 4 (deterministic) as the discovery group
        rng = np.random.default_rng(777)
        order = list(devs); rng.shuffle(order)
        ev = sorted(order[:4]); train = sorted(order[4:])
        return [f"ORACLE:{d}" for d in train], [f"ORACLE:{d}" for d in ev]
    raise KeyError(name)


# ---- STFT front end (domain-native params) ----
_WIN_CACHE = {}


def _hann(n, device):
    key = (n, str(device))
    if key not in _WIN_CACHE:
        _WIN_CACHE[key] = torch.hann_window(n, device=device)
    return _WIN_CACHE[key]


def stft_mag(iq, n_fft, hop):
    """iq: [B, 2, L] real torch tensor -> [B, 2, F, T] magnitude (center=False, Hann)."""
    B = iq.shape[0]
    x = iq.reshape(B * 2, iq.shape[-1])
    sp = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                    window=_hann(n_fft, iq.device), center=False, return_complex=True)
    return sp.abs().reshape(B, 2, n_fft // 2 + 1, -1)


def forward_shapes():
    """(name -> dict(iq, stft)) native shapes for a forward-pass sanity check."""
    out = {}
    for name in DEFAULT_DOMAINS:
        d = REGISTRY[name]
        out[name] = dict(iq=(2, d.window_len), stft=(2, *d.stft_shape))
    return out
