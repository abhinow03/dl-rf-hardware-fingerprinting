"""rffp.physics.features — classical impairment features (19-D) + protocol-suppressed
residual view (blind complex-LPC v1).

BRANCH-SAFE: this is a COPY of the Step-4 `classical_feats`/`classical_matrix` (mechanism_
validation.py), parameterized by native sample-rate `fs`, plus a new LPC residual. The
originals are never imported for production use (they execute heavy code under __main__);
the P1-G1 unit test imports `mechanism_validation.classical_matrix` (a module-level, side-
effect-free symbol) purely to check byte-for-byte agreement.

Rate note: the Step-4 19-D set is computed in NORMALIZED frequency (f in [-0.5, 0.5]) and
lag-1 phase (rad/sample), so it is already rate-invariant — the 19-D vector does NOT change
with `fs`. `fs` is carried for provenance and to expose a Hz-scaled CFO diagnostic
(`cfo_hz`), which is NOT part of the 19-D and never enters the model. This keeps the unit
test exact while honoring "parameterize rate-dependent ops."
"""
import numpy as np
from scipy.signal import lfilter
from scipy.linalg import solve_toeplitz

FEATURE_NAMES = [
    "cfo", "cfo_std", "env_mean", "env_std", "env_skew", "env_kurt", "papr",
    "iq_var_ratio", "iq_corr", "spec_centroid", "spec_spread", "spec_flatness",
    "spec_rolloff", "band_lo", "band_mid", "band_hi", "zcr", "I_kurt", "Q_kurt",
]
N_FEATURES = 19


def _as_complex(iq):
    """iq: [2, L] real (I, Q) -> complex [L]."""
    iq = np.asarray(iq)
    return iq[0].astype(np.float64) + 1j * iq[1].astype(np.float64)


def classical_features(iq, fs=None):
    """iq: [2, L] real IQ window -> 19-D impairment/statistics vector (float32).

    Verbatim port of Step-4 `classical_feats` (normalized-frequency, no preamble assumption).
    `fs` (native sample rate, Hz) is accepted for API/provenance but does not alter the 19-D
    (they are rate-normalized); see module docstring.
    """
    iq = np.asarray(iq)
    I, Q = iq[0].astype(np.float64), iq[1].astype(np.float64)
    x = I + 1j * Q
    env = np.abs(x); e2 = env ** 2
    eps = 1e-12
    # CFO via lag-1 autocorrelation phase-slope (generic; no sync/preamble)
    lag1 = x[1:] * np.conj(x[:-1])
    inc = np.angle(lag1)
    cfo = float(np.angle(lag1.mean() + 0j))
    cfo_std = float(inc.std())
    # envelope moments
    em, es = float(env.mean()), float(env.std())
    esk = float(((env - em) ** 3).mean() / (es ** 3 + eps))
    eku = float(((env - em) ** 4).mean() / (es ** 4 + eps))
    papr = float(e2.max() / (e2.mean() + eps))
    # IQ imbalance proxies
    iq_ratio = float(I.var() / (Q.var() + eps))
    iq_corr = float(np.corrcoef(I, Q)[0, 1]) if I.std() > 0 and Q.std() > 0 else 0.0
    # spectral shape (shifted PSD, normalized frequency)
    X = np.fft.fftshift(np.fft.fft(x))
    P = np.abs(X) ** 2; Ps = P.sum() + eps
    f = np.linspace(-0.5, 0.5, len(P))
    cent = float((f * P).sum() / Ps)
    spread = float(np.sqrt(((f - cent) ** 2 * P).sum() / Ps))
    flat = float(np.exp(np.log(P + eps).mean()) / (P.mean() + eps))
    cum = np.cumsum(P) / Ps
    rolloff = float(f[np.searchsorted(cum, 0.85)])
    t3 = len(P) // 3
    blo = float(P[:t3].sum() / Ps); bmid = float(P[t3:2 * t3].sum() / Ps); bhi = float(P[2 * t3:].sum() / Ps)
    zcr = float((np.abs(np.diff(np.sign(I))) > 0).mean())
    iku = float(((I - I.mean()) ** 4).mean() / (I.var() ** 2 + eps))
    qku = float(((Q - Q.mean()) ** 4).mean() / (Q.var() ** 2 + eps))
    return np.array([cfo, cfo_std, em, es, esk, eku, papr, iq_ratio, iq_corr,
                     cent, spread, flat, rolloff, blo, bmid, bhi, zcr, iku, qku], dtype=np.float32)


def classical_matrix(windows, fs=None):
    """windows: [Nw, 2, L] -> [Nw, 19] raw (unstandardized) float32."""
    return np.stack([classical_features(w, fs) for w in windows]).astype(np.float32)


def cfo_hz(iq, fs):
    """Rate-DEPENDENT diagnostic (NOT in the 19-D): normalized lag-1 CFO scaled to Hz."""
    x = _as_complex(iq)
    lag1 = x[1:] * np.conj(x[:-1])
    return float(np.angle(lag1.mean() + 0j)) / (2 * np.pi) * float(fs)


# ============================================================================
# Protocol-suppressed residual view (blind v1 — complex linear prediction)
# ============================================================================
def residual_view(iq, order=32, eps=1e-6):
    """Blind protocol-suppression via per-window complex linear prediction (autocorrelation
    method, order `order`). The LPC captures the predictable modulation structure; the
    residual e[n] = x[n] - sum_k a_k x[n-k] concentrates transients / nonlinearity / noise-
    floor structure where hardware impairments live.

    Returns (res, ratio):
      res   : [2, L] float32 (I_res, Q_res) — the residual channel, same length as input.
      ratio : residual_energy_ratio = ||e||^2 / ||x||^2 (diagnostic scalar in (0, ~1]).
    """
    iq = np.asarray(iq)
    L = iq.shape[1]
    x = _as_complex(iq)
    p = int(min(order, L - 1))
    if p < 1:
        return np.zeros((2, L), np.float32), 0.0
    # autocorrelation via zero-padded FFT: r[k] = sum_n x[n] conj(x[n-k])
    nfft = 1 << (2 * L - 1).bit_length()
    X = np.fft.fft(x, nfft)
    r = np.fft.ifft(np.abs(X) ** 2)[: p + 1]
    r = r.copy()
    r[0] = r[0].real + eps * (abs(r[0]) + 1.0)   # regularize the diagonal
    try:
        # Hermitian Toeplitz system R a = r[1:p+1]  (first col r[:p], first row conj(r[:p]))
        a = solve_toeplitz((r[:p], np.conj(r[:p])), r[1: p + 1])
    except Exception:
        return np.zeros((2, L), np.float32), 0.0
    # analysis filter A(z) = 1 - sum a_k z^-k  ->  e = lfilter([1, -a...], 1, x)
    e = lfilter(np.concatenate([[1.0 + 0j], -a]), [1.0 + 0j], x)
    ex = float(np.abs(x) ** 2 @ np.ones(L)) if L else 0.0
    ex = float((np.abs(x) ** 2).sum())
    ee = float((np.abs(e) ** 2).sum())
    ratio = ee / (ex + 1e-12)
    return np.stack([e.real, e.imag]).astype(np.float32), float(ratio)


def residual_batch(windows, order=32):
    """windows: [Nw, 2, L] -> (res[Nw, 2, L] float32, ratios[Nw] float32)."""
    res, ratios = [], []
    for w in windows:
        r, ratio = residual_view(w, order=order)
        res.append(r); ratios.append(ratio)
    return np.stack(res).astype(np.float32), np.asarray(ratios, np.float32)
