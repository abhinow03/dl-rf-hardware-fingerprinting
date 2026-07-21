#!/usr/bin/env python3
"""
Phase 2 Approach A — classical_b feature extractor for BLE D2 (2,1850) IQ @ 6 MS/s.
19-D, six families. Fully vectorized over a batch. NO training here.

Regions (fixed, justified by Phase-0b onset stats: onset ramp always within first ~13
samples, body constant-envelope, tail falls in last ~5):
  ONSET = [0:25]   BODY = [30:1820]   (body split into early/late halves for drift)

Feature order (19) — LOCKED after DEV pruning (see CLASSICAL_B_SPEC.md):
 F-CFO  (3): cfo_hz, cfo_drift_hz, phase_resid_std
 F-TRANS(5): rise_1090, ramp_lin, ramp_curv, fall_1090, onset_overshoot   <- P6 family
 F-MOD  (4): if_spread, if_skew, if_kurt, if_iqr
 F-SPEC (4): occ_bw, spec_spread, spec_flatness, spec_skew
 F-IQ   (1): dc_offset
 F-ENV  (2): env_cv, papr
DEV removed: spec_centroid (r=0.966 w/ cfo_hz), gain_imbalance & quad_error (F<1.2, dead).
"""
import numpy as np
from scipy import signal, stats

FS = 6_000_000.0
ONSET0, ONSET1 = 0, 25
BODY0, BODY1   = 30, 1820

FEATURE_NAMES = [
    "cfo_hz","cfo_drift_hz","phase_resid_std",                       # F-CFO
    "rise_1090","ramp_lin","ramp_curv","fall_1090","onset_overshoot", # F-TRANS (P6)
    "if_spread","if_skew","if_kurt","if_iqr",                         # F-MOD
    "occ_bw","spec_spread","spec_flatness","spec_skew",               # F-SPEC
    "dc_offset",                                                      # F-IQ
    "env_cv","papr",                                                  # F-ENV
]
FAMILY = {"F-CFO":[0,1,2],"F-TRANS":[3,4,5,6,7],"F-MOD":[8,9,10,11],
          "F-SPEC":[12,13,14,15],"F-IQ":[16],"F-ENV":[17,18]}

def extract(X):
    """X: (B,2,1850) float32 -> (B,19) float32."""
    X = np.asarray(X, dtype=np.float64)
    I, Q = X[:,0,:], X[:,1,:]
    z = I + 1j*Q
    p = I*I + Q*Q                                  # power (B,L)
    B, L = p.shape
    body_ref = np.median(p[:, 100:1750], axis=1) + 1e-20   # robust body power

    # ---------- F-TRANS (shape only) ----------
    ref = body_ref[:,None]
    def cross(mat, frac):                          # first index power crosses frac*ref
        return np.argmax(mat >= frac*ref, axis=1)
    i10 = cross(p, 0.1); i90 = cross(p, 0.9)
    rise_1090 = (i90 - i10).astype(np.float64)
    # fall edge on reversed signal
    pr = p[:, ::-1]
    j10 = np.argmax(pr >= 0.1*ref, axis=1); j90 = np.argmax(pr >= 0.9*ref, axis=1)
    fall_1090 = (j90 - j10).astype(np.float64)
    # ramp poly fit over ONSET window on power normalized by body_ref
    n = np.arange(ONSET0, ONSET1, dtype=np.float64)
    Vd = np.vstack([np.ones_like(n), n, n*n]).T    # (K,3) design [1,n,n^2]
    Pw = (p[:, ONSET0:ONSET1] / ref)               # (B,K)
    coef, *_ = np.linalg.lstsq(Vd, Pw.T, rcond=None)   # (3,B)
    ramp_lin  = coef[1]; ramp_curv = coef[2]
    onset_overshoot = p[:, :40].max(axis=1) / body_ref  # turn-on overshoot/ringing (shape, scale-free)

    # ---------- F-CFO / F-MOD (instantaneous frequency) ----------
    phase = np.unwrap(np.angle(z), axis=1)
    IF = np.diff(phase, axis=1) * (FS/(2*np.pi))   # (B,L-1) Hz
    ifb = IF[:, BODY0:BODY1]
    cfo_hz = ifb.mean(axis=1)
    half = ifb.shape[1]//2
    cfo_drift_hz = ifb[:, half:].mean(1) - ifb[:, :half].mean(1)
    # residual after linear phase fit over body
    pb = phase[:, BODY0:BODY1]; m = pb.shape[1]
    t = np.arange(m, dtype=np.float64); Vp = np.vstack([np.ones(m), t]).T
    cp, *_ = np.linalg.lstsq(Vp, pb.T, rcond=None)      # (2,B)
    resid = pb - (Vp @ cp).T
    phase_resid_std = resid.std(axis=1)
    if_spread = ifb.std(axis=1)
    if_skew = stats.skew(ifb, axis=1)
    if_kurt = stats.kurtosis(ifb, axis=1)
    if_iqr = np.percentile(ifb, 75, axis=1) - np.percentile(ifb, 25, axis=1)  # robust IF deviation

    # ---------- F-SPEC (body PSD) ----------
    f, Pxx = signal.welch(z[:, BODY0:BODY1], fs=FS, nperseg=256,
                          return_onesided=False, axis=1)
    f = np.fft.fftshift(f); Pxx = np.fft.fftshift(Pxx, axes=1)
    Psum = Pxx.sum(axis=1, keepdims=True) + 1e-30
    Pn = Pxx / Psum
    spec_centroid = (Pn * f[None,:]).sum(axis=1)                      # internal only
    spec_spread = np.sqrt((Pn * (f[None,:]-spec_centroid[:,None])**2).sum(axis=1))
    spec_skew = (Pn * (f[None,:]-spec_centroid[:,None])**3).sum(axis=1) / (spec_spread**3 + 1e-30)
    spec_flatness = np.exp(np.mean(np.log(Pxx+1e-30), axis=1)) / (Pxx.mean(axis=1)+1e-30)
    # occupied -10 dB BW
    peak = Pxx.max(axis=1, keepdims=True)
    mask = Pxx >= peak*0.1
    fmask = np.where(mask, f[None,:], np.nan)
    occ_bw = np.nanmax(fmask, axis=1) - np.nanmin(fmask, axis=1)

    # ---------- F-IQ / F-ENV (body) ----------
    Ib, Qb = I[:, BODY0:BODY1], Q[:, BODY0:BODY1]
    rms = np.sqrt((Ib*Ib+Qb*Qb).mean(axis=1))+1e-20
    dc_offset = np.sqrt(Ib.mean(1)**2 + Qb.mean(1)**2)/rms          # LO leakage (only working F-IQ)
    envb = np.sqrt(Ib*Ib+Qb*Qb)
    env_cv = envb.std(axis=1)/(envb.mean(axis=1)+1e-20)
    pb2 = p[:, BODY0:BODY1]
    papr = pb2.max(axis=1)/(pb2.mean(axis=1)+1e-20)

    feats = np.stack([
        cfo_hz, cfo_drift_hz, phase_resid_std,
        rise_1090, ramp_lin, ramp_curv, fall_1090, onset_overshoot,
        if_spread, if_skew, if_kurt, if_iqr,
        occ_bw, spec_spread, spec_flatness, spec_skew,
        dc_offset,
        env_cv, papr], axis=1).astype(np.float32)
    return feats

if __name__ == "__main__":
    import os
    ROOT="/home/docker/pw26_akp_01/ext_data/ble_xiao"
    X=np.asarray(np.load(os.path.join(ROOT,"Wired (indoors)/Ch1_R1/X_train.npy"),mmap_mode="r")[:256])
    F=extract(X)
    print("feat shape",F.shape,"names",len(FEATURE_NAMES))
    print("finite:", np.isfinite(F).all(), " per-col std>0:", (F.std(0)>0).all())
    for i,nm in enumerate(FEATURE_NAMES):
        print(f"  {i:2d} {nm:16s} mean={F[:,i].mean():+.4g} std={F[:,i].std():.4g} finite={np.isfinite(F[:,i]).all()}")
