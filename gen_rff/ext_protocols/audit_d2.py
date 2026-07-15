#!/usr/bin/env python3
"""
GEN-RFF EXT-PROTOCOLS PHASE 0b — D2 physical audit (BLE XIAO-ESP32C3x31, CNS2025).
CPU-only, read-only over the dataset. Emits JSON summary + PNGs to audit_out/.
No training, no preprocessing beyond what the audit itself needs.

Data layout discovered from bytes:
  ext_data/ble_xiao/<condition>/<collection>/{X,Y}_{train,test}.npy
  X: (N, 2, 1850) float32   -> I/Q, 2 channels x 1850 samples/window
  Y: (N, 31)      float64   -> one-hot over 31 same-model units
Documented (paper/DATA_AUDIT.md): fs = 6 MS/s, BLE channels Ch1/2/14/32,
  2x USRP B210 receivers (R1/R2), wired-indoor + wireless-indoor + 4 outdoor locs.
"""
import os, json, hashlib, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

# single-threaded BLAS is set via env by the caller
ROOT = "/home/docker/pw26_akp_01/ext_data/ble_xiao"
OUT  = "/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols/audit_out"
FS   = 6_000_000.0            # documented sample rate (Hz)
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260715)

def header(p):
    with open(p, "rb") as f:
        ver = np.lib.format.read_magic(f)
        shp, fort, dt = np.lib.format.read_array_header_1_0(f)
    return shp, dt

def collections():
    cols = []
    for xt in sorted(glob.glob(os.path.join(ROOT, "*", "*", "X_train.npy"))):
        d = os.path.dirname(xt)
        cond = os.path.basename(os.path.dirname(d))
        coll = os.path.basename(d)
        cols.append((f"{cond}/{coll}", d))
    return cols

def to_complex(rows):
    # rows: (n, 2, L) -> (n, L) complex
    return rows[:, 0, :].astype(np.float64) + 1j * rows[:, 1, :].astype(np.float64)

# ---------------- STEP 2: inventory ----------------
def step2_inventory(cols):
    inv = []
    for name, d in cols:
        rec = {"collection": name}
        for split in ("train", "test"):
            xs, xd = header(os.path.join(d, f"X_{split}.npy"))
            ys, yd = header(os.path.join(d, f"Y_{split}.npy"))
            rec[f"N_{split}"] = int(xs[0])
            rec[f"X_{split}_shape"] = list(xs)
            rec[f"Y_{split}_shape"] = list(ys)
            rec[f"X_{split}_dtype"] = str(xd)
            rec[f"Y_{split}_dtype"] = str(yd)
        # per-unit counts from one-hot Y_train
        Y = np.load(os.path.join(d, "Y_train.npy"), mmap_mode="r")
        Y = np.asarray(Y)
        onehot_ok = bool(np.all(Y.sum(axis=1) == 1))
        lab = Y.argmax(axis=1)
        cnt = np.bincount(lab, minlength=Y.shape[1])
        rec["n_units"] = int(Y.shape[1])
        rec["units_present"] = int((cnt > 0).sum())
        rec["onehot_valid"] = onehot_ok
        rec["per_unit_min"] = int(cnt.min())
        rec["per_unit_max"] = int(cnt.max())
        rec["per_unit_mean"] = float(cnt.mean())
        inv.append(rec)
        print(f"[inv] {name:26s} Ntr={rec['N_train']:6d} Nte={rec['N_test']:6d} "
              f"units={rec['units_present']}/{rec['n_units']} onehot={onehot_ok} "
              f"per-unit {rec['per_unit_min']}..{rec['per_unit_max']}")
    return inv

# ---------------- STEP 3: IQ sanity ----------------
def step3_iq_sanity(cols):
    # 3 segments from 3 different units, drawn from 3 different collections
    picks = [cols[0], cols[len(cols)//2], cols[-1]]
    results = []
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    for r, (name, d) in enumerate(picks):
        X = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
        Y = np.asarray(np.load(os.path.join(d, "Y_train.npy"), mmap_mode="r")).argmax(1)
        target_unit = r  # a different unit index per row
        idxs = np.where(Y == target_unit)[0]
        i = int(idxs[rng.integers(len(idxs))])
        seg = np.asarray(X[i])                      # (2, L)
        z = seg[0].astype(np.float64) + 1j*seg[1].astype(np.float64)
        L = z.size
        dur_us = L / FS * 1e6
        env = np.abs(z)
        # PSD (Welch), centered
        f, Pxx = signal.welch(z, fs=FS, nperseg=min(256, L), return_onesided=False)
        f = np.fft.fftshift(f); Pxx = np.fft.fftshift(Pxx)
        PdB = 10*np.log10(Pxx + 1e-20)
        # occupied bandwidth: -10 dB from peak
        peak = PdB.max(); mask = PdB >= peak - 10
        occ_bw = (f[mask].max() - f[mask].min())/1e6 if mask.any() else float("nan")
        f_center = f[np.argmax(PdB)]/1e6
        rec = dict(collection=name, seg_index=i, unit=target_unit, L=L,
                   dur_us=round(dur_us,2), occ_bw_MHz=round(float(occ_bw),3),
                   peak_freq_MHz=round(float(f_center),3),
                   dc_bin_dB=round(float(PdB[np.argmin(np.abs(f))]),2),
                   env_min=float(env.min()), env_max=float(env.max()),
                   clip_frac=float(np.mean(np.abs(seg) > 0.99*np.abs(seg).max())))
        results.append(rec)
        # plots: envelope, PSD, spectrogram
        t = np.arange(L)/FS*1e6
        axes[r,0].plot(t, env, lw=0.6); axes[r,0].set_title(f"{name}\nu{target_unit} |z| env")
        axes[r,0].set_xlabel("us")
        axes[r,1].plot(f/1e6, PdB, lw=0.7); axes[r,1].set_title("PSD (Welch, dB)")
        axes[r,1].set_xlabel("MHz"); axes[r,1].axvline(0, color='k', lw=0.4, ls=':')
        ff, tt, Sxx = signal.spectrogram(z, fs=FS, nperseg=128, noverlap=96, return_onesided=False)
        ff = np.fft.fftshift(ff); Sxx = np.fft.fftshift(Sxx, axes=0)
        axes[r,2].pcolormesh(tt*1e6, ff/1e6, 10*np.log10(Sxx+1e-20), shading='auto')
        axes[r,2].set_title("spectrogram"); axes[r,2].set_xlabel("us"); axes[r,2].set_ylabel("MHz")
        print(f"[iq ] {name:26s} seg#{i} u{target_unit} L={L} dur={dur_us:.1f}us "
              f"occBW={occ_bw:.2f}MHz fpk={f_center:+.2f}MHz clip={rec['clip_frac']:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step3_iq_sanity.png"), dpi=90)
    plt.close(fig)
    return results

# ---------------- STEP 4: burst / transient integrity ----------------
def step4_transient(cols, n_sample=3000):
    """Segments are pre-windowed (L=1850). Answer: is the turn-on transient captured,
    i.e. do windows start below noise and ramp up (onset present), or are they
    steady-state / mid-packet (flat envelope from sample 0)?
    Method: mean power envelope across many segments, aligned at window start,
    per collection; plus per-segment onset detection (first index crossing a
    fraction of that segment's own peak)."""
    summ = []
    fig, axes = plt.subplots(len(cols)//2 + len(cols)%2, 2, figsize=(13, 2.4*(len(cols)//2+len(cols)%2)))
    axes = np.array(axes).ravel()
    for k, (name, d) in enumerate(cols):
        X = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
        N = X.shape[0]; L = X.shape[2]
        sel = rng.choice(N, size=min(n_sample, N), replace=False)
        sel.sort()
        # power env in chunks to bound memory
        acc = np.zeros(L); onset_idx = []; peakpos = []
        first_last_ratio = []
        B = 512
        for s in range(0, len(sel), B):
            rows = np.asarray(X[sel[s:s+B]])           # (b,2,L)
            p = rows[:,0,:]**2 + rows[:,1,:]**2         # instantaneous power (b,L)
            acc += p.sum(axis=0)
            pk = p.max(axis=1, keepdims=True)
            thr = 0.2*pk                                # 20% of each seg's own peak
            above = p >= thr
            oi = above.argmax(axis=1)                   # first crossing
            onset_idx.extend(oi.tolist())
            peakpos.extend(p.argmax(axis=1).tolist())
            # ratio of mean power in first 5% vs last 5% of window
            fr = p[:, :max(1,L//20)].mean(axis=1)
            lr = p[:, -max(1,L//20):].mean(axis=1)
            first_last_ratio.extend((fr/(lr+1e-20)).tolist())
        mean_env = acc/len(sel)
        onset_idx = np.array(onset_idx); peakpos = np.array(peakpos)
        me_pk = mean_env.max()
        rec = dict(collection=name, L=int(L), n_sampled=int(len(sel)),
                   onset_idx_med=float(np.median(onset_idx)),        # first crossing of 20% of seg peak
                   onset_idx_p10=float(np.percentile(onset_idx,10)),
                   onset_idx_p90=float(np.percentile(onset_idx,90)),
                   peakpos_med=float(np.median(peakpos)),
                   # burst-alignment signature (mean power envelope, normalized to its own peak):
                   sample0_norm=float(mean_env[0]/me_pk),            # window START level (noise floor if <<1)
                   ramp_end_sample=int(np.argmax(mean_env >= 0.5*me_pk)),  # sample where env first reaches 50% peak
                   tail_norm=float(mean_env[-1]/me_pk),              # window END level (falloff if <<1)
                   body_norm=float(np.median(mean_env[L//4:3*L//4])/me_pk))
        summ.append(rec)
        ax = axes[k]
        ax.plot(np.arange(L), mean_env/me_pk, lw=0.8)
        ax.set_title(f"{name}  mean power env (norm)", fontsize=8)
        ax.set_xlabel("sample")
        print(f"[brt] {name:26s} start={rec['sample0_norm']:.3f} ramp_end(50%)={rec['ramp_end_sample']:2d} "
              f"body={rec['body_norm']:.2f} tail={rec['tail_norm']:.3f} onset_med={rec['onset_idx_med']:.0f}")
    for j in range(len(cols), len(axes)): axes[j].axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step4_mean_envelope.png"), dpi=90)
    plt.close(fig)
    return summ

# ---------------- STEP 5: SNR + session axis ----------------
def step5_snr(cols, n_sample=1500):
    summ = []
    for name, d in cols:
        X = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
        N = X.shape[0]; L = X.shape[2]
        sel = rng.choice(N, size=min(n_sample, N), replace=False); sel.sort()
        sig_p = []; noise_p = []
        B = 512
        for s in range(0, len(sel), B):
            rows = np.asarray(X[sel[s:s+B]])
            p = rows[:,0,:]**2 + rows[:,1,:]**2
            # burst-aligned windows: noise-only region is the silent window EDGES
            # (first/last 2 samples, verified << peak); signal = constant-envelope BODY.
            noise = np.concatenate([p[:, :2], p[:, -2:]], axis=1).mean(axis=1)
            sigb  = p[:, L//4:3*L//4].mean(axis=1)
            noise_p.extend(noise.tolist()); sig_p.extend(sigb.tolist())
        sig_p = np.array(sig_p); noise_p = np.array(noise_p)
        snr_db = 10*np.log10(sig_p/(noise_p+1e-20))
        rec = dict(collection=name, snr_db_med=float(np.median(snr_db)),
                   snr_db_p10=float(np.percentile(snr_db,10)),
                   snr_db_p90=float(np.percentile(snr_db,90)))
        summ.append(rec)
        print(f"[snr] {name:26s} SNR med={rec['snr_db_med']:.1f}dB "
              f"[{rec['snr_db_p10']:.1f},{rec['snr_db_p90']:.1f}]")
    return summ

def main():
    cols = collections()
    print(f"== {len(cols)} collections ==")
    out = {"root": ROOT, "fs_hz_documented": FS, "n_collections": len(cols),
           "collections": [c[0] for c in cols]}
    out["step2_inventory"]  = step2_inventory(cols)
    out["step3_iq_sanity"]  = step3_iq_sanity(cols)
    out["step4_transient"]  = step4_transient(cols)
    out["step5_snr"]        = step5_snr(cols)
    with open(os.path.join(OUT, "audit_d2_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE", os.path.join(OUT, "audit_d2_summary.json"))

if __name__ == "__main__":
    main()
