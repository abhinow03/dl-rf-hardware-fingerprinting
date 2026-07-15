# CLASSICAL_B_SPEC.md — Approach-A feature spec for BLE D2 (LOCKED)

*Phase 2 Stage A. Authority: `EXT_PROTOCOL_PLAN.md` (cca5be51…). This spec is **locked before
any eval-split contact**; Stage B runs the battery once against it. Extractor:
`features_ble_classical.py`. Input: `(2,1850)` I/Q @ 6 MS/s, per-segment amplitude-normalized.*

## Regions (fixed; justified by Phase-0b onset stats)
- `z = I + jQ`, `p = I²+Q²`. Body power reference `ref = median(p[100:1750])`.
- **ONSET = [0:25]** (turn-on ramp always within first ~13 samples), **BODY = [30:1820]**
  (constant-envelope interior, avoids onset ramp and tail falloff).
- Instantaneous frequency `IF = diff(unwrap(angle(z)))·fs/(2π)` (Hz).
- Body PSD = Welch, `nperseg=256`, two-sided, over BODY; `f` fftshifted; `Pn = Pxx/ΣPxx`;
  `centroid = Σ Pn·f` (internal).

## Locked 19-D feature vector (`classical_b`)
| # | name | family | definition |
|---|------|--------|-----------|
| 0 | cfo_hz | F-CFO | mean(IF) over BODY (carrier offset proxy) |
| 1 | cfo_drift_hz | F-CFO | mean(IF, late half) − mean(IF, early half) of BODY |
| 2 | phase_resid_std | F-CFO | std of unwrapped-phase residual after linear fit over BODY |
| 3 | rise_1090 | **F-TRANS** | (first idx p≥0.9·ref) − (first idx p≥0.1·ref) |
| 4 | ramp_lin | **F-TRANS** | linear coef of quadratic fit of (p/ref) vs n over ONSET |
| 5 | ramp_curv | **F-TRANS** | quadratic coef of that fit |
| 6 | fall_1090 | **F-TRANS** | rise_1090 mirror on time-reversed p |
| 7 | onset_overshoot | **F-TRANS** | max(p[:40]) / ref (turn-on overshoot/ringing) |
| 8 | if_spread | F-MOD | std(IF) over BODY (GFSK deviation) |
| 9 | if_skew | F-MOD | skew(IF) over BODY |
| 10 | if_kurt | F-MOD | kurtosis(IF) over BODY |
| 11 | if_iqr | F-MOD | P75−P25 of IF over BODY (robust deviation) |
| 12 | occ_bw | F-SPEC | −10 dB occupied bandwidth of body PSD |
| 13 | spec_spread | F-SPEC | sqrt(Σ Pn (f−centroid)²) |
| 14 | spec_flatness | F-SPEC | geo-mean/arith-mean of body PSD |
| 15 | spec_skew | F-SPEC | Σ Pn (f−centroid)³ / spread³ |
| 16 | dc_offset | F-IQ | |mean(I),mean(Q)| / body-RMS (LO leakage) |
| 17 | env_cv | F-ENV | std(|z|)/mean(|z|) over BODY |
| 18 | papr | F-ENV | max(p)/mean(p) over BODY |

**F-TRANS is the P6 family** (transient SHAPE only — every member is a ratio, count, or
normalized-power coefficient; absolute scale is dead per Phase-0b and never used).

## Standardization transforms (locked; part of the spec)
- **PRIMARY — per-collection robust (median/IQR), fit UNSUPERVISED within each collection.**
  Each collection ≈ one receiver/condition; centering per-collection removes the receiver-common
  CFO/gain offset while preserving within-collection unit structure. Legal at deploy time (no
  labels, fit on the incoming capture batch). This is the intended CFO receiver-shift mitigation.
- **COMPARISON — global train-fit z-score** (mean/std fit on train_units×train_collections,
  applied everywhere). Standard baseline; does NOT mitigate receiver CFO shift. Both variants are
  reported in T3/T-RX.

## DEV evidence (train_units × train_collections ONLY — no eval contact)
DEV set: 10,080 segs, 21 train units, 8 train collections. All 19 features finite, non-degenerate.
- **kNN-1 train-unit ID = 0.979** (chance 0.048), 70/30 split, global z-score.
- **Discrimination (ANOVA F):** cfo_hz 4176 ≫ ramp_lin 161, onset_overshoot 159, ramp_curv 152,
  spec_flatness 138, spec_spread 91, phase_resid_std 65, dc_offset 65, … min if_iqr 11.4.
  **CFO dominates** — the P6 narrowband prediction; quantified in Stage-B CFO-ablation.
- **Family ablation (kNN-1):** drop F-CFO→0.937 (largest), F-IQ→0.958, F-SPEC→0.969, F-ENV→0.975,
  F-MOD→0.984, F-TRANS→0.991 (transient adds little in *matched* condition; its value is
  cross-condition — Stage B tests this).
- **Remaining redundancy:** only `ramp_lin ~ ramp_curv` (r=−0.942), inherent to two poly
  coefficients of one ramp; both kept (P6 family richness), documented.
- **DEV pruning log (why 19, not the first draft):** removed **spec_centroid** (r=0.966 with
  cfo_hz — duplicate carrier-offset), **gain_imbalance** (F=1.1) and **quad_error** (F=0.7) —
  time-domain IQ-imbalance proxies are chance-level on this integrated-transceiver device.
  Added **spec_skew** (frequency-domain LO/imbalance proxy, F=39), **onset_overshoot** (F=159,
  strengthens P6), **if_iqr** (robust IF deviation, F=11).

*Locked 2026-07-15. No eval split was touched to derive any of the above.*
