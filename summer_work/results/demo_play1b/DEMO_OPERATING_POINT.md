# DEMO OPERATING POINT — DEMO-SIDE, NOT PAPER RESULTS

**Locked stack (consumed by the demo build):**

- **Encoder checkpoint:** Play-1 native-from-scratch, `runs/drone_native/seed2024/best.pt`
  (step 4000); secondary seed1234 for robustness.
- **Feature:** 512-D head-free (`get_encoder_output`), native 4096-sample windows @ 50 MS/s,
  STFT n_fft=256/hop=64, per-window unit-power standardize.
- **Burst construction (track):** group windows by track continuity; accumulate **N* = 120**
  windows/track (mean-pool, L2-normalize). Balanced multi-condition aggregation.
- **K-estimator:** **eigengap**.
- **Clustering / partition:** the eigengap partition (assisted mode = same algorithm forced to
  the operator-supplied K).

**Expected demo success (scenario success = correct K AND partition ARI>=0.6), best-of-seed:**

| scenario | unassisted | assisted (K known) |
|----------|-----------|--------------------|
| K=3      | 0.661 | 1.000 |
| K=4      | 0.500 | 1.000 |
| K=8      | 1.000 | 1.000 |

**N* = 120** (smallest N within 0.03 of best oracle-K@8 on primary seed2024).

**Gate:** WORKABLE: K<=4 unassisted=0.661 (0.5-0.7) / assisted=1.000 -> ship with assisted mode (eigengap@N120).
