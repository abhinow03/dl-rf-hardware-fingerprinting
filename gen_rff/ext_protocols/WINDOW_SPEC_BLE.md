# WINDOW_SPEC_BLE.md — D2 BLE (XIAO-ESP32C3×31) preprocessing spec

*Phase 1. Authority: `EXT_PROTOCOL_PLAN.md` (§1.3 native-spec doctrine, §4 Phase 1).
Decisions are **made here, not deferred**. No training, no model forward. All shapes below
were printed from a real batch of the actual `.npy` data (not asserted from the README).*

## 1. Native window — INHERITED (no re-windowing, no burst detection)
- **Window = the authors' segment as-shipped:** `X = (N, 2, 1850) float32`, I/Q, **1850 samples
  @ 6 MS/s = 308.36 µs**, onset-aligned, **per-segment amplitude-normalized** (peak |z| ≈ 0.0284,
  CV 0.019 — Phase 0b).
- **No re-segmentation, no energy/burst detector.** Phase 0b established each window is already a
  complete onset-aligned burst (noise-floor → ~2 µs turn-on ramp → constant-envelope GFSK body,
  duty 0.988 → falloff). Re-windowing would only discard the authors' onset registration.
- **Consequence logged (Phase-0b caveat, verbatim):** *"at 6 MS/s the onset is resolved to only
  ~12 samples (~2 µs), and the data is per-segment amplitude-normalized — so the transient's
  **shape** is available to a feature extractor but its absolute **amplitude/scale** is not; and
  the ~2 µs ramp cannot be cleanly separated into device-PA turn-on vs receiver AGC/filter
  settling from the bytes alone."*
- **Per-window standardization (plan §1.3):** data is already per-segment amplitude-normalized;
  Approach-A/B encoders apply their own zero-mean/unit-power standardization at load time on the
  (2,1850) window. No scale features survive — features must be shape/relative (documented for
  Phase 2 Approach-A: use CFO, envelope-shape/rise-curvature, spectral moments — NOT absolute
  power/RSSI proxies).

## 2. Encoder input shapes (dual-branch RFEncoder, native arm)
Printed from an 8-sample real batch of `Wired/Ch1_R1/X_train.npy`:
- **1D branch input:** `(B, 2, 1850) float32` — raw I/Q, fed directly.
- **2D branch (STFT):** `scipy.signal.stft(z, fs=25e6*, nperseg=128, noverlap=64, boundary=None,
  padded=False)` on complex `z = I + jQ`, complex full-band (return_onesided=False) →
  `Zxx (B, 128, 27)`. Stored as **2 real planes (real, imag)** → **2D-branch tensor
  `(B, 2, F=128, T=27) float32`**.
  - `nperseg=128` (≈ 21.3 µs at 6 MS/s) balances frequency resolution (128 bins over the ~1 MHz
    GFSK channel) against time frames; `hop=64` (50 % overlap) → **T=27** frames over the 1850-sample
    window. Chosen over (256,128)→T=13 (too few frames) and (64,32)→F=64 (coarse in frequency).
  - *fs is nominal; STFT bin geometry is unaffected by the absolute fs label — it drives only the
    physical axis units, not tensor shape.*
- **Sanity:** both branch inputs derive from the SAME (2,1850) window; batch dims agree; no NaN/inf.

## 3. NATIVE arm storage — ZERO-COPY (no duplicate window cache)
- **No cache written for the native arm.** Loaders index the existing per-collection `.npy`
  in place via `(collection, row)` with `np.load(..., mmap_mode='r')`; the STFT is computed
  on-the-fly per batch (cheap: (128,27) STFT over 2×1850).
- **Fixity** = the Phase-0b per-file sha256 manifest (`audit_out/fixity_sha256.md`, 24 `.npy`).
  The native arm adds **zero** bytes to disk.

## 4. B1 commensurability arm — resample 6 → 25 MS/s (WiSig-frozen transfer)
The frozen WiSig encoder expects **256 samples @ 25 MS/s**. BLE is 6 MS/s, so this arm — and ONLY
this arm — resamples to the WiSig spec (plan §1.3: "that arm alone uses resampling").
- **Resample:** `scipy.signal.resample_poly(x, up=25, down=6, axis=time)` (polyphase 25/6) →
  `1850 → 7709 samples` (still 308.36 µs, now @ 25 MS/s). Applied to (2,1850) → (2,7709) real I/Q,
  batched over segments (measured 121 µs/seg; full 877 k segs ≈ 106 s).
- **Slice to WiSig windows:** 256-sample windows, **stride 256 (non-overlapping)** →
  `floor((7709-256)/256)+1 = 30 windows/segment`.
- **Aggregation rule:** frozen-encoder embeds each 256-window; **segment embedding = mean-pool over
  its 30 window-embeddings** (burst-mean integration, plan §2 locked harness). Mean-pool happens in
  Phase 3 (GPU); Phase 1 caches only the resampled windows (the encoder INPUT).
- **Cache size decision (by printed numbers):** 877,418 segs × 30 win × (256×2×4 B) = **53.9 GB**.
  Gate = subsample only if > 80 GB → **not triggered**; build full at stride 256. (stride 128 →
  106 GB would breach the gate; stride 256 is the non-overlapping tiling and is kept.)
- **Cache location & fixity:** docker volume `/home/docker/pw26_akp_01/ext_cache/ble_b1_25msps/`
  (613 GB free; op must not leave < 20 GB free — 53.9 GB leaves ~559 GB ✓). One `.npz` per
  collection (`X_win (N,30,2,256) f32`, `y_idx (N,)`, `seg_row (N,)`), **sha256 per file** →
  `audit_out/b1_cache_fixity.md`. Never enters git (outside repo + gitignored artifacts).

## 5. What is NOT done here (Phase 2+)
No features, no encoders, no training, no clustering. Approach-A feature derivation, B1/B3 forwards,
and the T1–T3 battery are Phase 2/3.
