# GEN-RFF PHASE 4/5 — Systems-Level Protocol Router (deployable capstone demo)

*DEMO-SIDE / R&D sandbox on branch `experimental/generalized-rff`. Not part of the locked
paper; nothing here trains or modifies frozen assets. This is [FINDINGS.md](../FINDINGS.md)
§3 ("router over a monolithic generalist") made executable. Phase 5 adds the **BLE tier** from
[EXT_FINDINGS.md](../ext_protocols/EXT_FINDINGS.md).*

**Standing exception — mavicAir2 as demo content.** Everywhere else in this project mavicAir2
is a sealed eval group. **Here it is deliberately used as the demo's unknown-emitter content**
(the drones the base station must discover). It is never used to train the router (routing is
learned on WiSig + non-mavicAir2 DRFF + BLE *train* units only) and the tier encoders are frozen,
so no selection touches it. TEST/board-18 and BLE `eval_units` stay closed to router fitting.

## Architecture (Phase 5 — config-driven tier registry)

```
 raw IQ stream(s)
   │
   ▼  [1] TRACKER (replay.py) ── each emitter → one track per receiver (R=4), windows dealt
   │        round-robin in capture/session order; grouping from continuity ONLY, never labels
   ▼  [2] PROTOCOL ROUTER (router.py) ── RATE-AWARE spectral stats (absolute-frequency) →
   │        {wifi, drone_ocusync, ble, UNKNOWN} + confidence; per-class Mahalanobis novelty → UNKNOWN
   ▼  [3] TIER ENCODER (tiers.py) ── one encode() interface over a config-driven TIER_REGISTRY:
   │        T-A drone_native (4096@50, 512-D) · T-B frozen WiSig (256@25, 512-D) ·
   │        T-C fallback = frozen WiSig + classical-19 (UNKNOWN, 512-D) ·
   │        T-D BLE native B3 (1850@6, 128-D forward head, RAW — no calibration)
   ▼  [4] ACCUMULATOR (accumulate.py) ── per track, mean-pool + L2 @ N*=120 (partial flag if short)
   ▼  [5] BASE STATION (base_station.py) ── pool PER PROTOCOL GROUP; eigengap K (or assisted-K);
   │        spectral partition; namespaced fingerprint_id d<k>.<protocol>  (d*.drone / d*.wifi / d*.ble)
   ▼  global IDs + JSON  (out/associated_stream.jsonl → AR / multilateration layer)
   scoring.py (ONLY module that reads ground truth): routing acc, per-group ARI, correct-K
```

### TIER_REGISTRY (config-driven; adding a protocol = adding a row)

| protocol | tier | checkpoint | backend | dim | aux |
|---|---|---|---|---|---|
| `drone_ocusync` | T-A | `runs/drone_native/seed2024/best.pt` | `deep512` (get_encoder_output) | 512 | — |
| `wifi` | T-B | `runs/wisig_supcon_fft64/retrain_best/best_model.pt` | `deep512` | 512 | — |
| `unknown` | T-C | *(shares the frozen WiSig encoder)* | `deep512` + classical-19 | 512 | ✓ |
| `ble` | **T-D** | `runs/ble_native_s2024/best.pt` (sha256 `f1d7745e…`) | **`native128`** (forward head, native STFT nfft=128/hop=64) | **128** | — |

T-B and T-C share **one** frozen WiSig encoder instance (unchanged from Phase 4). The JSON
[contract](contract.py) is now **per-protocol dim** (512 for deep tiers, 128 for BLE); mixed-dim
associated streams validate cleanly (0 errors). Clustering is always **within** a protocol group,
so the differing embedding dimensions never meet.

## Design choices ↔ EXT_FINDINGS evidence (BLE tier)

| Choice | Why (EXT_FINDINGS) |
|---|---|
| **BLE tier = B3 native encoder** | B3 owns the **pooled** regime the base station runs in: pooled/diagnostic N=120 ARI **0.714** vs B1 0.284 / A 0.193 (EF2), best receiver-transfer 0.759 (EF7). |
| **Assisted + eigengap-K** | B3's deployable eigengap-K is the closest to truth (raw est-K 8.6 vs true 8, EF4); assisted-K remains the shipping primary, eigengap the unassisted secondary. |
| **RAW embeddings, no per-collection calibration** | B3 is **calibration-indifferent** — collection η² ≈ 0.000, so per-collection robust standardization is a no-op (EF5 / W3). The tier ships raw; only A needed calibration. |
| **N\* = 120 accumulation (unchanged)** | The wall is N-dependent (EF1 / F1); N=120 burst integration is where cross-condition discovery is viable for every tier. |
| **`full_s2024` seed frozen (not `s1234`)** | Near-tie on pooled discovery, but `full_s2024` holds a **+0.144 receiver-transfer margin**, and base-station pooling is receiver-shaped (EF7). |

## Router: rate-awareness (Phase-5 fix)

Frequency-dimensioned features (centroid, spread, 85%-rolloff, hop-rate proxy) and the
occupied-bandwidth fraction are now expressed in **absolute frequency**, referenced to the legacy
WiSig rate (`REF_RATE = 25 MS/s`): `value_abs = value_norm × (sample_rate / REF_RATE)`.

- **At 25 MS/s the factor is exactly 1.0 → every legacy WiSig feature vector is numerically
  unchanged** (verified bitwise, max|Δ| = 0.0e0 — regression evidence).
- Drone (50 MS/s) freq-dims scale ×2.0 exactly; BLE (6 MS/s) ×0.24 — so a 1 MHz GFSK BLE burst
  now reads as genuinely **narrowband** instead of "1/6 of the band". This is what lets the router
  separate BLE from wide WiFi/drone occupancy.

The novelty gate is a per-class Mahalanobis distance (99th-pct training threshold). **Named risk
retired:** a narrowband FM tone could in principle now look like narrowband BLE — measured, it does
not: the canonical FM sweep sits at Mahalanobis **78.6 from the BLE class** (its farthest class),
6.29 from drone (nearest), > threshold 5.59 → correctly UNKNOWN.

## Operating point

R = 4 receivers · N\* = 120 windows/track (mean-pool + L2) · per-protocol eigengap K-estimate
(spectral partition), assisted-K supported · router = LogisticRegression on 10 **rate-aware**
spectral features + per-class Mahalanobis(99th-pct) novelty gate over **4 classes**
(wifi / drone_ocusync / ble / unknown).

## JSON contract (Design-2 + `protocol` + per-protocol-dim amendment — see [contract.py](contract.py))

Receiver emits everything except `fingerprint_id`; the router adds `protocol` (+`protocol_conf`);
the base station adds `fingerprint_id` (namespaced). Example **BLE** associated message (128-D):

```json
{
  "receiver_id": "rx0", "track_id": "ble:3__rx0__s0__rep0", "timestamp": 0.0,
  "embedding": [ "...128 floats..." ], "rssi": -80.0, "aoa_deg": 46.08, "tdoa_ns": 0.0,
  "n_windows": 120, "partial": false, "class": "unknown",
  "protocol": "ble", "protocol_conf": 0.994, "fingerprint_id": "d1.ble"
}
```
`rssi`/`aoa_deg`/`tdoa_ns` are MOCKED from a fixed synthetic geometry and are **never** used for
grouping (clustering reads `embedding` only — runtime-proven label/geo invariance).

## Scenarios & results (5 repeats each; `results_gen/demo_router/scenario_results.csv`)

| scenario | mode | routing | success | mean group ARI |
|---|---|---|---|---|
| S1 drone×3 | assisted / unassisted | 1.00 | 1.00 / 1.00 | 1.00 |
| S1 drone×4 | assisted / unassisted | 1.00 | 1.00 / 0.80 | 1.00 / 0.857 |
| S2 mixed (2 drone + 2 wifi) | assisted / unassisted | 1.00 | 1.00 / 1.00 | drone 1.00, wifi 1.00 |
| S3 unknown (2 drone forced UNKNOWN) | assisted (K=2) | 1.00→UNKNOWN | report-only | 1.00 (T-C fallback) |
| **S4 ble×K, single-collection** (K=2,3,4) | assisted / unassisted | 1.00 | 1.00 | ble 1.00 |
| **S4 ble×K, pooled-collection** (K=2,3,4) | assisted / unassisted | 1.00 | 1.00 | ble 1.00 |
| **S6 grand-mixed (2 drone + 2 wifi + 2 ble)** | assisted / unassisted | 1.00 | 1.00 | drone / wifi / ble = 1.00 · 0 ID collisions |

**S1–S3 reproduce the Phase-4 build byte-for-byte** (regression gate: 8/8 cells identical).

**Gates:** R1 router acc 0.973 (≥0.95) · G-FM FM-sweep→UNKNOWN ✓ · G-RT 4-class held-out routing
1.00 (≥0.95) ✓ · G-REG S1–S3 8/8 ✓ · G-S4 pooled min routing 1.00 & assisted success 1.00 (≥0.8) ✓ ·
G-S6 routing 1.00, per-group assisted success 1.00 (≥0.8), 0 cross-protocol collisions ✓.
Label-confinement: proven (clustering reads embeddings only; ground truth confined to scoring.py).

## Honest caveats

- **S4 BLE cells are the small-K regime (K = 2–4).** ARI 1.00 there does **not** contradict the
  8-unit battery headline (pooled/diagnostic 0.714, EXT_FINDINGS EF2): fewer same-model units at
  N=120 with a collection-invariant encoder is the easy end. The demo runs K ≤ 4; scaling to the
  full 8+ same-model population inherits the battery's harder numbers.
- **Assisted-K is primary.** Unassisted eigengap holds here for K ≤ 4 but is the documented weak
  link at higher K (EF4). Ship with "operator supplies N".
- **BLE is single-dataset (D2), single-protocol, no day/aging axis, 2-unit val-selection**
  (EXT_FINDINGS EF8). The frozen tier's checkpoint was selected on a 2-unit validation ROC-AUC that
  saturated (W5); external validity (Zigbee / D1) is future work.
- **T-C is best-effort, not a trained tier** (its S3 number is honesty-tier, reported, no bar).
- **Single-receiver source data / mocked geometry.** rssi/aoa/tdoa are synthetic; the receiver
  round-robin is a replay device, not true simultaneous multi-sensor capture.
- **Zigbee placeholder.** No Zigbee tier is enrolled — **enrollment pending D1 access; a Zigbee
  emitter routes to UNKNOWN/T-C by design until then.**

## Protected assets

Frozen, off-repo (gitignored `summer_work/runs/`, read-only), sha256-verified before/after:
`best_model.pt` (WiSig) `03898f49…` · `drone_native seed2024` `2ef7fc25…` ·
**`ble_native_s2024` `f1d7745e…`** (the BLE tier T-D). See the PROTECTED ASSETS table in
[EXT_FINDINGS.md](../ext_protocols/EXT_FINDINGS.md#phase-5--router-integration-ble-tier-frozen-asset-registration).

## Run

```bash
cd ~/CAPSTONE/DL_model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  ~/rf_env/bin/python3 -u -m gen_rff.demo_router.run_demo
```
Outputs: `results_gen/demo_router/{scenario_results.csv, phase5_router_report.json}`,
`results_gen/demo_router/out/associated_stream.jsonl`.
