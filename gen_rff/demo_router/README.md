# GEN-RFF PHASE 4 — Systems-Level Protocol Router (deployable capstone demo)

*DEMO-SIDE / R&D sandbox on branch `experimental/generalized-rff`. Not part of the locked
paper; nothing here trains or modifies frozen assets. This is [FINDINGS.md](../FINDINGS.md)
§3 ("router over a monolithic generalist") made executable.*

**Standing exception — mavicAir2 as demo content.** Everywhere else in this project mavicAir2
is a sealed eval group. **Here it is deliberately used as the demo's unknown-emitter content**
(the drones the base station must discover). It is never used to train the router (routing is
learned on WiSig + non-mavicAir2 DRFF only) and the tier encoders are frozen, so no selection
touches it. TEST/board-18 stays closed.

## Architecture

```
 raw IQ stream(s)
   │
   ▼  [1] TRACKER (replay.py) ── each emitter → one track per receiver (R=4), windows dealt
   │        round-robin in capture/session order; grouping from continuity ONLY, never labels
   ▼  [2] PROTOCOL ROUTER (router.py) ── cheap rate-normalized spectral stats → {wifi,
   │        drone_ocusync, UNKNOWN} + confidence; Mahalanobis novelty gate → UNKNOWN
   ▼  [3] TIER ENCODER (tiers.py) ── one encode() interface, three tiers:
   │        T-A drone_native seed2024 (4096@50) · T-B frozen WiSig (256@25) ·
   │        T-C fallback = frozen WiSig + classical-19 (UNKNOWN)
   ▼  [4] ACCUMULATOR (accumulate.py) ── per track, mean-pool + L2 @ N*=120 (partial flag if short)
   ▼  [5] BASE STATION (base_station.py) ── pool PER PROTOCOL GROUP; eigengap K (or assisted-K);
   │        spectral partition; namespaced fingerprint_id d<k>.<protocol>
   ▼  global IDs + JSON  (out/associated_stream.jsonl → AR / multilateration layer)
   scoring.py (ONLY module that reads ground truth): routing acc, per-group ARI, correct-K
```

## Design choices ↔ FINDINGS evidence

| Choice | Why (FINDINGS) |
|---|---|
| **Router over a monolithic generalist** | The generalist line is CLOSED (§5.2): multi-domain SupCon, physics injection, residual, cross-condition, capacity, and channel-adversarial invariance all fail to beat frozen transfer at matched N (F2/F3/F7). Route-then-specialize is the honest deployable answer (§3). |
| **T-A native drone encoder** | Native training is the only method clearly above the N=120 averaging band (F1: native 0.792/0.835 vs frozen 0.729); it also gives clean count estimation (F8). |
| **T-B frozen WiSig encoder** | In-domain WiFi discovery; the specialist reference (DEV P2 ≈ 0.72) far exceeds any generalist (F7). |
| **T-C fallback = frozen WiSig + classical-19** | The validated *untrained* fallback: at N=120 frozen WiFi features reach 0.729 and classical-19 0.713 on unseen drones, eigengap K7/ARI 0.68 (F1). Adequate best-effort when the protocol is unenrolled. |
| **N*=120 accumulation** | The wall is N-dependent (F1): burst integration to N=120 closes most of the cross-domain gap for any feature set. |
| **Per-protocol-group clustering (never across groups)** | Protocols are separable up front; mixing them would only add confusable structure. Namespaced IDs keep drone/wifi discovery independent. |
| **Assisted-K as primary** | K≤4 unassisted discovery is the weak link (Play-1b + F-line); assisted ("operator says N") is the shipping mode, unassisted reported. |

## Operating point

R = 4 receivers · N* = 120 windows/track (mean-pool + L2) · per-protocol eigengap K-estimate
(spectral partition), assisted-K supported · router = LogisticRegression on ~10 spectral
features + Mahalanobis(99th-pct) novelty gate.

## JSON contract (Design-2 + `protocol` amendment — see [contract.py](contract.py))

Receiver emits everything except `fingerprint_id`; the router adds `protocol` (+`protocol_conf`);
the base station adds `fingerprint_id` (namespaced). Example associated message:

```json
{
  "receiver_id": "rx0", "track_id": "drone:mavicAir2_1__rx0__s0__rep0", "timestamp": 0.0,
  "embedding": [ "...512 floats..." ], "rssi": -80.0, "aoa_deg": 46.08, "tdoa_ns": 0.0,
  "n_windows": 120, "partial": false, "class": "unknown",
  "protocol": "drone_ocusync", "protocol_conf": 0.997, "fingerprint_id": "d1.drone"
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

**Gates:** R1 router acc 0.973 (≥0.95) + FM-sweep→UNKNOWN ✓ · G-S1 assisted ≥0.8 ✓ ·
G-S2 routing ≥0.95 & assisted ≥0.8 ✓ · G-S3 reported ✓. Label-confinement: proven (zeroing
geometry + scrambling ids leaves the partition identical).

## Honest caveats

- **Assisted-K is primary.** Unassisted eigengap is correct for K≤3 and mixed protocols but can
  miss K on 4 same-model drones (S1 drone×4 unassisted 0.80). Ship with "operator supplies N".
- **K ≤ 4 regime.** Same-model discovery beyond a few units inherits the single-fabrication-batch
  hardness documented for WiSig board-18 (locked EVAL_PROTOCOL §4.4) and the drone Play-1b limits.
- **T-C is best-effort, not a trained tier.** Its S3 number is the honesty tier — reported, no bar;
  it leans on burst integration (F1), not a drone-native representation.
- **The N=120 "success" is integration-assisted** (F1): most of the discovery quality at N=120 is
  averaging that any feature set gets — the router's value is *routing + per-protocol specialization*,
  not a novel cross-domain representation (the generalist line is closed, §5.2).
- **Single-receiver drone data / mocked geometry.** rssi/aoa/tdoa are synthetic; the receiver
  round-robin is a replay device, not true simultaneous multi-sensor capture.

## Run

```bash
cd ~/CAPSTONE/DL_model
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  ~/rf_env/bin/python3 -u -m gen_rff.demo_router.run_demo
```
Outputs: `results_gen/demo_router/{scenario_results.csv, phase4_router_report.json}`,
`results_gen/demo_router/out/associated_stream.jsonl`.
