# CAPSTONE DEMO — offline RF-fingerprint discovery pipeline

*DEMO-SIDE — NOT PAPER RESULTS.* No training happens here. `best_model.pt`, the TEST split
(board-18), and the M100 cache are untouched. The encoder is the frozen Play-1 native
checkpoint `summer_work/runs/drone_native/seed2024/best.pt`.

This build turns raw drone IQ into **discovered global emitter IDs**, implementing the locked
operating point from [`DEMO_OPERATING_POINT.md`](../summer_work/results/demo_play1b/DEMO_OPERATING_POINT.md)
under the **Design-2 base-station association** architecture.

## Pipeline (raw IQ → global IDs → JSON)

```
 replay.py      raw 50 MS/s IQ  ->  native 4096 windows  ->  round-robin deal to R
 (RF world)     receiver-drones ->  one track per receiver-emitter (accumulate <= N*=120)

 receiver.py    per track: unit-power -> STFT 256/64 -> frozen encoder (get_encoder_output,
 (drone-side)   512-D head-free) -> mean-pool + L2 = track fingerprint
                -> emit contract JSON {embedding, MOCK rssi/aoa/tdoa, class:"unknown"}

 base_station.py pool all track embeddings -> eigengap K (or assisted K) -> spectral
 (discovery)     partition -> assign global fingerprint_id d1..dK  [reads embeddings ONLY]

 scoring.py      join fingerprint_id <-> ground-truth emitter (LABELS USED FOR SCORING ONLY)
 (eval)          -> ARI, correct-K, confusion, association table
```

Supporting modules: [`encoder_frontend.py`](encoder_frontend.py) (window + STFT + encoder),
[`geometry.py`](geometry.py) (fixed synthetic layout → mock rssi/aoa/tdoa),
[`contract.py`](contract.py) + [`JSON_CONTRACT.md`](JSON_CONTRACT.md) (frozen message schema).

## Run it

```bash
cd ~/CAPSTONE/DL_model/demo
# env vars are also set inside every entrypoint; setting them here is belt-and-suspenders
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  ~/rf_env/bin/python3 -u run_demo.py all      # or: T1 | T2
```

> **Single-threaded BLAS is required.** Multithreaded BLAS segfaults natively on the many
> small linear-algebra calls (eigengap / spectral). Every entrypoint sets
> `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` on import.

Outputs: `../summer_work/results/demo_build/{scenario_results.csv, report.json}` and
`out/{associated_stream.jsonl, receiver_stream.jsonl}` (the streams feed the downstream AR
visualization layer).

## Scenarios

| id | tier | emitters | K |
|----|------|----------|---|
| **T1** | mixed-model | `mini3pro_2`, `mini4PRO_2` (the two never-trained val airframes) + `mavicAir2_1`, `mavicAir2_2` | 4 |
| **T2a** | same-model | `mavicAir2_{1,2,3}` | 3 |
| **T2b** | same-model | `mavicAir2_{1,2,3,4}` | 4 |

Each runs in both **assisted** (operator supplies K) and **unassisted** (eigengap estimates K)
modes, ×5 repeats with different segment draws. Success = *correct K AND partition ARI ≥ 0.6*
(unassisted) / *ARI ≥ 0.6* (assisted).

## The operating point (locked, from Play-1b)

Encoder `seed2024/best.pt` · native 4096 @ 50 MS/s · STFT 256/64 · per-window unit-power ·
512-D head-free · **N\*=120** windows/track (mean-pool + L2) · **eigengap** K-estimate ·
**spectral** partition. Assisted mode forces the same spectral partition to the supplied K.

## Honest caveats

- **Assisted is the primary shipping mode.** The Play-1b operating point ships with
  assisted-K because unassisted K-estimation across *arbitrary* mavicAir2 unit combinations is
  the weak link (all-subsets averages ≈ 0.66 for K=3, 0.50 for K=4). The demo also exercises
  unassisted for completeness.
- **These fixed unit sets are a favorable draw.** T2 uses the specific units `{1,2,3}` and
  `{1,2,3,4}`, which are well-separated; with only R=4 near-identical tracks per emitter they
  cluster cleanly, so the demo's T2 unassisted success is far above the all-subsets ceiling.
  The honest same-model difficulty is the Play-1b all-combinations number, not this run.
- **Track-grouping framing.** A "track" is one receiver's time-contiguous observation of one
  emitter; grouping into a fingerprint comes only from embedding similarity, never labels.
- **`mini4PRO_2` is data-limited** (~268 native windows). At R=4 its tracks accumulate ~67
  windows rather than the full 120 — reported honestly via each message's `n_windows`. Cross-
  model separation keeps T1 robust regardless.
- **Small-graph neighbor scaling.** The locked eigengap/spectral recipe uses `n_neighbors=15`;
  on the demo's tiny graphs (12–16 tracks) that is nearly a complete graph and erases
  structure, so `base_station._nn` scales it to `n//4` for small n while preserving 15 at the
  operating-point scale (n ≥ 60). This does not change the operating point.
- **Ground truth is only in `scoring.py`.** The simulated world (`replay`, `geometry`)
  legitimately knows which airframe produced a track — that is the RF reality. The discovery
  path (receiver → base station) never reads it; `base_station.associate` takes no label and
  reads only `embedding`. The runtime label-free proof (G3) zeroes rssi/aoa/tdoa and scrambles
  ids, then requires the clustering to be identical.

## Gates (all PASS in the shipped run)

- **G1** T1 success ≥ 0.8 both modes · **G2** T2 assisted success ≥ 0.8 ·
  **G3** grouping provably label-free (code-path audit + runtime invariance proof) ·
  **G4** JSON validates against the contract; embeddings present; `fingerprint_id` assigned
  only at the base station.
