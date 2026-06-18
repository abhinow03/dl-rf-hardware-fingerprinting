# summer_work/ — Open-World Emitter Discovery (the pivot)

Fast-reload context for future sessions. Read this first, then `PAPER_PLAN.md` and
`NEIGHBOR_ANALYSIS (1).md` in `~/CAPSTONE/`.

## What this is
A new pipeline built **on top of** the existing V4 encoder (`../model.py`, `../losses.py`),
not a fork. The capstone (closed-set RFFI on ORACLE, V4) stays as a benchmark.

## The contribution (locked)
> Open-world **discovery** of **individual same-model** emitters under **cross-domain transfer**.

Three novelty axes, none individually novel, but their intersection is empty in the
literature: (1) individual same-model device discrimination, (2) open-world discovery
(group unknowns, not just reject), (3) cross-domain transfer. Every prior *discovery*
paper is single-domain; every *transfer* paper is closed-set. That gap is ours.

## Why the DroneRF detour was abandoned
DroneRF has only 3–4 drone *models* and they differ by **protocol/waveform**, so the
model cheated by learning protocol, not hardware fingerprint — the exact closed-set trap.
Fix: learn the metric from MANY emitters of the SAME protocol (WiSig, 150 WiFi Tx) so the
only discriminator is hardware imperfection; then discover on same-model units.

## Datasets / roles
- **WiSig ManyTx** (`~/CAPSTONE/phase3_dataset/ManyTx.pkl`, 150 Tx, 256-sample IQ) —
  metric-learning pool. Many same-protocol identities.
- **WiSig ManySig** (`ManySig_zf.pkl`, 6 Tx × 12 Rx × 4 days) — cross-receiver/day
  transfer benchmark (baseline vs DRIFT etc.).
- **ORACLE** (`~/Desktop/neu_m044q5210./KRI-16Devices-RawData/`, 16 USRP, 8/14/20/26ft) —
  second within-domain benchmark; V4 already works.
- **Drones** (M100 7 units, DRFF-R2) — APPLICATION target, added LAST; downloads stalled,
  not needed for the first 80% of the work.

## Locked decisions (see session prompt Part 3 for full list)
- Train the metric **fresh** on WiSig ManyTx — do NOT seed from ORACLE V4 weights.
- 256-sample packets → STFT **n_fft=64, hop=16** → spectral input ≈ [2, 33, ~13]. Same
  config must stay valid for 4096-sample drone windows later (encoder GAP is length-agnostic).
  Do NOT concatenate packets (WiSig packets aren't time-contiguous).
- Clustering = `sklearn.cluster.HDBSCAN` (do NOT build standalone hdbscan). Cluster on
  RAW 128-D L2-normalized embeddings. DBSCAN + agglomerative as comparisons. UMAP/t-SNE
  for VISUALIZATION ONLY — never cluster on reduced dims.
- Metrics: ARI (headline), NMI, V-measure, purity, count error |K_est − K_true|.
- Splits: discovery set FULLY device-disjoint from metric-training set. Write exact
  held-out Tx IDs to disk every time.
- Run naming: `runs/wisig_supcon_fft64/` (descriptive; drop phaseN convention).

## Board / Tx assumption (LOCKED — protects the core claim)
WiSig ManyTx = 150 Tx across **20 boards**; Tx IDs encode **`board-antenna`** (e.g. `8-13`
= board 8, antenna 13). Tx-per-board ranges 2–16.

**Assumption (conservative, not verified from the pickle):** same-board antennas SHARE an RF
front-end (oscillator / PA / mixer) — i.e. they are not fully independent radios.
**Consequence:** the train/discover split MUST be **board-disjoint**, never per-Tx. A per-Tx
split could place sibling antennas of one physical front-end on both sides, leaking hardware
identity and inflating discovery scores — which would invalidate the "individual same-model
device discovery" claim. All ManyTx splits use `split_by='board'`.

Locked split: `runs/wisig_supcon_fft64/splits/split_manytx.json` — board-disjoint, seed 42,
**109 train Tx (15 boards) / 41 discover Tx (boards 4,5,16,18,20)**. Kept at 109/41 (richer
41-device discovery test); not tuned toward 120/30.

## Layout
- `shared.py` — re-exports parent `RFEncoder`, `SupervisedContrastiveLoss`, `get_temperature`.
- `datasets/` — one loader module per source (WiSig ManyTx first).
- `train/` — metric-encoder training scripts.
- `discover/` — open-world discovery engine (HDBSCAN + count + metrics).
- `runs/` — outputs, logs, locked split files, checkpoints.

## Roadmap (do not run ahead)
1. Setup ✅  2. WiSig ManyTx loader + device-disjoint split  3. Train metric encoder
(SupCon, FFT=64)  4. Discovery engine + validate on held-out WiSig  5. ORACLE through
new discovery  6. Cross-Rx/day transfer on ManySig  7. Classical baselines
(OpenMax, Proto+EVT, threshold).  Later: drones as headline target.

## Guardrails
GroupNorm only · SupCon NaN guard stays · `rm -rf __pycache__` before runs · all splits
device-disjoint with IDs on disk · cluster on raw embeddings · train fresh on WiSig ·
watch for receiver/date leakage (the analog of ORACLE's distance leakage).
