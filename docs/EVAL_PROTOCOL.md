# EVAL_PROTOCOL.md — Open-World Emitter Discovery Evaluation Protocol

**Status:** LOCKED rules for all future discovery tuning + reporting. Written this housekeeping
session (no training/clustering was run to produce it — it pins the rules that already govern the
numbers reported to date). Items marked **UNDER AUDIT** describe *current code behavior verbatim*
and are candidates for a controlled Step-2 audit; they are documented, **not changed**, here.

Canonical eval code referenced below: `discover/geometry_consolidate.py` (the most recent eval
script; `geometry_stage1.py` / `stage2.py` use identical metric/burst/noise logic).

---

## 0. Split arithmetic (audited from code + pkl, this session)

| quantity | value | source |
|---|---|---|
| Full WiSig catalog (plan-level, aspirational) | **174** Tx | `PAPER_PLAN.md:73`, `NEIGHBOR_ANALYSIS.md:174` — NOT this pkl |
| `ManyTx.pkl` actual subset on disk | **150** Tx across **20** boards | `pickle.load(ManyTx.pkl)['tx_list']` len = 150 |
| Devices dropped by filtering | **0** | loader keeps every Tx; 0 zero-signal Tx at eq=0 |
| Train (fit) devices | **109** | `split_manytx.json` `n_train` |
| Discover (held-out) devices | **41** | `split_manytx.json` `n_discover` |
| Held-out boards (grid rows) | 5 → rows **4, 5, 16, 18, 20** | `held_out_boards` |
| Train boards (grid rows) | 15 rows | complement |

**174 → 150:** the 174 is the size of the *full published WiSig transmitter catalog* cited in the
paper plan; the `ManyTx.pkl` we actually use is the 150-Tx ManyTx subset. There is **no min-signal
threshold or filtering** dropping devices — `load_manytx()` iterates all 150 Tx and every one has
>0 signals at eq=0 (verified this session). **150 → 109/41** is the board-disjoint split
(`make_split(split_by='board', n_discover=30, seed=42)`; it accumulates whole boards until
≥30 devices, landing on 41 across 5 rows).

**Disjointness (G1, verified):** no board appears on both sides (train rows ∩ discover rows = ∅);
no device appears on both sides. Assertion passes.

**Seed-123 slice (G3, verified):** the "18-device seed-123 slice" is drawn by `scatter_slice(held_tx,
18, 123)` **from the 41 discover devices only** (round-robin across the 5 held-out rows, seed 123).
Its 18 IDs are
`16-1, 16-19, 16-20, 16-5, 18-11, 18-16, 18-20, 18-7, 20-1, 20-14, 20-4, 4-1, 4-10, 4-11, 5-1, 5-16, 5-20, 5-5`
— all ⊆ the 41 discover pool, **zero overlap with the 109 training devices**. No leakage.

---

## 1. Metric definitions

Scored over **ALL** points with the locked noise rule (§2): each `-1` becomes its own singleton
cluster; ARI, NMI, purity share one support.

- **ARI** — `sklearn.metrics.adjusted_rand_score(true, pred_singletons)`. Chance = 0, perfect = 1.
- **NMI** — `sklearn.metrics.normalized_mutual_info_score(true, pred_singletons)`.
- **purity** — EXACT formula copied verbatim from `discover/geometry_consolidate.py`:
  ```python
  def purity(true, pred):
      if len(true) == 0: return float("nan")
      return sum(np.unique(true[pred == c], return_counts=True)[1].max()
                 for c in np.unique(pred)) / len(true)
  ```
  i.e. for each predicted cluster `c` (**including the noise label `-1`, which counts as one
  cluster here**), take the size of its majority true-device, sum over clusters, divide by the
  **total** number of points (incl. noise). Range (0, 1].
- **|K_est − K_true|** — count error. `K_est = len(np.unique(pred[pred != -1]))` (number of real
  clusters, noise excluded from the count). `K_true` = number of devices in the slice (18 for the
  standard slice). Report both `K_est` and the absolute error.

Also reported: **noise fraction** `= (pred == -1).mean()`; **intra-burst cosine** and
**nearest-other-centroid cosine** for geometry diagnostics.

---

## 2. HDBSCAN noise rule — **LOCKED** (audited 2026-07-08, Step-2 integrity)

Clusterer: `sklearn.cluster.HDBSCAN(min_cluster_size=15, metric="euclidean")` on the
L2-normalized burst-mean points. Points HDBSCAN cannot assign get label **`-1`**.

**LOCKED RULE — every `-1` point is scored as its own singleton cluster; ARI, NMI, and purity are
all computed over the SAME support (ALL points, with singletons); noise fraction is reported
separately.** Verbatim:
```python
def relabel_noise_as_singletons(pred):
    p = pred.copy(); nm = p == -1
    if nm.any():
        start = (p.max() + 1) if (p >= 0).any() else 0
        p[nm] = np.arange(start, start + nm.sum())      # each -1 -> unique new cluster id
    return p

def score_locked(pred, true):
    p = relabel_noise_as_singletons(pred)
    return dict(ARI=adjusted_rand_score(true, p), NMI=normalized_mutual_info_score(true, p),
                purity=purity(true, p),                     # §1 formula, over ALL points
                K_est=len(np.unique(pred[pred != -1])),     # real (non-noise) clusters, reported
                noise=float((pred == -1).mean()))
```

**Why the change (Step-2 finding):** the *old* rule computed ARI/NMI on the **kept** points only
(`pred != -1`) but purity over the **full** set (with `-1` as one big cluster) — two different
supports. That (a) produced the misleading recorded pair NMI 0.804 / purity 0.482 (the low purity
was just the noise cluster diluting a full-support metric while NMI saw only kept points), and
(b) **flattered any method that dumps hard points into noise**: excluding `-1` from ARI/NMI rewards
a clusterer for refusing to cluster. This mattered — under the old rule the CosFace head scored
ARI 0.645–0.667 on the seed-123 slice while labeling **45 % of bursts as noise**; under the locked
rule (which charges those 45 % as wrong singletons) the same head scores **0.350**. All numbers
from 2026-07-08 onward use `score_locked`.

**Purity implementation — VERIFIED CORRECT (audit removed):** the §1 `purity` formula was
re-derived by hand on the seed-123 slice and matches the code to machine precision
(`hand_purity_full == code_purity`, diff < 1e-9) for both 128-D and CosFace. The formula was never
wrong; the problem was the support mismatch above, now fixed by scoring purity over the same
singleton-augmented support as ARI/NMI.

---

## 3. Burst construction rule — **channel-leakage AUDIT PASSED** (2026-07-08)

> **PROVISIONAL PROTOCOL UPDATE (Step-2b decoherence, 2026-07-08 — not final until confirmed on
> TEST):** the scattered gain is **DEPLOYABLE via multi-receiver bursts**. Four burst protocols on
> the same 5 DEV slices (seeds 201–205 × mcs{13,15,17}, locked noise rule), decision table:
>
> | protocol | deployable | 128-D ARI | CosFace ARI | noise 128/CF | CosFace K_est |
> |---|---|---|---|---|---|
> | **P0** consecutive, one (rx,date) | yes | 0.558±0.040 | 0.368±0.022 | 0.16 / 0.44 | 43 (31–61) |
> | **P1** same date, 10 diff receivers | **yes (swarm)** | 0.630±0.029 | **0.772±0.037** | 0.13 / **0.008** | **16.3 (14–17)** |
> | **P2** same rx, across 4 dates | semi | 0.567±0.048 | 0.211±0.031 | 0.17 / 0.13 | 153 (119–189) |
> | **P3** across rx AND dates | no (ceiling) | 0.699±0.055 | 0.765±0.034 | 0.13 / 0.005 | 13.6 (13–14) |
>
> **The gain is RECEIVER diversity, not day diversity:** P1 (multi-rx only) ≈ P3 (multi-rx+day)
> for CosFace (0.772 vs 0.765); P2 (multi-day only) *collapses* CosFace (0.211, K=153). So the
> coherent-burst confound is a **per-receiver channel signature**; averaging across receivers
> removes it and exposes hardware. **CosFace is UN-RETIRED, but ONLY under P1:** it beats 128-D
> under P1 on **5/5** slices (0.783/0.780/0.777/0.817/0.705 vs 0.649/0.652/0.618/0.632/0.600) and
> its coherent-burst count blow-up (K 43) **vanishes** (K 16.3±0.9, noise 0.8 %).
>
> **PROVISIONAL locked protocol → P1 multi-receiver burst + CosFace head (m=0.20, s=32, d128) +
> HDBSCAN(mcs=15) + locked noise rule → ARI 0.772±0.037.** Deployability rests on grouping windows
> by **simultaneous multi-sensor co-observation of one emission (timestamp/rx metadata, NOT device
> id)** — exactly the drone-swarm architecture. WiSig cannot reconstruct true simultaneous captures,
> so P1 here approximates it by same-date/different-receiver grouping; that assumption must be
> stated in any claim. Not final until evaluated once on TEST (board 18) with these frozen
> hyperparameters. Bar it must clear there: honest P0 128-D 0.558. Artifacts:
> `results/step2b_decoherence/`.

**Shuffle-control result (Step-2 Test 2):** each WiSig device has exactly 18 rx × 4 dates × 50 =
**72 channel-coherent cells**, and "consecutive N=10" always falls **inside one (rx, date) cell**
(= one receiver, one day = maximally channel-coherent). We compared, on a DEV slice, **coherent**
bursts (current, 10 consecutive = same session) vs **scattered** bursts (10 windows each from a
DISTINCT (rx, date) cell = up to 48–72 independent receiver-day sessions, sharing only the
transmitter hardware). **Scattered did NOT collapse — it was ≥ coherent** (128-D −0.09, i.e.
scattered *higher*; CosFace scattered 0.802 vs coherent 0.396). Burst-mean is therefore
**denoising toward the hardware centroid, not exploiting within-session channel coherence** — the
fingerprint is hardware, not channel. `min_cluster_size` and consecutive grouping are retained as
the standard protocol; scattered-burst discovery is recorded as a stronger (channel-decohered)
diagnostic. The following is the standard construction (verbatim):

Discovery unit = **burst-mean, N = 10** (locked 2026-07-02). Built by, verbatim:
```python
def consec_windows(tx_data, tx, nwin):        # per-device windows
    iq = tx_data[tx]["iq"]
    return np.stack([W.standardize(iq[k].T.copy())
                     for k in range(min(nwin, iq.shape[0]))]).astype(np.float32)

def burst_pool(Femb, N):                      # group embeddings into bursts of N
    nb = Femb.shape[0] // N
    return unitrows(Femb[:nb*N].reshape(nb, N, -1).mean(1))   # mean-pool + L2-renorm
```

**What "consecutive" means today:** `tx_data[tx]["iq"]` is the Tx's signals **pooled across ALL
receivers and ALL capture dates**, concatenated in loader storage order (rx-major, then date; see
`wisig_manytx.load_manytx`). `consec_windows` takes the **first `min(nwin, N)` rows of that pooled
array** and `burst_pool` groups them into consecutive blocks `[0:N], [N:2N], …`, mean-pools each
block, and L2-renormalizes. So a burst is 10 windows **consecutive in the pooled-array storage
order** — it is **device-blind** (never grouped by true label for scoring) but it is **NOT
guaranteed to be same-receiver or same-capture**: a single burst can straddle the rx/date boundary
wherever the concatenation seam falls.

> Resolved: the shuffle control above (§3 header) ran and **passed** — scattered ≥ coherent, so the
> consecutive/storage-order grouping does not leak a receiver/day cue that inflates the ladder.

---

## 4. Slice-drawing procedure, dev/test partition, and the one-touch rule

**Slice drawing:** `scatter_slice(pool, n, seed)` — group pool devices by board/row, shuffle each
row by `seed`, then round-robin one device per row until `n` are collected (spreads the slice
across grid rows so it isn't a single spatial cluster). Standard slice size `n = 18`.

**Dev/test partition (`splits/discovery_partition.json`, seed 777, board-disjoint):**
the 41 discover devices are split into DEV and TEST by shuffling the 5 held-out boards with
`np.random.default_rng(777)` and accumulating whole boards into DEV until DEV ≥ ⌈0.60·41⌉ = 25,
remainder → TEST. Result:

- **DEV (25 devices, boards 4/5/16/20)** — used for **ALL** future tuning; every tuning slice is
  drawn **only from `dev_tx`** (25-device pool, draw 18-device scattered slices from it).
  `4-1, 4-10, 4-11, 5-1, 5-16, 5-20, 5-5, 16-1, 16-16, 16-19, 16-20, 16-5, 20-1, 20-12, 20-14,
  20-15, 20-16, 20-18, 20-19, 20-20, 20-3, 20-4, 20-5, 20-7, 20-8`
- **TEST (16 devices, board 18)** — touched **exactly once** at the very end with frozen
  hyperparameters; **never** used for tuning, model selection, or threshold picking.
  `18-1, 18-10, 18-11, 18-12, 18-13, 18-14, 18-15, 18-16, 18-17, 18-2, 18-20, 18-4, 18-5, 18-7,
  18-8, 18-9`

Properties: board-disjoint DEV↔TEST (dev rows {4,5,16,20} ∩ test row {18} = ∅), both disjoint from
the 109 training devices, TEST ≥ 15. TEST = one fully held-out grid row → a genuine
independent-row generalization test.

> **Caveat (must honor):** the already-spent seed-123 tuning slice **included board-18 devices**
> (18-11, 18-16, 18-20, 18-7). So the current "locked operating point" (CosFace m=0.35/s=16,
> merge@0.5) was tuned partly on what is now TEST. Those numbers must be **re-derived on DEV-only
> slices** before the one-touch TEST evaluation; the TEST number is only clean if no hyperparameter
> ever saw board 18 after this partition was fixed.

**One-touch rule:** the following are evaluated **once**, with hyperparameters frozen on DEV,
and are **never tuned on**: the **TEST** slice (board 18), and every **cross-domain** set —
**ManySig, ORACLE, M100, DRFF-R2** (and any drone/drive set). One evaluation, reported as-is,
pass or fail. No going back to retune after seeing them.

### 4.1 DRFF-R2 field semantics — CORRECTION (Step-5)

Each DRFF-R2 clean record carries `TD` (airframe id, e.g. `mavicAir2_3`), `U`, `D`, `C`,
`Height`, `State`. Two corrections to earlier (Step-3b) usage:

- **`TD` = airframe** (the same-model unit we discover). Unchanged, correct.
- **`U` = USRP RECEIVER NUMBER** — the DRFF-R2 capture used **two clock-synchronized
  USRP-2943 receivers** (`u1`, `u2`). `U` is **not** a within-airframe channel condition.
  **Step-3b is superseded on this point:** its confound probe lumped `U` into the nuisance
  group `{D, C, U, Height, State}` predicted within a model. That treated the
  **receiver-diversity axis** (the direct analog of WiSig's `rx`) as if it were channel noise.
  Recorded here as a correction; the Step-3b files are kept, not deleted.

**Consequence for coverage (why no drone multi-rx test exists).** Receiver coverage is almost
entirely single-receiver per airframe. Of 23 usable airframes only **6** have files from both
receivers, and within `mavicAir2` only **1** (`mavicAir2_1`); within `mavicAir2s` only **1**.
An R=2 cross-receiver same-model burst test therefore **cannot be run** on the drone set
(needs ≥4 both-receiver airframes) — reported and skipped, not forced. DRFF stays effectively
**single-receiver**, which is exactly why it is the cross-domain single-rx row in the mechanism
table; it is **not** evidence about receiver diversity.

**Step-3b session-disjoint holdout — re-verified, stands.** The clean same-model probe defined
`session = (airframe, U, D, C)` and held whole sessions out disjointly, with `U` **included** in
the session key. So held-out test windows differ from train in genuine capture structure
(receiver and/or distance/channel). The **0.60** mavicAir2 (OPT-B, logreg/mlp burst) and **0.833**
mavicAir2s (OPT-B, mlp burst) numbers are valid cross-capture generalization and **need no
re-run**. A stricter receiver-*disjoint* same-model probe is infeasible (only `mavicAir2_1` spans
both receivers).

---

## 4.3 THE locked scoring harness (Step-7 — single source of truth for all DRFF/discovery numbers)

Two harnesses previously produced different frozen mavicAir2 E1 oracle-K@8 (step3c 0.218 = k-means
only; step6 0.341 = spectral nn-graph + max-of-methods + a different window-build seed). Locked once,
here, to kill the discrepancy. **Every headline number is reported under this harness:**

- **Window-build seed = FIXED `777`** (matches the step3c/step5 citations). Deterministic windows
  every run — removes build-seed variance. *(step6 used 1234; its numbers are superseded for citation.)*
- **oracle-K (CEILING) = report k-means AND spectral SEPARATELY, never max-of-methods.** Max is silent
  method-selection that inflates the ceiling. k-means: `n_init=10, seed 0`. spectral: `nearest_neighbors`
  affinity, `n_neighbors=15, seed 0`.
- **HDBSCAN = locked noise rule** (singletons → own clusters; ARI/NMI/purity over ALL points; noise
  fraction reported separately). Fixed `min_cluster_size` grid — DRFF `{5,7}`, WiSig `{13,15,17}`;
  **headline = MEAN ARI over the grid, full grid reported.** Per-result argmax over `mcs` is selection.
- **Bursts** = mean of N=10 windows, L2-renormalized; balanced to the min per-class count before scoring.

**Locked DRFF headline** (`results/step7_test_and_harness/drff_headline_locked.csv`, frozen encoder unless
noted; oracle-K = CEILING):

| method | E1 HDBSCAN(mean) | E1 oracle km / sp | E0 oracle km / sp | E1 kNN-1 |
|---|---|---|---|---|
| frozen-512 | 0.071 | **0.297 / 0.298** | 0.055 / 0.078 | 0.672 |
| classical-19 | 0.245 | 0.285 / 0.299 | 0.032 / 0.062 | 0.613 |
| hybrid PCA-64+19 | 0.206 | **0.415 / 0.541** | 0.121 / 0.076 | 0.840 |
| adapted-R1 (E1) | 0.043 | 0.331 / 0.315 | — | 0.707 |

Frozen E1 oracle-K@8 = **km 0.297 / sp 0.298** is the canonical cross-domain number. Adaptation gives no
reliable lift (Δ noise-level, sign-unstable across build seeds; HDBSCAN worse); the no-retrain hybrid is
the strongest cross-domain result (spectral 0.541) but fragile (see step5). Cross-domain wall holds
vs multi-board in-domain (DEV P2 ≈ 0.72).

## 4.4 One-touch TEST — SPENT 2026-07-10 (board 18, 16 devices)

Board-18 TEST was evaluated **exactly once** (`results/step7_test_and_harness/test_table.csv`); no
re-runs, no post-hoc parameter changes. It is now **closed** — never to be re-evaluated or tuned on.

| row | protocol | headline | TEST | DEV ref | drift |
|---|---|---|---|---|---|
| A1 | P1 CosFace + HDBSCAN mcs15 (provisional op-point) | ARI | **0.219** | 0.727 | 0.508 |
| A2 | P1 512-D | oracle-K@16 km | 0.424 | 0.799 | 0.375 |
| A3 | P2 512-D (single-rx) | oracle-K@16 km | 0.359 | 0.720 | 0.361 |
| A4 | P0 512-D (coherent) | oracle-K@16 km | 0.352 | 0.703 | 0.351 |

**Every row drifts 0.35–0.51, far above the pre-registered ~0.07 expectation — reported as-is.** Cause:
board-18 is a single fabrication board (16 same-fab devices), whereas DEV draws 16 devices across 4
boards. Global K=16 clustering of same-fab siblings collapses (HDBSCAN finds only K_est 4–10), while
kNN-1 stays high (0.89–0.99, gap 0.43–0.58) — the fingerprint info is present locally; **global K-way
partition on one fab batch is what fails.** Single-board separability is strongly K/board-dependent
(cf. step5 board-20 K8 = 0.895 vs board-18 K16 = 0.36). Paper consequence: a single fabrication batch of
16 same-model WiFi devices is about as hard to discover as cross-domain drones. The CosFace op-point was
originally selected in seed-123-era tuning (which touched board-18), then re-validated on clean DEV in
Step-2b; A1 above is its **first and only** TEST evaluation.

---

## 5. Seeds / reporting policy

Every reported discovery number = **mean ± std over ≥ 5 slices × ≥ 3 clustering seeds**
(≥ 15 measurements). Slice seeds vary the device sub-sample (from DEV only); clustering seeds vary
any stochastic step of the clustering pipeline.

> **Note (honest):** the current `sklearn.cluster.HDBSCAN` call is **deterministic** given the data
> (no RNG), so "≥3 clustering seeds" today has no effect unless a stochastic element is added
> (burst subsampling, a random UMAP/projection pre-step, or bootstrap resampling of bursts). Until
> such an element is defined, report the ≥5-slice mean±std and state that clustering is
> deterministic. Defining the clustering-seed axis is a Step-2 item.

Single-slice, single-seed numbers (e.g. the historical seed-123 0.540 / 0.667) are **diagnostic
only** and must not be quoted as the headline result.

---

## 6. Frozen assets (this session)

| asset | path | sha256 |
|---|---|---|
| encoder (frozen, never overwritten) | `runs/wisig_supcon_fft64/retrain_best/best_model.pt` | `03898f49061916c7ffbb0c112575f949ff3ce700042e4e3e98d43ca1ecc19bd5` |
| main split (109/41) | `runs/wisig_supcon_fft64/splits/split_manytx.json` | `d16c48fa8e187dbdb08165fc6e3ca16516904db6fd625d60090df25352eaf7c8` |
| dev/test partition | `runs/wisig_supcon_fft64/splits/discovery_partition.json` | `cb1caa544263aba314aaeb1ac40cc5bbb593bef56210003d7b23c55b03f4597d` |

Backbone stays FROZEN; `best_model.pt` is never retrained or overwritten. Data domains M100 /
ORACLE / DRFF-R2 / drone / drive remain SEALED (one-touch, §4).
