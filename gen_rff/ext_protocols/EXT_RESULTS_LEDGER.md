# EXT-PROTOCOLS — RESULTS LEDGER

> **EXT-PROTOCOLS — NOT PAPER RESULTS.** These are R&D-sandbox numbers on the generalized-RFF
> branch. Citable numbers live in `FINDINGS.md` / the paper ledgers. This file tracks the
> Approach-A/B comparative battery on new PHYs (BLE first).

Authority: `EXT_PROTOCOL_PLAN.md` (cca5be51…), `splits_ext_ble.json` (69ad8d94…),
`CLASSICAL_B_SPEC.md` (locked). Battery run 2026-07-15, CPU-only, single-threaded BLAS,
5 seeds, ARI vs true unit labels on burst-mean embeddings. Metric headline = **k-means
oracle-K=8 ARI**; spectral reported alongside (unstable at low point-count — see note).

## Phase 2 — Approach A (classical_b, 19-D) on BLE D2

### T1 — Transferability (linear probe + kNN-1 on the 8 eval units; global-z; chance 0.125)
| regime | N | probe acc | kNN-1 acc |
|---|---|---|---|
| canonical (outdoor) | 1 | 1.000 | 1.000 |
| canonical (outdoor) | 10 | 1.000 | 1.000 |
| diagnostic (matched) | 1 | 0.917 | 1.000 |
| diagnostic (matched) | 10 | 0.976 | 1.000 |

Eval units are **perfectly linearly separable** in classical_b space (probe/kNN ≈ 1.0 ≫ 0.125).
The fingerprint is trivially *present*; the hard part is *unsupervised discovery* (T2/T3), where
cross-condition structure — not separability — governs.

### T2 — Discovery (eval_units, N=120; robust transform, 5 seeds)
| regime | k-means ARI | spectral ARI | eigengap est-K | correct-K rate | HDBSCAN found-K / ARI / noise |
|---|---|---|---|---|---|
| **canonical (outdoor)** | **0.863 ± 0.000** | 0.115 | 3.2 | 0.00 | 28 / 0.443 / 0.01 |
| diagnostic (matched) | 0.193 ± 0.037 | 0.036 | 7.6 | 0.00 | 64 / 0.192 / 0.00 |

Note the **inversion**: canonical (4 outdoor locations, one receiver) clusters far better than
diagnostic (8 wired+wireless-indoor collections spanning R1/R2 receivers). Matched-condition here
mixes *more* receivers/channels → more per-unit CFO spread → harder discovery. Estimated-K
under-counts (eigengap finds 3–4 macro-groups, not 8) → **correct-K rate 0.00**; HDBSCAN
over-fragments (28–64 micro-clusters) but its ARI (0.44 canonical) confirms real structure.
Deployable K-estimation is the weak link, not the representation.

### T3 — N-sweep (oracle-K=8 ARI, both transforms)
| transform | regime | N=1 | N=10 | N=30 | N=120 |
|---|---|---|---|---|---|
| **robust** | canonical | 0.536 | **0.925** | 0.873 | 0.863 |
| robust | diagnostic | 0.176 | 0.182 | 0.198 | 0.193 |
| global-z | canonical | 0.767 | 0.784 | 0.786 | 0.814 |
| global-z | diagnostic | 0.139 | 0.152 | 0.140 | 0.153 |
*(k-means; spectral omitted — unstable, see note.)*

- **Burst integration helps then saturates** (robust canonical 0.54→0.93 by N=10, flat after) —
  consistent with F1/P5 (integration dominates, then a ceiling). global-z is flatter in N (already
  integrates the receiver-common offset out via the global mean).
- **Transform comparison:** per-collection **robust beats global-z on canonical** (0.863 vs 0.814
  at N=120) and, importantly, **on diagnostic** (0.193 vs 0.153) — the per-collection centering is
  doing its intended job (removing receiver-common CFO/gain offset). Neither transform rescues the
  diagnostic cross-receiver spread to canonical levels.

### T-RX — Receiver-disjoint arm (fit R1 robust stats → cluster R2, eval_units, N=120)
| variant | k-means ARI | spectral ARI |
|---|---|---|
| full (R1-fit stats on R2) | **0.571 ± 0.031** | 0.209 |
| CFO-ablated (drop F-CFO) | 0.496 ± 0.044 | 0.226 |
| R2-matched reference (R2's own stats) | 0.540 | — |

- **The deployable transform transfers across receivers:** R1-fit stats applied to R2 (0.571) even
  *slightly exceed* R2's own stats (0.540) — receiver-shift does not break the standardization.
- **CFO ablation:** dropping F-CFO costs **Δ=−0.075** (0.571→0.496) — CFO carries a real chunk of
  the cross-receiver individuating signal, but the majority survives without it → receiver-shift
  does **not** all ride on CFO.

### Per-family ablation (canonical, N=120, robust, k-means ARI)
| set | ARI | Δ vs FULL |
|---|---|---|
| FULL (19-D) | 0.863 | — |
| drop F-MOD | 0.838 | **−0.026** (largest real cost) |
| drop F-CFO | 0.855 | −0.008 |
| drop F-IQ | 0.855 | −0.008 |
| drop F-TRANS | 0.862 | −0.002 |
| drop F-ENV | 0.863 | 0.000 |
| **drop F-SPEC** | **1.000** | **+0.137** |

**F-SPEC is channel/location-confounded** (finding, not bug): removing the spectral family yields
*perfect* cross-location discovery. Isolation probe: dropping `occ_bw` alone → 0.863 (no change),
`spec_flatness` → 0.989, whole family → 1.000. The spectral shape tracks the 4 outdoor propagation
environments, fragmenting same-unit clusters across locations. **Candidate refinement for Phase 4**
(a `classical_b_minus_spec` 15-D variant), logged but **NOT** retro-applied to the locked spec
(that would be tuning to eval). Under the per-collection-centered representation F-CFO/F-TRANS/F-ENV
are near-redundant at the saturated operating point; **F-MOD (GFSK deviation stats) is the single
most valuable retained family**.

### T5 — Compute
| metric | value |
|---|---|
| feature extraction | **0.220 ms / segment** (1 core, incl. STFT + Welch PSD) |
| feature count | 19 |
| feature cache (all 877 k segs) | 68.4 MB |
| training data required | **zero** (Approach A is unsupervised at deploy; structural T4 property) |

## Pre-registered prior check-ins
- **P3 (A beats B1 frozen at low N cross-domain):** *A's number logged, awaiting B1 (Phase 3).*
  Approach-A canonical discovery **k-means ARI = 0.863 (N=120), 0.925 (N=10)** is the floor B1
  frozen-WiSig transfer must beat. Recorded now; verdict deferred to Phase 3.
- **P6 VERDICT (BLE, from the F-TRANS ablation) — verbatim:**
  > **PARTIALLY REFUTED on BLE.** P6 predicted the individuating information on narrowband
  > protocols would concentrate in **CFO + transient envelope**. On BLE D2 at the burst-mean
  > discovery operating point (canonical, N=120, oracle-K=8), dropping the **entire F-TRANS
  > transient family changes ARI by only Δ=−0.002** (0.863→0.862) — the turn-on transient carries
  > essentially **no marginal discovery signal** here, refuting P6's transient clause. The **CFO
  > clause holds more**: in the receiver-disjoint transfer test, dropping F-CFO costs Δ=−0.075
  > (0.571→0.496), so CFO is a genuine cross-receiver cue — but under the deployable per-collection
  > centering (which removes the receiver-common CFO offset by design) F-CFO's marginal value
  > shrinks to Δ=−0.008. Net: on BLE the discovery signal is **not transient-concentrated**, and
  > CFO matters chiefly as a receiver-shared offset the deployable transform is built to neutralize.
  > **Caveat (Phase-0b):** the transient here is coarsely sampled (~2 µs / ~12 samples @ 6 MS/s)
  > and amplitude-normalized, so this is a **dataset-limited** refutation, not a universal claim
  > about BLE turn-on transients.

## Methodology note (honesty)
The battery was run once. One numerical fix was applied mid-run and re-run: an **IQR floor** on the
per-collection robust transform (a quantized `occ_bw` had IQR=0 in Loc4 → division blew up, giving a
spurious ARI≈0.010). This is a robust-scaler stability guard, not a results-tuning change; the locked
19-D feature set was untouched (see `CLASSICAL_B_SPEC.md` amendment). Spectral-clustering ARI is
reported but **not** used as headline — with per-collection-centered embeddings and few points at
high N its kNN-graph is often disconnected (sklearn warns), making it unstable; k-means is the
reliable oracle-K metric.
