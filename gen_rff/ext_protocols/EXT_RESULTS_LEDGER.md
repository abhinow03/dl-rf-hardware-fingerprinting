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
- **P3 (A beats B1 frozen at low N cross-domain):** *A's number logged in Phase 2; **verdict now
  RESOLVED in Phase 3a below** (CONFIRMED at low N, inverts at high N).* Approach-A canonical discovery
  **k-means ARI = 0.863 (N=120), 0.925 (N=10)** was the floor; B1 = 0.597 (N=10) / 0.945 (N=120).
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

## Phase 3a — Approach B / Arm B1 (frozen WiSig transfer, 512-D) on BLE D2

Authority: `EXT_PROTOCOL_PLAN.md` (cca5be51…), `splits_ext_ble.json` (69ad8d94…). Frozen encoder
`runs/wisig_supcon_fft64/retrain_best/best_model.pt` (**03898f49…**, read-only, unchanged after run).
Embedding stage = `RFEncoder.get_encoder_output` → **512-D** (LayerNorm[ cross-attn(256) ⊕ spectral(256) ]),
STFT front-end nfft=64/hop=16 (`stft_for(256)`, the fft64 front-end) — the **same stage+dim as every
prior F-series transfer eval** (FINDINGS F1 frozen@N). Input = B1 25 MS/s cache (N,30,2,256); per
segment: forward all 30 windows → mean-pool → L2-normalize. Extraction GPU-only, no weight update.
Battery run 2026-07-16, single-threaded BLAS, 5 seeds, ARI on burst-mean embeddings. Two transforms
mirror A: **raw** = L2-renormed burst mean (the canonical frozen-transfer point, replaces A's global-z
comparison); **robust** = per-collection median/IQR (same IQR-floor safeguard as A).

### T1 — Transferability (linear probe + kNN-1 on the 8 eval units; raw L2; chance 0.125)
| regime | N | probe acc | kNN-1 acc |
|---|---|---|---|
| canonical (outdoor) | 1 | 0.532 | 0.994 |
| canonical (outdoor) | 10 | 0.523 | 0.983 |
| diagnostic (matched) | 1 | 0.387 | 0.999 |
| diagnostic (matched) | 10 | 0.442 | 0.964 |

kNN-1 ≈ 1.0 (the fingerprint is locally present), but the **linear probe is far weaker (0.53 canonical
N=1)** — frozen-WiSig embeddings are kNN-separable but **not linearly separable by unit** at segment
resolution (non-linear tangle). Contrast A, where classical_b is *perfectly linearly* separable
(probe 1.0). This gap is the crux of the B2 gate below.

### T2 — Discovery (eval_units, N=120, 5 seeds)
| transform | regime | k-means ARI | spectral | estK | correct-K | HDBSCAN found-K / ARI / noise |
|---|---|---|---|---|---|---|
| **robust** | **canonical** | **0.945 ± 0.000** | 0.431 | 11.0 | 0.00 | **9.8 / 0.956 / 0.00** |
| robust | diagnostic | 0.284 ± 0.009 | 0.037 | 5.6 | 0.00 | 63.6 / 0.191 / 0.01 |
| raw | canonical | 0.924 ± 0.005 | 0.415 | 10.0 | 0.00 | 9.0 / 0.965 / 0.01 |
| raw | diagnostic | 0.229 ± 0.013 | 0.038 | 3.2 | 0.00 | 60.2 / 0.191 / 0.01 |

The frozen embedding yields a **far more clusterable canonical geometry than A**: deployable HDBSCAN
(no oracle-K) finds **≈9–10 clusters with ARI 0.956–0.965** — versus A's 28 fragments / 0.443. K-estimation
still over/under-shoots 8 (correct-K 0.00, as with A), but the *representation* clusters cleanly.

### T3 — N-sweep (oracle-K=8 k-means ARI, both transforms)
| transform | regime | N=1 | N=10 | N=30 | N=120 |
|---|---|---|---|---|---|
| **robust** | canonical | 0.174 | 0.597 | 0.806 | **0.945** |
| robust | diagnostic | 0.096 | 0.241 | 0.279 | 0.284 |
| raw | canonical | 0.217 | 0.483 | 0.673 | 0.924 |
| raw | diagnostic | 0.094 | 0.182 | 0.224 | 0.229 |

**Monotone strong integration gain, weak at low N** (robust canonical 0.174→0.597→0.806→0.945) — the
textbook **F1 frozen-encoder signature**: per-segment embeddings are noisy, burst-mean integration
denoises them, no early saturation. This is the *opposite* shape to A (front-loaded: 0.536 at N=1,
peaks 0.925 at N=10, then flat/declines).

### T-RX — Receiver-disjoint arm (fit R1 robust stats → cluster R2, eval_units, N=120)
| variant | k-means ARI | spectral ARI |
|---|---|---|
| full (R1-fit stats on R2) | 0.454 ± 0.016 | 0.271 |
| R2-matched reference (R2's own stats) | 0.439 | — |
R1-fit stats transfer to R2 (0.454 ≥ 0.439 matched) — like A, the deployable transform survives
receiver shift — but the **absolute level is below A's** (0.571).

### T5 — Compute (vs A)
| metric | A (classical_b) | B1 (frozen WiSig) |
|---|---|---|
| ms / segment | 0.220 (1-core CPU) | **0.638 (GPU)** / 69.5 (1-core CPU) |
| embedding dim | 19 | 512 |
| params | 0 (unsupervised) | 1,490,944 (frozen, pretrained on WiSig WiFi) |
| embedding cache (877 k segs) | 68.4 MB | 1798.7 MB |
| training data required | zero | zero (frozen; no BLE labels used) |
B1 needs a **GPU** to be fast (69.5 ms/seg on one CPU core — ~316× A) and a **26× larger** cache.

### A-vs-B1 side-by-side (all shared cells; k-means oracle-K=8 ARI unless noted)
| cell | A | B1 | winner |
|---|---|---|---|
| **canonical robust N=1** | **0.536** | 0.174 | **A** (+0.362) |
| **canonical robust N=10** | **0.925** | 0.597 | **A** (+0.328) |
| canonical robust N=30 | 0.873 | 0.806 | A (+0.067) |
| **canonical robust N=120** | 0.863 | **0.945** | **B1** (+0.082) |
| **diagnostic robust N=120** (pooled) | 0.193 | **0.284** | **B1** (+0.091) |
| diagnostic robust N=10 | 0.182 | 0.241 | B1 (+0.059) |
| diagnostic robust N=30 | 0.198 | 0.279 | B1 (+0.081) |
| T2 canonical HDBSCAN ARI (deployable-K) | 0.443 | **0.956** | **B1** (+0.513) |
| T-RX full R1→R2 | **0.571** | 0.454 | A (+0.117) |
| T1 canonical probe N=1 | **1.000** | 0.532 | A |
| T1 canonical kNN-1 N=1 | 1.000 | 0.994 | ~tie |

**Pooled-diagnostic flag (the regime where B can still earn its keep):** **YES — B1 shows a real
advantage there.** Diagnostic (matched-condition, R1+R2 pooled) k-means ARI **0.284 (B1) vs 0.193 (A)**,
+0.091, and B1 leads at every diagnostic N except N=1. B1 also wins big on **deployable clustering**
(canonical HDBSCAN 0.956 vs 0.443) and on **high-N canonical** (0.945 vs 0.863). B1 loses decisively at
**low N** (N≤10) and on **receiver transfer**. Net: the frozen encoder buys *integration-limited* gains
(needs N≳30 and a GPU) but genuinely improves the hardest pooled cross-receiver cell and the
no-oracle-K deployable path.

### Pre-registered prior verdicts (Phase 3a)
- **P3 VERDICT (A beats B1 frozen at low N cross-domain) — verbatim:**
  > **CONFIRMED at low N; INVERTS at high N.** Pre-registered floor: A canonical **N=120 0.863 /
  > N=10 0.925**. B1 frozen-WiSig on the same cells: **N=10 0.597** (A wins by +0.328) and **N=1
  > 0.174 vs A 0.536** (A wins by +0.362) — P3's *low-N* claim holds cleanly, A beats frozen transfer
  > when bursts are few. But at **N=120 B1 = 0.945 > A 0.863** (B1 +0.082): with enough burst
  > integration the frozen encoder overtakes the classical floor. P3 is a **low-N** statement and is
  > upheld as such; it does not extend to the integration-saturated regime.
- **P1 CHECK-IN (F1 N-curve shape) — verbatim:**
  > **MATCHES F1.** B1's N-curve is the canonical frozen-encoder shape F1 predicts — **weak at low N,
  > large monotone integration gain, no early saturation**: robust canonical 0.174 (N=1) → 0.597
  > (N=10) → 0.806 (N=30) → 0.945 (N=120). Single-segment frozen embeddings are noisy and only
  > kNN-separable (probe 0.53), so discovery is poor at N=1; burst-mean integration denoises them and
  > ARI climbs steadily. This is the opposite of Approach A's front-loaded curve (0.536 at N=1, peak
  > 0.925 at N=10, then flat), confirming the two approaches occupy **different N-regimes** — A owns
  > low-N, B1 owns high-N.

### B2 GATE (pre-registered; decide by numbers, no execution)
Rule: B2 partial-unfreeze runs in Phase 3b **ONLY IF** B1 canonical probe(N=1) ≥ 0.60 **OR** B1
diagnostic N=120 oracle-K ARI ≥ 0.30.
- canonical probe(N=1) = **0.532** ≥ 0.60 ? **NO** (near-miss).
- diagnostic N=120 oracle-K ARI = **0.284** ≥ 0.30 ? **NO** (near-miss).
- **→ B2 SKIPPED.** Both conditions fail (both narrowly). Justification (pre-registered): a partial
  head-only unfreeze on a frozen encoder ≡ training a linear head ≡ the **linear probe already run**
  (T1 canonical probe 0.532 is the *best* a linear head can do), and cross-condition discovery
  (0.284) sits below the bar — consistent with the R1–R3 precedent that head-only adaptation adds
  little over the frozen readout. B1's genuine wins (high-N canonical, pooled diagnostic, deployable
  HDBSCAN) come from **burst integration + clustering geometry, not from a trainable head**, so
  partial-unfreeze is not indicated. *(B3 native-from-scratch is a separate arm, Phase 3c.)*

## Methodology note (honesty)
The battery was run once. One numerical fix was applied mid-run and re-run: an **IQR floor** on the
per-collection robust transform (a quantized `occ_bw` had IQR=0 in Loc4 → division blew up, giving a
spurious ARI≈0.010). This is a robust-scaler stability guard, not a results-tuning change; the locked
19-D feature set was untouched (see `CLASSICAL_B_SPEC.md` amendment). Spectral-clustering ARI is
reported but **not** used as headline — with per-collection-centered embeddings and few points at
high N its kNN-graph is often disconnected (sklearn warns), making it unstable; k-means is the
reliable oracle-K metric.

## Phase 3c — Approach B / Arm B3 (native from-scratch SupCon) on BLE D2

**PRE-REGISTERED before any training (written 2026-07-16, BEFORE run):**
Authority `EXT_PROTOCOL_PLAN.md` (cca5be51…), `splits_ext_ble.json` (69ad8d94…),
`WINDOW_SPEC_BLE.md` (3541aad6…). Native dual-branch RFEncoder trained FRESH (random init),
SupCon on unit labels, constant tau=0.5, NO augmentation (no CFO aug — doctrine; no time
jitter — onset-aligned). 128-D L2-normalized head is the discovery embedding (native-arm
precedent; cf. B1 used 512-D frozen encoder output). Trained on train_units(21) ×
train_collections(8) = 400,951 segments.

- **PRIMARY (P4 resolves here):** pooled-**diagnostic N=120 oracle-K ARI > 0.334** (B1 0.284 + 0.05)
  — the one regime where a learned encoder still earns its keep.
- **SECONDARY:** canonical N=120 ≥ 0.90 (parity band with B1 0.945); N=10 canonical reported vs A 0.925.
- **P5 tested** via the N-sweep convergence band (does N=120 canonical converge into a band across
  A / B1 / B3, or does one method escape it?).
- **Mechanism hypothesis (logged, not a success gate):** SupCon positives spanning collections
  (receivers/channels/links) inside balanced P×S batches push per-unit invariance to exactly the
  axes that pool in the diagnostic regime → predicts the diagnostic gain over B1.

*(Results, verdicts, and the three-way A/B1/B3 table are appended below after the run.)*

### Phase 3c RESULTS — B3 native from-scratch (appended after run, 2026-07-16)

Native dual-branch RFEncoder, fresh random init, SupCon tau=0.5 (constant), NO augmentation,
balanced **P×S = 21×16 = 336** (all train units/batch; segments drawn uniformly across the 8
train_collections). AdamW lr=5e-4 (cosine, 5% warmup) wd=1e-4, grad-clip 1.0, AMP. 12 epochs =
14,328 steps/run. **epoch-1 = 176.7 s → projected 0.59 h/run (<6 h, no epoch halving).** Wall:
full runs 34.9 / 35.3 min, de10 16.5 min, de5 8.9 min. Discovery embedding = **128-D L2-normed
projection head** (`forward()`; native-arm precedent — cf. B1's 512-D frozen encoder output).
Frozen WiSig `best_model.pt` (03898f49…) untouched/read-only; authorities unchanged.

**Checkpoint selection (pre-registered, honored):** primary = val-unit(2) burst-mean N=10 pairwise
-cosine ROC-AUC on train_collections; tie-break = val SupCon loss. **The val-AUC saturated at 1.000
from epoch 1** (only 2 held-out val units → trivially separable), so selection fell entirely to the
**val-loss tie-break, which favors early epochs** (val SupCon loss rises as the encoder specializes
to train units). Selected epochs: full_s2024 **ep2**, full_s1234 **ep1**, de10 ep1, de5 ep5.
*Honest limitation:* a 2-unit val signal is too weak to discriminate late checkpoints; the protocol
therefore effectively early-stops. Reported as-is (no post-hoc change). Collapse guard (per-dim
emb-std) stayed healthy (~0.055) every epoch, all runs — no collapse.

**Selected-checkpoint sha256 (off-repo, gitignored):**
`full_s2024 f1d7745eaa72…` · `full_s1234 e480589e142a…` · `de10_s2024 a4eaf7d31104…` · `de5_s2024 de08691f74e9…`

#### T1 — probe + kNN-1 (raw L2, chance 0.125) — full_s2024 / full_s1234
| regime | N | probe | kNN-1 |
|---|---|---|---|
| canonical | 1 | 0.928 / 0.929 | 0.965 / 0.951 |
| canonical | 10 | 0.963 / 0.974 | 0.989 / 0.991 |
| diagnostic | 1 | 0.912 / 0.908 | 0.971 / 0.941 |
| diagnostic | 10 | 0.920 / 0.917 | 0.986 / 0.982 |
Native embeddings are **linearly separable** (probe 0.93) — unlike B1 frozen (probe 0.53) — the
SupCon head learned a linear unit geometry.

#### T2 — Discovery (N=120, full_s2024; 5 seeds)
| transform | regime | k-means ARI | spectral | estK | HDBSCAN found-K / ARI / noise |
|---|---|---|---|---|---|
| robust | canonical | 0.891 ± 0.000 | 0.111 | 3.6 | 31.0 / 0.371 / 0.00 |
| robust | diagnostic | **0.699 ± 0.004** | 0.023 | 5.2 | 64.0 / 0.192 / 0.00 |
| raw | canonical | 0.850 ± 0.000 | 0.105 | 8.6 | 22.6 / 0.531 / 0.00 |
| raw | diagnostic | 0.701 ± 0.005 | 0.053 | 3.2 | 55.2 / 0.255 / 0.01 |
Deployable-K (HDBSCAN) on **canonical is B3's weak spot**: it over-fragments (K≈22–31, ARI 0.37–0.53)
— **B1 still wins that cell (K≈10, ARI 0.956).** B3's win is the **oracle diagnostic** regime.

#### T3 — N-sweep (oracle-K k-means ARI), full_s2024
| transform | regime | N=1 | N=10 | N=30 | N=120 |
|---|---|---|---|---|---|
| robust | canonical | 0.755 | 0.892 | 0.889 | 0.891 |
| robust | diagnostic | 0.702 | 0.699 | 0.698 | 0.699 |
| raw | canonical | 0.726 | 0.860 | 0.852 | 0.850 |
| raw | diagnostic | 0.705 | 0.696 | 0.693 | 0.701 |
**B3 is essentially FLAT in N** (0.755→0.891 canonical; ~0.70 diagnostic at *every* N incl. N=1) — the
learned per-segment embedding is already discriminative, so **no burst integration is needed**. This
is the opposite of B1 (F1 integration curve) and stronger-at-low-N than A.

#### T-RX — receiver-disjoint (fit R1 robust → cluster R2, N=120)
| run | full R1→R2 | R2-matched ref |
|---|---|---|
| full_s2024 | **0.831 ± 0.000** | 0.831 |
| full_s1234 | 0.687 ± 0.000 | 0.831 |
Best receiver transfer of all three approaches (A 0.571, B1 0.454). Seed spread is notable here
(0.687–0.831) — receiver transfer is B3's least seed-stable cell.

#### T4 — data-efficiency curve (N=120, seed 2024; robust / raw k-means ARI)
| train_units | canonical robust | diagnostic robust | canonical raw | diagnostic raw |
|---|---|---|---|---|
| 5 | 0.554 | 0.586 | 0.781 | 0.588 |
| 10 | 0.849 | 0.786 | 0.840 | 0.801 |
| 21 (full) | 0.891 | 0.699 | 0.850 | 0.701 |
Canonical rises monotonically with train-unit count (data-hungry: needs ≥10 units). Diagnostic
**already exceeds the pre-registered 0.334 bar with only 5 units** (0.586) and peaks at 10 units
(0.786) — the cross-collection invariance the mechanism predicts is learnable from few units.

#### T5 — Compute (three-way)
| metric | A | B1 frozen | B3 native |
|---|---|---|---|
| embedding dim | 19 | 512 | 128 |
| params | 0 | 1.49 M (frozen) | 1.49 M (trained) |
| GPU ms/seg | — (0.22 CPU) | 0.638 | **0.142** |
| 1-core CPU ms/seg | 0.220 | 69.5 | 17.9 |
| embedding cache | 68.4 MB | 1798.7 MB | 451.0 MB |
| training | none | none | 4 runs; full 34.9/35.3 min (14,328 steps), GPU |

### Three-way A / B1 / B3 (all shared cells; robust k-means oracle-K ARI unless noted)
| cell | A | B1 | B3 (s2024) | B3 seed-mean |
|---|---|---|---|---|
| canonical N=1 | 0.536 | 0.174 | 0.755 | 0.810 |
| canonical N=10 | **0.925** | 0.597 | 0.892 | 0.893 |
| canonical N=30 | 0.873 | 0.806 | 0.889 | 0.889 |
| canonical N=120 | 0.863 | **0.945** | 0.891 | 0.891 |
| **diagnostic N=120 (pooled)** | 0.193 | 0.284 | **0.699** | **0.714** |
| diagnostic N=1 | 0.176 | 0.096 | 0.702 | 0.702 |
| T1 canonical probe N=1 | **1.000** | 0.532 | 0.928 | 0.929 |
| T2 canonical HDBSCAN ARI (deployable-K) | 0.443 | **0.956** | 0.371 (raw 0.531) | — |
| T2 diagnostic HDBSCAN ARI | 0.192 | 0.191 | 0.192 (raw 0.255) | — |
| T-RX full R1→R2 | 0.571 | 0.454 | **0.831** | 0.759 |
*(B3 seed-mean over full_s2024/full_s1234. diagnostic N=120: 0.699/0.730. canonical N=120: 0.891/0.891.)*

### PRE-REGISTERED SCORING
- **PRIMARY — pooled-diagnostic N=120 oracle-K ARI > 0.334:** **PASS (decisively).** B3 = **0.699
  (s2024) / 0.730 (s1234), mean 0.714** — +0.38 over the bar, +0.43 over B1 (0.284), +0.52 over A
  (0.193). The learned encoder earns its keep in the one regime that matters.
- **SECONDARY — canonical N=120 ≥ 0.90 (parity band w/ B1 0.945):** **NARROWLY MISSED — 0.891** (both
  seeds), 0.009 short of 0.90; sits in-band above A (0.863), below B1 (0.945). N=10 canonical: B3
  0.892/0.895 vs **A 0.925** — A still edges the low-N canonical cell.

### P4 VERDICT (verbatim)
> **P4 CONFIRMED on BLE.** A learned encoder was predicted to still earn its keep in the pooled
> cross-condition (diagnostic) regime where classical/frozen fail. B3 native SupCon delivers
> **diagnostic N=120 oracle-K ARI = 0.714 (seed-mean), 0.699/0.730 per seed** — versus A 0.193 and
> B1 0.284 — clearing the pre-registered 0.334 bar by +0.38 and roughly **2.5× the best prior
> approach.** The gain is present at *every* N (diagnostic ARI ≈ 0.70 even at N=1) and appears with as
> few as 5 training units (T4: 0.586). The mechanism hypothesis holds: SupCon positives spanning
> receivers/channels/links inside balanced batches push per-unit invariance onto exactly the axes
> that the diagnostic regime pools. **Caveat:** B3 does *not* dominate everywhere — B1 frozen still
> wins the deployable-K canonical HDBSCAN cell (0.956 vs 0.53) and high-N canonical oracle (0.945 vs
> 0.891), and A still wins low-N canonical. B3's value is specifically **cross-condition robustness**,
> not a uniform ceiling lift.

### P5 VERDICT (verbatim)
> **P5 CONFIRMED — canonical N=120 converges into a shared band; no method escapes it.** At the
> canonical (single-receiver, cross-location) N=120 operating point the three approaches land in a
> narrow band: **A 0.863 / B3 0.891 / B1 0.945** (width 0.082, all in [0.86, 0.95]). No representation
> escapes upward — canonical discovery is **representation-saturated**, a shared ceiling set by the
> data, not the encoder. Divergence appears **only in the harder pooled-diagnostic regime** (A 0.193 /
> B1 0.284 / B3 0.714), where B3 escapes the band decisively. Net: the encoder choice is nearly
> irrelevant at the easy canonical operating point and decisive at the hard cross-condition one.

### Honest seed-variance note
Two full-data seeds (2024/1234). **Canonical N=120 is rock-stable** (0.891 / 0.891). **Diagnostic
N=120** has modest spread (0.699 / 0.730, mean 0.714, ±0.016). **T-RX is the least stable cell**
(0.831 / 0.687, spread 0.144) — receiver-transfer geometry depends on init. All PRIMARY-relevant
cells clear their bars under both seeds; the SECONDARY near-miss (0.891) is seed-invariant. Per-seed
5-seed clustering std within each run is ≤0.024 on all headline cells (tables show ±).
