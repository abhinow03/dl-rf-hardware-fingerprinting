# EXT-PROTOCOLS — FINDINGS (BLE D2)

> **NOT PAPER RESULTS.** R&D-sandbox findings on the generalized-RFF branch. Numbers trace to
> `EXT_RESULTS_LEDGER.md` (Phases 2 / 3a / 3c batteries) and `audit_out/`. Citable numbers live in
> `FINDINGS.md` / the paper ledgers. Authority: `EXT_PROTOCOL_PLAN.md` (cca5be51…),
> `splits_ext_ble.json` (69ad8d94…). Dataset: BLE D2 (Seeed XIAO-ESP32C3 ×31, GFSK 1 Mbps, USRP
> B210, 12 collections). Metric = ARI vs true unit labels on burst-mean embeddings; headline =
> k-means oracle-K=8 ("assisted-K"); HDBSCAN/eigengap = "deployable-K".

Three approaches compared: **A** = classical_b 19-D hand features (Phase 2); **B1** = frozen WiSig
WiFi encoder transferred (512-D, Phase 3a); **B3** = native from-scratch SupCon encoder (128-D,
Phase 3c). **B2** (fine-tune-from-WiFi) was **skipped by pre-registered gate**. No new eval-split
experiments were run in Phase 4; one chartered-but-untabulated cell was surfaced from cached results
(B3 eigengap est-K / correct-K — see M1). All why-analysis (§W) is **train_units-only**.

---

## STAGE 1 — Master tables

### M1 — full comparative battery (canonical / diagnostic)
Robust-transform k-means ARI unless noted; B3 = full_s2024 (seed-mean where two seeds shown). Chance 0.125.

| test / cell | A (19-D) | B1 frozen (512) | B2 | B3 native (128) |
|---|---|---|---|---|
| **T1 probe** canon N1 / N10 | 1.000 / 1.000 | 0.532 / 0.523 | *skipped* | 0.928 / 0.963 |
| **T1 kNN-1** canon N1 / N10 | 1.000 / 1.000 | 0.994 / 0.983 | *(gate)* | 0.965 / 0.989 |
| **T1 probe** diag N1 / N10 | 0.917 / 0.976 | 0.387 / 0.442 | | 0.912 / 0.920 |
| **T3 canon** N=1 | 0.536 | 0.174 | | 0.755 |
| **T3 canon** N=10 | **0.925** | 0.597 | | 0.892 |
| **T3 canon** N=30 | 0.873 | 0.806 | | 0.889 |
| **T3 canon** N=120 | 0.863 | **0.945** | | 0.891 |
| **T3 diag** N=1 | 0.176 | 0.096 | | 0.702 |
| **T3 diag** N=120 (pooled) | 0.193 | 0.284 | | **0.699 / 0.730 (μ 0.714)** |
| **T2 canon** est-K / correct-K | 3.2 / 0.00 | 11.0 / 0.00 | | 3.6 / 0.20 (raw 8.6 / 0.20) |
| **T2 canon** HDBSCAN found-K / ARI | 28 / 0.443 | **9.8 / 0.956** | | 31 / 0.371 (raw 22.6 / 0.531) |
| **T2 diag** est-K / correct-K | 7.6 / 0.00 | 5.6 / 0.00 | | 5.2 / 0.20 |
| **T2 diag** HDBSCAN found-K / ARI | 64 / 0.192 | 63.6 / 0.191 | | 64 / 0.192 (raw 55 / 0.255) |
| **T-RX** R1→R2 (N=120) | 0.571 | 0.454 | | **0.831 / 0.687 (μ 0.759)** |
| **T4** 5u / 10u / 21u canon | — | — | | 0.554 / 0.849 / 0.891 |
| **T4** 5u / 10u / 21u diag | — | — | | 0.586 / 0.786 / 0.699 |
| **T5** embedding dim | 19 | 512 | | 128 |
| **T5** params | 0 | 1.49 M (frozen) | | 1.49 M (trained) |
| **T5** GPU / 1-core CPU ms/seg | — / 0.220 | 0.638 / 69.5 | | **0.142** / 17.9 |
| **T5** cache (877k) / training | 68 MB / none | 1799 MB / none | | 451 MB / 4 runs, 35 min |

**B2 gate (pre-registered, values shown):** run B2 iff canonical probe(N=1) ≥ 0.60 **OR** diagnostic
N=120 ARI ≥ 0.30. B1 gave **0.532** and **0.284** — both near-miss → **B2 SKIPPED** (head-only
fine-tune ≡ the linear probe already measured; R1–R3 precedent).

### M2 — regime map (winner per cell; the study's actual verdict — no single-winner line)
| regime × K-mode | low-N (N≤10) | high-N (N=120) |
|---|---|---|
| **canonical / assisted-K** (oracle 8) | **B3** @N1 (0.755) → **A** @N10 (0.925) | **B1** (0.945) |
| **canonical / deployable-K** (HDBSCAN) | — | **B1** (0.956) |
| **diagnostic / assisted-K** (pooled) | **B3** (0.702) | **B3** (0.714) |
| **diagnostic / deployable-K** (HDBSCAN) | — | **B3**-raw (0.255) ≳ tie (A/B1 0.19) |
| **receiver-transfer** T-RX | — | **B3** (0.759) |

Reading: **A** owns canonical low-integration (N=10 assisted). **B1** owns canonical high-N and the
only strong deployable-K cell (canonical HDBSCAN). **B3** owns every cross-condition cell — pooled
diagnostic at all N, receiver transfer, and canonical N=1. The winner is **regime-dependent**.

---

## STAGE 2 — Why-analysis (train_units only; descriptive, no eval contact)

### W1 — Rediscovery (ridge R², predict A's 19 locked features from B1 / B3 embeddings)
Overall test-R² (train segments, 70/30, ridge α=10): **B1 0.110, B3 0.118** — both LOW.

| family | B1 R² | B3 R² |
|---|---|---|
| F-CFO | 0.30 | **0.34** |
| F-TRANS | 0.03 | 0.10 |
| F-MOD | **0.13** | 0.04 |
| F-SPEC | 0.13 | 0.10 |
| F-IQ | 0.01 | 0.09 |
| F-ENV | 0.01 | 0.04 |

**Neither learned encoder is a linear re-encoding of the classical physics.** The single most
recoverable family is **F-CFO** (carrier offset), yet even that caps at R²≈0.3. Both encoders
separate units near-perfectly (kNN≈1.0) while carrying only weak *linear* traces of the hand
features → they exploit **largely orthogonal discriminative structure**. *Caveat: R² measures
linear decodability only; nonlinear CFO/MOD dependence could be higher.*

### W2 — Variance decomposition (multivariate η² = between-group / total, z-scored dims)
| method | unit η² | collection η² | unit/coll |
|---|---|---|---|
| A (19-D) | 0.144 | **0.249** | 0.58 |
| B1 (512) | 0.169 | 0.096 | 1.76 |
| **B3 (128)** | **0.979** | **0.000** | **≫1000** |

**The mechanism, quantified.** In A's raw features **collection (receiver/channel/link) variance
dominates unit variance** (0.249 > 0.144) — this is *why* A collapses in the pooled diagnostic regime
and needs per-collection centering. B1 partially suppresses collection variance (unit > coll). **B3
drove collection variance to ≈0 while unit variance is ≈all of it** — the SupCon cross-collection
positives learned near-total nuisance invariance. This is the direct evidence for P4 / EF2 and is
visually confirmed in `w4_umap.png` (B3 unit-manifolds intermix all collections).

### W3 — Calibration-indifference (train-side collection η² before/after per-collection robust std)
| method | coll η² raw | coll η² robust | Δ removed | unit η² (robust) |
|---|---|---|---|---|
| A (19-D) | 0.249 | 0.024 | **−0.225** | 0.173 |
| B1 (512) | 0.096 | 0.001 | −0.095 | 0.185 |
| **B3 (128)** | 0.000 | 0.001 | **≈0** | 0.976 |

Per-collection robust standardization **rescues A** (removes 0.225 of collection variance — the bulk),
**helps B1** (0.095), and is **irrelevant to B3** (nothing to remove). This predicts the eval
transform-sensitivity exactly: A robust 0.863 vs global-z 0.814 (Δ+0.049, matters); B1 robust 0.945
vs raw 0.924 (Δ+0.021, small); B3 robust 0.891 vs raw 0.850 canonical / 0.699 vs 0.701 diagnostic
(≈0, indifferent). **B3 is calibration-free by construction.**

### W4 — Plots (`audit_out/`, from already-cached eval embeddings — not new experiments)
- `w4_umap.png` — UMAP per method × {by-unit, by-collection}. A: unit clusters split by collection;
  B1: tangled overlapping blob; **B3: clean per-unit manifolds, collections fully intermixed.**
- `w4_eigengap.png` — normalized-Laplacian spectra (canonical N=120 robust). Only **B1 shows a
  structured rising tail** (eigengap → est-K≈11); A and B3-robust are over-connected/flat (est-K 3–4).
- `w4_t3_ncurves.png` — A front-loaded, B1 integration-backed (rises with N), **B3 flat** in N.
- `w4_t4_dataeff.png` — B3 canonical rises with train-units; diagnostic clears the 0.334 bar at 5 units.

### W5 — Early-epoch selection note (no new evals; design lesson)
Checkpoint selection was pre-registered as **val-unit(2) burst-mean N=10 ROC-AUC (primary) →
val SupCon loss (tie-break)**. With only **2 held-out val units the ROC-AUC saturated at 1.000 from
epoch 1**, so selection fell entirely to the val-loss tie-break, which **favored very early epochs**
(full_s2024 ep2, full_s1234 ep1) because val SupCon loss rises as the encoder specializes to train
units. Two readings: **(i)** the protocol correctly early-stops before over-specialization (and B3
still wins its regime); **(ii)** a 2-unit val signal is too weak to rank late checkpoints, so the
selected encoder is under-trained and B3's ceiling may be *higher* than reported. Both are honest;
neither was resolved post-hoc (no re-selection on eval). **Design lesson:** future arms need a
val set with ≥6–8 held-out units (or a discovery-ARI val proxy) so the primary signal doesn't
saturate. Collapse guard (per-dim emb-std ≈0.055) stayed healthy all runs — no collapse.

---

## STAGE 3 — Findings

- **EF1 — N-regime complementarity (F1 echo).** The three approaches occupy different N-regimes:
  **A is front-loaded** (canonical 0.536→0.925 by N=10, then flat/declines — features are per-segment
  discriminative), **B1 is integration-backed** (0.174→0.945, the textbook F1 frozen-encoder curve —
  noisy per-segment, denoised by burst-mean), **B3 is flat** (0.755→0.891 — learned per-segment
  discriminability, no integration needed). Evidence: T3, `w4_t3_ncurves.png`. Caveat: single dataset.

- **EF2 — The pooled-regime wall, and B3's escape (P4).** Classical (A) and frozen-transfer (B1)
  hit a wall in the pooled cross-condition (diagnostic) regime: **A 0.193, B1 0.284** at N=120.
  **B3 escapes it — 0.714 (seed-mean), +0.43 over B1, ~3.7× A**, present at every N (0.70 even at
  N=1) and from as few as 5 training units (T4). **Mechanism (W2):** B3's collection η² ≈ 0 vs unit
  η² 0.979 — SupCon positives spanning receivers/channels/links inside balanced batches learned the
  nuisance invariance the pooled regime demands. This is the study's central result and confirms P4.

- **EF3 — P5 convergence band on canonical.** At the canonical N=120 operating point all three land
  in a narrow band: **A 0.863 / B3 0.891 / B1 0.945** (width 0.082). No representation escapes upward
  — canonical discovery is **representation-saturated** (a data-set ceiling, not an encoder ceiling).
  Divergence lives only in the harder pooled regime. Encoder choice is nearly irrelevant on easy
  canonical, decisive on hard diagnostic.

- **EF4 — Deployable-K split verdict.** With no oracle K, **B1 is the standout**: canonical HDBSCAN
  recovers K≈10 at ARI **0.956** (vs A 0.443, B3 0.371) — B1's frozen geometry is the most cleanly
  *clusterable* at deployable-K on canonical. But **no method recovers K=8 by eigengap** in the
  pooled regime (correct-K ≈0), and in the *pooled* regime all deployable-K ARIs collapse to ≈0.19–0.26.
  Note B3's eigengap is transform-dependent (raw est-K 8.6/correct 0.20 — the study's best — vs robust
  3.6). Doctrine: **the demo uses assisted/eigengap-K, not HDBSCAN**, because the demo regime is pooled
  where HDBSCAN helps no one. B1's HDBSCAN win is real but confined to single-receiver canonical.

- **EF5 — Calibration hierarchy (W3).** Per-collection robust standardization **rescues A** (removes
  0.225 collection η², the bulk; without it A's pooled discovery is receiver-dominated), **helps B1**
  (0.095), and is **irrelevant to B3** (collection η² already 0). B3 is calibration-free; A is
  calibration-dependent. Matches eval transform deltas exactly.

- **EF6 — P6 transient refutation (verbatim, Phase 2, dataset-limited).**
  > **PARTIALLY REFUTED on BLE.** Dropping the entire F-TRANS transient family changes canonical
  > N=120 ARI by only Δ=−0.002; the turn-on transient carries essentially no marginal discovery
  > signal. The CFO clause holds more (drop-F-CFO costs Δ=−0.075 in receiver-transfer) but shrinks to
  > Δ=−0.008 under per-collection centering. **Caveat:** the transient is coarsely sampled (~2 µs /
  > ~12 samples @ 6 MS/s) and amplitude-normalized — a dataset-limited refutation, not a universal
  > claim about BLE turn-on transients.

- **EF7 — Receiver-transfer ordering.** Fitting the deployable transform on R1 and clustering R2:
  **B3 0.831/0.687 (μ 0.759) > A 0.571 > B1 0.454.** The native encoder transfers across receivers
  best; the frozen WiFi encoder worst. **Seed-instability caveat:** T-RX is B3's least stable cell
  (0.831 vs 0.687 across two seeds, spread 0.144) — receiver-transfer geometry depends on init.

- **EF8 — Limitations.** (i) **2-unit val / early-epoch selection** (W5): B3 checkpoints are likely
  under-trained; its ceiling may exceed reported. (ii) **Quarantined classical-18 (pre-registered
  prediction, untested here):** Phase-2 found F-SPEC channel/location-confounded (drop-F-SPEC →
  canonical 1.000); we predict a `classical_b_minus_spec` 15-D (or an 18-D dropping the single worst
  `spec_flatness`) would improve A's cross-location discovery on a **D1** dataset — logged, **not**
  retro-applied to the locked spec (that would be tuning to eval). (iii) **No day/aging axis** in D2
  (session-disjoint by condition, not time) → **no aging/temporal-stability claims**. (iv) **Single
  dataset, single protocol (BLE)** so far → external validity to Zigbee/other PHYs is open (Phase-5).

### PRIOR SCORECARD (P1–P6 final)
- **P1 (B1 weak low-N, converges at N=120):** **HELD.** B1 0.174 (N=1) → 0.945 (N=120), into the band.
- **P2 (B2 fine-tune fails to beat B3):** **N/A — B2 skipped by gate** (near-miss: probe 0.532<0.60,
  diag 0.284<0.30). Directionally supported (B3 0.714 ≫ B1 frozen-readout 0.284; head-only ≡ that
  readout), but not formally executed.
- **P3 (A beats B1 at low N cross-domain):** **HELD at low N, inverts high-N.** A 0.925 vs B1 0.597
  (N=10); B1 0.945 > A 0.863 (N=120).
- **P4 (B3 native winner if enough units/sessions):** **HELD in the regime that matters** — B3 wins
  pooled diagnostic (0.714), T-RX (0.759), canonical N=1; the "enough units" clause confirmed by T4
  (needs ≥10 for canonical, ≥5 for diagnostic). Not a *uniform* winner (A/B1 own specific canonical cells).
- **P5 (A/B1/B3 converge into one band at N=120):** **HELD on canonical** (0.863–0.945), **refuted on
  diagnostic** (B3 escapes: 0.714 vs 0.19–0.28). Convergence is regime-specific.
- **P6 (narrowband info in CFO+transient, favors A):** **PARTIALLY REFUTED** (transient ≈0; CFO is a
  receiver-shared offset the transform neutralizes). Dataset-limited.

---

## STAGE 4 — Demo-tier recommendation (design memo)

**Operating regime = POOLED.** The base station pools receivers, so the demo's discovery regime is
the **pooled / diagnostic** one (multiple receivers/conditions mixed), *not* single-receiver canonical.
Candidates scored on (pooled N=120 ARI, T-RX, est-K behavior, T5 compute, data needs):

| criterion | A | B1 | **B3** |
|---|---|---|---|
| pooled N=120 (diagnostic) | 0.193 | 0.284 | **0.714** |
| receiver transfer (T-RX) | 0.571 | 0.454 | **0.759** |
| deployable est-K | flat (3) | struct. (11) | **raw 8.6 (closest to 8)** |
| GPU ms/seg / emb dim | — / 19 | 0.638 / 512 | **0.142 / 128** |
| training data need | none | none (WiFi-pretrained) | 21 labeled units (5–10 works) |

**Recommendation (numbers-backed):** **B3 native embeddings + assisted/eigengap-K** for the pooled
demo. In the regime the base station actually runs (pooled), B3 beats B1 by **+0.43** and A by
**+0.52** on discovery ARI, transfers across receivers best, has the closest deployable eigengap-K
(raw 8.6), the fastest GPU inference (0.142 ms/seg) and smallest embedding (128-D). **B1 is the
unassisted-canonical alternative** — its deployable HDBSCAN K-recovery (0.956) is unmatched but only
in single-receiver canonical, which is **not** the base-station regime, so that advantage does not
transfer to the demo. **A is the zero-training CPU fallback** if no GPU/labels are available, but it
is weak in the pooled regime (0.193). Honest caveat: B3's edge assumes labeled BLE units for training
and a GPU at enrollment; both hold for the demo. *Router-side BLE spectral-stats routing class +
Mahalanobis refit is Phase-5 work, out of scope here.*
