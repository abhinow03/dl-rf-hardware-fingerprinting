# GEN-RFF SANDBOX — FINDINGS

*R&D branch `experimental/generalized-rff`. DEMO-SIDE / negative-and-structural results —
paper-2 seed material, not part of the locked paper (§6). Every claim below carries its
artifact path, the exact table/cell, and the committed script that regenerates it.*

**Provenance note.** Result artifacts live under `results_gen/` and `runs_gen/`, which are
**gitignored** (code-only branch). They are reproducible from the committed scripts named
per claim. Paths are given relative to `~/CAPSTONE/DL_model/`. Checkpoints are `*.pt`
(gitignored); JSON/CSV summaries are the citable records.

---

## 1. Question and setup

**Hypothesis (the "generalist").** A single protocol-agnostic RF-fingerprint encoder,
trained across multiple device populations/protocols with physics priors (a classical-19
feature token + an LPC protocol-residual channel), will *discover* emitters of an unseen
protocol/domain better than (a) a frozen single-domain encoder transferred cold, approaching
(b) a native in-domain encoder.

**Pool & claim cell.** Leave-one-protocol-out with **holdout = DRFF** (drone, OcuSync
5.8 GHz, native 50 MS/s): train identities = **WiSig 109** (WiFi 802.11a, 25 MS/s) **+
ORACLE 12** (WiFi 802.11a, 5 MS/s) — **121 identities, no drone data of any kind**; eval =
the **8 mavicAir2** airframes, touched only at final eval.
Split + disjointness asserts (WiSig TEST/board-18 excluded everywhere; mavicAir2 never in a
train pool): `results_gen/splits_lopo.json` (holdout=DRFF → train 121 / eval 8), produced by
`gen_rff/bench/lopo.py::write_all_lopo_splits` (seed 777).

**Evaluation.** The locked scoring harness ported verbatim from
`summer_work/EVAL_PROTOCOL.md §4.3` into `gen_rff/bench/harness.py` (oracle-K = k-means AND
spectral, reported separately; HDBSCAN mean over the DRFF mcs grid {5,7}; bursts = mean of N
windows, balanced to min per class). Reproduction of the locked cells to ±0.02 was gated in
Phase 0 (`results_gen/verify_report.json`, key `V3`: DRFF frozen E1 km/sp **0.297/0.298**
exact, classical HDB-mean **0.245** exact, WiSig DEV P2 km **0.714** vs locked ~0.72).

**Phase-2 training run** (the claim-cell result under scrutiny here):
`gen_rff/train/train_lopo.py` → `results_gen/phase2_lopo_drff_report.json` (GenRFEncoder
1.84 M params, SupCon τ=0.5, P8×V4 cross-condition positives, 10 k steps, selection composite
frozen before training, selected step 5000).

---

## 2. Findings (strongest evidence first)

### F1 — THE WALL IS N-DEPENDENT; burst integration washes out most of the transfer gap.
At small bursts the cross-domain gap is large; at N=120 the deep encoders and even classical
features converge into one band. mavicAir2 8-way, E1 bursts, cap 1500
(`results_gen/phase2b_part1_controls.csv`, `gen_rff/verify/phase2b_part1_controls.py`):

| method | N=10 oracle-km | **N=120 oracle-km / sp** |
|---|---|---|
| frozen WiSig-only (OPT-B 256) | 0.137 | **0.729 / 0.777** |
| classical-19 (OPT-B 256) | 0.174 | **0.713 / 0.666** |
| native drone-trained (native 4096) | 0.276 | **0.792 / 0.835** |
| generalist step-5000 (native 4096) | 0.197 | **0.625 / 0.666** |

At N=120 all four sit in **0.62–0.79** — heavy averaging, not representation, drives most of
the number. The **residual representational gap** that native training actually buys is
frozen-WiFi **0.729** vs native **0.792/0.835** at N=120 (≈0.06–0.11 km/sp). Everything else
is integration.

### F2 — MULTI-DOMAIN WIFI-FAMILY TRAINING DID NOT IMPROVE TRANSFER.
The generalist is the **worst deep method at N=120** (0.625 < frozen 0.729 < native 0.792);
[`phase2b_part1_controls.csv`]. This is the cap-robust comparison (N=120 collapses each pool
to ~10 bursts/af regardless of cap). At **N=10 the comparison is cap-confounded** (see §4a):
against the locked wall (0.297, cap 320) the generalist (0.197) is lower, but at *matched*
cap 1500 the frozen encoder is only 0.137 — so N=10 yields no clean verdict either direction,
and in **neither** case does the generalist reach the native reference (0.276 @N10, 0.792
@N120). Net: training a WiFi-family generalist bought nothing transferable that the frozen
WiSig encoder didn't already have. [`phase2_lopo_drff_report.json` final_eval;
`phase2b_part1_controls.csv`]

### F3 — PHYSICS-TOKEN INJECTION HURTS.
Ablation A1 (physics token removed, physics-side branch-dropout off) *improves* both axes vs
the full recipe: mav@N10 **0.197 → 0.227 (+0.030)**, WiSig-DEV self-cell **0.392 → 0.514
(+0.122)** [`results_gen/phase2b_part2_ablation.csv`, rows `full_recipe_CITED` / `A1_no_physics`;
`gen_rff/train/ablation.py`]. The token was *used* during training — physics grad-norm share
ranged 0.06–0.33 across checkpoints [`phase2_lopo_drff_report.json` curve, `phys_grad_share`]
— yet its net effect is negative. Injected classical priors, as fused here (concat+FC), are
not a free lunch.

### F4 — THE LPC RESIDUAL IS THE ONE VINDICATED INGREDIENT.
Ablation A2 (residual removed, 2-ch stems) *degrades* the recipe: WiSig self-cell **0.392 →
0.260 (−0.132)**, mav@N120 **0.625 → 0.489 (−0.136)**; mav@N10 ≈ neutral (+0.007)
[`phase2b_part2_ablation.csv`, `A2_no_residual`]. So the protocol-suppressing residual channel
carries real signal. **Caveat — it is rate-dependent:** residual-energy-ratio is 0.001 (DRFF
50 MS/s) and 0.107 (WiSig 25 MS/s) but **0.987 (ORACLE 5 MS/s OFDM)** — blind LPC extracts
almost nothing on low-oversampling OFDM [`results_gen/verify_report.json`, key `V4`
`residual_energy_ratio`].

### F5 — CROSS-CONDITION POSITIVES TRADE TRANSFER AGAINST IN-DOMAIN.
Ablation A3 (positives unconstrained within device) gives the **best transfer** (mav@N10
**0.272**, mav@N120 **0.806**) but the **worst in-domain** (WiSig self **0.174**)
[`phase2b_part2_ablation.csv`, `A3_no_crosscond`]. Where invariance pressure is placed is a
real, load-bearing dial — which is exactly what an explicit domain/channel-adversarial
objective would target (§5).

### F6 — ORACLE WAS UNLEARNABLE UNDER THIS RECIPE, AND HURT TRANSFER.
Per-domain SupCon loss: WiSig fell to ~2.2–2.4 in every arm while **ORACLE stayed ~3.4 flat**
— A1 3.434→3.432, A2 3.457→3.416, A3 3.441→3.413, and even **A5 at 30 k steps only 3.442→3.27**
[`phase2b_part2_ablation.json` `oracle_loss_note` / `dloss_trajectories`; and
`phase2_lopo_drff_report.json` curve VAL-B plateau ~0.51]. Removing ORACLE (A4) *improves*
transfer (mav@N10 +0.050, mav@N120 0.625→0.781) [`phase2b_part2_ablation.csv`, `A4_no_oracle`].
A small-identity (12), low-rate (5 MS/s), 2-condition domain whose residual is useless (F4)
contributed no learnable structure and diluted the shared pool.

### F7 — THE GENERALIST'S IN-DOMAIN COST IS INTRINSIC.
The generalist scores WiSig-DEV self-cell **0.392** vs the **0.72** WiSig specialist reference
[`phase2_lopo_drff_report.json` `self_cells.wisig_dev`; specialist ref = EVAL_PROTOCOL §4.3
"DEV P2 ≈ 0.72"]. This near-halving is **not** ORACLE-poisoning (A4 self = **0.397** ≈ full)
and **not** undertraining (A5 30 k self = **0.325**, if anything worse)
[`phase2b_part2_ablation.csv`]. It is intrinsic to the multi-domain SupCon + 4-ch/physics
setup relative to the dedicated CosFace/512-D specialist pipeline.

### F8 — COUNT ESTIMATION DID NOT TRANSFER WITH QUALITY.
At N=120 the generalist's eigengap estimate is **K=5** (true 8) with ARI@K_est 0.535
[`phase2_lopo_drff_report.json` final_eval.N120 `eigengap_Kest`/`ARI_at_Kest`], whereas the
native drone stack (Play-1b) recovered K cleanly (eigengap nailed K=8 on seed1234; K=4 battery
partition ARI ≈ 0.80; N=120 oracle-K ≈ 0.895)
[`summer_work/results/demo_play1b/DEMO_OPERATING_POINT.md` + `report.json`]. Good clustering
geometry (low K-error) did not come along for the cross-domain ride.

---

## 3. What the evidence supports and rules out

**Supports**
- **A router architecture over a monolithic generalist** — protocol-classify → *native*
  encoder per domain → classical + heavy-averaging (N large) fallback. Justified by F1
  (native is the only thing above the averaging band) + F2 (the generalist adds nothing) +
  F8 (native count-estimation is far better).
- **Fast per-domain enrollment as the deployment answer**, since heavy burst integration
  (F1) already closes most of the discovery gap for *any* feature set once you have windows
  of a new emitter.
- **Genuine cross-*family* data diversity as the precondition for any true generalist** —
  F6 shows adding a same-family-but-degenerate domain (ORACLE) hurts; a real generalist would
  need protocol families that are actually different and individually learnable.

**Rules out / deprioritizes**
- **Physics-token injection as built** (F3) — concat+FC fusion of a classical-19 token is
  net-negative; do not carry it forward unchanged.
- **WiFi-family multi-domain SupCon as a transfer mechanism** (F2, F1) — it does not beat
  frozen transfer; the averaging regime dominates.
- **Capacity / longer training as the binding constraint** (F7 via A5; the model already
  fits WiSig to kNN-1 ≈ 0.99) — the ceiling is representational/objective, not budget.

---

## 4. Reconciliation notes and known discrepancies

**(a) frozen@N10 = 0.137 (Part-1) vs 0.297 (locked wall) — cause pinned from code.**
Both numbers come from the *identical* pipeline — same frozen encoder
(`runs/wisig_supcon_fft64/retrain_best/best_model.pt`), same OPT-B 256-sample mavicAir2
windows built by `gen_rff/bench/lopo.py::_build_mavicAir2_optb`, same E1 construction
(round-robin over (D,C) cells, `nb = len(idx)//N`, N=10; `lopo.py:150` == `phase2b_part1_
controls.py:50`), and the same `harness.oracle_km_sp(unit(bp), bl, 8)`. **The only difference
is the OPT-B window cap**: the locked 0.297 uses `cap = DRFF_CAP = 320` (`lopo.py:40`, the
path `reproduce_drff_baselines` calls), whereas Part-1 uses `cap = CAP = 1500`
(`phase2b_part1_controls.py:40`). At cap 320 each airframe yields ~32 N=10 bursts; at cap 1500
~150. Oracle-K@8 (k-means over the *whole* burst set) on the **cross-domain-frozen** embedding
fragments more as the point count grows, so ARI falls 0.297 → 0.137; N=120 is insensitive
because it collapses the pool back to ~10 bursts/af regardless of cap. This is **proven**, not
inferred: Phase-0 V3 reproduced the locked 0.297 **exactly** at cap 320
(`results_gen/verify_report.json` `V3` cell "DRFF frozen-512 E1 oracle-km": reproduced 0.297,
diff −0.0), so cap is the sole moving part. Consequence: **oracle-K@8 at small N is a
burst-count-sensitive statistic on noisy cross-domain features and should not be quoted as a
fixed "wall"** — the cap-robust comparison is N=120 (F1/F2).

**(b) Phase-2 "STRONG @ 0.625" is superseded.** The Phase-2 checkpoint banded the run STRONG
on the literal oracle-K@8 ≥ 0.50 threshold at N=120 (0.625). The N-matched control (F1/F2)
supersedes that reading: 0.625 is *below* frozen (0.729) and classical (0.713) at the same N,
so it reflects burst-averaging, not a transfer win. The honest verdict is F2, not "STRONG".

**(c) ORACLE minimal-cache caveats.** ORACLE windows come from a minimal cache built
read-only from raw SigMF (`gen_rff/data/loaders.py::build_oracle_cache`): the raw cf32 stream
carries ~2.6 % out-of-range saturation spikes (~2^65) that overflow variance, so samples are
**clamped to ±5** before standardizing and only >50 %-corrupt windows are dropped. This is a
registration/forward source, **not** the V4 ORACLE benchmark pipeline; ORACLE numbers here
are indicative only (and F6 shows ORACLE was unlearnable regardless).

---

## 5. Open card — DANN / channel-adversarial closing experiment

*Pre-registered here; the next session appends its result, pass or fail, without moving these
goalposts.*

- **Design.** Take the one vindicated config — **residual-only, no physics-token, no ORACLE**
  (train = WiSig 109; GenRFEncoder `use_physics=False, use_residual=True`) — and add a
  **domain/channel-adversarial head**: a discriminator on the 512-D pre-projection trained to
  predict the WiSig *condition* (rx/date cell) through a gradient-reversal layer, so the
  encoder is pushed toward channel-invariant, hardware-only features. Same LOPO-DRFF eval,
  same locked harness, same frozen selection formula.
- **Why this lever.** F5 shows the transfer/in-domain balance is highly sensitive to *how*
  invariance is imposed via positive construction; DANN makes that invariance an explicit,
  differentiable objective rather than a sampler side-effect — the single most direct attack
  on the diagnosed failure (domain-specific, non-transferring features, F2/F7).
- **Success criterion (pre-registered).** mav@N120 oracle-km **> 0.729** (clears the frozen
  bar at matched N, i.e. a real per-fingerprint gain, not averaging) **AND** WiSig-DEV
  self-cell not worse than the residual-only baseline. Anything less = the generalist
  direction is not worth further investment; pivot to the router (§3).
### 5.1 Pre-registered prior + bands (Phase 3, frozen 2026-07-12 BEFORE training)

**Expectation: LOW.** DANN removes *domain-predictive* structure from the representation; but
the diagnosed failure of the generalist line is that drone-relevant information is **absent
from the training pool**, not present-but-entangled (F2/F6). With **one** training protocol
(WiSig only, ORACLE dropped per F6) the adversary cannot enforce *protocol* invariance at all.
This run therefore tests the strongest available variant — **channel-adversarial invariance**:
suppress WiSig receiver/session structure (F5 showed invariance *placement* is a real,
load-bearing dial). Success would mean channel-invariance pressure learned on WiSig
*generalizes to the unseen drone domain's channel structure*.

**Bands (mavicAir2 oracle-K@8 km, part-1 harness, N=10 / N=120):**
- **SIGNAL:** N10 ≥ 0.35 (clears the locked frozen wall 0.297 meaningfully) **OR** N120 ≥ 0.80
  (clears frozen 0.729 by ~0.07).
- **NULL:** below both → the generalist architecture line is **CLOSED** on this pool; router +
  data-diversity stand as the conclusions (F1/F2/F3/F4/F6 cited).

*This single pre-registered run has no rescue attempts, no λ sweep, no second seed. The 5.1
bands are the operative criteria for Phase 3 (they refine the generic §5 success criterion
above for this specific channel-adversarial run).*

### 5.2 Result (Phase 3) — BAND = NULL; generalist line CLOSED

*One pre-registered run, seed 1234, no sweeps/rescues/second-seed.
`gen_rff/train/train_dann.py` → `results_gen/phase3_dann_report.json`,
`runs_gen/dann/best.pt`.*

**Design as run.** WiSig-109 only (ORACLE dropped, F6); GenRFEncoder residual ON (F4),
physics token OFF (F3), cross-condition positives ON (F5). Channel adversary = MLP(512→256→K)
on the 512-D pre-projection via gradient reversal, λ ramp 0→0.3 over the first 30% of 10 k
steps then flat. Selection = VAL-A oracle-km/0.72 + VAL-A kNN-1 (argmax; no ORACLE), frozen.
**Adversary granularity decided from batch stats before training:** distinct (rx,date)/batch
= 6.28 of 16 vs distinct rx/batch = 2.36 of 4 → **rx-only, 4-way**. (The frozen Phase-2
`WISIG_CAP=250`, taken in rx-major storage order, exposes only ~4 receivers across the pool —
a faithful consequence of the fixed config; not retuned.)

**Adversary-accuracy trajectory (chance = 0.25):** first 0.625 → peak 0.938 → last 0.688,
mean over final 20% = **0.762**. The adversary stayed **well above chance throughout** →
**WEAK-PRESSURE**: λ=0.3 did not drive the encoder to receiver-invariance (the 4-way receiver
signal from the capped pool was never suppressed). Per the pre-registration this is reported,
not rescued.

**Final table — mavicAir2 8-way, part-1 harness (all cells internally comparable; N=10 cap
1500; §4a for the locked-vs-part1 N10 note):**

| method | N=10 km | **N=120 km / sp** | eigengap K / ARI@K | WiSig self |
|---|---|---|---|---|
| frozen WiSig-only | 0.137 (locked 0.297) | **0.729 / 0.777** | 7 / 0.68 | 0.72 (specialist) |
| native drone-trained | 0.276 | **0.792 / 0.835** | 6 / 0.69 | — |
| A3 no-crosscond | 0.272 | **0.806** | — | 0.174 |
| generalist (Phase 2) | 0.197 | **0.625 / 0.666** | 5 / 0.535 | 0.392 |
| **DANN (Phase 3)** | **0.219** | **0.534 / 0.631** | 7 / 0.631 | 0.494 |

[`results_gen/phase3_dann_report.json` final_eval + context_rows; `phase2b_part1_controls.csv`.]

**Band verdict:** **NULL** — mav oracle-K@8 km = **0.219 @N10 (<0.35) and 0.534 @N120 (<0.80)**,
below both bands. DANN's N=120 (0.534) is in fact the **worst** of every method measured, below
the untrained-transfer frozen baseline (0.729): channel-adversarial pressure, even the weak
pressure achieved here, **degraded** cross-domain transfer rather than improving it. (In-domain
WiSig self-cell did rise, 0.392→0.494, consistent with F3/F7 — physics-off helps in-domain —
but that is not the tested axis.)

**Closing statement — the generalist architecture line is CLOSED on this pool.** Architecture
space exhausted: multi-domain SupCon (F2), physics-token injection (F3), residual views (F4,
helpful but insufficient), cross-condition positives (F5), capacity / longer training (F7 via
A5), and now channel-adversarial invariance (this run) **all fail to beat the frozen baseline
at matched N**. The N=120 convergence band (F1) shows burst integration — not learned
representation — carries cross-domain discovery for every feature set on this data. The
remaining levers are **(i) genuine data diversity** (protocol families that are actually
different and individually learnable, unlike degenerate ORACLE, F6) and **(ii) per-domain
native enrollment behind a protocol router** (§3), which already delivers the demo operating
point (~0.89 @N120 in-domain). No further monolithic-generalist runs are warranted on this
pool.

---

## 6. Relation to the locked paper

**Nothing in this document alters the locked paper's tables or the frozen assets** — the
WiSig encoder, splits, drone-native checkpoint, and demo operating point are untouched
(sha256 unchanged across all gen-rff sessions; see the session git audits). These are
sandbox / future-work results. The strongest candidate for **paper-2 (or a future-work
paragraph)** is **F1**: the *N-dependence of the cross-domain wall* — that burst integration
alone lifts frozen-WiFi, classical, and native features into one 0.62–0.79 band at N=120,
and that the residual representational gap native training buys is only ~0.06–0.11. F2/F3/F6
are the supporting negative results (a WiFi-family generalist + physics-token injection does
not beat frozen transfer; degenerate domains poison the pool), which motivate the router
architecture (§3) as the honest deployment story.
