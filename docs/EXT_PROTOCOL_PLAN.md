# EXT-PROTOCOL COMPARATIVE STUDY — Zigbee + BLE Tiers
## Approach A (Spectral/Heuristic) vs Approach B (Trained Encoder)

Status: PLAN — no experiments run. Branch: `experimental/generalized-rff`, all work isolated in `gen_rff/ext_protocols/` (demo_router/ touched only in Phase 5 via the tier registry). Frozen assets read-only. TEST/board-18 closed. mavicAir2 demo-content exception unchanged.

---

## 0. Framing and pre-registered priors

This is not a from-zero study. Approach A vs Approach B is the **same comparison the sandbox already ran on drones** — classical-19 physics features vs learned encoders (frozen / adapted / native). Running it on two new PHYs makes this a **replication-and-generalization study of the F-series findings**, which is paper-2's first real chapter. That is the scientific value: either the WiFi→drone pattern replicates on Zigbee/BLE (strong generalization claim), or it diverges (informative — e.g., narrowband GFSK/O-QPSK may be CFO-dominated enough that classical features win outright).

**Scope separation (important):** A vs B compares *fingerprinting tiers* (WHO is emitting). Protocol *routing* (WHAT is emitting) stays spectral-statistic + per-class Mahalanobis regardless of the A/B outcome — routing was never the contested question; the router already generalizes (1.00 on unseen airframes).

**Pre-registered priors** (written before any result; from FINDINGS.md F1–F8, Play-1, R1–R3):

| # | Prior | Source precedent |
|---|-------|------------------|
| P1 | B1 (frozen WiSig transfer) is weak at low N, converges toward the common band at N=120 | F1 (the wall is N-dependent) |
| P2 | B2 (fine-tuning from WiFi weights) fails to beat B3 (from scratch) | R1–R3 adaptation failures |
| P3 | A beats B1 at low N cross-domain | classical 0.245 vs frozen 0.071 (DRFF) |
| P4 | B3 (native) is the overall winner *if* the dataset has enough units/sessions | Play-1 (0.756 oracle-K@8) |
| P5 | At N=120, A / B1 / B3 converge into one band; burst integration dominates | F1 |
| P6 | Divergence candidate: on narrowband protocols the individuating information may be concentrated in CFO + transient envelope, favoring A more than on wideband OFDM | open — the study's novel cell |

Gates and battery are fixed before results; no iteration past the battery; every phase ends at a STOP checkpoint.

---

## 1. Data strategy

### 1.1 Hard requirements per dataset (GO/NO-GO checklist)
- Raw IQ (not demodulated/decoded, not RSSI/CSI)
- **≥ 8 same-model units** (6 = floor) with per-unit labels — the study is same-model discovery; different-model classification sets are useless here
- ≥ 2 capture sessions/days (session-disjoint splits; the receiver-locked-burst poison lesson applies to any single-session set)
- Known sample rate + capture chain documentation
- License permitting research use; actually downloadable

### 1.2 Candidates (verify in Phase 0 — availability of the first two is NOT yet confirmed)

**Zigbee (802.15.4, O-QPSK DSSS, 2 MHz ch @ 2.4 GHz):**
1. **CC2530 × 54** (arXiv 2108.04436): 54 same-model TI CC2530 units, USRP N210 @ 10 MS/s, 1280-sample preambles, capture blocks spanning 18 months incl. device aging. Format-perfect (same-model, many units, multi-session, aging axis). **Public availability unverified — Phase 0 task #1.**
2. **SDR4IoT BLE & Zigbee** (Zenodo record 4639390): raw IQ, USRP N210, w-iLab.2 testbed nodes, multiple positions/emitters. Confirmed public. **Unit-model homogeneity and per-unit counts need audit** (testbed motes are often identical-model fleets — promising but unproven).

**BLE (GFSK, 1–2 MHz ch @ 2.4 GHz):**
1. **Seeed XIAO ESP32-C3 × 31** (arXiv 2510.09940): 31 same-model units, USRP B210, raw IQ, multiple environments / BLE channels / receivers — format-perfect, and the receiver axis lets us re-test the receiver-diversity control. **Availability unverified — Phase 0 task #2.**
2. **SDR4IoT** (above) — BLE side, same audit caveat.
3. **NEU BT+WiFi 10-chipset dataset** (arXiv 2303.13538): public, raw IQ, 72 GB, multi-day, WiFi+BT from the same combo chips. Only 10 devices, heterogeneous models → **secondary use only**: router training data, cross-standard same-emitter analysis, and the paper's "BT needs >1024-sample inputs" data point. Not a discovery eval set.
4. Fallbacks (only if 1–3 fail): Uzundurukan smartphone BT database (27 phones, several same-model subgroups; note it is **BT Classic, not BLE** — protocol mismatch must be documented); Wearable BT/BLE physical-layer dataset (MDPI Data 2024; likely model-level, few same-model units).

### 1.3 Preprocessing (native-spec doctrine, per Play-1)
- **No forced commensurability with the WiSig 25 MS/s spec.** Each protocol gets a native window spec: window length covering the preamble/transient at the dataset's native rate (Zigbee: 802.15.4 preamble at 10 MS/s ≈ 1280 samples per the CC2530 set; BLE: preamble + access address, plus the **transient** — BT literature says the turn-on transient carries strong fingerprint content, so windowing must include it where captures allow).
- Resample only if a dataset's rate is impractical; log the decision memo (M100 multi-rate precedent: decide explicitly, don't stall).
- Per-window unit-power standardization; STFT sized to the window (print tensor shape sanity check before any training); energy-based burst detection mirroring the drff pipeline; cached .npz + sha256; `splits_ext.json` per protocol.
- Splits: **unit-disjoint eval group held out entirely** (mavicAir2 pattern — no windows, no stats, no checkpoint signal), 2 val units held out from the pool, session-disjoint val segments within train units.
- B1 (frozen WiSig) additionally needs a commensurable 256@25-equivalent windowing arm — that arm alone uses resampling to the WiSig spec, documented separately.

---

## 2. Comparative framework (locked battery)

Locked harness: burst-mean discovery, both oracle-K and estimated-K reported, fixed seeds (2 seeds for any trained arm), fixed battery.

| Test | What | Metric | Purpose |
|------|------|--------|---------|
| T1 | Transferability gate | Supervised linear probe acc + kNN-1 on held-out units | Gate ≥ 0.30-over-chance precedent; cheap kill-switch per arm |
| T2 | Discovery | ARI oracle-K (k-means + spectral), estimated-K (eigengap, correct-K rate), HDBSCAN deployable | The locked headline metric |
| T3 | N-sweep | T2 at N ∈ {1, 10, 30, 120} | Tests P1/P5 directly; the F1 replication cell |
| T4 | Data efficiency | B3 trained on 25/50/100% of pool units; A requires zero training data (report as structural property, quantified) | The axis where A can win even if B3 tops raw ARI |
| T5 | Compute | Params, ms/window (CPU + 4090), feature-extraction cost | Deployment argument for the router tiers |
| T6 | Router impact | Routing acc with 4 enrolled classes, Mahalanobis threshold re-check, FM smoke re-run | Integration safety |

**Why-analysis (the "know why" requirement), attached to Phase 4:**
- Per-feature ablation of A: which physics features carry each protocol (CFO? transient envelope? spectral moments?)
- Probe of B3 embeddings against A's features (linear predictability): does the encoder rediscover the classical physics, or find something orthogonal?
- N-curves interpreted against F1 (integration vs representation decomposition)
- UMAP/eigengap spectra per protocol per method

Confound controls inherited wholesale: session-disjoint bursts, unit-disjoint eval, CFO augmentation OFF for training (CFO is a cue — and for Approach A it is explicitly a *feature*), single-threaded BLAS for clustering runs, `rm -rf __pycache__`.

---

## 3. Architectural plan (extend demo_router/ without breaking Drone/WiFi)

1. **All new science code/data in `gen_rff/ext_protocols/`** — datasets, caches, features, encoders, eval. demo_router/ is untouched until Phase 5.
2. **Tier registry pattern**: `tiers.py` becomes config-driven — `TIER_REGISTRY = {protocol: backend}` where a backend implements `embed(track_windows) -> vector` (dim may differ per protocol; safe because the base station clusters per protocol group and never across — namespaced IDs already guarantee this). Zigbee/BLE tiers plug in as registry entries pointing at the Phase-4 winner (classical extractor or encoder checkpoint).
3. **Router refit**: per-class Mahalanobis re-fit with the two new classes; UNKNOWN gate percentile re-derived; FM smoke re-run. Routing features unchanged (occupied-BW/spread/flatness already discriminate narrowband protocols well — verify, don't assume).
4. **JSON contract**: `protocol` gains values `zigbee`/`ble`; embedding dim documented per protocol; `fingerprint_id` namespace extends (`d1.zigbee`, …). AR layer consumes unchanged (it keys on fingerprint_id + protocol, not embedding dim).
5. **REGRESSION GATE (non-negotiable)**: after router refit, S1–S3 re-run must reproduce prior results (scenario_results.csv comparison in the report). New scenarios: S4 zigbee×K, S5 ble×K, S6 grand-mixed (drone + wifi + zigbee or ble simultaneously — the new money shot). Gates for S4/S5 set from Phase-4 measured numbers *before* scenario runs, S6 routing ≥ 0.95.
6. Git discipline unchanged: gen_rff-only diffs, frozen sha256 proofs, no merge to main, results_gen/ gitignored. Note: merge-debt to main grows with this work — schedule the reviewed merge after Phase 5.

---

## 4. Phased execution (each phase = one session, STOP at checkpoint)

> **Disk-budget amendment (2026-07-13, post reclaim-audit):** the original ≥100 GB-free download floor is replaced. Rule now: SAFE-only reclaim executed (~52 GB free), and **no download or preprocessing operation may leave <20 GB free**. ORACLE-raw (manifest R1) retirement deferred — not required for this study. Manifest R4 (`processed/drff_r2`) reclassified verify-then-PROTECT (likely the live DRFF cache).

**Phase 0 — Dataset acquisition + audit (CPU-only).**
Verify availability/licenses of CC2530×54 and XIAO×31; download SDR4IoT + NEU sets; count same-model units, sessions, per-unit sample volume, SNR; inspect raw IQ (spectrograms, burst structure). Deliverable: `DATA_AUDIT.md` + GO/NO-GO per protocol against §1.1.
**Gate:** a protocol with no qualifying dataset drops to **router-enrollment-only** (spectral recognition → T-C fallback), documented, and exits the study. If both fail, the study collapses to that cheap version — decided here, not after sunk training.

**Phase 1 — Preprocessing + splits + caches.**
Native window spec memo per protocol; burst extraction; caches + sha256; `splits_ext.json` (unit-disjoint eval group, val units, session-disjoint segments); the B1 commensurable-windowing arm. Deliverable: cache stats table + spec memo.

**Phase 2 — Approach A (the floor).**
Derive `classical_z` and `classical_b` feature sets (~19-D each for comparability with classical-19): CFO estimate (preamble-based), IQ-imbalance proxies, transient envelope statistics (rise time, curvature — BT-transient literature), spectral moments, occupied BW, PSD flatness. Run T1–T3 battery. Deliverable: full A results table. A's numbers become the floor every B arm must beat *and* the S4/S5 gate inputs if A wins.

**Phase 3 — Approach B (GPU sessions).**
- **B1 frozen WiSig** (cheapest, run first): frozen best_model.pt on the commensurable windows → T1–T3. Tests P1/P3 immediately.
- **B2 fine-tune** (optional arm — predicted to fail per P2; include for documentation value or skip; decide at the Phase-2 checkpoint): head-only and partial-unfreeze mirrors of R1/R2.
- **B3 native from scratch**: Play-1 recipe — dual-branch RFEncoder re-instantiated for the native input spec, SupCon on unit labels, constant tau=0.5, GroupNorm, balanced per-unit batches, AdamW+cosine+AMP, checkpoint selection on held-out val units' clustering + val gap (composite stated before launch), 2 seeds. Plus the T4 data-efficiency arms.

**Phase 4 — Comparative benchmark + verdict.**
One table: {A, B1, B2, B3} × {T1–T5} × {zigbee, ble}. Score each pre-registered prior P1–P6 (held/refuted). Why-analysis (§2). Pick the per-protocol demo-tier winner on the composite (discovery ARI at the demo operating point N=120, tempered by T4/T5 if within noise). Deliverable: `EXT_FINDINGS.md` with F-numbered findings, full artifact provenance — paper-2 material.

**Phase 5 — Router integration.**
Registry entries, router refit, regression gate S1–S3, new scenarios S4–S6, README update (design-choice ↔ EXT_FINDINGS table), git audit. Deliverable: extended scenario_results.csv + updated demo.

---

## 5. Honest risk register

- **Dataset risk is the whole ballgame.** Both format-perfect candidates (CC2530×54, XIAO×31) have unverified public availability. Phase 0 exists to spend one CPU session resolving this before anything else is invested. Do not substitute different-model datasets to keep the study alive — that changes the question.
- **Single-session datasets** would reintroduce the channel-coherence confound; if a candidate has only one session, its discovery numbers carry a mandatory caveat or the set is rejected.
- **BT Classic ≠ BLE** — different modulation index, packet structure, hopping. Any fallback to the smartphone BT database must be labeled a protocol substitution.
- **B3 may be data-starved**: Play-1 had 15 airframes for training; if a protocol's pool is (say) 20 units, the train/val/eval split leaves ~12–14 — comparable, workable. Below ~10 pool units, B3's prior weakens and A likely wins by default; note this in the verdict rather than stretching.
- **Scenario-table inflation**: as with S1–S3, perfect small-K demo numbers are integration validation, not performance claims. Citable numbers come from the T1–T3 battery on the full unit pools.
