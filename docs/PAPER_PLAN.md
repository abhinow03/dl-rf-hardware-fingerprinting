# Paper Plan — Open-Set RF Emitter Discovery

Working doc for the internship paper. Scope: the open-set discovery **model only**.
Localization / multilateration is explicitly out of scope (it belongs to the larger
capstone pipeline). Compiled June 2026.

---

## 1. Working title (pick later)

- "Open-Set Discovery of Unseen RF Emitters via Metric-Learned Hardware Fingerprints"
- "Discovering and Counting Unknown Emitters: Open-Set RF Fingerprinting that
  Generalizes Across Devices and Domains"

---

## 2. Contribution (locked)

A metric-learning encoder that **discovers** emitters it never saw in training —
clustering individual devices at the hardware-fingerprint level — rather than the
dominant "reject the rogue device" framing. Three concrete claims:

1. **Discovery, not rejection.** Given signals from emitters absent in training, the
   model groups them into distinct device identities (Unknown_A, Unknown_B, ...),
   evaluated as a clustering problem, not binary known/unknown detection.
2. **Cross-domain generalization.** The fingerprint metric, trained on one device
   population, transfers to discover emitters of a *different type/domain* unseen in
   training (e.g., trained on WiFi, discovers drones).
3. **(Phase 2) Count estimation.** Estimating the *number* of unknown emitters
   automatically, without assuming K. Staged — core discovery first, counting second.

Out of scope, stated as future work: AOA/TOA/RSSI/Doppler localization and
multilateration (data gap + separate problem; part of the larger pipeline).

---

## 3. Positioning (from the survey — differentiate explicitly)

- vs **open-set SEI** (OpenMax, Prototypical+EVT, Xie 2021, Guo 2024): they reject
  unknowns; we discover and cluster them. Use them as baselines (adapted).
- vs **2026 open-set UAV clustering** (arXiv 2603.24268): class/type-level discovery
  from signal semantics; we do individual-*device* discovery via hardware fingerprint,
  plus cross-domain transfer.
- vs **UAVSig** (MILCOM 2024): closed-set same-model fingerprinting; we are open-set.
- vs **Hiles 2025 generic RFF framework** (EDA/RFEC named as tasks): we instantiate,
  evaluate, and stress-test discovery with count estimation and transfer.

One-sentence delta to lead with: *end-to-end open-set discovery and counting of
individual emitters that generalizes to device populations and domains unseen in
training.*

---

## 4. Problem formulation

- Training set: signals from a set of **known** devices D_train, with labels.
- Test set: signals from devices D_test, **disjoint** from D_train (device-disjoint
  split — no device appears in both).
- Encoder f: signal → L2-normalized embedding in R^128.
- Task A (core): cluster D_test embeddings into device identities; evaluate clustering
  quality against true device labels (which are held out, used only for scoring).
- Task B (phase 2): estimate |D_test| (number of distinct emitters) without it being given.
- Key principle carried from V4 work: embeddings must encode hardware identity, not
  distance/SNR/channel. Verified explicitly (see robustness experiments).

---

## 5. Datasets and splits

All device-disjoint. Three roles:

**Transfer SOURCE (training corpus, many devices):**
- WiSig — 174 off-the-shelf WiFi Tx (free, UCLA CORES + GitHub). Primary source.
- LoRa-RFFI — 60 COTS LoRa devices (free, IEEE DataPort). Optional 2nd source /
  second protocol for a stronger generalization claim.

**Discovery TARGET (unseen emitters, evaluation):**
- A multi-unit drone set — Feb-2026 UAV RF dataset (26 units / 8 models, multiple
  units per model) **[VERIFY free access/license]**, and/or UAVSig (free, UCLA CORES,
  same-model drones). Drones = the cross-domain transfer target.

**Second benchmark (within-domain, already works):**
- ORACLE — 16 USRP X310, your locked 12/4 split. Reuse V4 results as a sanity anchor.

Split discipline: every experiment uses a device-disjoint train/test split. Document
exact device IDs per split (as you did with ORACLE's held-out 4) for reproducibility.

---

## 6. Model (reuse V4 — do not rebuild)

Locked architecture: dual-branch encoder (1D ResNet on time-domain IQ + STFT/2D ResNet
on spectral [2,129,61]), cross-attention fusion + residual + LayerNorm, projection head
FC(512→256→128) + L2 norm, GroupNorm throughout. SupCon loss, AdamW, cosine LR + warmup,
AMP, grad clip. The paper's methodological content is the **discovery procedure** and the
**transfer + robustness** evaluation built on this encoder, plus:
- Optional **distance-invariance loss term** (ties into robustness experiments; you
  planned this already).
- Inference-encoder parameter audit (report the real deployed param count).

---

## 7. Discovery + counting method

**Inference pipeline:** embed test signals → (optional dim-reduce) → cluster → assign IDs.

- **Phase 1 (count known):** cluster with K = true number of test devices. Measures
  pure embedding quality. Clustering: k-means / spectral / agglomerative on cosine.
- **Phase 2 (count unknown):** estimate number of emitters. Candidate methods:
  silhouette / eigengap / DBSCAN / GCD-style count estimators (e.g., CiPR's reference
  score). Report estimated-vs-true count error.

---

## 8. Baselines (adapt to the discovery setting)

These are mostly rejection methods — adapting them to discovery is part of the design
and reinforces the contribution.

- **OpenMax** — replace softmax tail; use its reject scoring, then cluster rejected set.
- **Prototypical networks + EVT** — episodic embedding + extreme value rejection, then cluster.
- **Similarity-threshold** — naive: threshold on cosine to form groups. The "trivial discovery" baseline.

Report all three on the same splits/metrics as the proposed method.

---

## 9. Metrics

- **Clustering quality:** NMI, ARI, V-measure, cluster purity (purity is what you
  already report — keep for continuity, add NMI/ARI which reviewers expect).
- **Counting (phase 2):** |K_est − K_true|, and purity/NMI at the estimated K.
- **Open-set sanity:** AUROC for known-vs-unknown separation (optional, ties to baselines).
- **Robustness curves:** metric vs distance, vs SNR, vs day/receiver shift.
- **Embedding sanity:** sim_gap, eval_sim (carry from V4), t-SNE for the paper figure.

Minimum publishable bar: proposed method beats all three baselines on NMI/ARI for
within-domain discovery AND shows non-trivial cross-domain transfer. Stretch: phase-2
counting within a small error.

---

## 10. Experiment matrix

| ID | Experiment | Train | Test | Purpose |
|----|-----------|-------|------|---------|
| E1 | Within-domain discovery (drones) | drone-set devices (subset) | held-out drone devices | Core result |
| E2 | Within-domain discovery (ORACLE) | 12 ORACLE | 4 held-out | Second benchmark (works already) |
| E3 | **Cross-domain transfer** | WiSig (and/or LoRa) | drones (unseen domain) | Headline generalization claim |
| E4 | Baselines vs ours | same as E1–E3 | same | OpenMax, Proto+EVT, threshold |
| E5 | Ablations | — | — | What carries the result (Sec. 11) |
| E6 | Robustness | — | — | Distance / SNR / day-shift invariance |
| E7 | **Phase 2** count estimation | E1/E3 setups | — | Auto-count (staged, after E1–E6) |

---

## 11. Ablations (E5)

- Single-branch (time only) vs single-branch (spectral only) vs dual-branch.
- Cross-attention fusion vs simple concat.
- SupCon vs triplet vs the distance-invariance-augmented loss.
- GroupNorm vs BatchNorm (you have evidence BatchNorm collapses — show it).
- Embedding dim sweep (128 vs alternatives), window size.

---

## 12. Timeline (sequence, not dates — venue/dates come after results)

1. **Data ingest + verify** — WiSig + drone set onto mars-4090; confirm formats,
   device counts, licenses. (Gate: don't train until splits are locked.)
2. **Reproduce baselines** — OpenMax, Proto+EVT, threshold on ORACLE first (known-good data).
3. **E1 + E2** — within-domain discovery, proposed method. → **First results checkpoint.**
4. **Show prof** the E1/E2 numbers + plan. Adjust before investing in transfer.
5. **E3** — cross-domain transfer (the headline).
6. **E4** — baselines across E1–E3.
7. **E5 + E6** — ablations and robustness.
8. **E7** — phase-2 counting (if time).
9. **Write-up** — draft → figures → internal review → arXiv preprint → pick CFP/venue.

Each step: one change, diagnose, confirm, proceed (the V1→V4 discipline). CONTEXT.md
per phase as usual.

---

## 13. Risks and mitigations

- **Eval devices too similar (the ORACLE merge problem).** Mitigation: pick a drone
  set with genuine hardware diversity; report inter-device similarity up front so a
  reviewer sees the data isn't degenerate.
- **Drone dataset access.** Feb-2026 set license unverified — UAVSig (free) is the fallback.
- **Cross-domain transfer underperforms.** Acceptable as a finding if framed honestly
  (where transfer holds / breaks is itself a result); keep within-domain as the floor.
- **Scope creep toward localization.** Hard line: out of scope, future work only.
- **Timeline vs peer review.** Internship deliverable = submitted + arXiv preprint,
  not "accepted" (review runs months).

---

## 14. Deliverables

- Paper draft (4–6 pp conference format) + arXiv preprint.
- Reproducible code + locked split definitions.
- Figures: architecture, t-SNE, robustness curves, results tables, transfer result.
- This plan + the literature survey as the prof-facing package.

---

## 15. Open decisions to confirm

- [ ] Drone dataset: Feb-2026 multi-unit set vs UAVSig (depends on access) — verify.
- [ ] Transfer source: WiSig alone, or WiSig + LoRa for a 2-protocol claim?
- [ ] Prof's baseline expectations — confirm OpenMax + Proto/EVT + threshold is enough.
- [ ] Target venue shortlist (after first results): MILCOM / GLOBECOM / ICASSP / DySPAN / workshop.
