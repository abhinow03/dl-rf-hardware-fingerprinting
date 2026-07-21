# Neighbor Analysis & Novelty Defense

Deep-search pass to confirm how much novelty survives. Compiled June 2026.
Read alongside LITERATURE_SURVEY.md and PAPER_PLAN.md.

---

## 0. Verdict (honest)

Method-level novelty is largely **taken**. Open-world/discovery RFFI via
contrastive metric learning + clustering is published (OpenRFI, AAAI 2025);
open-set + clustering + auto-count for UAVs is published (KTH, 2026). You
**cannot** claim: "first open-world RFFI," "metric learning for open-set SEI,"
"clustering unknowns," or "unknown-count estimation" as novel.

What survives is a **narrow applied contribution**: the unoccupied intersection of
(1) individual same-model device discrimination, (2) open-world discovery, and
(3) cross-domain transfer — demonstrated on drones. No single paper occupies all
three. This is publishable as a new-setting + empirical-study paper at applied
venues; it is **not** a method-first paper unless a real method delta is added
(see §4).

**Key structural finding (the strongest defense).** After a second deep pass, a
clean split appears in the literature: every *discovery* paper (OpenRFI, KTH) is
**single-domain**, and every *transfer / cross-domain / cross-receiver* paper
(DRIFT, source-free adaptation, Fourier+MMD, DeepCRF) is **closed-set** (a fixed,
known set of transmitters generalized across receivers/days/channels). Nobody does
open-world **discovery under domain shift**. That intersection is genuinely empty,
and the cross-receiver robustness methods become techniques you *borrow* and
*baseline against*, not competitors. This is the sentence to build the paper on.

---

## 1. Killer neighbors (must cite AND baseline against)

### OpenRFI — Open-world RFFI via Augmented Semi-supervised Learning (AAAI 2025)
Han et al., Beihang + Northeastern. Code: github.com/ShuaS2020/OpenRFI
- Claims **first** open-set → open-world extension for RFFI: discovers and further
  classifies unknown emitters, not just rejects them.
- Method: Roinformer backbone (Informer encoder + RoPE) → SimCLR self-supervised
  pretraining (AWGN + permutation augmentations) → instance-level similarity loss
  + prototype-group local entropy regularization (built on ORCA / OpenNCD / LegoGCD).
- Evaluated on temporal RF signal datasets (ZigBee / WiFi-type), device-level.
- Exact setup: a 32-class RFF dataset, 10 classes selected (5 known / 5 novel),
  only 10% of known samples labeled (semi-supervised, OpenNCD-style).
- **Limitations = your openings:** single-domain (no cross-domain transfer); total
  class count assumed known (no count estimation); not drones; not same-model-unit
  stressed. This is your single most important baseline.

### KTH — Incremental Open-Set Classification of Unknown UAVs (arXiv 2603.24268, 2026)
Liu et al., KTH + Aalborg.
- OSR (Mahalanobis + 3-sigma) → cluster rejected samples (K-Means/GMM) with
  **auto-count** (elbow + composite validity) → incremental learning with replay.
- Dual-branch CNN + transformer embedding; loss = center + separation + CE.
- Dataset: 24 UAV **classes/types** (18 known, 6 unknown) — DroneRFa-style.
- **Delta vs you:** type/model-level (semantic), not individual same-model units;
  single-domain; CE-anchored embedding (not pure metric). Your same-model-unit +
  transfer + pure-contrastive framing differs.

---

## 2. Tier-2 neighbors (cite; context, not direct scoops)

Open-set RFFI (rejection-oriented — they detect unknowns, don't discover):
- Multi-Task Prototype Learning (MTPL), Sensors 2025 — classify + reconstruct +
  prototype-cluster; EVT rejection.
- Improved Prototype Learning, Wang et al. (arXiv 2306.13895) — consistency reg +
  label smoothing; open-set rejection.
- Prototypical Networks + EVT, Appl. Sci. 2023 — ZigBee + 10 USRP X310; OpenMax
  comparison. (Note: X310 setup ~ ORACLE; useful baseline precedent.)
- FSST + Supervised Contrastive Learning open classifier, Huang et al. 2023/2024 —
  contrastive fingerprints + rogue detection.
- Robust open-set SEI with class-irrelevant features, Zhou et al., TIFS 2025.

Generic framework / generalization framing:
- Generic ML framework for RFF (SEI/EDA/RFEC), Hiles et al., arXiv 2510.09775 (2025)
  — explicitly frames the open-set obstacle as "when has the comparator seen enough
  emitters to generalize to unknown ones." Cite to motivate your transfer study.

Cross-domain / cross-receiver SEI (closed-set — transfer exists, but NOT for discovery).
Read for *solutions to borrow*, and use as robustness baselines:
- Cross-Domain Generalization for SEI via Fourier Phase + MMD, IEEE 2025 — solution:
  intra-domain invariants from Fourier phase + knowledge distillation; inter-domain
  invariants via MMD feature alignment. Closed-set.
- DRIFT: Cross-Receiver Generalization via Feature Disentanglement + Adversarial
  Training (arXiv 2510.09405) — solution: disentangle Tx-identity from receiver/channel
  features, adversarial (DANN-style) alignment. ManySig. Closed-set. Best robustness baseline.
- Source-Free Cross-Receiver Adaptation (arXiv 2512.16648, 2025) — adapts to a target
  receiver without source data. ManySig + HackRF. Closed-set.
- DeepCRF (Kong et al. 2024) — solution: model-inspired augmentation + supervised
  contrastive loss + decision fusion; ~99.5% on UNSEEN channels (CSI-based). Closed-set.
- SA2SEI: few-shot SEI via self-supervised + adversarial augmentation — few-shot
  transfer to target emitters.
- Federated RFFI via unsupervised contrastive learning, Shen et al., TIFS 2024.
- Channel-Robust Receiver-Independent RFFI (arXiv 2512.12070, 2025).
Takeaway: techniques (disentanglement, MMD, adversarial alignment, contrastive
augmentation) are mature for *closed-set* transfer — borrow them to make your
*discovery* robust to receiver/day shift. That recombination is part of your delta.

GCD method toolbox (vision; borrow for count-under-shift, §4):
- ORCA, OpenNCD, LegoGCD (what OpenRFI builds on); CiPR / Component-Adaptive
  Clustering / OpenGCD — estimate the unknown class count without assuming K.

Drone individual / same-model fingerprinting (closed-set — your data analogs):
- GENESYS Hovering UAVs — 7 identical DJI M100 (your device-level spine), closed-set.
- UAVSig (MILCOM 2024) — same-model drone detect + localize + fingerprint, closed-set.
- HSCP (arXiv 2512.08983) — VERIFIED a model *pruning* paper (spectral clustering of
  layers/channels for edge compression); uses UAV-M100 only as a closed-set
  classification benchmark. NOT a discovery competitor. Confirms M100 fingerprints
  are cleanly separable closed-set (reassuring precondition).

---

## 3. The white space (state exactly this)

No paper occupies the conjunction:
- **individual same-model units** (not drone *types*), AND
- **open-world discovery** (discover + group unknown emitters, not reject), AND
- **cross-domain transfer** (train one domain, discover unknown emitters in another),
- demonstrated on **drones**.

OpenRFI has open-world + device-level but is single-domain, known-count, non-drone.
KTH has drone + discovery + auto-count but is type-level, single-domain.
Cross-domain SEI has transfer but is closed-set.

Your contribution sentence: *"We study open-world discovery of individual emitters
under cross-domain transfer, at the same-model-unit level, and demonstrate it on
drone hardware fingerprints — a setting prior open-world RFFI (single-domain,
non-drone) and prior open-set drone work (type-level) do not address."*

---

## 4. To strengthen beyond "applied only" (optional, raises venue ceiling)

Pick at least one genuine method delta:
- **Unknown-count estimation under domain shift.** OpenRFI assumes known count; KTH
  estimates it in-domain. Robust count estimation that survives cross-domain transfer
  is an open, real method contribution.
- **Distance/channel-invariance loss that measurably improves transfer.** You already
  have the invariance machinery; show it helps cross-domain discovery, not just eval.
- **Pure-metric vs CE-anchored ablation.** Demonstrate SupCon embeddings transfer
  open-world better than KTH's CE-anchored embeddings. Turns a framing claim into a result.

---

## 5. Required baselines (non-negotiable for credibility)

- **OpenRFI** (adapt their open-world pipeline to your data) — primary baseline.
- **KTH** OSR+cluster+auto-count — secondary baseline.
- OpenMax, Prototypical+EVT, similarity-threshold — classical open-set floor.
Report all on the same device-disjoint, same-model and cross-domain splits.

---

## 6. Venue calibration

- Realistic targets given applied novelty: MILCOM, GLOBECOM workshops, IEEE DySPAN,
  IEEE CCNC, IEEE ICMLCN, or a counter-UAS / wireless-security workshop.
- A method delta from §4 + strong cross-domain results could justify a stronger
  venue or a journal (IEEE TIFS / Open J. Comms / IoT-J).
- Internship deliverable = submitted + arXiv preprint, not "accepted."

---

## 6b. Datasets — refined (keeping M100 + DRFF-R2)

Roles matter more than size. You need: many source identities (to learn the metric),
a cross-domain/receiver axis (for the transfer claim), and same-model drone units
(for the application). Mapped:

- **M100 (4.5 GB) + DRFF-R2 hover subset** — KEEP. Same-model drone units; the
  device-level + application core. M100 is SigMF/X310 (ORACLE-like pipeline).
- **WiSig (free, subsets of limited size)** — the "even better" addition for the
  *discovery* axis: up to 174 transmitters (the ManyTx subset gives many identities),
  so the metric is forced to generalize, not memorize. Source domain for transfer.
- **WiSig-ManySig (~2.2 GB, on HuggingFace)** — the "even better" addition for the
  *transfer* axis: 6 Tx × 12 Rx × 4 days, 256-sample 802.11a/g packets. It is the
  *standard* cross-receiver / cross-day benchmark, so using it lets you compare
  directly against DRIFT / source-free baselines — instant credibility for the
  cross-domain claim, tiny on disk. Caveat: only 6 Tx, so it is for the
  receiver/day *robustness* demonstration, not many-unknown discovery.
- **ORACLE** — KEEP as second within-domain benchmark (already works).
- Optional: HackRF multi-device set (cross-device source); LoRa-RFFI (60 same-model,
  clean second protocol).

Recommended spine: learn metric on WiSig (ManyTx) → discovery on M100 + DRFF-R2
held-out units → cross-receiver/day transfer demonstrated on WiSig-ManySig (with
DRIFT/source-free as baselines) and on DRFF-R2's multi-USRP hover data → ORACLE as
anchor. This covers all three novelty axes with mostly small, free downloads.

---

## 7. Reading still worth doing

- OpenRFI full method + code — replicate as baseline; confirm exact datasets and that
  it assumes known count.
- HSCP (2512.08983) — uses your 7-M100 data; confirm closed- vs open-set to avoid a
  same-data scoop.
- Hiles framework — borrow its EDA generalization framing for your motivation.
