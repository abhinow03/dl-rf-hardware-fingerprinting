# RF Hardware Fingerprinting & Open-World Emitter Discovery

Deep-learning **RF hardware fingerprinting**: identifying and discovering physical radio
transmitters from the tiny, hardware-specific imperfections baked into their I/Q signal
(carrier-frequency offset, I/Q imbalance, phase-noise, transient shape) — *not* their protocol
payload. The system is **open-world**: it does not just classify enrolled radios, it **discovers
and clusters emitters it has never seen** by embedding each signal and grouping by similarity.

One `RFEncoder` backbone (~**1.5M params**) is trained across four RF domains (WiFi, drone
OcuSync, ORACLE USRP, BLE) and driven by a **protocol-router demo** that turns raw I/Q streams
into named device fingerprints end-to-end.

```
 raw I/Q  ─►  ROUTER  ─►  TIER ENCODER  ─►  ACCUMULATOR  ─►  BASE STATION  ─►  device IDs
 (antenna)   which       RFEncoder →        pool N=120       cluster per       d1.ble
             protocol?    embedding          windows, L2      protocol group    d2.drone …
```

---

## Directory tree

```
final-project/
├── README.md               ← you are here
├── pyproject.toml          installable package (pip install -e .)
├── requirements.txt        pinned runtime deps
├── src/rffp/               ← ALL importable code (the `rffp` package)
│   ├── config.py           ★ single place for every dataset / checkpoint / output path (env-overridable)
│   ├── models/             encoder (dual-branch RFEncoder), losses (SupCon), gen_encoder
│   ├── data/               per-domain loaders, domain registry, STFT front-end, WiSig reader
│   ├── physics/            classical hand-crafted RF features (LPC residual, spectral stats)
│   ├── training/           train_oracle · train_wisig{,_hardneg,_triplet} · train_lopo · train_dann · ablation · train_b3_ble
│   ├── evaluation/         evaluate · inference · scoring (locked scorer) · bench/ (LOPO harness) · verify/ (gates)
│   └── discovery/          ← open-world discovery
│       ├── router/         the shippable protocol-router demo (run_demo, router, tiers, base_station …)
│       ├── wisig/          Phase-2 WiSig discovery probes (burst, geometry, leakage controls)
│       └── ext_ble/        BLE same-model study (batteries A / B1 / B3, feature extraction)
├── docs/                   all findings & protocol ledgers (see docs index below) + results/ figures
└── paper/                  IEEE conference draft (main.tex)
```

**Module count:** `models` 4 · `data` 5 · `physics` 2 · `training` 10 · `evaluation` 10 ·
`discovery` 40. Every module is import-checked against the installed package.

---

## Quickstart & usage

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                    # installs rffp + deps from pyproject.toml
#   (or: pip install -r requirements.txt)

# 2. point at YOUR data & checkpoints (only step that needs local paths — see src/rffp/config.py)
export RFFP_RUNS=/path/to/checkpoints     # frozen encoder .pt files
export RFFP_WISIG=/path/to/ManyTx.pkl     # WiSig dataset
export RFFP_DRFF=/path/to/drff_r2         # DRFF drone captures
export RFFP_ORACLE=/path/to/KRI-16Devices-RawData
export RFFP_BLE=/path/to/ble_xiao

# 3. run the end-to-end open-world discovery demo
python3 -m rffp.discovery.router.run_demo      # or the console script:  rffp-demo
```

| Task | Command |
|---|---|
| **Open-world router demo** (headline) | `python3 -m rffp.discovery.router.run_demo` |
| Train ORACLE closed-set encoder | `python3 -m rffp.training.train_oracle` |
| Train WiSig metric encoder (SupCon) | `python3 -m rffp.training.train_wisig` |
| Cross-domain LOPO benchmark | `python3 -m rffp.evaluation.bench.lopo` |
| ORACLE eval (t-SNE + purity) | `python3 -m rffp.evaluation.evaluate` |
| Open-set inference demo | `python3 -m rffp.evaluation.inference --demo` |
| Verification gates (V1–V5) | `python3 -m rffp.evaluation.verify.run_verify` |

> **Every path is resolved in `src/rffp/config.py`** and overridable by the `RFFP_*` env vars
> above — no code edits needed to relocate data. Datasets and `.pt` checkpoints live off-repo
> (gitignored); point the env vars at your copies. Edge-deployable **ONNX** encoders are exported
> separately (see the `FINAL_MODEL` bundle / `docs/`).

---

## Pipeline & architecture

### The encoder — dual-branch `RFEncoder` (`rffp/models/encoder.py`)

```
 raw I/Q [B,2,L] ──┬─► TimeBranch     1-D CNN  (wide-kernel stem + 4 residual blocks)      ─► 256-D
                   └─► SpectralBranch 2-D CNN  over STFT magnitude [B,2,F,T]               ─► 256-D
                                    │
              cross-attention (query = time, key/value = spectral) + residual
                                    │
                        concat → LayerNorm(512)
                          ├─► get_encoder_output()  →  512-D  (deep tiers: drone, WiFi)
                          └─► projection head → L2  →  128-D  (metric head: BLE, SupCon target)
```

| Design choice | Value / rationale |
|---|---|
| **Two streams** | time domain catches burst-transient shape; frequency domain catches CFO / I-Q imbalance / phase noise |
| **Normalization** | **GroupNorm** (stable with tiny batches) + **Squeeze-and-Excitation** channel attention |
| **Fusion** | `nn.MultiheadAttention(256, 4)` cross-attention, residual-added (prevents init collapse) |
| **Loss** | **Supervised Contrastive (SupCon)** with annealed temperature → same-device pulls together |
| **Output** | 512-D fusion embedding, or 128-D L2-normalized metric embedding |
| **Window sizes** | BLE **1850 @ 6 MS/s** · WiSig **256 @ 25 MS/s** · drone **4096 @ 50 MS/s** · ORACLE **4096 @ 5 MS/s** |

### The discovery system (`rffp/discovery/router/`)

1. **Router** — rate-aware spectral features → LogisticRegression → `{wifi, drone, ble, unknown}`, with a per-class **Mahalanobis novelty gate** (classical, not a neural net).
2. **Tier encoder** — routes to the matching frozen `RFEncoder` checkpoint per protocol.
3. **Accumulator** — pools **N\*=120** windows per track (mean + L2) into one fingerprint.
4. **Base station** — clusters fingerprints **within each protocol group** via **eigengap-K** + spectral partition; assigns namespaced IDs (`d1.ble`, `d2.drone`).
5. **Scoring** — the **locked scorer** (`rffp/evaluation/scoring.py`): HDBSCAN noise points count as singleton clusters, so a method can't inflate ARI by dumping hard emitters into "noise".

---

## Results summary

**Phase 1 — ORACLE closed-set** (16 *identical* USRP X310 units — fingerprinting at the limit of detectability)
- Within-device similarity **0.787** (target >0.75 ✓); held-out KNN purity **57.4%**; open-set discovered **2 of 4** held-out devices. Distance-invariant (train 8/14 ft, eval sim **0.986** at 20–26 ft). *Identical-hardware units are near the detectability floor — motivated the pivot to open-world discovery.*

**Phase 2 — WiSig & the N-dependent wall** (`docs/FINDINGS.md`)
- The cross-domain gap is **N-dependent**: at N=10 methods sit at 0.14–0.28; at **N=120 they converge to 0.62–0.79** — heavy burst integration, not representation, drives most of the discovery gain.

  | method (mavicAir2 8-way) | N=10 | **N=120 (km / sp)** |
  |---|---|---|
  | native drone-trained (4096) | 0.276 | **0.792 / 0.835** |
  | frozen WiSig (OPT-B 256) | 0.137 | **0.729 / 0.777** |
  | classical-19 | 0.174 | **0.713 / 0.666** |
- In-domain WiSig (locked scorer): DEV **0.727**; one-touch held-out TEST (board-18) **0.219** (spent, never re-run).

**Phase 3 — BLE same-model study** (`docs/EXT_FINDINGS.md`)
- Native-from-scratch **B3** owns the pooled regime: pooled N=120 ARI **0.714** vs B1 0.284 / A 0.193; best receiver-transfer **0.759**. B3 is **calibration-indifferent** (collection η² ≈ 0) → ships raw.

**Phase 4/5 — Protocol-router demo** (`docs/ROUTER_DEMO.md`)
- Router accuracy **0.973**; grand-mixed scenario (drone + WiFi + BLE) routing **1.00**, per-group ARI **1.00**, **0 cross-protocol ID collisions**. Operating point: N\*=120, eigengap-K, assisted-K primary.

> Ledgers of record: `docs/EVAL_PROTOCOL.md` (splits, locked noise rule, board-18 TEST),
> `docs/FINDINGS.md` (F1–F8, DANN null), `docs/EXT_FINDINGS.md` (BLE EF1–EF8),
> `docs/ROUTER_DEMO.md` (system + gates). Paper draft in `paper/main.tex`.

---

## Documentation index (`docs/`)

| File | What |
|---|---|
| `EVAL_PROTOCOL.md` | Locked evaluation protocol: splits, HDBSCAN noise rule, burst construction, board-18 one-touch TEST |
| `FINDINGS.md` | Cross-domain findings F1–F8 + DANN null result |
| `EXT_FINDINGS.md` | BLE same-model study EF1–EF8, variance decomposition, calibration-indifference |
| `ROUTER_DEMO.md` | Protocol-router system design, evidence, gates, honest caveats |
| `PAPER_PLAN.md` · `CONTEXT.md` · `NEIGHBOR_ANALYSIS.md` | Paper framing, project context, related-work analysis |
| `results/` | ORACLE Phase-1 reports + t-SNE figures |

## Notes on scope & reproducibility
- **Datasets and checkpoints are external** (gitignored); set the `RFFP_*` env vars. Raw dataset paths are the *only* external requirement.
- The core pipeline (models, data, physics, evaluation, router demo, training entrypoints) is **import-verified** against the installed package. Full end-to-end runs additionally require the datasets/checkpoints and (for training) a CUDA GPU.
- A few deep benchmark/verify scripts (`bench/lopo`, `verify/run_verify`) can cross-check against an extended R&D research tree; where that tree is absent, those sub-checks skip gracefully with a clear message.
- Mocked demo geometry (rssi/aoa/tdoa) is synthetic and never used for clustering; localization is out of scope.
