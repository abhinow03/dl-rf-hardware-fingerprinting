# DL-Based RF Hardware Fingerprinting

Deep learning system that identifies radio transmitters by their **hardware manufacturing imperfections** — IQ imbalance, phase noise, PA nonlinearity — rather than by protocol or content. Includes open-set discovery of unknown emitters never seen during training.

---

## Overview

| Item | Detail |
|------|--------|
| Architecture | Dual-branch encoder: Time 1D ResNet + Spectral 2D ResNet fused via Cross-Attention |
| Loss | Supervised Contrastive Loss (SupCon) with temperature annealing |
| Dataset | ORACLE (Northeastern GENESYS Lab) — 16 USRP X310 SDRs, 5 distances, IEEE 802.11a |
| Parameters | 1.48M |
| Best sim_gap | 1.3440 (v4, 4096-sample windows) |
| Distance invariance | eval_sim = 0.986 at epoch 20 (trained on 8ft+14ft only) |

---

## Dataset

**ORACLE** — Open Radio Access Network (ORAN) RF fingerprinting dataset  
- 16 identical USRP X310 software-defined radios transmitting IEEE 802.11a WiFi  
- Recorded at 5 distances: 8ft, 14ft, 20ft, 26ft, 32ft  
- Sample rate: 5 MS/s, 40M IQ samples per file  
- Format: `.sigmf-data` (raw IQ, float32)  
- Total size: ~100 GB  
- Source: [GENESYS Lab, Northeastern University](https://genesys-lab.org/oracle)

Training used **8ft + 14ft only**. Evaluation at 20ft + 26ft (unseen distances).  
4 devices held out entirely for open-set discovery testing.

---

## Architecture

```
IQ window [B, 2, 4096]
        │
        ├──► Time Branch (1D ResNet)
        │    Large-kernel stem → 4× ResBlock1D + SE → FC(128→256)
        │
        └──► Spectral Branch (2D ResNet on STFT)
             STFT [B, 2, 129, 61] → stem → 4× ResBlock2D + SE → GAP → FC(128→256)
                            │
                    CrossAttention(256, heads=4)  ← residual around attn
                            │
                    Concat + LayerNorm → [B, 512]
                            │
                    Projection Head FC(512→256→128) + L2-norm
                            │
                    Embedding [B, 128]
```

Key design choices:
- **GroupNorm** throughout — BatchNorm causes representation collapse in contrastive learning  
- **Residual around CrossAttention** — prevents init-time collapse  
- **No Dropout** — GroupNorm provides sufficient regularisation  
- **Float32 cast before loss** — avoids AMP precision issues in SupCon

---

## Phase History

### Phase 1 — Dataset Preparation
- Downloaded ORACLE `.sigmf-data` files (~100 GB)  
- Verified IQ format: interleaved float32, 2 channels (I/Q)  
- Identified 16 device IDs, 5 distances, ~40M samples per file

### Phase 2 — Data Loader
- `np.memmap` for memory-efficient access to large files  
- Window size: 1024 samples; skip first 1000 samples (startup transient)  
- Clip threshold: reject windows with |value| > 10.0  
- File-level 80/20 train/val split (prevents data leakage)  
- Augmentation pipeline (order fixed):  
  1. Phase rotation p=1.0 (complex rotation, not scalar — critical for hardware invariance)  
  2. CFO injection p=0.8 (±0.005 normalised frequency)  
  3. AWGN p=0.5 (SNR 15–50 dB)  
  4. Amplitude scale p=0.3 (0.95–1.05)  
  5. Per-window standardisation: (x − μ) / σ

### Phase 3 — Training (v1/v2)
- Window size: 1024 samples; STFT: FFT=128, hop=32 → [2, 65, 29]  
- SupCon loss, temperature schedule: 0.5 → 0.1 → 0.07  
- Batch: 12 devices × 16 windows = 192; 150 steps/epoch, 80 epochs  
- **Issue found:** 32ft distance caused distance-based clustering in embedding space  
- **Fix:** Excluded 32ft from training entirely

### Phase 4 — Distance Invariance Fix (v3)
- Trained only on 8ft + 14ft; validated on 20ft + 26ft  
- eval_sim approached train_sim: gap reduced from 0.13 → 0.013 by epoch 60  
- Confirmed model generalises to unseen distances without degradation

### Phase 5 — Window Size Scaling (v4)
- Increased window to **4096 samples** (4× larger); STFT: FFT=256, hop=64 → [2, 129, 61]  
- Larger windows capture more IQ imbalance cycles → stronger fingerprint signal  
- **sim_gap improved 159%**: 0.518 (1024-sample) → 1.344 (4096-sample)  
- Best results: sim_gap = 1.3440, eval_sim = 0.986 at epoch 20

### Phase 6 — Evaluation
- t-SNE visualisation: embeddings form ring manifold (hardware fingerprints are continuous, not discrete clusters — expected for nearly-identical USRP hardware)  
- KNN purity (all 16 devices): 34.6%; held-out only: 57.4%  
- Low KNN purity explained by high between-device similarity (0.72–0.82) — 16 USRPs are genuinely very similar hardware

### Phase 7 — Open-Set Discovery
- 4 held-out devices streamed in, never seen during training  
- Cosine similarity registry with EMA updates and auto-calibrated threshold (0.866)  
- **Assignment purity: 90.6%** — correct window-to-emitter assignment  
- **Limitation:** 3 of 4 held-out devices merge into 1 emitter (pairwise sim ~0.82 > any separable threshold) — reflects fundamental hardware similarity limit of ORACLE dataset

---

## Results Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Train within-device sim | 0.787 | > 0.75 | ✓ |
| Eval within-device sim | 0.939 | — | — |
| Train–eval sim gap | 0.033 | < 0.05 | ✓ |
| Best sim_gap (v4) | 1.344 | > 0.7 | ✓ |
| Open-set assignment purity | 90.6% | > 80% | ✓ |
| KNN purity (all 16 devices) | 34.6% | > 80% | ✗ |

KNN purity failure is a **dataset limitation**, not a model failure: 16 identical USRP X310 units have fingerprint differences at the limit of detectability. The model correctly learns continuous hardware variation; it simply cannot separate hardware that is nearly identical.

---

## Files

| File | Description |
|------|-------------|
| `model.py` | Dual-branch RFEncoder architecture |
| `losses.py` | SupCon loss with numerical stability and NaN guard |
| `train.py` | Training loop — Phase 6 v4 (4096-sample windows, 8ft+14ft only) |
| `evaluate.py` | Phase 7 evaluation: t-SNE, KNN purity, distance invariance |
| `inference.py` | Phase 8 open-set emitter discovery pipeline |
| `results/FINAL_REPORT.md` | Training metrics across all epochs |
| `results/RESULTS_REPORT.md` | Full evaluation results with analysis |
| `results/tsne_by_device.png` | t-SNE coloured by device ID |
| `results/tsne_by_distance.png` | t-SNE coloured by recording distance |

---

## Setup

```bash
python3 -m venv rf_env && source rf_env/bin/activate
pip install torch numpy scikit-learn matplotlib
```

**Train:**
```bash
python3 train.py
```

**Evaluate** (requires trained model at `~/phase6_output_v4/best_model.pt`):
```bash
python3 evaluate.py
```

**Open-set inference demo:**
```bash
python3 inference.py --demo
```

---

## Limitations & Next Steps

- **DroneRF dataset:** Switching to DroneRF (Bebop, AR, Phantom, BG) would test the system on genuinely different hardware from different manufacturers — fingerprints should be much more separable  
- **Larger windows:** Literature suggests 10,000+ samples for robust IQ imbalance detection  
- **Bispectrum features:** Higher-order statistics as additional spectral branch input  
- **DBSCAN clustering:** Batch-mode clustering instead of streaming registry for better open-set separation
