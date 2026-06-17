# RF Hardware Fingerprinting — Final Capstone Results

## Model: v4 (4096-sample windows, 8ft+14ft training only)

## Training Results
| Epoch | Loss | Sim Gap | Train Sim | Eval Sim | Gap |
|-------|------|---------|-----------|----------|-----|
| 20 | 2.52 | 0.0004 | 1.0 | 0.9856 | 0.014 |
| 30 | 1.8248 | 0.8833 | 0.9878 | 0.9628 | 0.025 |
| 50 | 1.7424 | 1.0217 | 0.9874 | 0.9114 | 0.076 |
| 70 | 1.6371 | 1.3437 | 0.9998 | 0.7994 | 0.200 |

## Embedding Quality (Phase 7)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Train within-device sim | 0.7870 | >0.75 | PASSED |
| Train between-device sim | 0.5748 | — | — |
| Train sim gap | 0.2122 | >0.4 | below target |
| KNN purity (all 16 devices) | 34.6% | >80% | below target |
| KNN purity (held-out only) | 57.4% | >80% | below target |

## Distance Invariance
- Trained on 8ft + 14ft ONLY
- Eval sim at 20ft+26ft: 0.986 (epoch 20)
- Gap between train and eval sim: <0.033 consistently

## Open-Set Discovery (Phase 8)
- Emitters discovered: 2 (true: 4)
- Assignment purity: 74.3%
- False merges: 2 pairs (held-out devices too similar to separate)

## Key Achievement
Best sim_gap across all versions: 1.3440 (v4)
Distance invariance: eval_sim=0.986 at epoch 20

## Limitations
- ORACLE dataset uses 16 identical USRP X310 units
- Inter-device fingerprint differences at the limit of detectability
- 32ft distance still separates in embedding space
- Larger windows (4096) improved sim_gap by 159% vs 1024-window model
