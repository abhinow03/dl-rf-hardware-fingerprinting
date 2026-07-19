# DISK RECLAIM — Inventory + Manifest (NO DELETIONS this session)

*Generated 2026-07-13 on mars-4090. Purpose: identify reclaimable space to clear the project's
100 GB-free floor before EXT-PROTOCOLS Phase 0 downloads. **This session performed zero deletions.**
Deletion is a separate, explicitly-approved session. Every SAFE line is cross-checked against
FINDINGS.md / ledgers; REVIEW items require owner sign-off and are NEVER auto-safe.*

## df baseline
```
/dev/nvme1n1p5  488G  440G  23G  96%  /home     ← 23 GB free; floor target = 100 GB free (need +77 GB)
```

## ⚠️ Hard reality check (read first)
- **This user's entire home is only ~104 GB.** The volume shows 440 GB used because **~336 GB lives
  outside this home** (other users/dirs on the shared `/home`) — not reclaimable from in-scope trees.
- **The ≥150 GB reclaim target is not achievable here.** Even deleting everything except PROTECTED
  assets + the venv tops out at **~92 GB** reclaimable.
- **Clearing the 100 GB floor (+77 GB) is *just barely* possible, and only by deleting the 39 GB
  ORACLE raw (a REVIEW decision) plus caches plus at least one non-RF project.** SAFE-only frees
  ~28 GB (→ ~51 GB free) — short of the floor, **but already far more than the actual D2 BLE
  download needs (a few GB).** If the 100 GB floor is a soft safety margin rather than a hard byte
  requirement, SAFE-only reclaim is sufficient to proceed and the ORACLE-raw decision can be deferred.

## PROTECTED — verified this session, never touch
| asset | sha256 | status |
|---|---|---|
| `summer_work/runs/wisig_supcon_fft64/retrain_best/best_model.pt` | `03898f49…` | ✓ verified (==EVAL_PROTOCOL §6) |
| `summer_work/runs/drone_native/seed2024/best.pt` | `2ef7fc25…` | ✓ verified |
| `summer_work/runs/drone_native/seed1234/best.pt` | `eb310d88…` | ✓ verified |
| `CAPSTONE/phase6_output_v4/best_model.pt` (ORACLE-V4) | `5d9ccfa4…` | ✓ verified |
| `summer_work/runs/wisig_supcon_fft64/splits/split_manytx.json` | `d16c48fa…` | ✓ verified |
| `results_gen/splits_lopo.json` | `8743448e…` | ✓ verified |

Also PROTECTED (not re-hashed, keep): `results_gen/oracle_cache_*.npz` + `oracle_min_cache.npz`
(locked caches), `gen_rff/` (committed), `FINDINGS.md`, `DATA_AUDIT.md`, all CONTEXT/README docs,
and raw sources still in use → `phase3_dataset/ManyTx.pkl` (4.18 GB, WiSig ManyTx source),
`phase3_dataset/ManySig_zf.pkl` (1.57 GB, DRFF-R2 source).

## MANIFEST — ranked, with cumulative reclaim

### SAFE (regenerable or verified-duplicate; not cited by any ledger/FINDINGS)
| # | path | size | cum GB | evidence |
|---|---|---|---|---|
| S1 | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct` | 15 GB | 15.0 | LLM model cache, **non-RF**, re-downloadable from HF; not referenced anywhere in project |
| S2 | `~/Downloads/ManyTx.pkl.zip` | 2.5 GB | 17.5 | byte-identical (`a8fc3e35`) to phase3 copy; extracted `ManyTx.pkl` present → redundant |
| S3 | `~/CAPSTONE/phase3_dataset/ManyTx.pkl.zip` | 2.5 GB | 20.0 | archive of extant `ManyTx.pkl` (4.18 GB present) → redundant |
| S4 | `~/.cache/pip` | 4.0 GB | 24.0 | pip wheel cache, always regenerable |
| S5 | `~/.cache/torch` | 2.7 GB | 26.7 | torch hub cache, regenerable |
| S6 | `~/CAPSTONE/phase3_dataset/ManySig_zf.zip` | 1.5 GB | 28.2 | archive of extant `ManySig_zf.pkl` (1.57 GB present) → redundant |
| S7 | all `__pycache__/` (home-wide) | 0.5 GB | 28.7 | bytecode, regenerated on import |
| S8 | `summer_work/runs/drff_adapt/{R1,R2,R3}` | 0.13 GB | 28.8 | superseded DRFF adaptation runs; demo uses `drone_native/seed2024`, not these; not cited |
| S9 | `runs_gen/lopo_drff/{A1..A5,smoke_A1..A4}` | 0.06 GB | 28.9 | ablation/smoke checkpoints; results captured in `results_gen/phase2b_part2_ablation.json`; selected run = `seed1234` (kept) |

**SAFE subtotal ≈ 28.9 GB → projected free ≈ 52 GB** (still < 100 GB floor; ≫ enough for the D2 BLE download).

### REVIEW (plausibly reclaimable — owner decision required, never auto-safe)
| # | path | size | cum GB | why REVIEW (not SAFE) |
|---|---|---|---|---|
| R1 | `~/Desktop/neu_m044q5210./KRI-16Devices-RawData/{8,14,20,26}ft` | 39 GB | 39.0 | **ORACLE raw (.sigmf).** Only the 4 *used* distances remain (extras already deleted). Reclaimable **only if** the ORACLE line is retired (FINDINGS F6: ORACLE poisons/unlearnable; trained model `5d9ccfa4` + `oracle_cache_*.npz` already exist) **AND** the locked phase-1 oracle-closed-set never needs to re-derive from raw. **Biggest single lever; owner call.** |
| R2 | `~/splatfusion` | 11 GB | 50.0 | **non-RF project** (gaussian-splatting); out of RF scope — owner's other work |
| R3 | `~/aadhya` | 6.4 GB | 56.4 | **non-RF project** (uav-llm); out of scope — owner's other work |
| R4 | `~/Desktop/processed/drff_r2` | 3.4 GB | 59.8 | processed DRFF-R2 features — **may be the in-use "DRFF-R2 npz"**; verify against loaders before touching |
| R5 | `~/Downloads` (excl. S2 zip) | ~3.1 GB | 62.9 | misc downloads, contents unaudited |
| R6 | `~/Desktop/processed/m100` | 0.44 GB | 63.3 | only M100 artifact present; protected list names "M100 sigmf" — keep unless confirmed derived |
| R7 | `~/.cache/huggingface/datasets/json` | 0.15 GB | 63.5 | possibly RF-adjacent cached dataset |

**REVIEW subtotal ≈ 63.5 GB.**

### Not reclaimable / keep
- `~/rf_env` (5.6 GB) — active venv, in use.
- `~/CAPSTONE/phase3_dataset/*.pkl` (5.75 GB) — PROTECTED raw sources (WiSig ManyTx + DRFF-R2).
- system journal 129.6 MB — system-managed, out of scope.
- No large stray logs, no core dumps found. Backups (`*_bak/*_old`): 4 files, 256 KB total (trivial; REVIEW).

## Projected free space
| scenario | freed | projected free | clears 100 GB floor? |
|---|---|---|---|
| current | — | 23 GB | — |
| **SAFE-only** | +28.9 GB | **~52 GB** | ✗ (but ample for the few-GB D2 BLE download) |
| SAFE + ORACLE raw (R1) | +67.9 GB | **~91 GB** | ✗ (short by ~9 GB) |
| SAFE + all REVIEW | +92.4 GB | **~115 GB** | ✓ (requires deleting ORACLE raw + both non-RF projects) |

## Recommendation
1. **If the 100 GB floor is a hard rule:** the only way to reach it is SAFE + ORACLE-raw (R1) + a
   couple of REVIEW items (R2/R4/R5). This forces the **ORACLE-retirement decision** — confirm the
   locked phase-1 oracle work will never re-derive from raw before deleting R1.
2. **If the floor is a safety margin:** run **SAFE-only** (~52 GB free) — that comfortably fits the
   D2 XIAO BLE download (a few GB) with room to spare, and defers the ORACLE decision entirely.
3. The ≥150 GB target cannot be met in this home; if that much is genuinely required, it must come
   from elsewhere on the volume (outside this user's ~104 GB home).

*No `rm` was executed. This manifest is the input to a separate, approved deletion session.*
