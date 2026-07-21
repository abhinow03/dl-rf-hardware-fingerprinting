"""Central configuration: every external asset path in one overridable place.

All raw datasets, model checkpoints and output directories are resolved here so a
fresh clone never needs code edits — only environment variables (or dropping assets
at the documented defaults). This is the single file to touch when relocating data.

Environment overrides (all optional; effective default shown):

    RFFP_RUNS      frozen checkpoint root      <repo>/summer_work/runs
    RFFP_RESULTS   pipeline output root        <repo>/results_gen
    RFFP_WISIG     WiSig ManyTx.pkl            ~/CAPSTONE/phase3_dataset/ManyTx.pkl
    RFFP_DRFF      DRFF drone capture dir      ~/Desktop/processed/drff_r2
    RFFP_ORACLE    ORACLE raw-data dir         ~/Desktop/neu_m044q5210./KRI-16Devices-RawData
    RFFP_BLE       BLE (Xiao) capture dir      /home/docker/pw26_akp_01/ext_data/ble_xiao

Example (fresh machine):
    export RFFP_RUNS=/data/rffp/checkpoints
    export RFFP_WISIG=/data/rffp/ManyTx.pkl
"""
import os


def _abs(v: str) -> str:
    return os.path.abspath(os.path.expanduser(v))


def _env(name: str, default: str) -> str:
    return _abs(os.environ.get(name, default))


# repo root = two levels up from this file (src/rffp/config.py -> <repo>)
REPO_ROOT = _abs(os.path.join(os.path.dirname(__file__), "..", ".."))

# --- raw datasets ----------------------------------------------------------
WISIG_PKL   = _env("RFFP_WISIG",  "~/CAPSTONE/phase3_dataset/ManyTx.pkl")
DRFF_DIR    = _env("RFFP_DRFF",   "~/Desktop/processed/drff_r2")
ORACLE_DIR  = _env("RFFP_ORACLE", "~/Desktop/neu_m044q5210./KRI-16Devices-RawData")
BLE_DIR     = _env("RFFP_BLE",    "/home/docker/pw26_akp_01/ext_data/ble_xiao")

# --- frozen checkpoints (off-repo, gitignored) -----------------------------
RUNS_DIR      = _env("RFFP_RUNS", os.path.join(REPO_ROOT, "summer_work", "runs"))
DRONE_CKPT    = os.path.join(RUNS_DIR, "drone_native", "seed2024", "best.pt")
WIFI_CKPT     = os.path.join(RUNS_DIR, "wisig_supcon_fft64", "retrain_best", "best_model.pt")
BLE_CKPT      = os.path.join(RUNS_DIR, "ble_native_s2024", "best.pt")
WISIG_SPLITS  = os.path.join(RUNS_DIR, "wisig_supcon_fft64", "splits")

# --- outputs ---------------------------------------------------------------
RESULTS_DIR = _env("RFFP_RESULTS", os.path.join(REPO_ROOT, "results_gen"))
