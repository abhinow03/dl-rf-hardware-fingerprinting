"""STEP-7 ONE-TOUCH TEST + HARNESS UNIFICATION — two eval-only tasks.

TASK A — PRE-REGISTERED TEST EVALUATION (board 18, 16 devices, ONE touch). Spends the only
sealed WiSig devices exactly once. Rows fixed BEFORE results; all reported regardless; zero re-runs:
  A1  P1 bursts (multi-rx, N=10) + CosFace head (m0.20/s32/d128) + HDBSCAN mcs=15, locked noise.
  A2  P1 bursts, head-free 512-D, HDBSCAN mcs{13,15,17} (all three, no selection) + oracle-K@16 (km,sp).
  A3  P2 bursts (single-rx, multi-date), head-free 512-D, same scoring.
  A4  P0 bursts (coherent),           head-free 512-D, same scoring.
Metrics/row: ARI/NMI/purity/K_est(vs16)/noise + kNN-1 + intra/inter gap. DEV@16 reference beside
each + |TEST-DEV| drift (pre-registered expectation ~0.07; larger reported as-is, single-board caveat).
Provenance note (verbatim): the CosFace op-point was originally selected during seed-123-era tuning
(which touched board-18 devices), then re-validated on clean DEV slices in Step-2b; A1 is its first and
ONLY TEST evaluation.

TASK B — HARNESS UNIFICATION (kills the 0.218-vs-0.341 oracle discrepancy). Define THE harness once
(oracle-K = km AND sp reported SEPARATELY, never max-of-methods; HDBSCAN = locked noise, mean over a
pre-declared mcs grid, no argmax; window-build seed FIXED = 777). Recompute the paper's DRFF headline
cells under it -> drff_headline_locked.csv. Sanity: (i) step6 'no adaptation lift' still holds; (ii)
step3c cross-domain << in-domain wall still holds. Flag loudly if either flips.

FROZEN best_model.pt + frozen CosFace head. NO tuning of any kind. Eval-only. The adapted-R1 E1 cell
(required by Task B(b)) is the SOLE read-only load of an adaptation checkpoint; it is never used for
TEST and the checkpoint is not modified — flagged in the report.

  python3 results/step7_test_and_harness/test_and_harness.py
"""
import os, sys, json, csv
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity"),
          os.path.join(_SW, "results", "step2b_decoherence"),
          os.path.join(_SW, "results", "step4_mechanism_validation")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
import geometry_consolidate as GC
from integrity import score_locked, hdbscan_pred, unitrows, bursts_coherent
from decohere import bursts_p1_multirx, bursts_p2_multidate
from mechanism_validation import classical_matrix           # guarded import (no rerun)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
PART      = os.path.join(RUN_DIR, "splits", "discovery_partition.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
R1_CKPT   = os.path.join(_SW, "runs", "drff_adapt", "R1", "best.pt")   # read-only, Task B(b) only
DRFF_DIR  = os.path.expanduser("~/Desktop/processed/drff_r2")
OUT       = _HERE

# ---- THE LOCKED HARNESS (fixed here, mirrored into EVAL_PROTOCOL.md) ----
HARNESS_SEED = 777                 # window-build seed policy: fixed -> deterministic windows every run
MCS_WISIG = [13, 15, 17]           # K~16-18 regime grid
MCS_DRFF  = [5, 7]                 # K~8 regime grid
N = 10
WIN = 256
NTAPS = 129
COS_M, COS_S = 0.20, 32
DRFF_CAP = 320
WIN_FULL = 100000                  # "full available windows" per device
SLICE_SEEDS = [201, 202, 203, 204, 205]

def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# ============================================================================
# LOCKED SCORING PRIMITIVES
# ============================================================================
def knn_purity(X, y, ks=(1, 5)):
    Xn = unit(X); k = min(max(ks) + 1, len(Xn))
    nn = NearestNeighbors(n_neighbors=k).fit(Xn); _, ind = nn.kneighbors(Xn)
    yi = np.unique(y, return_inverse=True)[1]; out = {}
    for kk in ks:
        nb = ind[:, 1:kk + 1]; out[kk] = float((yi[nb] == yi[:, None]).mean())
    return out

def cos_gap(X, y):
    Xn = unit(X); yi = np.unique(y, return_inverse=True)[1]; intra, inter = [], []
    for a in np.unique(yi):
        mm = yi == a
        if mm.sum() < 2: continue
        C = Xn[mm] @ Xn[mm].T; iu = np.triu_indices(mm.sum(), 1)
        intra.append(C[iu].mean()); inter.append((Xn[mm] @ Xn[~mm].T).mean())
    return float(np.mean(intra)), float(np.mean(inter))

def oracle_km_sp(bp, bl, K):
    """oracle-K CEILING: k-means AND spectral, reported SEPARATELY (never max)."""
    yi = np.unique(bl, return_inverse=True)[1]
    km = float(adjusted_rand_score(yi, KMeans(K, n_init=10, random_state=0).fit_predict(bp)))
    try:
        sp = SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                n_neighbors=15).fit_predict(bp)
        sp = float(adjusted_rand_score(yi, sp))
    except Exception:
        sp = float("nan")
    return km, sp

def hdbscan_grid(bp, bl, mcs_grid):
    """locked noise rule per mcs; returns per-mcs score dicts + MEAN ARI (no argmax selection)."""
    per = {mcs: score_locked(hdbscan_pred(bp, mcs), bl) for mcs in mcs_grid}
    mean_ari = float(np.mean([per[m]["ARI"] for m in mcs_grid]))
    return per, mean_ari

def full_score(bp, bl, K, mcs_grid, with_oracle=True):
    bpu = unit(bp)
    per, hdb_mean = hdbscan_grid(bpu, bl, mcs_grid)
    kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl)
    out = dict(hdb_per_mcs=per, hdb_mean_ARI=hdb_mean, knn1=kp[1], knn5=kp[5],
               intra=intra, inter=inter, gap=intra - inter, n_bursts=int(len(bl)))
    if with_oracle:
        km, sp = oracle_km_sp(bpu, bl, K)
        out["oracleK_km"] = km; out["oracleK_sp"] = sp
    return out

# ============================================================================
# FROZEN ENCODER + reproduced CosFace head (seed42, == step2b)
# ============================================================================
print("loading ManyTx (eq=0)...")
TXD, _ = W.load_manytx(eq=0, verbose=False)
print("[0] frozen encoder + reproduced CosFace head (seed42, m=%.2f s=%d) ..." % (COS_M, COS_S))
sp_split = json.load(open(SPLIT)); part = json.load(open(PART))
train_tx = sp_split["train_tx"]; dev_tx = part["dev_tx"]; test_tx = part["test_tx"]; n_cls = len(train_tx)
model = RFEncoder().cuda()
model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
model.eval()
for pp in model.parameters(): pp.requires_grad_(False)
tt = np.concatenate([GC.consec_windows(TXD, tx, GC.TRAIN_PER_DEV) for tx in train_tx])
ty = np.concatenate([np.full(min(GC.TRAIN_PER_DEV, TXD[tx]["iq"].shape[0]), i)
                     for i, tx in enumerate(train_tx)])
Ftr = GC.extract512(model, tt); del tt
scaler = StandardScaler().fit(Ftr)
head = GC.train_cosface(torch.from_numpy(scaler.transform(Ftr).astype(np.float32)).cuda(),
                        torch.from_numpy(ty).cuda(), n_cls, COS_M, COS_S)
del Ftr

def build_wisig_cache(tx_list):
    cache = {}
    for tx in tx_list:
        t = GC.consec_windows(TXD, tx, WIN_FULL); Wn = t.shape[0]
        e512 = GC.extract512(model, t)
        ecf = GC.apply_head(head, {tx: scaler.transform(e512).astype(np.float32)})[tx]
        cache[tx] = dict(f512=e512, fcos=ecf,
                         rx=TXD[tx]["rx"][:Wn].copy(), date=TXD[tx]["date"][:Wn].copy(), nwin=Wn)
    return cache

print(f"caching TEST ({len(test_tx)}) + DEV ({len(dev_tx)}) devices (full windows) ...")
TESTC = build_wisig_cache(test_tx)
DEVC  = build_wisig_cache(dev_tx)
print("    TEST windows/device:", {tx: TESTC[tx]["nwin"] for tx in test_tx[:3]}, "...")

# ============================================================================
# WiSig burst builders (identical to step2b/step4)
# ============================================================================
def bursts_for(cache, tx_list, proto, emb_key, seed):
    bp, bl = [], []
    for di, tx in enumerate(tx_list):
        c = cache[tx]; e = c[emb_key]
        if proto == "P0":  b = bursts_coherent(e, N)
        elif proto == "P1": b = bursts_p1_multirx(e, c["rx"], c["date"], seed=seed)[0]
        elif proto == "P2": b = bursts_p2_multidate(e, c["rx"], c["date"], seed=seed)[0]
        if b is None: continue
        bp.append(b); bl.append(np.full(len(b), di))
    return np.concatenate(bp), np.concatenate(bl)

def dev_ref(proto, emb_key, K, mcs_grid, scorer_key, with_oracle=True):
    """DEV@K reference: mean over 5 slices of the headline metric(s)."""
    vals = defaultdict(list)
    for s in SLICE_SEEDS:
        sl = GC.scatter_slice(dev_tx, K, s)
        bp, bl = bursts_for(DEVC, sl, proto, emb_key, s)
        r = full_score(bp, bl, K, mcs_grid, with_oracle=with_oracle)
        vals["hdb_mean_ARI"].append(r["hdb_mean_ARI"]); vals["knn1"].append(r["knn1"])
        vals["gap"].append(r["gap"])
        if with_oracle: vals["oracleK_km"].append(r["oracleK_km"]); vals["oracleK_sp"].append(r["oracleK_sp"])
        for m in mcs_grid: vals[f"hdb_mcs{m}"].append(r["hdb_per_mcs"][m]["ARI"])
    return {k: float(np.nanmean(v)) for k, v in vals.items()}

# ============================================================================
# TASK A — ONE-TOUCH TEST (board 18, 16 devices)  [computed ONCE, no re-runs]
# ============================================================================
KTEST = 16
print("\n" + "=" * 80)
print("TASK A — ONE-TOUCH TEST EVALUATION (board 18, 16 devices). Computed once, no re-runs.")
print("=" * 80)
test_rows = []

def emit_test_row(rowid, proto, emb, emb_key, mcs_grid, with_oracle, ktest=KTEST):
    seed = SLICE_SEEDS[0]                                  # fixed single build seed for TEST bursts
    bp, bl = bursts_for(TESTC, test_tx, proto, emb_key, seed)
    r = full_score(bp, bl, ktest, mcs_grid, with_oracle=with_oracle)
    ref = dev_ref(proto, emb_key, ktest, mcs_grid, rowid, with_oracle=with_oracle)
    # headline metric: A1 -> HDBSCAN mcs15 ARI; A2-A4 -> oracle-K km (+ HDBSCAN mean shown too)
    head_name = "hdb_mcs15_ARI" if rowid == "A1" else "oracleK_km"
    test_head = (r["hdb_per_mcs"][15]["ARI"] if rowid == "A1" else r["oracleK_km"])
    dev_head  = (ref.get("hdb_mcs15") if rowid == "A1" else ref.get("oracleK_km"))
    drift = abs(test_head - dev_head) if dev_head is not None else float("nan")
    dev_hdb_mean = ref.get("hdb_mean_ARI")
    row = dict(row=rowid, protocol=proto, emb=emb,
               TEST_hdb_mean_ARI=round(r["hdb_mean_ARI"], 3),
               DEV_hdb_mean_ARI=round(dev_hdb_mean, 3),
               drift_hdb_mean=round(abs(r["hdb_mean_ARI"] - dev_hdb_mean), 3),
               TEST_knn1=round(r["knn1"], 3), TEST_gap=round(r["gap"], 3),
               TEST_nbursts=r["n_bursts"], headline_metric=head_name,
               TEST_headline=round(test_head, 3), DEV_headline=round(dev_head, 3),
               drift=round(drift, 3), drift_ok=(drift <= 0.07))
    for m in mcs_grid:
        sc = r["hdb_per_mcs"][m]
        row[f"TEST_hdb_mcs{m}_ARI"] = round(sc["ARI"], 3)
        row[f"TEST_hdb_mcs{m}_NMI"] = round(sc["NMI"], 3)
        row[f"TEST_hdb_mcs{m}_purity"] = round(sc["purity"], 3)
        row[f"TEST_hdb_mcs{m}_Kest"] = int(sc["K_est"])
        row[f"TEST_hdb_mcs{m}_noise"] = round(sc["noise"], 3)
        row[f"DEV_hdb_mcs{m}_ARI"] = round(ref.get(f"hdb_mcs{m}", float('nan')), 3)
    if with_oracle:
        row["TEST_oracleK_km"] = round(r["oracleK_km"], 3); row["TEST_oracleK_sp"] = round(r["oracleK_sp"], 3)
        row["DEV_oracleK_km"] = round(ref.get("oracleK_km", float('nan')), 3)
        row["DEV_oracleK_sp"] = round(ref.get("oracleK_sp", float('nan')), 3)
    test_rows.append(row)
    orc = (f" | oracle@16 km={row.get('TEST_oracleK_km')} sp={row.get('TEST_oracleK_sp')} "
           f"(DEV km={row.get('DEV_oracleK_km')})") if with_oracle else ""
    print(f"  [{rowid}] {proto} {emb}: HDBmean={row['TEST_hdb_mean_ARI']} "
          f"(mcs " + "/".join(str(row[f'TEST_hdb_mcs{m}_ARI']) for m in mcs_grid) + ")"
          + orc + f" | kNN1={row['TEST_knn1']} gap={row['TEST_gap']} "
          f"| HEADLINE {head_name}: TEST={row['TEST_headline']} DEV={row['DEV_headline']} "
          f"drift={row['drift']} ({'OK' if row['drift_ok'] else 'ABOVE 0.07'})")

emit_test_row("A1", "P1", "CosFace",  "fcos", [15], with_oracle=False)
emit_test_row("A2", "P1", "512D",     "f512", MCS_WISIG, with_oracle=True)
emit_test_row("A3", "P2", "512D",     "f512", MCS_WISIG, with_oracle=True)
emit_test_row("A4", "P0", "512D",     "f512", MCS_WISIG, with_oracle=True)

# ============================================================================
# TASK B — HARNESS UNIFICATION (DRFF headline, seed 777, km/sp separate, HDBSCAN mean)
# ============================================================================
print("\n" + "=" * 80)
print("TASK B — HARNESS UNIFICATION: DRFF headline recomputed under THE locked harness (seed 777)")
print("=" * 80)
import re
TAPS_B = firwin(NTAPS, 0.9)
def decimate_B(iq):
    x = iq.astype(np.float32); pad = min(3 * NTAPS, x.shape[1] - 1)
    return filtfilt(TAPS_B, [1.0], x, axis=1, padlen=pad)[:, ::2]
manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
pat = re.compile(r'(.+?)_(\d+)_hover')
af_files = defaultdict(list)
for r in manifest["clean"]:
    af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
all_af = sorted(af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
model_of = {a: pat.match(a + "_hover").group(1) for a in all_af}
EVAL_AF = [a for a in all_af if model_of[a] == "mavicAir2"]

def build_mavicAir2_windows(cap=DRFF_CAP, seed=HARNESS_SEED):
    rng = np.random.default_rng(seed)
    Xt, af, seg, Ds, Cs = [], [], [], [], []
    gseg = 0
    for ai, a in enumerate(EVAL_AF):
        files = af_files[a][:]; rng.shuffle(files)
        units = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]): units.append((fn, si))
        rng.shuffle(units)
        got = 0; zc = {}
        for fn, si in units:
            if got >= cap: break
            if fn not in zc:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zc[fn] = dict(dec=decimate_B(z["iq"]), sb=z["seg_bounds"], D=str(z["D"]), C=int(z["C"]))
            zz = zc[fn]; off, ln = zz["sb"][si]; o2, l2 = off // 2, ln // 2
            nw = l2 // WIN
            if nw < 1: continue
            take = min(nw, cap - got, 12); segw = zz["dec"][:, o2:o2 + l2]
            for k in range(take):
                Xt.append(W.standardize(segw[:, k*WIN:(k+1)*WIN].astype(np.float32)))
                af.append(ai); seg.append(gseg); Ds.append(zz["D"]); Cs.append(zz["C"])
            got += take; gseg += 1
    return np.stack(Xt).astype(np.float32), np.array(af), np.array(seg), np.array(Ds), np.array(Cs)

print("[B] building mavicAir2 windows (seed 777) + frozen 512-D + classical-19 ...")
Xt, afd, segd, Dd, Cd = build_mavicAir2_windows()
Xs = W.compute_stft_batch(Xt)
F512_frozen = GC.extract512(model, Xt)
CLS19 = StandardScaler().fit_transform(classical_matrix(Xt)).astype(np.float32)
print(f"    {len(Xt)} windows, {len(np.unique(segd))} segments, {len(EVAL_AF)} airframes")

# adapted-R1 512-D (SOLE read-only checkpoint load; never used for TEST)
class Adapter(nn.Module):
    def __init__(self, dim=512, bott=128):
        super().__init__()
        self.ln = nn.LayerNorm(dim); self.down = nn.Linear(dim, bott); self.up = nn.Linear(bott, dim)
    def forward(self, h): return h + self.up(F.relu(self.down(self.ln(h))))

print("[B] loading R1 adaptation checkpoint READ-ONLY (Task B(b) adapted-R1 cell only) ...")
r1_state = torch.load(R1_CKPT, map_location="cuda", weights_only=False)
r1_base = RFEncoder().cuda(); r1_base.load_state_dict(r1_state["base"], strict=True); r1_base.eval()
r1_adapter = Adapter().cuda(); r1_adapter.load_state_dict(r1_state["adapter"], strict=True); r1_adapter.eval()
for p in list(r1_base.parameters()) + list(r1_adapter.parameters()): p.requires_grad_(False)
@torch.no_grad()
def r1_enc512(Xt_, Xs_, batch=1024):
    f = np.empty((Xt_.shape[0], 512), np.float32)
    xt = torch.from_numpy(Xt_); xs = torch.from_numpy(Xs_)
    for i in range(0, Xt_.shape[0], batch):
        with torch.amp.autocast('cuda'):
            h = r1_base.get_encoder_output(xt[i:i+batch].cuda(), xs[i:i+batch].cuda())
            f[i:i+batch] = r1_adapter(h).float().cpu().numpy()
    return f
F512_R1 = r1_enc512(Xt, Xs)

# burst builders (matched-index for hybrid)
def build_E0(emb):
    bp, bl = [], []
    for sg in np.unique(segd):
        idx = np.where(segd == sg)[0]
        for k in range(len(idx)//N):
            ch = idx[k*N:(k+1)*N]; bp.append(emb[ch].mean(0)); bl.append(afd[ch][0])
    return np.array(bp), np.array(bl)
def E1_index_bursts(seed=HARNESS_SEED):
    r = np.random.default_rng(seed); out = []
    for a in np.unique(afd):
        idx = np.where(afd == a)[0]; cells = defaultdict(list)
        for i in idx: cells[(Dd[i], Cd[i])].append(i)
        ck = list(cells.keys()); nb = len(idx)//N
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < N and ci < 2000:
                c = ck[ci % len(ck)]
                if cells[c]: pick.append(cells[c][r.integers(len(cells[c]))])
                ci += 1
            if len(pick) == N: out.append((np.array(pick), a))
    return out
def balance_idx(bursts, seed=0):
    r = np.random.default_rng(seed); lbl = np.array([b[1] for b in bursts])
    per = min(int((lbl == a).sum()) for a in np.unique(lbl)); keep = []
    for a in np.unique(lbl):
        ii = np.where(lbl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    return [bursts[i] for i in sorted(keep)]
def mod_bursts(feat, bursts): return np.stack([feat[idx].mean(0) for idx, _ in bursts])
def balance_arr(bp, bl, seed=0):
    r = np.random.default_rng(seed); per = min(int((bl == a).sum()) for a in np.unique(bl)); keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    k = np.array(sorted(keep)); return bp[k], bl[k]

KD = 8
E1_bursts = balance_idx(E1_index_bursts())
E1_labels = np.array([b[1] for b in E1_bursts])
cls_b = mod_bursts(CLS19, E1_bursts)
enc_b = mod_bursts(F512_frozen, E1_bursts)
# hybrid PCA-64+19 (== step5 best variant): standardize blocks, PCA-64 on encoder, concat, unit
def hybrid_pca64(cls_bm, enc_bm):
    Zc = StandardScaler().fit_transform(cls_bm)
    Zp = StandardScaler().fit_transform(PCA(n_components=min(64, enc_bm.shape[0], 512),
                                            random_state=0).fit_transform(enc_bm))
    return unit(np.concatenate([Zc, Zp], axis=1))

locked_rows = []
def emit_locked(method, proto, bp, bl, K=KD):
    r = full_score(bp, bl, K, MCS_DRFF, with_oracle=True)
    row = dict(method=method, protocol=proto,
               HDBSCAN_mean_ARI=round(r["hdb_mean_ARI"], 3),
               HDBSCAN_mcs5_ARI=round(r["hdb_per_mcs"][5]["ARI"], 3),
               HDBSCAN_mcs7_ARI=round(r["hdb_per_mcs"][7]["ARI"], 3),
               oracleK8_kmeans=round(r["oracleK_km"], 3), oracleK8_spectral=round(r["oracleK_sp"], 3),
               kNN1=round(r["knn1"], 3), gap=round(r["gap"], 3), n_bursts=r["n_bursts"])
    locked_rows.append(row)
    print(f"  {method:>16} {proto}: HDBmean={row['HDBSCAN_mean_ARI']} (mcs5 {row['HDBSCAN_mcs5_ARI']}/"
          f"mcs7 {row['HDBSCAN_mcs7_ARI']}) | oracle@8 km={row['oracleK8_kmeans']} sp={row['oracleK8_spectral']} "
          f"| kNN1={row['kNN1']} gap={row['gap']} nb={row['n_bursts']}")
    return row

# frozen 512-D  (E0 + E1)
emit_locked("frozen-512", "E0", *balance_arr(*build_E0(F512_frozen)))
emit_locked("frozen-512", "E1", enc_b, E1_labels)
# classical-19  (E0 + E1)
emit_locked("classical-19", "E0", *balance_arr(*build_E0(CLS19)))
emit_locked("classical-19", "E1", cls_b, E1_labels)
# hybrid PCA-64+19 (E0 + E1)
E0c_bp, E0c_bl = build_E0(CLS19); E0e_bp, _ = build_E0(F512_frozen)
_hb0 = hybrid_pca64(E0c_bp, E0e_bp)
emit_locked("hybrid-pca64+19", "E0", *balance_arr(_hb0, E0c_bl))
emit_locked("hybrid-pca64+19", "E1", hybrid_pca64(cls_b, enc_b), E1_labels)
# adapted-R1 (E1 only, per Task B(b))
emit_locked("adapted-R1", "E1", mod_bursts(F512_R1, E1_bursts), E1_labels)

# ============================================================================
# SANITY CONFIRMATIONS
# ============================================================================
def get(method, proto, col):
    return next(r for r in locked_rows if r["method"] == method and r["protocol"] == proto)[col]
frozen_e1_km, frozen_e1_sp = get("frozen-512", "E1", "oracleK8_kmeans"), get("frozen-512", "E1", "oracleK8_spectral")
r1_e1_km, r1_e1_sp = get("adapted-R1", "E1", "oracleK8_kmeans"), get("adapted-R1", "E1", "oracleK8_spectral")
lift_km = r1_e1_km - frozen_e1_km; lift_sp = r1_e1_sp - frozen_e1_sp
no_lift = (lift_km <= 0.03 and lift_sp <= 0.03)
sanity1 = (f"{'CONFIRMED' if no_lift else 'FLIP!!'}: adapted-R1 E1 oracle vs frozen — km {r1_e1_km:.3f} vs "
           f"{frozen_e1_km:.3f} ({lift_km:+.3f}), sp {r1_e1_sp:.3f} vs {frozen_e1_sp:.3f} ({lift_sp:+.3f}). "
           f"Step-6 'no adaptation lift' {'holds' if no_lift else 'DOES NOT hold'} under THE harness.")

# in-domain reference (from this session's TEST A3 P2 512-D oracle km, and step5 same-board K8 0.895)
indomain_km = next(r for r in test_rows if r["row"] == "A3").get("TEST_oracleK_km")  # TEST P2 single-rx
crossdomain_km = max(frozen_e1_km, get("frozen-512", "E0", "oracleK8_kmeans"))
wall_holds = crossdomain_km < indomain_km - 0.15
sanity2 = (f"{'CONFIRMED' if wall_holds else 'FLIP!!'}: cross-domain DRFF frozen oracle km "
           f"E1={frozen_e1_km:.3f}/E0={get('frozen-512','E0','oracleK8_kmeans'):.3f} << in-domain "
           f"(TEST P2 single-rx oracle km={indomain_km:.3f}; step5 same-board K8=0.895). "
           f"Cross-domain wall {'holds' if wall_holds else 'DOES NOT hold'}.")

print("\n=== SANITY ===")
print("(i) ", sanity1)
print("(ii)", sanity2)

# ============================================================================
# SAVE
# ============================================================================
def wcsv(fn, rows, fields=None):
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        keys = fields or sorted(set().union(*[r.keys() for r in rows]))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

test_cols = ["row", "protocol", "emb", "headline_metric", "TEST_headline", "DEV_headline", "drift",
             "drift_ok", "TEST_hdb_mean_ARI", "DEV_hdb_mean_ARI", "drift_hdb_mean",
             "TEST_hdb_mcs13_ARI", "TEST_hdb_mcs15_ARI",
             "TEST_hdb_mcs17_ARI", "TEST_hdb_mcs13_NMI", "TEST_hdb_mcs15_NMI", "TEST_hdb_mcs17_NMI",
             "TEST_hdb_mcs13_purity", "TEST_hdb_mcs15_purity", "TEST_hdb_mcs17_purity",
             "TEST_hdb_mcs13_Kest", "TEST_hdb_mcs15_Kest", "TEST_hdb_mcs17_Kest",
             "TEST_hdb_mcs13_noise", "TEST_hdb_mcs15_noise", "TEST_hdb_mcs17_noise",
             "TEST_oracleK_km", "TEST_oracleK_sp", "DEV_oracleK_km", "DEV_oracleK_sp",
             "TEST_knn1", "TEST_gap", "TEST_nbursts"]
wcsv("test_table.csv", test_rows, test_cols)
wcsv("drff_headline_locked.csv", locked_rows,
     ["method", "protocol", "HDBSCAN_mean_ARI", "HDBSCAN_mcs5_ARI", "HDBSCAN_mcs7_ARI",
      "oracleK8_kmeans", "oracleK8_spectral", "kNN1", "gap", "n_bursts"])

LOCKED_HARNESS = dict(
    window_build_seed=HARNESS_SEED,
    window_seed_policy="FIXED seed 777 for the DRFF window build (matches step3c/step5 citations) -> "
                       "deterministic windows every run; justify: reproducibility, removes build-seed variance.",
    oracleK_policy="report k-means AND spectral SEPARATELY (never max-of-methods); justify: max is silent "
                   "method-selection that inflates the ceiling.",
    hdbscan_policy=f"locked noise rule (singletons -> own clusters; ARI/NMI/purity over ALL points; noise "
                   f"reported separately). Fixed mcs grid (DRFF {MCS_DRFF}, WiSig {MCS_WISIG}); headline = "
                   f"MEAN ARI over grid, full grid reported; justify: per-result argmax over mcs is selection.",
    burst_policy="burst-mean N=10, L2-renormalized; bursts balanced to min per-class count before scoring.")

report = dict(
    taskA=dict(
        one_touch_statement="TEST (board 18, 16 devices) touched EXACTLY ONCE. All 4 pre-registered rows "
            "(A1-A4) computed a single time and reported regardless of outcome. ZERO re-runs, zero "
            "parameter changes after seeing results, no added rows, adaptation checkpoints excluded from TEST.",
        provenance_note="The CosFace op-point (m=0.20/s=32/d128) was originally selected during seed-123-era "
            "tuning (which touched board-18 devices), then re-validated on clean DEV slices in Step-2b; A1 is "
            "its FIRST and ONLY TEST evaluation.",
        drift_expectation=0.07,
        single_board_caveat="TEST is a single board (16 same-fab devices) — a known structural caveat; larger "
            "drift is reported as-is with no re-runs, and echoes the board-20 fineness control (same-board "
            "devices cluster differently from multi-board DEV slices).",
        rows=test_rows),
    taskB=dict(locked_harness=LOCKED_HARNESS, drff_headline=locked_rows,
        adapted_R1_note="adapted-R1 E1 is the SOLE read-only load of an adaptation checkpoint "
            "(runs/drff_adapt/R1/best.pt), required by Task B(b); it is NEVER used for TEST and the checkpoint "
            "file is not modified. Top guardrail 'no adaptation checkpoints used anywhere' is otherwise honored.",
        sanity_no_adaptation_lift=sanity1, sanity_crossdomain_wall=sanity2,
        both_confirmed=bool(no_lift and wall_holds)),
    guardrails="eval-only; frozen best_model.pt + frozen CosFace head; NO tuning/selection after results; "
               "adaptation checkpoints not modified (R1 read-only for one cell); M100 + TEST-after-touch sealed.")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)

print("\n" + "=" * 80)
print("G1:", report["taskA"]["one_touch_statement"])
print("saved -> results/step7_test_and_harness/ (test_table.csv, drff_headline_locked.csv, report.json)")
print("CHECKPOINT — TEST spent once; harness locked; frozen weights only; M100 untouched.")
