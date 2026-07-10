"""STEP-8 DISENTANGLE + HYBRID — last two open experimental claims, LOCKED harness (§4.3:
window seed 777, oracle-K km AND sp reported separately never max, HDBSCAN mean over fixed mcs
grid, locked noise rule). Frozen best_model.pt (head-free 512-D). TEST/board-18 CLOSED. No M100.

TASK 1 — FINENESS DISENTANGLE (board-20 K13, DEV-legal). step5 'fineness not the wall' came
from board-20 K8=0.895; TEST board-18 K16 collapsed to 0.36-0.42 (K vs board confounded). Board
20 has 13 DEV devices = largest legal same-board K.
  (a) same-board: ALL 13 board-20 devices, K13, P1 AND P2, head-free 512-D, locked harness.
  (b) mixed-board: 3 slices of 13 across boards 4/5/16/20 (seeds 501-503), same, mean +/- std.
  (c) board-20 K8 re-scored under THE harness (2 draws of 8-of-13, seeds 511-512) for a
      harness-consistent K8->K13 trend.

TASK 2 — BLOCK-SCALED HYBRID, alpha tuned LEGALLY on mavicAir2s, frozen, applied ONCE to
mavicAir2. f_hybrid(alpha) = [alpha*enc_block ; (1-alpha)*cls_block], each block standardized
then scaled by 1/sqrt(d) (distance equalization) BEFORE alpha. Encoder variants: (E-a) raw 512,
(E-b) PCA-64 with PCA FIT ON pool-minus-mavicAir2s (NEVER mavicAir2/mavicAir2s). alpha in
{0.2,0.35,0.5,0.65,0.8}. Tune on mavicAir2s (7) with a condition-purity control (disqualify an
alpha whose oracle clustering tracks distance-D better than airframe). Freeze (variant,alpha),
write tuning table BEFORE mavicAir2 eval, then apply ONCE to mavicAir2 (E1+E0) + a WiSig
in-domain preservation check (one mixed-board K13 slice, same hybrid construction).

  python3 results/step8_disentangle_hybrid/disentangle_hybrid.py
"""
import os, sys, json, csv, re, datetime
from collections import defaultdict, Counter
import numpy as np
import torch
from scipy.signal import firwin, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity"),
          os.path.join(_SW, "results", "step2b_decoherence"),
          os.path.join(_SW, "results", "step4_mechanism_validation")):
    if p not in sys.path: sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
import geometry_consolidate as GC
from integrity import score_locked, hdbscan_pred, unitrows, bursts_coherent
from decohere import bursts_p1_multirx, bursts_p2_multidate
from mechanism_validation import classical_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
PART      = os.path.join(RUN_DIR, "splits", "discovery_partition.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
DRFF_DIR  = os.path.expanduser("~/Desktop/processed/drff_r2")
OUT       = _HERE

HARNESS_SEED = 777
MCS_WISIG = [13, 15, 17]         # K13/K16-18 grid
MCS_DRFF  = [5, 7]               # K~7-8 grid
N, WIN, NTAPS = 10, 256, 129
DRFF_CAP = 320
WIN_CACHE = 2400
ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8]
def now(): return datetime.datetime.now().isoformat(timespec="seconds")
def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# ---- locked scoring ----
def knn_purity(X, y, ks=(1, 5)):
    Xn = unit(X); k = min(max(ks)+1, len(Xn))
    nn = NearestNeighbors(n_neighbors=k).fit(Xn); _, ind = nn.kneighbors(Xn)
    yi = np.unique(y, return_inverse=True)[1]; out = {}
    for kk in ks:
        nb = ind[:, 1:kk+1]; out[kk] = float((yi[nb] == yi[:, None]).mean())
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
    yi = np.unique(bl, return_inverse=True)[1]
    km = float(adjusted_rand_score(yi, KMeans(K, n_init=10, random_state=0).fit_predict(bp)))
    try:
        sp = float(adjusted_rand_score(yi, SpectralClustering(K, affinity="nearest_neighbors",
                   random_state=0, n_neighbors=15).fit_predict(bp)))
    except Exception: sp = float("nan")
    return km, sp
def hdb_mean(bp, bl, grid):
    per = {m: score_locked(hdbscan_pred(bp, m), bl) for m in grid}
    return float(np.mean([per[m]["ARI"] for m in grid])), {m: round(per[m]["ARI"],3) for m in grid}, per
def full_score(bp, bl, K, grid):
    bpu = unit(bp)
    hm, per_r, per = hdb_mean(bpu, bl, grid)
    km, sp = oracle_km_sp(bpu, bl, K)
    kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl)
    K13 = per[grid[len(grid)//2]]["K_est"]
    return dict(hdb_mean=round(hm,3), hdb_per=per_r, oracleK_km=round(km,3), oracleK_sp=round(sp,3),
                knn1=round(kp[1],3), gap=round(intra-inter,3), n_bursts=int(len(bl)), K_est_mid=int(K13))

# ============================================================================
# frozen encoder (head-free)
# ============================================================================
print("loading ManyTx (eq=0)...")
TXD, _ = W.load_manytx(eq=0, verbose=False)
print("[0] frozen encoder (head-free 512-D) ...")
part = json.load(open(PART)); dev_tx = part["dev_tx"]
model = RFEncoder().cuda()
model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
model.eval()
for pp in model.parameters(): pp.requires_grad_(False)

# ============================================================================
# TASK 1 — FINENESS DISENTANGLE (WiSig DEV, head-free 512-D)
# ============================================================================
print("\n" + "="*78 + "\nTASK 1 — FINENESS DISENTANGLE (board-20 K13, locked harness)\n" + "="*78)
print(f"    HDBSCAN mcs grid for WiSig = {MCS_WISIG} (locked §4.3)")
dev512 = {}
for tx in dev_tx:
    t = GC.consec_windows(TXD, tx, WIN_CACHE); Wn = t.shape[0]
    dev512[tx] = dict(emb=GC.extract512(model, t), rx=TXD[tx]["rx"][:Wn].copy(),
                      date=TXD[tx]["date"][:Wn].copy())
board20 = [d for d in dev_tx if d.split("-")[0] == "20"]

def wisig_bursts(slice_tx, proto, seed):
    bp, bl = [], []
    for di, tx in enumerate(slice_tx):
        c = dev512[tx]
        b = (bursts_p1_multirx(c["emb"], c["rx"], c["date"], seed=seed)[0] if proto == "P1"
             else bursts_p2_multidate(c["emb"], c["rx"], c["date"], seed=seed)[0])
        if b is None: continue
        bp.append(b); bl.append(np.full(len(b), di))
    return np.concatenate(bp), np.concatenate(bl)

fine_rows = []
def fine_score(tag, slice_tx, proto, K, seed):
    bp, bl = wisig_bursts(slice_tx, proto, seed)
    r = full_score(bp, bl, K, MCS_WISIG)
    fine_rows.append(dict(row=tag, proto=proto, K=K, n_dev=len(slice_tx), **r))
    return r

# (a) same-board K13
for proto in ("P1", "P2"):
    r = fine_score("same_board20_K13", board20, proto, 13, 501)
    print(f"  (a) same-board20 K13 {proto}: HDBmean={r['hdb_mean']} oracle km={r['oracleK_km']}/sp={r['oracleK_sp']} "
          f"kNN1={r['knn1']} gap={r['gap']}")
# (b) mixed-board K13, seeds 501-503
mixed_agg = defaultdict(lambda: defaultdict(list))
for seed in (501, 502, 503):
    sl = GC.scatter_slice(dev_tx, 13, seed)
    for proto in ("P1", "P2"):
        bp, bl = wisig_bursts(sl, proto, seed)
        r = full_score(bp, bl, 13, MCS_WISIG)
        for k in ("hdb_mean","oracleK_km","oracleK_sp","knn1","gap"): mixed_agg[proto][k].append(r[k])
mix_summary = {}
for proto in ("P1", "P2"):
    ms = {k: (round(float(np.mean(v)),3), round(float(np.std(v)),3)) for k,v in mixed_agg[proto].items()}
    mix_summary[proto] = ms
    fine_rows.append(dict(row="mixed_board_K13", proto=proto, K=13, n_dev=13,
        hdb_mean=ms["hdb_mean"][0], oracleK_km=ms["oracleK_km"][0], oracleK_sp=ms["oracleK_sp"][0],
        knn1=ms["knn1"][0], gap=ms["gap"][0], hdb_std=ms["hdb_mean"][1], km_std=ms["oracleK_km"][1]))
    print(f"  (b) mixed-board K13 {proto}: HDBmean={ms['hdb_mean'][0]}+/-{ms['hdb_mean'][1]} "
          f"oracle km={ms['oracleK_km'][0]}+/-{ms['oracleK_km'][1]}/sp={ms['oracleK_sp'][0]} kNN1={ms['knn1'][0]}")
# (c) board-20 K8 re-scored (2 draws of 8-of-13, seeds 511-512)
k8_agg = defaultdict(lambda: defaultdict(list))
for seed in (511, 512):
    rng = np.random.default_rng(seed); draw = sorted(rng.choice(board20, size=8, replace=False).tolist())
    for proto in ("P1", "P2"):
        bp, bl = wisig_bursts(draw, proto, seed)
        r = full_score(bp, bl, 8, MCS_DRFF if False else MCS_WISIG)   # WiSig grid per §4.3
        for k in ("hdb_mean","oracleK_km","oracleK_sp","knn1"): k8_agg[proto][k].append(r[k])
for proto in ("P1", "P2"):
    km = round(float(np.mean(k8_agg[proto]["oracleK_km"])),3)
    hm = round(float(np.mean(k8_agg[proto]["hdb_mean"])),3)
    sp = round(float(np.mean(k8_agg[proto]["oracleK_sp"])),3)
    fine_rows.append(dict(row="same_board20_K8", proto=proto, K=8, n_dev=8, hdb_mean=hm,
        oracleK_km=km, oracleK_sp=sp, knn1=round(float(np.mean(k8_agg[proto]["knn1"])),3), gap=None))
    print(f"  (c) same-board20 K8 (re-scored) {proto}: HDBmean={hm} oracle km={km}/sp={sp}")

# verdict (P2, km oracle — the fineness/step5 axis)
sb13 = next(r for r in fine_rows if r["row"]=="same_board20_K13" and r["proto"]=="P2")["oracleK_km"]
mx13 = mix_summary["P2"]["oracleK_km"][0]
sb8  = next(r for r in fine_rows if r["row"]=="same_board20_K8" and r["proto"]=="P2")["oracleK_km"]
gap13 = mx13 - sb13
if gap13 > 0.15:
    T1 = (f"FINENESS BITES & SCALES with K: same-board K13 oracle km={sb13} << mixed K13 {mx13} "
          f"(gap {gap13:+.3f}); K8->K13 same-board trend {sb8}->{sb13}. TEST collapse generalizes -> "
          f"'same-batch discovery difficulty grows with K'.")
elif sb13 >= 0.6 and mx13 >= 0.6:
    T1 = (f"BOARD-18 IDIOSYNCRATIC: same-board K13 km={sb13} ~= mixed K13 {mx13} and both high; "
          f"same-board K8->K13 {sb8}->{sb13} stays high -> TEST is a hard single-batch instance, not K-scaling.")
else:
    T1 = (f"K-SCALING DOMINATES regardless of board mixing: same-board K13={sb13}, mixed K13={mx13}, "
          f"same-board K8={sb8} — both K13 rows low; difficulty is K-driven, board mixing secondary.")
print("\n[T1] " + T1)

# ============================================================================
# TASK 2 — BLOCK-SCALED HYBRID
# ============================================================================
print("\n" + "="*78 + "\nTASK 2 — BLOCK-SCALED HYBRID (alpha tuned on mavicAir2s, frozen, one-shot mavicAir2)\n" + "="*78)
TAPS_B = firwin(NTAPS, 0.9)
def decimate_B(iq):
    x = iq.astype(np.float32); pad = min(3*NTAPS, x.shape[1]-1)
    return filtfilt(TAPS_B, [1.0], x, axis=1, padlen=pad)[:, ::2]
manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
pat = re.compile(r'(.+?)_(\d+)_hover')
af_files = defaultdict(list)
for r in manifest["clean"]: af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
all_af = sorted(af_files, key=lambda t:(t.rsplit("_",1)[0], int(t.rsplit("_",1)[1])))
model_of = {a: pat.match(a+"_hover").group(1) for a in all_af}
MAVIC2  = [a for a in all_af if model_of[a]=="mavicAir2"]
MAVIC2S = [a for a in all_af if model_of[a]=="mavicAir2s"]
PCA_FIT_AF = [a for a in all_af if model_of[a] not in ("mavicAir2","mavicAir2s")]

def build_drff(af_list, cap=DRFF_CAP, seed=HARNESS_SEED):
    rng = np.random.default_rng(seed)
    Xt, af, seg, Ds, Cs, Hs = [], [], [], [], [], []
    gseg = 0
    for ai, a in enumerate(af_list):
        files = af_files[a][:]; rng.shuffle(files); units = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]): units.append((fn, si))
        rng.shuffle(units); got = 0; zc = {}
        for fn, si in units:
            if got >= cap: break
            if fn not in zc:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zc[fn] = dict(dec=decimate_B(z["iq"]), sb=z["seg_bounds"], D=str(z["D"]),
                              C=int(z["C"]), H=str(z["Height"]))
            zz = zc[fn]; off, ln = zz["sb"][si]; o2, l2 = off//2, ln//2; nw = l2//WIN
            if nw < 1: continue
            take = min(nw, cap-got, 12); segw = zz["dec"][:, o2:o2+l2]
            for k in range(take):
                Xt.append(W.standardize(segw[:, k*WIN:(k+1)*WIN].astype(np.float32)))
                af.append(ai); seg.append(gseg); Ds.append(zz["D"]); Cs.append(zz["C"]); Hs.append(zz["H"])
            got += take; gseg += 1
    return (np.stack(Xt).astype(np.float32), np.array(af), np.array(seg),
            np.array(Ds), np.array(Cs), np.array(Hs))

# ---- H1: PCA basis fit on pool-minus-mavicAir2s (never mavicAir2/mavicAir2s) ----
print(f"[H1] fitting frozen PCA-64 basis on {len(PCA_FIT_AF)} airframes (NO mavicAir2/mavicAir2s): {PCA_FIT_AF}")
Xt_pca, _, _, _, _, _ = build_drff(PCA_FIT_AF, cap=DRFF_CAP)
F_pca_fit = GC.extract512(model, Xt_pca)
PCA64 = PCA(n_components=64, random_state=0).fit(F_pca_fit)
print(f"     PCA-64 fit on {len(F_pca_fit)} windows; explained var ratio sum={PCA64.explained_variance_ratio_.sum():.3f}")

# ---- burst builders (matched-index) ----
def E1_index_bursts(af, D, C, seed=HARNESS_SEED):
    r = np.random.default_rng(seed); out = []
    for a in np.unique(af):
        idx = np.where(af==a)[0]; cells = defaultdict(list)
        for i in idx: cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx)//N
        for _ in range(nb):
            r.shuffle(ck); pick=[]; ci=0
            while len(pick)<N and ci<2000:
                c=ck[ci%len(ck)]
                if cells[c]: pick.append(cells[c][r.integers(len(cells[c]))])
                ci+=1
            if len(pick)==N: out.append((np.array(pick), a))
    return out
def E0_index_bursts(af, seg):
    out=[]
    for sg in np.unique(seg):
        idx=np.where(seg==sg)[0]
        for k in range(len(idx)//N): out.append((idx[k*N:(k+1)*N], af[idx[k]]))
    return out
def balance_idx(bursts, seed=0):
    r=np.random.default_rng(seed); lbl=np.array([b[1] for b in bursts])
    per=min(int((lbl==a).sum()) for a in np.unique(lbl)); keep=[]
    for a in np.unique(lbl):
        ii=np.where(lbl==a)[0]; r.shuffle(ii); keep+=ii[:per].tolist()
    return [bursts[i] for i in sorted(keep)]
def mod_bursts(feat, bursts): return np.stack([feat[idx].mean(0) for idx,_ in bursts])
def burst_modal(labelarr, bursts):
    return np.array([Counter(labelarr[idx]).most_common(1)[0][0] for idx,_ in bursts])

# ---- frozen hybrid construction ----
def block_scale(x, d):
    Z = StandardScaler().fit_transform(x)
    return Z / np.sqrt(d)
def make_hybrid(enc_bm, cls_bm, alpha, variant):
    enc_used = enc_bm if variant == "Ea" else PCA64.transform(enc_bm)
    d_enc = enc_used.shape[1]
    eb = block_scale(enc_used, d_enc); cb = block_scale(cls_bm, 19)
    return unit(np.concatenate([alpha*eb, (1-alpha)*cb], axis=1))

# ---- H2: tune alpha on mavicAir2s (condition-purity control) ----
print("\n[H2] tuning (variant, alpha) on mavicAir2s (7) — condition-disjoint E1, condition-purity control")
tune_start = now()
Xt2s, af2s, seg2s, D2s, C2s, H2s = build_drff(MAVIC2S)
enc2s = GC.extract512(model, Xt2s); cls2s = classical_matrix(Xt2s)
b2s = balance_idx(E1_index_bursts(af2s, D2s, C2s))
lab2s = np.array([b[1] for b in b2s]); K2s = len(np.unique(lab2s))
enc_bm2s = mod_bursts(enc2s, b2s); cls_bm2s = mod_bursts(cls2s, b2s)
modalD = burst_modal(D2s, b2s); modalH = burst_modal(H2s, b2s)
yi_af = np.unique(lab2s, return_inverse=True)[1]
yi_D  = np.unique(modalD, return_inverse=True)[1]
tune_rows = []
for variant in ("Ea", "Eb"):
    for alpha in ALPHAS:
        Hh = make_hybrid(enc_bm2s, cls_bm2s, alpha, variant)
        hm, per_r, _ = hdb_mean(Hh, lab2s, MCS_DRFF)
        km, spx = oracle_km_sp(Hh, lab2s, K2s)
        # condition-purity control on the oracle kmeans@K clustering
        kmpred = KMeans(K2s, n_init=10, random_state=0).fit_predict(Hh)
        ari_af = float(adjusted_rand_score(yi_af, kmpred))
        ari_D  = float(adjusted_rand_score(yi_D, kmpred))
        disq = ari_D > ari_af
        tune_rows.append(dict(variant=variant, alpha=alpha, hdb_mean=round(hm,3),
            oracleK_km=round(km,3), oracleK_sp=round(spx,3),
            ctrl_ari_airframe=round(ari_af,3), ctrl_ari_distanceD=round(ari_D,3),
            disqualified=bool(disq)))
        print(f"    {variant} a={alpha}: HDBmean={hm:.3f} oracle km={km:.3f}/sp={spx:.3f} | "
              f"ctrl airframe-ARI={ari_af:.3f} vs D-ARI={ari_D:.3f} {'DISQUALIFIED(channel)' if disq else 'ok'}")
# select: HDBSCAN mean primary, oracle km secondary, among non-disqualified
elig = [r for r in tune_rows if not r["disqualified"]]
pool_sel = elig if elig else tune_rows
best = max(pool_sel, key=lambda r: (r["hdb_mean"], r["oracleK_km"]))
FROZEN = (best["variant"], best["alpha"]); tune_end = now()
print(f"\n[H2] FROZEN choice = variant {FROZEN[0]}, alpha {FROZEN[1]} "
      f"(HDBmean={best['hdb_mean']}, oracle km={best['oracleK_km']}; selected {tune_start} .. {tune_end}, "
      f"BEFORE any mavicAir2 eval)")

# ============================================================================
# H3: ONE-SHOT mavicAir2 eval with frozen (variant, alpha)
# ============================================================================
print(f"\n[H3] ONE-SHOT mavicAir2 eval with FROZEN {FROZEN} (E1 + E0) @ {now()}")
Xt2, af2, seg2, D2, C2, H2 = build_drff(MAVIC2)
enc2 = GC.extract512(model, Xt2); cls2 = classical_matrix(Xt2)
K2 = len(np.unique(af2))
final_rows = []
for proto, bursts in (("E1", balance_idx(E1_index_bursts(af2, D2, C2))),
                      ("E0", balance_idx(E0_index_bursts(af2, seg2)))):
    lab = np.array([b[1] for b in bursts])
    enc_bm = mod_bursts(enc2, bursts); cls_bm = mod_bursts(cls2, bursts)
    Hh = make_hybrid(enc_bm, cls_bm, FROZEN[1], FROZEN[0])
    r = full_score(Hh, lab, K2, MCS_DRFF)
    final_rows.append(dict(method=f"hybrid-weighted({FROZEN[0]},a{FROZEN[1]})", protocol=proto,
        HDBSCAN_mean_ARI=r["hdb_mean"], oracleK8_kmeans=r["oracleK_km"], oracleK8_spectral=r["oracleK_sp"],
        kNN1=r["knn1"], gap=r["gap"], n_bursts=r["n_bursts"]))
    print(f"    mavicAir2 {proto}: HDBmean={r['hdb_mean']} oracle km={r['oracleK_km']}/sp={r['oracleK_sp']} "
          f"kNN1={r['knn1']} gap={r['gap']}")

# existing rows (from drff_headline_locked.csv) for side-by-side
EXISTING = []
hl = os.path.join(_SW, "results", "step7_test_and_harness", "drff_headline_locked.csv")
if os.path.exists(hl):
    for r in csv.DictReader(open(hl)):
        if r["protocol"] in ("E0","E1"):
            EXISTING.append(dict(method=r["method"], protocol=r["protocol"],
                HDBSCAN_mean_ARI=float(r["HDBSCAN_mean_ARI"]), oracleK8_kmeans=float(r["oracleK8_kmeans"]),
                oracleK8_spectral=float(r["oracleK8_spectral"]), kNN1=float(r["kNN1"]), gap=float(r["gap"]),
                n_bursts=int(r["n_bursts"])))

# ---- in-domain preservation check: one mixed-board K13 slice, SAME hybrid construction ----
print(f"\n[H3b] in-domain preservation: mixed-board K13 slice (seed 501), P2, same hybrid construction")
sl = GC.scatter_slice(dev_tx, 13, 501)
enc_id, cls_id, lab_id = [], [], []
for di, tx in enumerate(sl):
    t = GC.consec_windows(TXD, tx, WIN_CACHE); Wn = t.shape[0]
    e = dev512[tx]["emb"]; c = classical_matrix(t)
    rx, date = dev512[tx]["rx"], dev512[tx]["date"]
    be = bursts_p2_multidate(e, rx, date, seed=501)[0]
    bc = bursts_p2_multidate(c, rx, date, seed=501)[0]     # same seed/rx/date -> aligned bursts
    if be is None or bc is None: continue
    enc_id.append(be); cls_id.append(bc); lab_id.append(np.full(len(be), di))
enc_id = np.concatenate(enc_id); cls_id = np.concatenate(cls_id); lab_id = np.concatenate(lab_id)
Hh_id = make_hybrid(enc_id, cls_id, FROZEN[1], FROZEN[0])
r_hyb_id = full_score(Hh_id, lab_id, 13, MCS_WISIG)
r_enc_id = full_score(unit(enc_id), lab_id, 13, MCS_WISIG)
print(f"    in-domain encoder-only: oracle km={r_enc_id['oracleK_km']}/sp={r_enc_id['oracleK_sp']} HDBmean={r_enc_id['hdb_mean']}")
print(f"    in-domain HYBRID:       oracle km={r_hyb_id['oracleK_km']}/sp={r_hyb_id['oracleK_sp']} HDBmean={r_hyb_id['hdb_mean']}")
drop = r_enc_id["oracleK_km"] - r_hyb_id["oracleK_km"]
indomain_verdict = ("PRESERVES in-domain" if drop <= 0.10 else "DEGRADES in-domain") + \
                   f" (encoder km {r_enc_id['oracleK_km']} -> hybrid {r_hyb_id['oracleK_km']}, drop {drop:+.3f})"

# ---- H4 verdict ----
he1 = next(r for r in final_rows if r["protocol"]=="E1")
best_existing_hdb = 0.245   # classical-19 E1 HDBSCAN(mean)
best_existing_orsp = 0.541  # hybrid PCA-64+19 E1 oracle sp
lift_hdb = he1["HDBSCAN_mean_ARI"] - best_existing_hdb
lift_orsp = he1["oracleK8_spectral"] - best_existing_orsp
if lift_hdb >= 0.03 or lift_orsp >= 0.03:
    kind = "REAL LIFT"
elif lift_hdb >= -0.03 and lift_orsp >= -0.03:
    kind = "TIE"
else:
    kind = "WORSE"
H4 = (f"{kind}: weighted hybrid mavicAir2 E1 HDBmean={he1['HDBSCAN_mean_ARI']} (vs best existing 0.245, "
      f"{lift_hdb:+.3f}), oracle sp={he1['oracleK8_spectral']} (vs 0.541, {lift_orsp:+.3f}), "
      f"km={he1['oracleK8_kmeans']}. In-domain: {indomain_verdict}.")
print("\n[H4] " + H4)

# ============================================================================
# SAVE
# ============================================================================
def wcsv(fn, rows, fields=None):
    if not rows: return
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        keys = fields or sorted(set().union(*[r.keys() for r in rows]))
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
wcsv("fineness_k13.csv", fine_rows,
     ["row","proto","K","n_dev","hdb_mean","oracleK_km","oracleK_sp","knn1","gap","hdb_std","km_std","K_est_mid","n_bursts"])
wcsv("hybrid_alpha_dev.csv", tune_rows)
wcsv("hybrid_final.csv", final_rows + EXISTING)

report = dict(
    task1=dict(hdbscan_grid=MCS_WISIG, rows=fine_rows, mixed_summary=mix_summary, verdict=T1,
        note="same-board = ALL 13 board-20 DEV devices (one slice); mixed = 3x scatter_slice(13) over "
             "boards 4/5/16/20 (seeds 501-503); K8 = 2 draws of 8-of-13 board-20 (seeds 511-512). Head-free 512-D."),
    task2=dict(
        H1_pca=dict(fit_airframes=PCA_FIT_AF, excluded="mavicAir2 + mavicAir2s (never fit on eval/tune sets)",
                    n_fit_windows=int(len(F_pca_fit)), explained_var=float(PCA64.explained_variance_ratio_.sum()),
                    fit_source="build_drff(PCA_FIT_AF, seed 777) frozen 512-D features"),
        H2_tuning=dict(tuning_set="mavicAir2s (7)", alpha_grid=ALPHAS, variants=["Ea=raw512","Eb=PCA64"],
                       selection="HDBSCAN(mean) primary, oracle-K@7 km secondary, among non-disqualified",
                       condition_control="disqualify if oracle-kmeans@7 ARI vs distance-D > ARI vs airframe",
                       rows=tune_rows, frozen_variant=FROZEN[0], frozen_alpha=FROZEN[1],
                       selected_at=tune_end, selected_before_eval=True),
        H3_mavicAir2=dict(frozen=list(FROZEN), rows=final_rows, existing_rows=EXISTING,
                          in_domain=dict(encoder=r_enc_id, hybrid=r_hyb_id, verdict=indomain_verdict)),
        H4_verdict=H4),
    guardrails="frozen encoder head-free; no gradient; locked harness (seed 777, km/sp separate, HDBSCAN "
               "mean over fixed mcs grid, locked noise); TEST/board-18 CLOSED; M100 untouched; mavicAir2 "
               "touched ONCE (Task 2); fixed 2-variant x 5-alpha grid, no expansion.")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)
print("\nsaved -> results/step8_disentangle_hybrid/ (fineness_k13.csv, hybrid_alpha_dev.csv, hybrid_final.csv, report.json)")
print("CHECKPOINT — frozen; locked harness; TEST/M100 sealed; mavicAir2 touched once; battery complete.")
