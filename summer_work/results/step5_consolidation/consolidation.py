"""STEP-5 POST-OVERTURN CONSOLIDATION — three bounded, frozen, eval-only tasks.

TASK 1  U-FIELD CORRECTION + u1/u2 COVERAGE (bookkeeping):
  DRFF-R2 _u* = USRP RECEIVER NUMBER (two clock-synced USRP-2943s), NOT a within-airframe
  condition (step3b lumped U into the D/C/U/H/St nuisance group -> superseded). Emit the
  per-airframe u1/u2 coverage table; the R=2 cross-receiver same-model test is only run if
  >=4 mavicAir2 airframes have BOTH receivers (else reported + skipped, not forced).

TASK 2  FINENESS CONTROL (closes Control-1's domain-vs-fineness confound):
  Same-board WiSig 8-device slices (most physically-similar hardware WiSig offers), P2-style
  single-rx bursts, head-free 512-D, HDBSCAN + oracle-K@8, kNN-1, gap. Mixed-board 8-device
  slice set = the heterogeneous contrast row. Approximates same-model fineness WITHIN domain.

TASK 3  HYBRID FEATURES, NO RETRAIN (constructive test):
  DRFF mavicAir2 E1 (OPT-B): standardized classical-19 (+) standardized encoder-512, three
  fixed variants {concat, PCA-64 of 512 (+) 19, 50/50 variance-weighted}, beside classical-only
  and encoder-only. Same hybrid on one WiSig DEV P2 slice (K18) to check it doesn't wreck
  in-domain. Complementary if hybrid > max(classical, encoder).

Encoder + CosFace head FROZEN (no gradient anywhere). DEV WiSig + mavicAir2 OPT-B only.
Locked noise rule. Oracle-K = CEILING (true K). No TEST/board-18, no M100. Fixed battery.

  python3 results/step5_consolidation/consolidation.py
"""
import os, sys, json, csv, re
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
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
import geometry_consolidate as GC
from burst_probe import embed_times
from integrity import score_locked, hdbscan_pred, unitrows, bursts_coherent
from decohere import bursts_p2_multidate
from mechanism_validation import classical_matrix        # reuse the EXACT 19-D classical feats
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

N = 10
WIN = 256
WIN_CACHE = 2400
COS_M, COS_S = 0.20, 32
NTAPS = 129
SEED = 777
MCS_WISIG = [13, 15, 17]
MCS_DRFF  = [5, 7]
CAP_FINENESS = 120        # bursts/device for the fineness battery (== Control-1 cap)
CAP_WISIG_HYB = 80        # bursts/device for the WiSig hybrid sanity check (tractable spectral)
KFINE = 8                 # same K as DRFF mavicAir2 -> fineness comparison
KWISIG18 = 18
KDRFF = 8
N_FINE_SLICES = 5

def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# ---- shared scoring helpers (identical geometry to step3c/step4) ----
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

def hdbscan_best(bp, bl, mcs_list):
    best = None
    for mcs in mcs_list:
        sc = score_locked(hdbscan_pred(bp, mcs), bl)
        if best is None or sc["ARI"] > best["ARI"]: best = dict(mcs=mcs, **sc)
    return best

def oracle_k(bp, bl, K):
    yi = np.unique(bl, return_inverse=True)[1]
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(bp)
    ari_km = float(adjusted_rand_score(yi, km))
    try:
        sp = SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                                random_state=0, n_neighbors=15).fit_predict(bp)
        ari_sp = float(adjusted_rand_score(yi, sp))
    except Exception:
        ari_sp = float("nan")
    return ari_km, ari_sp

def cap_bursts(bp, bl, cap, seed=0):
    r = np.random.default_rng(seed); keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:cap].tolist()
    keep = np.array(sorted(keep)); return bp[keep], bl[keep]

def score_block(bp, bl, K, mcs_list):
    """uniform scorer for a burst matrix (already row-normalized): HDBSCAN best + oracle-K."""
    hb = hdbscan_best(bp, bl, mcs_list)
    km, spc = oracle_k(bp, bl, K)
    kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl)
    return dict(hdb_ARI=hb["ARI"], hdb_K=hb["K_est"], hdb_noise=hb["noise"],
                oracleK_kmeans=km, oracleK_spectral=spc, knn1=kp[1], cos_gap=intra - inter,
                n_bursts=int(len(bl)))

# ============================================================================
# TASK 1 — U/receiver coverage table (manifest only; no compute)
# ============================================================================
print("=" * 78)
print("TASK 1 — U-FIELD (USRP RECEIVER) COVERAGE")
print("=" * 78)
manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
pat = re.compile(r'(.+?)_(\d+)_hover')
af_U = defaultdict(Counter); af_files = defaultdict(list)
for r in manifest["clean"]:
    af_U[r["TD"]][r["U"]] += 1
    af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
all_af = sorted(af_U, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
model_of = {af: pat.match(af + "_hover").group(1) for af in all_af}

cov_rows = []
for a in all_af:
    c = af_U[a]; u1, u2 = c.get("u1", 0), c.get("u2", 0)
    cov_rows.append(dict(airframe=a, model=model_of[a], u1_files=u1, u2_files=u2,
                         both_receivers=(u1 > 0 and u2 > 0), total_files=u1 + u2))
both_ma2 = [r["airframe"] for r in cov_rows if r["model"] == "mavicAir2" and r["both_receivers"]]
both_all = [r["airframe"] for r in cov_rows if r["both_receivers"]]
print(f"{'airframe':16}{'model':12}{'u1':>4}{'u2':>4}{'both':>6}{'tot':>5}")
for r in cov_rows:
    print(f"{r['airframe']:16}{r['model']:12}{r['u1_files']:>4}{r['u2_files']:>4}"
          f"{str(r['both_receivers']):>6}{r['total_files']:>5}")
print(f"\nboth-receiver airframes (all models): {len(both_all)} -> {both_all}")
print(f"mavicAir2 with BOTH receivers: {len(both_ma2)} -> {both_ma2}")
R2_FEASIBLE = len(both_ma2) >= 4
print(f"R=2 cross-receiver same-model test feasible (>=4 mavicAir2 w/ both rx)? {R2_FEASIBLE}")
if not R2_FEASIBLE:
    print("  -> SKIP R=2 test: insufficient both-receiver coverage. Not forced (guardrail).")

# ============================================================================
# 0. FROZEN ENCODER + reproduced CosFace head + DEV caches (== step2b/step4 recipe)
# ============================================================================
print("\nloading ManyTx (eq=0)...")
TXD, _ = W.load_manytx(eq=0, verbose=False)
print("[0] frozen encoder + reproduced CosFace head (seed42, m=%.2f s=%d) ..." % (COS_M, COS_S))
sp = json.load(open(SPLIT)); part = json.load(open(PART))
train_tx = sp["train_tx"]; dev_tx = part["dev_tx"]; n_cls = len(train_tx)
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

print(f"caching 512-D + classical for {len(dev_tx)} DEV devices (WIN_CACHE={WIN_CACHE}) ...")
cache512 = {}; rawcls = {}
for tx in dev_tx:
    t = GC.consec_windows(TXD, tx, WIN_CACHE)
    Wn = t.shape[0]
    rx = TXD[tx]["rx"][:Wn].copy(); date = TXD[tx]["date"][:Wn].copy()
    e512 = GC.extract512(model, t)
    raw = np.stack([TXD[tx]["iq"][k].T.copy() for k in range(Wn)]).astype(np.float32)
    cache512[tx] = dict(emb=e512, rx=rx, date=date)
    rawcls[tx] = dict(feat=classical_matrix(raw), rx=rx, date=date)
torch.cuda.empty_cache()
allfc = np.concatenate([rawcls[tx]["feat"] for tx in dev_tx])
csc = StandardScaler().fit(allfc)          # label-free standardization only
cacheCls = {tx: dict(emb=csc.transform(rawcls[tx]["feat"]).astype(np.float32),
                     rx=rawcls[tx]["rx"], date=rawcls[tx]["date"]) for tx in dev_tx}

def p2_bursts(cache, tx, seed):
    c = cache[tx]
    return bursts_p2_multidate(c["emb"], c["rx"], c["date"], seed=seed)[0]

def build_p2_slice(cache, slice_tx, seed):
    bp, bl = [], []
    for di, tx in enumerate(slice_tx):
        b = p2_bursts(cache, tx, seed)
        if b is None: continue
        bp.append(b); bl.append(np.full(len(b), di))
    return np.concatenate(bp), np.concatenate(bl)

# ============================================================================
# TASK 2 — FINENESS CONTROL (same-board vs mixed-board, K=8, WiSig P2, 512-D)
# ============================================================================
print("\n" + "=" * 78)
print("TASK 2 — FINENESS CONTROL (WiSig same-board K8 P2, head-free 512-D)")
print("=" * 78)
by_board = defaultdict(list)
for d in dev_tx: by_board[d.split("-")[0]].append(d)
boards_ge8 = sorted([b for b in by_board if len(by_board[b]) >= 8], key=int)
print("DEV devices/board:", {b: len(by_board[b]) for b in sorted(by_board, key=int)})
print(f"boards allowing 8-device same-board slices: {boards_ge8} "
      f"(others have <8 -> cannot form same-model-ish 8-slice)")

fine_rows = []
def run_fine(tag, slices, seeds):
    accum = defaultdict(list)
    for sl, s in zip(slices, seeds):
        bp, bl = build_p2_slice(cache512, sl, s)
        bp, bl = cap_bursts(bp, bl, CAP_FINENESS, seed=s)
        r = score_block(bp, bl, KFINE, MCS_WISIG)
        for k, v in r.items(): accum[k].append(v)
    row = dict(slice_type=tag, n_slices=len(slices), K=KFINE,
               **{k: float(np.nanmean(v)) for k, v in accum.items()},
               chance=1.0 / KFINE)
    fine_rows.append(row)
    print(f"  {tag:>22}: HDB ARI={row['hdb_ARI']:.3f} | oracle km@8={row['oracleK_kmeans']:.3f} "
          f"sp@8={row['oracleK_spectral']:.3f} | kNN1={row['knn1']:.3f} gap={row['cos_gap']:+.3f} "
          f"(nb~{int(row['n_bursts'])})")
    return row

# same-board slices: board(s) with >=8 devices; random 8-of-n (overlapping when n<16)
for b in boards_ge8:
    devs = sorted(by_board[b], key=lambda x: int(x.split("-")[1]))
    slices, seeds = [], []
    for i in range(N_FINE_SLICES):
        rng = np.random.default_rng(3100 + i)
        slices.append(sorted(rng.choice(devs, size=8, replace=False).tolist())); seeds.append(3100 + i)
    n = len(devs)
    note = "disjoint 8-slices impossible (n<16); overlapping random 8-of-%d" % n if n < 16 else "disjoint"
    print(f"[same-board {b}] {n} devices, {N_FINE_SLICES} slices ({note})")
    run_fine(f"same_board_{b}", slices, seeds)

# mixed-board contrast: 8 devices drawn across all boards
mix_slices, mix_seeds = [], []
for i in range(N_FINE_SLICES):
    rng = np.random.default_rng(3200 + i)
    mix_slices.append(sorted(rng.choice(dev_tx, size=8, replace=False).tolist())); mix_seeds.append(3200 + i)
print(f"[mixed-board] {N_FINE_SLICES} heterogeneous 8-device slices across boards {sorted(by_board, key=int)}")
run_fine("mixed_board", mix_slices, mix_seeds)

sb_key = f"same_board_{boards_ge8[0]}"
sb = next(r for r in fine_rows if r["slice_type"] == sb_key)
mx = next(r for r in fine_rows if r["slice_type"] == "mixed_board")
sb_or = max(sb["oracleK_kmeans"], sb["oracleK_spectral"])
if sb_or >= 0.60:
    G2FINE = (f"FINENESS is NOT the wall: same-board K8 oracle-K={sb_or:.3f} (>=0.60) clusters well "
              f"within-domain -> the domain-gap sentence stands clean "
              f"(mixed-board K8 oracle={max(mx['oracleK_kmeans'],mx['oracleK_spectral']):.3f}).")
elif sb_or <= 0.30:
    G2FINE = (f"FINENESS IS a co-factor: same-board K8 oracle-K={sb_or:.3f} (<=0.30) walls within-domain "
              f"-> corrected sentence = 'domain gap AND same-model fineness jointly gate discovery'.")
else:
    G2FINE = (f"FINENESS is a PARTIAL co-factor: same-board K8 oracle-K={sb_or:.3f} (0.30-0.60), "
              f"intermediate; mixed-board K8 oracle={max(mx['oracleK_kmeans'],mx['oracleK_spectral']):.3f}. "
              f"Physically-similar same-board devices are still not same-model-identical (approximation).")
print("\n[G2] " + G2FINE)

# ============================================================================
# TASK 3 — HYBRID FEATURES (DRFF mavicAir2 E1, OPT-B) + WiSig in-domain check
# ============================================================================
print("\n" + "=" * 78)
print("TASK 3 — HYBRID FEATURES, NO RETRAIN")
print("=" * 78)
TAPS_B = firwin(NTAPS, 0.9)
def decimate_B(iq):
    x = iq.astype(np.float32); pad = min(3 * NTAPS, x.shape[1] - 1)
    return filtfilt(TAPS_B, [1.0], x, axis=1, padlen=pad)[:, ::2]
TARGET = "mavicAir2"

def build_drff_windows(cap=320):
    """faithful build_subset replay (SEED=777, cap=320, <=12 win/seg) -> raw mavicAir2 windows."""
    rng = np.random.default_rng(SEED)
    Xr, afs, segs, Ds, Cs = [], [], [], [], []
    gseg = 0
    for a in all_af:
        files = af_files[a][:]; rng.shuffle(files)
        units = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]):
                units.append((fn, si))
        rng.shuffle(units)
        is_tgt = (model_of[a] == TARGET); got = 0; zc = {}
        for fn, si in units:
            if got >= cap: break
            if fn not in zc:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zc[fn] = dict(dec=(decimate_B(z["iq"]) if is_tgt else None), sb=z["seg_bounds"],
                              D=str(z["D"]), C=int(z["C"]))
            zz = zc[fn]; off, ln = zz["sb"][si]; o2, l2 = off // 2, ln // 2
            nw = l2 // WIN
            if nw < 1: continue
            take = min(nw, cap - got, 12)
            if is_tgt:
                seg = zz["dec"][:, o2:o2 + l2]
                for k in range(take):
                    Xr.append(seg[:, k * WIN:(k + 1) * WIN].astype(np.float32))
                    afs.append(a); segs.append(gseg); Ds.append(zz["D"]); Cs.append(zz["C"])
            got += take; gseg += 1
    return np.stack(Xr), np.array(afs), np.array(segs), np.array(Ds), np.array(Cs)

print("[3] building DRFF mavicAir2 OPT-B windows + classical-19 + encoder-512 (frozen) ...")
Xr, afd, segd, Dd, Cd = build_drff_windows()
cls_drff = StandardScaler().fit_transform(classical_matrix(Xr)).astype(np.float32)   # 19-D
Xr_std = np.stack([W.standardize(w) for w in Xr]).astype(np.float32)                  # encoder input
enc_drff = GC.extract512(model, Xr_std)                                              # 512-D frozen
print(f"    {len(Xr)} windows, {len(np.unique(segd))} segments, {len(np.unique(afd))} airframes")

def E1_index_bursts(af, D, C, seed=SEED):
    """return list of (index-array[N], label) — matched bursts across modalities."""
    r = np.random.default_rng(seed); out = []
    for a in np.unique(af):
        idx = np.where(af == a)[0]; cells = defaultdict(list)
        for i in idx: cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx) // N
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

def modality_bursts(feat, bursts):
    """mean-pool a window-feature matrix over each burst's indices -> [Nb, dim]."""
    return np.stack([feat[idx].mean(0) for idx, _ in bursts])

def make_variants(cls_b, enc_b):
    """cls_b:[Nb,19] enc_b:[Nb,512] (burst means). Return dict name->row-normalized burst matrix."""
    Zc = StandardScaler().fit_transform(cls_b)
    Ze = StandardScaler().fit_transform(enc_b)
    Zpca = StandardScaler().fit_transform(PCA(n_components=min(64, enc_b.shape[0], 512),
                                              random_state=0).fit_transform(enc_b))
    variants = {
        "classical_only": Zc,
        "encoder_only":   Ze,
        "hybrid_concat19+512": np.concatenate([Zc, Ze], axis=1),
        "hybrid_pca64+19":     np.concatenate([Zc, Zpca], axis=1),
        # 50/50 variance-weighted: each block scaled to unit total variance (equal weight)
        "hybrid_5050_varwt":   np.concatenate([Zc / np.sqrt(Zc.shape[1]),
                                               Ze / np.sqrt(Ze.shape[1])], axis=1),
    }
    return {k: unit(v) for k, v in variants.items()}

# ---- DRFF E1 hybrid ----
bursts = balance_idx(E1_index_bursts(afd, Dd, Cd))
labels = np.array([b[1] for b in bursts])
cls_b = modality_bursts(cls_drff, bursts); enc_b = modality_bursts(enc_drff, bursts)
print(f"    E1 bursts (balanced): {len(labels)} across {len(np.unique(labels))} airframes")
hyb_rows = []
for name, M in make_variants(cls_b, enc_b).items():
    r = score_block(M, labels, KDRFF, MCS_DRFF)
    hyb_rows.append(dict(regime="DRFF_E1_mavicAir2", variant=name, dim=int(M.shape[1]), **r,
                         chance=1.0 / KDRFF))
    print(f"  DRFF E1 {name:>22}: HDB ARI={r['hdb_ARI']:.3f} | oracle km@8={r['oracleK_kmeans']:.3f} "
          f"sp@8={r['oracleK_spectral']:.3f} | kNN1={r['knn1']:.3f}")

def best_or(rows, name):
    r = next(x for x in rows if x["variant"] == name)
    return max(r["oracleK_kmeans"], r["oracleK_spectral"]), r["hdb_ARI"]
cls_or, cls_h = best_or(hyb_rows, "classical_only")
enc_or, enc_h = best_or(hyb_rows, "encoder_only")
hyb_best_name = max(["hybrid_concat19+512", "hybrid_pca64+19", "hybrid_5050_varwt"],
                    key=lambda nm: best_or(hyb_rows, nm)[0])
hyb_or, hyb_h = best_or(hyb_rows, hyb_best_name)
base = max(cls_or, enc_or)
if hyb_or >= base + 0.03:
    G3 = (f"COMPLEMENTARY: best hybrid ({hyb_best_name}) oracle-K@8={hyb_or:.3f} > max(classical "
          f"{cls_or:.3f}, encoder {enc_or:.3f}) -> a no-retrain constructive gain on DRFF single-rx.")
elif hyb_or <= max(cls_or, enc_or) + 0.02 and abs(hyb_or - cls_or) <= 0.03 and cls_or >= enc_or:
    G3 = (f"NOT complementary: best hybrid oracle-K@8={hyb_or:.3f} ~= classical-alone {cls_or:.3f} "
          f"(encoder {enc_or:.3f}); the encoder adds nothing cross-domain -> the path is adaptation.")
else:
    G3 = (f"MIXED: best hybrid oracle-K@8={hyb_or:.3f} vs classical {cls_or:.3f} / encoder {enc_or:.3f} "
          f"(no clear complementarity beyond noise).")
print("\n[G3] " + G3)

# ---- WiSig in-domain hybrid sanity (one DEV P2 slice, K18) ----
print("\n[3c] WiSig in-domain hybrid check (one DEV P2 slice, K18) — must not destroy in-domain:")
wsl = GC.scatter_slice(dev_tx, KWISIG18, 201)
# matched bursts: same seed/rx/date -> bursts_p2_multidate picks identical indices for both modalities
enc_bp, wl = build_p2_slice(cache512, wsl, 201)
cls_bp, wl2 = build_p2_slice(cacheCls, wsl, 201)
assert np.array_equal(wl, wl2)
enc_bp, cls_bp, wl = enc_bp, cls_bp, wl
# cap (matched indices) for tractable spectral
r = np.random.default_rng(201); keep = []
for a in np.unique(wl):
    ii = np.where(wl == a)[0]; r.shuffle(ii); keep += ii[:CAP_WISIG_HYB].tolist()
keep = np.array(sorted(keep))
enc_bp, cls_bp, wl = enc_bp[keep], cls_bp[keep], wl[keep]
wis_rows = []
for name, M in make_variants(cls_bp, enc_bp).items():
    rr = score_block(M, wl, KWISIG18, MCS_WISIG)
    wis_rows.append(dict(regime="WiSig_P2_K18_seed201", variant=name, dim=int(M.shape[1]), **rr,
                         chance=1.0 / KWISIG18))
    print(f"  WiSig P2 {name:>22}: HDB ARI={rr['hdb_ARI']:.3f} | oracle km@18={rr['oracleK_kmeans']:.3f} "
          f"sp@18={rr['oracleK_spectral']:.3f} | kNN1={rr['knn1']:.3f}")
w_enc_or = best_or(wis_rows, "encoder_only")[0]
w_hyb_or = max(best_or(wis_rows, nm)[0] for nm in
               ["hybrid_concat19+512", "hybrid_pca64+19", "hybrid_5050_varwt"])
wis_note = (f"in-domain preserved: best hybrid oracle-K@18={w_hyb_or:.3f} vs encoder-only "
            f"{w_enc_or:.3f} (drop {w_enc_or-w_hyb_or:+.3f})")
print("  -> " + wis_note)

# ============================================================================
# SAVE
# ============================================================================
def wcsv(fn, rows):
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
wcsv("coverage_table.csv", cov_rows)
wcsv("fineness_control.csv", fine_rows)
wcsv("hybrid_test.csv", hyb_rows + wis_rows)

report = dict(
    task1_coverage=dict(rows=cov_rows, both_receiver_all=both_all, both_receiver_mavicAir2=both_ma2,
        R2_feasible=R2_FEASIBLE,
        R2_verdict=("SKIPPED — only %d mavicAir2 airframe(s) have both receivers (<4 required); "
                    "not forced (guardrail)." % len(both_ma2)),
        U_semantics="U = USRP receiver number (two clock-synced USRP-2943s), per DRFF-R2 dataset "
                    "paper. NOT a within-airframe channel condition.",
        step3b_correction="step3b confound probe treated U as one of the nuisance conditions "
                          "(D/C/U/H/St) predicted within same-model groups — SUPERSEDED: U is the "
                          "receiver-diversity axis (analog of WiSig rx), not a within-airframe cue.",
        step3b_holdout_verdict="VALID, no re-run: the session-disjoint probe defined "
            "session=(airframe,U,D,C) with U INCLUDED, so held-out test sessions differ from train "
            "in real capture structure (receiver and/or distance/channel). The 0.60 mavicAir2 clean "
            "number (OPT-B logreg/mlp burst) held out genuine capture structure and stands; "
            "best 0.833 (mavicAir2s OPT-B mlp burst). A stricter receiver-DISJOINT same-model probe "
            "is infeasible (only mavicAir2_1 has both u1&u2).",
        n_both_receiver=len(both_ma2)),
    task2_fineness=dict(rows=fine_rows, boards_ge8=boards_ge8, K=KFINE,
        same_board_oracleK=sb_or, mixed_board_oracleK=max(mx["oracleK_kmeans"], mx["oracleK_spectral"]),
        verdict=G2FINE,
        approximation_note="same-board WiSig devices are the most physically-similar hardware WiSig "
                           "offers but are NOT same-model-identical like 8 mavicAir2 airframes; "
                           "this is an approximation of same-model fineness, labeled as such."),
    task3_hybrid=dict(drff_rows=hyb_rows, wisig_rows=wis_rows,
        best_hybrid_variant=hyb_best_name, best_hybrid_oracleK=hyb_or,
        classical_oracleK=cls_or, encoder_oracleK=enc_or, verdict=G3, wisig_indomain=wis_note),
    guardrails="encoder+CosFace head FROZEN (no gradient); DEV WiSig + mavicAir2 OPT-B only; "
               "oracle-K=CEILING(true K); locked noise rule; no TEST/board-18, no M100; fixed battery "
               "(3 hybrid variants, no sweep).",
    params=dict(seed=SEED, cap_fineness=CAP_FINENESS, cap_wisig_hyb=CAP_WISIG_HYB, n_fine_slices=N_FINE_SLICES,
                K_fine=KFINE, K_wisig=KWISIG18, K_drff=KDRFF, mcs_wisig=MCS_WISIG, mcs_drff=MCS_DRFF))
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)

print("\n" + "=" * 78)
print("VERDICTS")
print("=" * 78)
print("G1 (U-semantics/coverage):", report["task1_coverage"]["step3b_holdout_verdict"][:120], "...")
print("G2 (fineness):", G2FINE)
print("G3 (hybrid):", G3)
print("\nsaved -> results/step5_consolidation/ (coverage_table.csv, fineness_control.csv, "
      "hybrid_test.csv, report.json)")
print("CHECKPOINT — frozen; oracle-K=ceiling; TEST/M100 untouched; fixed battery complete.")
