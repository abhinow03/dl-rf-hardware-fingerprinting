"""STEP-3c DRFF-R2 DISCOVERY-GAP DIAGNOSIS (mavicAir2, confound-clean, OPT-B).
Supervised probe passes (session-disjoint 0.60) but unsupervised discovery fails (ARI~0.07,
K collapses). Is this a FIXABLE geometry problem (linear-but-not-density-separable) or a WALL
(single-rx has no decohering axis)? Fixed battery, DIAGNOSIS not tuning. Encoder FROZEN, head
NOT refit (CosFace embeddings reused from cache), eval-only. Test-3 uses NO gradient training:
transforms are label-free or fit on WiSig-TRAIN devices only. Oracle-K k-means = ceiling, not a
discovery result.

  python3 results/step3c_discovery_gap/discovery_gap.py
"""
import os, sys, json, csv
from collections import defaultdict
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity"), _HERE):
    if p not in sys.path: sys.path.insert(0, p)
from shared import RFEncoder
import wisig_manytx as W
import geometry_consolidate as GC
from integrity import score_locked, hdbscan_pred
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

SEED = 777
N = 10
MCS = [5, 7]
KTRUE = 8
OUT = _HERE
FEAT = os.path.join(_SW, "results", "step3b_drff_smoke", "features_B.npz")

def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# ---- load cached OPT-B mavicAir2 windows ----
z = np.load(FEAT, allow_pickle=True)
model = z["model"].astype(str); m = model == "mavicAir2"
af = z["af"].astype(str)[m]; seg = z["seg"][m]; D = z["D"].astype(str)[m]; C = z["C"].astype(str)[m]
E = {"512D": z["F512"][m], "CosFace": z["FCOS"][m], "128D": z["F128"][m]}
print(f"mavicAir2 OPT-B: {m.sum()} windows, 8 airframes, {len(np.unique(seg))} segments")

# ---- burst builders (E0 coherent within-segment; E1 across conditions) ----
def build_E0(emb):
    bp, bl = [], []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        for k in range(len(idx)//N):
            ch = idx[k*N:(k+1)*N]; bp.append(emb[ch].mean(0)); bl.append(af[ch][0])
    return np.array(bp), np.array(bl)
def build_E1(emb, seed=SEED):
    r = np.random.default_rng(seed); bp, bl = [], []
    for a in np.unique(af):
        idx = np.where(af == a)[0]
        cells = defaultdict(list)
        for i in idx: cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx)//N
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < N and ci < 2000:
                c = ck[ci % len(ck)]
                if cells[c]: pick.append(cells[c][r.integers(len(cells[c]))])
                ci += 1
            if len(pick) == N: bp.append(emb[pick].mean(0)); bl.append(a)
    return np.array(bp), np.array(bl)
def balance(bp, bl, seed=0):
    r = np.random.default_rng(seed); per = min(int((bl==a).sum()) for a in np.unique(bl))
    keep = []
    for a in np.unique(bl):
        ii = np.where(bl==a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    keep = np.array(sorted(keep)); return bp[keep], bl[keep]

def knn_purity(X, y, ks=(1,5)):
    Xn = unit(X); nn = NearestNeighbors(n_neighbors=max(ks)+1).fit(Xn)
    _, ind = nn.kneighbors(Xn)
    out = {}
    yi = np.unique(y, return_inverse=True)[1]
    for k in ks:
        nb = ind[:, 1:k+1]                    # drop self
        same = (yi[nb] == yi[:, None]).mean()
        out[k] = float(same)
    return out

def cos_gap(X, y):
    Xn = unit(X); yi = np.unique(y, return_inverse=True)[1]
    intra, inter = [], []
    for a in np.unique(yi):
        mm = yi == a
        if mm.sum() < 2: continue
        Cc = Xn[mm] @ Xn[mm].T; iu = np.triu_indices(mm.sum(), 1)
        intra.append(Cc[iu].mean()); inter.append((Xn[mm] @ Xn[~mm].T).mean())
    return float(np.mean(intra)), float(np.mean(inter))

def disc(bp, bl):
    """best HDBSCAN over mcs on L2-normalized burst-means -> (ARI, K, noise, mcs)."""
    best = None
    for mcs in MCS:
        sc = score_locked(hdbscan_pred(unit(bp), mcs), bl)
        if best is None or sc["ARI"] > best["ARI"]: best = dict(mcs=mcs, **sc)
    return best

rows_geo, rows_e, rows_fix = [], [], []

# =========================== TEST 1 — geometry (E0, per embedding) ===========================
print("\n[TEST 1] geometry probe (E0 coherent burst-means):")
CEILING = 0.60   # session-disjoint supervised mavicAir2 OPT-B burst acc (Step-3b), chance 0.125
print(f"  (a) supervised readout ceiling = {CEILING:.2f} burst acc (session-disjoint, chance 0.125)")
for name in ("512D", "CosFace", "128D"):
    bp, bl = balance(*build_E0(E[name]))
    kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl); d = disc(bp, bl)
    rows_geo.append(dict(emb=name, knn1=kp[1], knn5=kp[5], intra_cos=intra, inter_cos=inter,
                         cos_gap=intra-inter, ARI_hdbscan=d["ARI"], K=d["K_est"], noise=d["noise"]))
    print(f"  {name:>8}: kNN1={kp[1]:.3f} kNN5={kp[5]:.3f} (chance {1/8:.3f}) | "
          f"intra={intra:.3f} inter={inter:.3f} gap={intra-inter:+.3f} | "
          f"HDBSCAN ARI={d['ARI']:.3f} K={d['K_est']}(t8) noise={d['noise']:.3f}")

# =========================== TEST 2 — E0 vs E1 (the P2 analog) ===============================
print("\n[TEST 2] single-rx decoherence axis (E0 coherent vs E1 multi-condition):")
for name in ("512D", "CosFace"):
    for proto, fn in (("E0_coherent", build_E0), ("E1_multicond", build_E1)):
        bp, bl = balance(*fn(E[name]))
        kp = knn_purity(bp, bl); d = disc(bp, bl); sc_full = score_locked(hdbscan_pred(unit(bp), d["mcs"]), bl)
        rows_e.append(dict(emb=name, proto=proto, n_bursts=len(bl), knn1=kp[1], **{k:sc_full[k] for k in ("ARI","NMI","purity","K_est","noise")}))
        print(f"  {name:>8} {proto:>13}: ARI={sc_full['ARI']:.3f} NMI={sc_full['NMI']:.3f} "
              f"pur={sc_full['purity']:.3f} K={sc_full['K_est']}(t8) noise={sc_full['noise']:.3f} "
              f"kNN1={kp[1]:.3f} nb={len(bl)}")

# pick E-best (embedding+proto with best HDBSCAN ARI) for Test 3
ebest = max(rows_e, key=lambda r: r["ARI"])
print(f"  -> E-best = {ebest['emb']} {ebest['proto']} (ARI {ebest['ARI']:.3f})")
be_emb = ebest["emb"]; be_fn = build_E0 if ebest["proto"]=="E0_coherent" else build_E1
BP, BL = balance(*be_fn(E[be_emb]))
base = disc(BP, BL)

# =========================== TEST 3 — cheap frozen geometry fixes ============================
print("\n[TEST 3] cheap fixes on E-best (no gradient; oracle-K = ceiling):")
def hdb_ari(X):
    b = None
    for mcs in MCS:
        a = adjusted_rand_score(BL, np.array([str(x) for x in hdbscan_pred(X, mcs)]))
        b = a if b is None else max(b, a)
    return float(b)

# baseline
print(f"  baseline (L2 + HDBSCAN)          ARI={base['ARI']:.3f} K={base['K_est']}")
rows_fix.append(dict(fix="baseline_L2_HDBSCAN", ARI=base["ARI"], note=f"K={base['K_est']}"))

# (a) L2 + per-dim standardize (whiten) then HDBSCAN
Xw = StandardScaler().fit_transform(unit(BP))
ari_w = hdb_ari(Xw)
print(f"  (a) L2+whiten + HDBSCAN          ARI={ari_w:.3f}")
rows_fix.append(dict(fix="L2_whiten_HDBSCAN", ARI=ari_w, note="per-dim standardize"))

# (b) LDA fit on WiSig-TRAIN devices only (frozen source-domain transform), applied to DRFF
print("  (b) fitting LDA on WiSig-train devices (frozen encoder, source labels only) ...")
TXD, _ = W.load_manytx(eq=0, verbose=False)
sp = json.load(open(os.path.join(_SW, "runs/wisig_supcon_fft64/splits/split_manytx.json")))
train_tx = sp["train_tx"]
mdl = RFEncoder().cuda(); mdl.load_state_dict(torch.load(
    os.path.join(_SW, "runs/wisig_supcon_fft64/retrain_best/best_model.pt"),
    map_location="cuda", weights_only=True), strict=True); mdl.eval()
for pp in mdl.parameters(): pp.requires_grad_(False)
tt = np.concatenate([GC.consec_windows(TXD, tx, GC.TRAIN_PER_DEV) for tx in train_tx])
ty = np.concatenate([np.full(min(GC.TRAIN_PER_DEV, TXD[tx]["iq"].shape[0]), i) for i, tx in enumerate(train_tx)])
Fw = GC.extract512(mdl, tt); del tt
lda = LinearDiscriminantAnalysis().fit(Fw, ty)     # WiSig-train ONLY (no DRFF labels)
Xl = lda.transform(BP)                              # apply frozen transform to DRFF burst-means
ari_lda = hdb_ari(unit(Xl))
print(f"  (b) WiSig-LDA transform+HDBSCAN  ARI={ari_lda:.3f}  (LDA dims={Xl.shape[1]})")
rows_fix.append(dict(fix="WiSigLDA_HDBSCAN", ARI=ari_lda, note=f"LDA fit on 109 train dev, dims={Xl.shape[1]}"))

# (c) oracle-K ceilings (use true K=8 -> DIAGNOSTIC, not a discovery result)
km = KMeans(n_clusters=KTRUE, n_init=10, random_state=0).fit_predict(unit(BP))
ari_km = float(adjusted_rand_score(BL, km))
sp_c = SpectralClustering(n_clusters=KTRUE, affinity="nearest_neighbors", random_state=0).fit_predict(unit(BP))
ari_sp = float(adjusted_rand_score(BL, sp_c))
km_l = KMeans(n_clusters=KTRUE, n_init=10, random_state=0).fit_predict(unit(Xl))
ari_kml = float(adjusted_rand_score(BL, km_l))
print(f"  (c) ORACLE-K CEILING (true K=8, NOT discovery):")
print(f"        k-means@8 (E-best)         ARI={ari_km:.3f}")
print(f"        spectral@8 (E-best)        ARI={ari_sp:.3f}")
print(f"        k-means@8 (WiSig-LDA)      ARI={ari_kml:.3f}")
rows_fix += [dict(fix="ORACLE_kmeans@8_Ebest", ARI=ari_km, note="ceiling, true K"),
             dict(fix="ORACLE_spectral@8_Ebest", ARI=ari_sp, note="ceiling, true K"),
             dict(fix="ORACLE_kmeans@8_WiSigLDA", ARI=ari_kml, note="ceiling, true K")]

# =========================== VERDICT ============================
knn1_best = max(r["knn1"] for r in rows_geo)
e0 = next(r for r in rows_e if r["emb"]==be_emb and r["proto"]=="E0_coherent")["ARI"]
e1 = next(r for r in rows_e if r["emb"]==be_emb and r["proto"]=="E1_multicond")["ARI"]
fix_best = max(r["ARI"] for r in rows_fix if not r["fix"].startswith("ORACLE"))
oracle_best = max(r["ARI"] for r in rows_fix if r["fix"].startswith("ORACLE"))
chance_knn = 1/8

fixable = (knn1_best > 2*chance_knn) and (e1 > e0 + 0.05 or fix_best > 0.15 or oracle_best > 0.40)
V1 = "FIXABLE" if fixable else "WALL"
print("\n=== VERDICT ===")
print(f"V1: {V1}")
print(f"    kNN1 best={knn1_best:.3f} (chance {chance_knn:.3f}); E0={e0:.3f} E1={e1:.3f}; "
      f"cheap-fix best={fix_best:.3f}; oracle-K best={oracle_best:.3f}")

report = dict(group="mavicAir2", decimation="OPT-B", ceiling_supervised=CEILING,
              test1_geometry=rows_geo, test2_E0E1=rows_e, e_best=ebest,
              test3_fixes=rows_fix, knn1_best=knn1_best, E0=e0, E1=e1,
              cheapfix_best=fix_best, oracleK_best=oracle_best, V1=V1)
json.dump(report, open(os.path.join(OUT, "discovery_gap_report.json"), "w"), indent=2, default=str)
def wcsv(fn, rows):
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
wcsv("test1_geometry.csv", rows_geo); wcsv("test2_E0E1.csv", rows_e); wcsv("test3_fixes.csv", rows_fix)
print("\nSaved -> results/step3c_discovery_gap/")
