"""STEP-4 MECHANISM VALIDATION — is "unsupervised same-model discovery requires RECEIVER
diversity" the right sentence? Two frozen, eval-only controls.

CONTROL 1 (kills/confirms the receiver-diversity vs domain-gap confound):
  WiSig DEV, same 5 slices (seeds 201-205). Build P0/P1/P2 bursts (rx/date metadata only).
  Per protocol x embedding {512-D, CosFace}: HDBSCAN(best mcs) ARI + ORACLE-K k-means@18 &
  spectral@18 (CEILING, true K) + kNN-1/5 purity + intra/inter cosine gap. P1 is the
  multi-RECEIVER row; P2 is the single-rx multi-DATE row (the within-domain analog of the
  DRFF single-rx wall). If P2 oracle-K is HIGH while it walls cross-domain -> the DRFF wall
  is DOMAIN-GAP, not receiver-diversity; if P2 oracle-K is LOW while P1 is high -> receiver
  diversity is cleanly the axis.

CONTROL 2 (is the wall physics or our encoder?):
  A no-DL classical impairment-feature baseline (CFO via lag-1 autocorr phase-slope — generic,
  no preamble assumption; envelope/spectral/IQ-imbalance/PSD-shape stats; ~19 dims, standardized).
  Burst-mean N=10 like everything else. Run in the SAME regimes: WiSig P1, WiSig P2, and DRFF
  mavicAir2 E0/E1 (OPT-B). HDBSCAN + oracle-K. Beside the encoder numbers.

Encoder + CosFace head FROZEN (no gradient). DEV WiSig only; mavicAir2 clean group on DRFF
(OPT-B). Locked noise rule. Oracle-K everywhere = CEILING (uses true K), never a discovery
result. No TEST/board-18, no M100 reprocessing, no scaling. DRFF encoder numbers reused from
step3c (cited), not recomputed.

  python3 results/step4_mechanism_validation/mechanism_validation.py
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
          os.path.join(_SW, "results", "step2b_decoherence")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
import geometry_consolidate as GC
from burst_probe import embed_times
from integrity import score_locked, hdbscan_pred, unitrows, bursts_coherent
# the EXACT deployable burst builders from Step-2b (guaranteed identical to the decoherence run)
from decohere import bursts_p1_multirx, bursts_p2_multidate
from sklearn.preprocessing import StandardScaler
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
MCS_JITTER = [13, 15, 17]
MCS_DRFF   = [5, 7]
SLICE_SEEDS = [201, 202, 203, 204, 205]
COS_M, COS_S = 0.20, 32
KWISIG = 18          # devices per DEV slice -> oracle-K true K
KDRFF  = 8           # mavicAir2 airframes
CAP_BURSTS = 120     # bursts/device kept for the oracle-K/geometry battery (tractable spectral)
NTAPS = 129
SEED = 777

def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# ============================================================================
# CLASSICAL IMPAIRMENT FEATURES (no learning; per-window scalar stats)
# ============================================================================
def classical_feats(w):
    """w: [2,256] real IQ -> ~19-D impairment/statistics vector. NO preamble assumption."""
    I, Q = w[0].astype(np.float64), w[1].astype(np.float64)
    x = I + 1j * Q
    env = np.abs(x); e2 = env ** 2
    eps = 1e-12
    # CFO via lag-1 autocorrelation phase-slope (generic estimator, no sync/preamble)
    lag1 = x[1:] * np.conj(x[:-1])
    inc = np.angle(lag1)
    cfo = float(np.angle(lag1.mean() + 0j))
    cfo_std = float(inc.std())
    # envelope moments
    em, es = float(env.mean()), float(env.std())
    esk = float(((env - em) ** 3).mean() / (es ** 3 + eps))
    eku = float(((env - em) ** 4).mean() / (es ** 4 + eps))
    papr = float(e2.max() / (e2.mean() + eps))
    # IQ imbalance proxies
    iq_ratio = float(I.var() / (Q.var() + eps))
    iq_corr = float(np.corrcoef(I, Q)[0, 1]) if I.std() > 0 and Q.std() > 0 else 0.0
    # spectral shape (shifted PSD)
    X = np.fft.fftshift(np.fft.fft(x))
    P = np.abs(X) ** 2; Ps = P.sum() + eps
    f = np.linspace(-0.5, 0.5, len(P))
    cent = float((f * P).sum() / Ps)
    spread = float(np.sqrt(((f - cent) ** 2 * P).sum() / Ps))
    flat = float(np.exp(np.log(P + eps).mean()) / (P.mean() + eps))
    cum = np.cumsum(P) / Ps
    rolloff = float(f[np.searchsorted(cum, 0.85)])
    t3 = len(P) // 3
    blo = float(P[:t3].sum() / Ps); bmid = float(P[t3:2 * t3].sum() / Ps); bhi = float(P[2 * t3:].sum() / Ps)
    zcr = float((np.abs(np.diff(np.sign(I))) > 0).mean())
    iku = float(((I - I.mean()) ** 4).mean() / (I.var() ** 2 + eps))
    qku = float(((Q - Q.mean()) ** 4).mean() / (Q.var() ** 2 + eps))
    return np.array([cfo, cfo_std, em, es, esk, eku, papr, iq_ratio, iq_corr,
                     cent, spread, flat, rolloff, blo, bmid, bhi, zcr, iku, qku], dtype=np.float32)

def classical_matrix(windows):
    """windows: [Nw,2,256] -> [Nw,19] (raw, unstandardized)."""
    return np.stack([classical_feats(w) for w in windows])

# ============================================================================
# geometry probes (identical to step3c)
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
    except Exception as e:
        ari_sp = float("nan")
    return ari_km, ari_sp

def cap_bursts(bp, bl, cap, seed=0):
    """cap bursts per device to keep oracle/geometry tractable and balanced."""
    r = np.random.default_rng(seed); keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:cap].tolist()
    keep = np.array(sorted(keep)); return bp[keep], bl[keep]

# ============================================================================
# 0. FROZEN ENCODER + reproduced CosFace head (== Step-2b recipe, seed42)
# ============================================================================

if __name__ == "__main__":
    print("loading ManyTx (eq=0)...")
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

    # ============================================================================
    # cache 512-D, CosFace, and CLASSICAL features for the DEV devices
    # ============================================================================
    print(f"caching 512-D/CosFace/classical for {len(dev_tx)} DEV devices (WIN_CACHE={WIN_CACHE}) ...")
    cache = {"512D": {}, "CosFace": {}, "Classical": {}}
    raw_class = {}
    for tx in dev_tx:
        t = GC.consec_windows(TXD, tx, WIN_CACHE)                    # [Wn,2,256] standardized (encoder path)
        Wn = t.shape[0]
        rx = TXD[tx]["rx"][:Wn].copy(); date = TXD[tx]["date"][:Wn].copy()
        e512 = GC.extract512(model, t)
        ecf = GC.apply_head(head, {tx: scaler.transform(e512).astype(np.float32)})[tx]
        # classical features computed on RAW IQ windows (physical impairments; no encoder)
        raw = np.stack([TXD[tx]["iq"][k].T.copy() for k in range(Wn)]).astype(np.float32)
        fc = classical_matrix(raw)
        cache["512D"][tx] = dict(emb=e512, rx=rx, date=date)
        cache["CosFace"][tx] = dict(emb=ecf, rx=rx, date=date)
        raw_class[tx] = dict(feat=fc, rx=rx, date=date)
    torch.cuda.empty_cache()
    # global classical scaler on all DEV windows (label-free standardization only)
    allfc = np.concatenate([raw_class[tx]["feat"] for tx in dev_tx])
    csc = StandardScaler().fit(allfc)
    for tx in dev_tx:
        cache["Classical"][tx] = dict(emb=csc.transform(raw_class[tx]["feat"]).astype(np.float32),
                                      rx=raw_class[tx]["rx"], date=raw_class[tx]["date"])
    print(f"    classical dims = {allfc.shape[1]}")

    # ============================================================================
    # CONTROL 1 (+ classical WiSig for CONTROL 2) — P0/P1/P2 over 5 DEV slices
    # ============================================================================
    PROTO_FN = {
        "P0": lambda c, s: bursts_coherent(c["emb"], N),
        "P1": lambda c, s: bursts_p1_multirx(c["emb"], c["rx"], c["date"], seed=s)[0],
        "P2": lambda c, s: bursts_p2_multidate(c["emb"], c["rx"], c["date"], seed=s)[0],
    }
    def build_wisig(emb_name, slice_tx, proto, seed):
        bp, bl = [], []
        for di, tx in enumerate(slice_tx):
            b = PROTO_FN[proto](cache[emb_name][tx], seed)
            if b is None: continue
            bp.append(b); bl.append(np.full(len(b), di))
        return np.concatenate(bp), np.concatenate(bl)

    print("\n=== CONTROL 1 (+ classical) — WiSig P0/P1/P2, 5 DEV slices (seeds 201-205) ===")
    protos = ["P0", "P1", "P2"]
    embs = ["512D", "CosFace", "Classical"]
    agg = defaultdict(lambda: defaultdict(list))     # (proto,emb) -> metric -> [per-seed]
    slices = {s: GC.scatter_slice(dev_tx, KWISIG, s) for s in SLICE_SEEDS}
    for proto in protos:
        for emb_name in embs:
            for s in SLICE_SEEDS:
                bp, bl = build_wisig(emb_name, slices[s], proto, s)
                bp, bl = cap_bursts(bp, bl, CAP_BURSTS, seed=s)
                hb = hdbscan_best(bp, bl, MCS_JITTER)
                km, spc = oracle_k(bp, bl, KWISIG)
                kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl)
                m = agg[(proto, emb_name)]
                m["hdb_ARI"].append(hb["ARI"]); m["hdb_K"].append(hb["K_est"]); m["hdb_noise"].append(hb["noise"])
                m["orak_km"].append(km); m["orak_sp"].append(spc)
                m["knn1"].append(kp[1]); m["knn5"].append(kp[5])
                m["intra"].append(intra); m["inter"].append(inter); m["gap"].append(intra - inter)
            mm = agg[(proto, emb_name)]
            print(f"  {proto} {emb_name:>9}: HDB ARI={np.mean(mm['hdb_ARI']):.3f} | "
                  f"ORACLE km@18={np.mean(mm['orak_km']):.3f} sp@18={np.nanmean(mm['orak_sp']):.3f} | "
                  f"kNN1={np.mean(mm['knn1']):.3f} gap={np.mean(mm['gap']):+.3f}")

    def ms(vals): return float(np.mean(vals)), float(np.std(vals))
    c1_rows = []
    for proto in protos:
        for emb_name in embs:
            mm = agg[(proto, emb_name)]
            c1_rows.append(dict(
                proto=proto, emb=emb_name,
                hdb_ARI=ms(mm["hdb_ARI"])[0], hdb_ARI_std=ms(mm["hdb_ARI"])[1],
                oracleK_kmeans18=ms(mm["orak_km"])[0], oracleK_spectral18=float(np.nanmean(mm["orak_sp"])),
                knn1=ms(mm["knn1"])[0], knn5=ms(mm["knn5"])[0],
                intra_cos=ms(mm["intra"])[0], inter_cos=ms(mm["inter"])[0], cos_gap=ms(mm["gap"])[0],
                hdb_K=ms(mm["hdb_K"])[0], hdb_noise=ms(mm["hdb_noise"])[0], chance=1.0 / KWISIG))

    # ============================================================================
    # CONTROL 2 (DRFF) — reproduce mavicAir2 OPT-B raw windows, classical E0/E1
    #   (encoder DRFF numbers are reused from step3c; here we add the classical baseline)
    # ============================================================================
    print("\n=== CONTROL 2 — DRFF mavicAir2 OPT-B classical baseline (E0/E1) ===")
    TAPS_B = firwin(NTAPS, 0.9)
    manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
    pat = re.compile(r'(.+?)_(\d+)_hover')
    af_files = defaultdict(list)
    for r in manifest["clean"]:
        af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
    all_af = sorted(af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
    model_of = {af: pat.match(af + "_hover").group(1) for af in all_af}
    TARGET = "mavicAir2"

    def decimate_B(iq):
        x = iq.astype(np.float32); pad = min(3 * NTAPS, x.shape[1] - 1)
        return filtfilt(TAPS_B, [1.0], x, axis=1, padlen=pad)[:, ::2]

    # faithful replay of build_subset (SEED=777, CAP=320, <=12 win/seg) capturing RAW windows
    def build_drff_raw(cap=320):
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
            is_tgt = (model_of[a] == TARGET)
            got = 0; zcache = {}
            for fn, si in units:
                if got >= cap: break
                if fn not in zcache:
                    z = np.load(os.path.join(DRFF_DIR, fn))
                    zcache[fn] = dict(dec=(decimate_B(z["iq"]) if is_tgt else None), sb=z["seg_bounds"],
                                      D=str(z["D"]), C=int(z["C"]))
                zc = zcache[fn]; off, ln = zc["sb"][si]; o2, l2 = off // 2, ln // 2
                nw = l2 // WIN
                if nw < 1: continue
                take = min(nw, cap - got, 12)
                if is_tgt:
                    seg = zc["dec"][:, o2:o2 + l2]
                    for k in range(take):
                        Xr.append(seg[:, k * WIN:(k + 1) * WIN].astype(np.float32))
                        afs.append(a); segs.append(gseg); Ds.append(zc["D"]); Cs.append(zc["C"])
                got += take; gseg += 1
        return (np.stack(Xr), np.array(afs), np.array(segs), np.array(Ds), np.array(Cs))

    Xr, afd, segd, Dd, Cd = build_drff_raw()
    print(f"    mavicAir2 OPT-B raw windows: {len(Xr)} across {len(np.unique(segd))} segments, "
          f"{len(np.unique(afd))} airframes")
    Fcl = StandardScaler().fit_transform(classical_matrix(Xr)).astype(np.float32)

    def build_E0(emb):
        bp, bl = [], []
        for sgid in np.unique(segd):
            idx = np.where(segd == sgid)[0]
            for k in range(len(idx) // N):
                ch = idx[k * N:(k + 1) * N]; bp.append(emb[ch].mean(0)); bl.append(afd[ch][0])
        return np.array(bp), np.array(bl)
    def build_E1(emb, seed=SEED):
        r = np.random.default_rng(seed); bp, bl = [], []
        for a in np.unique(afd):
            idx = np.where(afd == a)[0]; cells = defaultdict(list)
            for i in idx: cells[(Dd[i], Cd[i])].append(i)
            ck = list(cells.keys()); nb = len(idx) // N
            for _ in range(nb):
                r.shuffle(ck); pick = []; ci = 0
                while len(pick) < N and ci < 2000:
                    c = ck[ci % len(ck)]
                    if cells[c]: pick.append(cells[c][r.integers(len(cells[c]))])
                    ci += 1
                if len(pick) == N: bp.append(emb[pick].mean(0)); bl.append(a)
        return np.array(bp), np.array(bl)
    def balance(bp, bl, seed=0):
        r = np.random.default_rng(seed); per = min(int((bl == a).sum()) for a in np.unique(bl))
        keep = []
        for a in np.unique(bl):
            ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
        keep = np.array(sorted(keep)); return bp[keep], bl[keep]

    c2_drff = []
    for proto, fn in (("E0", build_E0), ("E1", build_E1)):
        bp, bl = balance(*fn(Fcl))
        bpu = unit(bp)
        hb = hdbscan_best(bpu, bl, MCS_DRFF)
        km, spc = oracle_k(bpu, bl, KDRFF)
        kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl)
        c2_drff.append(dict(regime=f"DRFF_{proto}", emb="Classical", hdb_ARI=hb["ARI"],
                            oracleK_kmeans8=km, oracleK_spectral8=spc, knn1=kp[1], cos_gap=intra - inter,
                            hdb_K=hb["K_est"], hdb_noise=hb["noise"], n_bursts=len(bl), chance=1.0 / KDRFF))
        print(f"  DRFF {proto} Classical: HDB ARI={hb['ARI']:.3f} | ORACLE km@8={km:.3f} sp@8={spc:.3f} | "
              f"kNN1={kp[1]:.3f} gap={intra-inter:+.3f} nb={len(bl)}")

    # ============================================================================
    # reused encoder numbers (with provenance) for the mechanism table
    # ============================================================================
    STEP2B = "results/step2b_decoherence/full_grid.csv"
    STEP3C = "results/step3c_discovery_gap/discovery_gap_report.json"
    STEP3M = "results/step3_m100_smoke/discovery_decoherence.csv"
    d3c = json.load(open(os.path.join(_SW, STEP3C)))
    def c1(proto, emb, metric):
        return next(r for r in c1_rows if r["proto"] == proto and r["emb"] == emb)[metric]

    # ============================================================================
    # THE MECHANISM TABLE  (primary encoder embedding = 512-D, head-free, comparable)
    # columns: encoder HDBSCAN ARI | encoder oracle-K ARI | classical HDBSCAN ARI |
    #          classical oracle-K ARI | kNN-1 | intra-inter gap
    # ============================================================================
    NA = "n/a"
    mech = []
    # WiSig P1 (multi-receiver) — 512-D from Control 1 (this session); CosFace noted separately
    mech.append(dict(regime="WiSig P1 (multi-rx)", receiver_diversity="YES (10 rx)",
        enc_hdbscan=round(c1("P1", "512D", "hdb_ARI"), 3),
        enc_oracleK=round(max(c1("P1", "512D", "oracleK_kmeans18"), c1("P1", "512D", "oracleK_spectral18")), 3),
        cls_hdbscan=round(c1("P1", "Classical", "hdb_ARI"), 3),
        cls_oracleK=round(max(c1("P1", "Classical", "oracleK_kmeans18"), c1("P1", "Classical", "oracleK_spectral18")), 3),
        knn1=round(c1("P1", "512D", "knn1"), 3), gap=round(c1("P1", "512D", "cos_gap"), 3),
        prov="512D/classical: this session (Control 1)"))
    # WiSig P2 (single-rx, multi-date) — the within-domain analog of the DRFF wall
    mech.append(dict(regime="WiSig P2 (single-rx,multi-date)", receiver_diversity="NO (1 rx, 4 dates)",
        enc_hdbscan=round(c1("P2", "512D", "hdb_ARI"), 3),
        enc_oracleK=round(max(c1("P2", "512D", "oracleK_kmeans18"), c1("P2", "512D", "oracleK_spectral18")), 3),
        cls_hdbscan=round(c1("P2", "Classical", "hdb_ARI"), 3),
        cls_oracleK=round(max(c1("P2", "Classical", "oracleK_kmeans18"), c1("P2", "Classical", "oracleK_spectral18")), 3),
        knn1=round(c1("P2", "512D", "knn1"), 3), gap=round(c1("P2", "512D", "cos_gap"), 3),
        prov="512D/classical: this session (Control 1)"))
    # M100 single-rx — existing numbers only (no reprocessing per guardrail)
    mech.append(dict(regime="M100 (single-rx, cross-domain)", receiver_diversity="NO (1 rx)",
        enc_hdbscan=0.002, enc_oracleK=NA, cls_hdbscan=NA, cls_oracleK=NA, knn1=NA, gap=NA,
        prov=f"enc_hdbscan: {STEP3M} (D1 512D best ~0.002); rest n/a (M100 not reprocessed)"))
    # DRFF single-rx E0 (coherent) — reused from step3c
    d3c_e0 = next(r for r in d3c["test1_geometry"] if r["emb"] == "512D")
    mech.append(dict(regime="DRFF E0 (single-rx, coherent)", receiver_diversity="NO (1 rx)",
        enc_hdbscan=round(d3c["E0"], 3), enc_oracleK=NA,
        cls_hdbscan=round(next(r for r in c2_drff if r["regime"] == "DRFF_E0")["hdb_ARI"], 3),
        cls_oracleK=round(max(next(r for r in c2_drff if r["regime"] == "DRFF_E0")["oracleK_kmeans8"],
                              next(r for r in c2_drff if r["regime"] == "DRFF_E0")["oracleK_spectral8"]), 3),
        knn1=round(d3c_e0["knn1"], 3), gap=round(d3c_e0["cos_gap"], 3),
        prov=f"enc: {STEP3C} (E0 512D); classical: this session (Control 2)"))
    # DRFF single-rx E1 (multi-condition) — reused from step3c + oracle from step3c ceiling
    mech.append(dict(regime="DRFF E1 (single-rx, multi-cond)", receiver_diversity="NO (1 rx)",
        enc_hdbscan=round(d3c["E1"], 3), enc_oracleK=round(d3c["oracleK_best"], 3),
        cls_hdbscan=round(next(r for r in c2_drff if r["regime"] == "DRFF_E1")["hdb_ARI"], 3),
        cls_oracleK=round(max(next(r for r in c2_drff if r["regime"] == "DRFF_E1")["oracleK_kmeans8"],
                              next(r for r in c2_drff if r["regime"] == "DRFF_E1")["oracleK_spectral8"]), 3),
        knn1=round(d3c["knn1_best"], 3), gap=round(next(r for r in d3c["test1_geometry"] if r["emb"] == "512D")["cos_gap"], 3),
        prov=f"enc: {STEP3C} (E1 512D, oracle-K ceiling); classical: this session (Control 2)"))

    # ============================================================================
    # VERDICTS
    # ============================================================================
    p1_enc_ok = c1("P1", "512D", "oracleK_kmeans18")
    p2_enc_ok = max(c1("P2", "512D", "oracleK_kmeans18"), c1("P2", "512D", "oracleK_spectral18"))
    p1_cf = c1("P1", "CosFace", "hdb_ARI"); p2_cf = c1("P2", "CosFace", "hdb_ARI")
    drff_e1_oracle = d3c["oracleK_best"]

    # G1: receiver-diversity vs domain-gap
    if p2_enc_ok <= 0.30 and p1_enc_ok >= 0.50:
        G1 = f"RECEIVER-DIVERSITY confirmed: P2 single-rx oracle-K={p2_enc_ok:.3f} LOW while P1 multi-rx oracle-K={p1_enc_ok:.3f} HIGH."
    elif p2_enc_ok >= 0.50:
        G1 = f"DOMAIN-GAP: P2 single-rx within-domain oracle-K={p2_enc_ok:.3f} HIGH -> DRFF wall is substantially domain-gap, not receiver-diversity. REWRITE the sentence."
    else:
        G1 = f"MIXED: P2 oracle-K={p2_enc_ok:.3f} (P1={p1_enc_ok:.3f}) -> both contribute; frame quantitatively."

    # G2: method independence (classical vs encoder on single-rx)
    cls_p2 = max(c1("P2", "Classical", "oracleK_kmeans18"), c1("P2", "Classical", "oracleK_spectral18"))
    cls_p1 = c1("P1", "Classical", "oracleK_kmeans18")
    cls_drff = max(next(r for r in c2_drff if r["regime"] == "DRFF_E1")["oracleK_kmeans8"],
                   next(r for r in c2_drff if r["regime"] == "DRFF_E1")["oracleK_spectral8"])
    if cls_drff <= 0.30 and cls_p2 <= 0.35:
        G2 = f"METHOD-INDEPENDENT: classical also walls single-rx (DRFF E1 oracle={cls_drff:.3f}, WiSig P2 oracle={cls_p2:.3f}); classical P1={cls_p1:.3f}."
    elif cls_drff >= 0.40:
        G2 = f"ENCODER-SPECIFIC: classical clusters DRFF single-rx (oracle={cls_drff:.3f}) where encoder walls -> wall is our encoder, not physics. REPORT LOUDLY."
    else:
        G2 = f"UNINFORMATIVE/mixed: classical DRFF oracle={cls_drff:.3f}, P2={cls_p2:.3f}, P1={cls_p1:.3f}."

    print("\n" + "=" * 78)
    print("MECHANISM TABLE (primary embedding = 512-D, head-free; oracle-K = CEILING)")
    print("=" * 78)
    hdr = f"{'regime':>32} {'rxDiv':>16} {'encHDB':>7} {'encOrK':>7} {'clsHDB':>7} {'clsOrK':>7} {'kNN1':>6} {'gap':>7}"
    print(hdr)
    for m in mech:
        print(f"{m['regime']:>32} {m['receiver_diversity']:>16} "
              f"{str(m['enc_hdbscan']):>7} {str(m['enc_oracleK']):>7} {str(m['cls_hdbscan']):>7} "
              f"{str(m['cls_oracleK']):>7} {str(m['knn1']):>6} {str(m['gap']):>7}")
    print("\nCosFace (with-head) contrast: P1 HDB ARI=%.3f  P2 HDB ARI=%.3f (reused step2b: 0.772 / 0.211)"
          % (p1_cf, p2_cf))
    print("\n=== VERDICTS ===")
    print("G1 (receiver-diversity vs domain-gap):", G1)
    print("G2 (method-independent vs encoder-specific):", G2)

    # does the sentence survive?
    if "RECEIVER-DIVERSITY confirmed" in G1:
        SENT = "SURVIVES as-is: single-rx geometry is interleaved within-domain too; receiver diversity is the axis."
    elif "DOMAIN-GAP" in G1:
        SENT = "WRONG / must be rewritten: within-domain single-rx clusters given true K -> the DRFF wall is domain-gap."
    else:
        SENT = "NEEDS QUALIFICATION: receiver diversity helps but does not alone explain the DRFF wall; state both quantitatively."
    print("SENTENCE:", SENT)

    # ============================================================================
    # save
    # ============================================================================
    def wcsv(fn, rows):
        with open(os.path.join(OUT, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    wcsv("control1_wisig.csv", c1_rows)
    wcsv("control2_drff_classical.csv", c2_drff)
    wcsv("mechanism_table.csv", mech)
    report = dict(
        control1_wisig=c1_rows, control2_drff_classical=c2_drff, mechanism_table=mech,
        cosface_contrast=dict(P1_hdb=p1_cf, P2_hdb=p2_cf),
        G1_receiver_vs_domain=G1, G2_method_independence=G2, sentence_verdict=SENT,
        params=dict(slice_seeds=SLICE_SEEDS, cap_bursts=CAP_BURSTS, K_wisig=KWISIG, K_drff=KDRFF,
                    classical_dims=int(allfc.shape[1]), mcs_wisig=MCS_JITTER, mcs_drff=MCS_DRFF),
        provenance=dict(step2b=STEP2B, step3c=STEP3C, step3_m100=STEP3M),
        guardrails="encoder+head frozen; oracle-K=ceiling(true K); DEV WiSig + mavicAir2 OPT-B; "
                   "no TEST/board-18; no M100 reprocessing; no scaling; locked noise rule")
    json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)
    print("\nsaved -> results/step4_mechanism_validation/ (mechanism_table.csv, control1_wisig.csv, "
          "control2_drff_classical.csv, report.json)")
    print("CHECKPOINT — encoder+head frozen, oracle-K=ceiling, TEST/M100 untouched.")
