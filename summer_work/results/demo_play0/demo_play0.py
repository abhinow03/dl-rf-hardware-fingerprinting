"""PLAY 0 — DEMO-SCALE RE-SCORING. EVAL-ONLY, NO GPU, NO TRAINING. DEMO-SIDE numbers, kept in a
SEPARATE ledger — NEVER merged into paper tables. Question: do current pipelines become demo-viable
at demo conditions (small K, large burst N, full-bandwidth features)? Runs in parallel with Play 1:
uses CACHED encoder features (features_B.npz F512) only; all classical features on CPU. Writes ONLY
to results/demo_play0/. TEST/board-18 CLOSED. M100 untouched.

DATA: DRFF mavicAir2 (8) — already spent as eval, legal to score. Encoder/hybrid from cached
seed-777/cap-320 OPT-B F512 (features_B.npz). classical-19 recomputed on the SAME (reproduced) OPT-B
windows (CPU). Lever-3 rich classical on NATIVE 50 MS/s segments (CPU).

LEVER 1 small-K: methods {classical-19, frozen-512, hybrid-pca64+19 (ad-hoc, cached)} on ALL C(8,4)=70
  and C(8,3)=56 airframe subsets. E1 N=10. HDBSCAN mean over mcs{3,5} + oracle-K@{K} km/sp. Report
  mean+/-std ARI, frac ARI>=0.6, frac correct K_est.
LEVER 2 burst-N: full-8 AND K=4 battery, N in {10,50,200,per-segment}; classical-19 + hybrid. E1 where
  N allows, else per-segment fallback (a tracked continuous emission = one object; temporal continuity
  is free info every fielded system uses).
LEVER 3 full-band native classical (50 MS/s, per-segment, FIXED list below, ~36 dims): CFO-trajectory
  stats, envelope/transient shape, full-band spectral moments + band-energy ratios (outer band matters),
  IQ-imbalance proxies, log-PSD polynomial shape coeffs. Score 8-way + K=4 beside classical-19; plus one
  combo row = full-band classical (+) block-scaled encoder (fixed 50/50 block equalization, no alpha).

GATE: DEMO-VIABLE if any method/config: K=4 mean ARI>=0.6 AND correct-K frac>=0.5 AND K=3 mean>=0.7.

  python3 results/demo_play0/demo_play0.py
"""
import os, sys, json, csv, re, itertools
from collections import defaultdict, Counter
import numpy as np
from scipy.signal import firwin, filtfilt, welch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity"),
          os.path.join(_SW, "results", "step4_mechanism_validation")):
    if p not in sys.path: sys.path.insert(0, p)
import wisig_manytx as W
from integrity import score_locked, hdbscan_pred
from mechanism_validation import classical_matrix            # the 25 MS/s classical-19 (guarded import)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

DRFF_DIR = os.path.expanduser("~/Desktop/processed/drff_r2")
FEATB    = os.path.join(_SW, "results", "step3b_drff_smoke", "features_B.npz")
OUT      = _HERE
WIN, N, CAP, NTAPS, SEED = 256, 10, 320, 129, 777
TAPS_B = firwin(NTAPS, 0.9)
def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
def decimate_B(iq):
    x = iq.astype(np.float32); pad = min(3*NTAPS, x.shape[1]-1)
    return filtfilt(TAPS_B, [1.0], x, axis=1, padlen=pad)[:, ::2]

# ============================================================================
# 0. CACHED encoder F512 (+metadata) ; reproduce OPT-B windows in cached order (CPU)
# ============================================================================
print("[0] loading cached F512 (features_B.npz) — NO GPU ...")
z = np.load(FEATB, allow_pickle=True)
mm = z["model"].astype(str); sel = mm == "mavicAir2"
F512 = z["F512"][sel].astype(np.float32)
AF = z["af"][sel].astype(str); SEG = z["seg"][sel].astype(int)
D = z["D"][sel].astype(str); C = z["C"][sel].astype(int); H = z["H"][sel].astype(str)
U = z["U"][sel].astype(str)
print(f"    cached mavicAir2: {len(F512)} windows, {len(np.unique(SEG))} segments, {len(np.unique(AF))} airframes")

manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
pat = re.compile(r'(.+?)_(\d+)_hover')
af_files = defaultdict(list)
for r in manifest["clean"]: af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
all_af = sorted(af_files, key=lambda t:(t.rsplit("_",1)[0], int(t.rsplit("_",1)[1])))
model_of = {a: pat.match(a+"_hover").group(1) for a in all_af}

print("[0] reproducing build_subset('B', all_af) window order on CPU (decimate mavicAir2 only) ...")
def reproduce_windows():
    """Faithful replica of drff_smoke.build_subset('B', all_af, SEED=777): advance rng/gseg over ALL
    airframes (counters need only seg_bounds), decimate+extract window VALUES for mavicAir2 only.
    Returns per-mavicAir2-window: std window, af, gseg, and per-gseg native (fn,off,ln)."""
    rng = np.random.default_rng(SEED)
    sb_cache = {}
    def segbounds(fn):
        if fn not in sb_cache:
            sb_cache[fn] = np.load(os.path.join(DRFF_DIR, fn))["seg_bounds"]
        return sb_cache[fn]
    Xs, afs, segs = [], [], []
    seg_native = {}                        # gseg -> (fn, off, ln)
    gseg = 0
    for a in all_af:
        is_t = (model_of[a] == "mavicAir2"); got = 0
        files = af_files[a][:]; rng.shuffle(files)
        units = []
        for fn in files:
            for si in range(segbounds(fn).shape[0]): units.append((fn, si))
        rng.shuffle(units)
        dec_cache = {}
        for fn, si in units:
            if got >= CAP: break
            off, ln = segbounds(fn)[si]; o2, l2 = off // 2, ln // 2
            nw = l2 // WIN
            if nw < 1: continue
            take = min(nw, CAP - got, 12)
            if is_t:
                if fn not in dec_cache:
                    dec_cache[fn] = decimate_B(np.load(os.path.join(DRFF_DIR, fn))["iq"])
                segw = dec_cache[fn][:, o2:o2 + l2]
                for k in range(take):
                    Xs.append(W.standardize(segw[:, k*WIN:(k+1)*WIN].astype(np.float32)))
                    afs.append(a); segs.append(gseg)
                seg_native[gseg] = (fn, int(off), int(ln))
            got += take; gseg += 1
    return np.stack(Xs).astype(np.float32), np.array(afs), np.array(segs), seg_native
Xw, af_r, seg_r, SEG_NATIVE = reproduce_windows()
# verify alignment with cached F512 rows
assert len(Xw) == len(F512), f"window count mismatch {len(Xw)} vs {len(F512)}"
assert np.array_equal(af_r, AF) and np.array_equal(seg_r, SEG), "reproduced order != cached order"
print(f"    reproduced {len(Xw)} windows; alignment vs cached F512 metadata: VERIFIED")

print("[0] computing classical-19 (25 MS/s) on reproduced windows (CPU) ...")
CLS19 = StandardScaler().fit_transform(classical_matrix(Xw)).astype(np.float32)
MAVIC = sorted(np.unique(AF).tolist())
KFULL = len(MAVIC)

# ============================================================================
# scoring primitives (LOCKED harness: HDBSCAN mean over grid; oracle km AND sp separate)
# ============================================================================
def grid_for(K): return [3, 5] if K <= 4 else [5, 7]
def hdb_mean(bp, bl, grid):
    per = {m: score_locked(hdbscan_pred(bp, m), bl) for m in grid}
    ari = float(np.mean([per[m]["ARI"] for m in grid]))
    kest = float(np.mean([per[m]["K_est"] for m in grid]))
    return ari, kest
def oracle_km_sp(bp, bl, K):
    yi = np.unique(bl, return_inverse=True)[1]
    if len(bp) < K: return float("nan"), float("nan")
    km = float(adjusted_rand_score(yi, KMeans(K, n_init=10, random_state=0).fit_predict(bp)))
    try:
        sp = float(adjusted_rand_score(yi, SpectralClustering(K, affinity="nearest_neighbors",
                   random_state=0, n_neighbors=min(15, len(bp)-1)).fit_predict(bp)))
    except Exception: sp = float("nan")
    return km, sp
def score_set(bp, bl, K):
    bpu = unit(bp); grid = grid_for(K)
    ari, kest = hdb_mean(bpu, bl, grid)
    km, sp = oracle_km_sp(bpu, bl, K)
    return dict(hdb_ari=ari, hdb_kest=kest, oracle_km=km, oracle_sp=sp,
                correctK=int(round(kest) == K), n=len(bl))

# ---- burst / feature builders over cached rows ----
def E1_index_bursts(keep_af, Nb, seed=SEED):
    r = np.random.default_rng(seed); out = []
    for a in keep_af:
        idx = np.where(AF == a)[0]; cells = defaultdict(list)
        for i in idx: cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx) // Nb
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < Nb and ci < 3000:
                c = ck[ci % len(ck)]
                if cells[c]: pick.append(cells[c][r.integers(len(cells[c]))])
                ci += 1
            if len(pick) == Nb: out.append((np.array(pick), a))
    return out
def perseg_index_bursts(keep_af):
    out = []
    for a in keep_af:
        for sg in np.unique(SEG[AF == a]):
            idx = np.where((AF == a) & (SEG == sg))[0]
            if len(idx) >= 1: out.append((idx, a))
    return out
def balance(bursts, seed=0):
    r = np.random.default_rng(seed); lbl = np.array([b[1] for b in bursts])
    if len(np.unique(lbl)) < 2: return bursts
    per = min(int((lbl == a).sum()) for a in np.unique(lbl)); keep = []
    for a in np.unique(lbl):
        ii = np.where(lbl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    return [bursts[i] for i in sorted(keep)]
def bmeans(feat, bursts): return np.stack([feat[idx].mean(0) for idx, _ in bursts])
def hybrid_adhoc(cls_bm, enc_bm):
    Zc = StandardScaler().fit_transform(cls_bm)
    nc = min(64, enc_bm.shape[0], 512)
    Ze = StandardScaler().fit_transform(PCA(n_components=nc, random_state=0).fit_transform(enc_bm))
    return unit(np.concatenate([Zc, Ze], axis=1))

def build_method(method, bursts):
    lbl = np.array([b[1] for b in bursts])
    if method == "classical-19":  M = unit(bmeans(CLS19, bursts))
    elif method == "frozen-512":  M = unit(bmeans(F512, bursts))
    elif method == "hybrid-pca64+19": M = hybrid_adhoc(bmeans(CLS19, bursts), bmeans(F512, bursts))
    return M, lbl

# ============================================================================
# LEVER 1 — small-K subsets
# ============================================================================
print("\n" + "="*76 + "\nLEVER 1 — small-K subsets (C(8,4)=70, C(8,3)=56)\n" + "="*76)
lever1_rows = []; l1_detail = {}
for Ksub in (4, 3):
    subsets = list(itertools.combinations(MAVIC, Ksub))
    for method in ("classical-19", "frozen-512", "hybrid-pca64+19"):
        aris, kms, corr = [], [], []
        for sub in subsets:
            bursts = balance(E1_index_bursts(sub, N))
            if len(bursts) < Ksub: continue
            M, lbl = build_method(method, bursts)
            s = score_set(M, lbl, Ksub)
            aris.append(s["hdb_ari"]); kms.append(s["oracle_km"]); corr.append(s["correctK"])
        aris = np.array(aris); kms = np.array(kms); corr = np.array(corr)
        row = dict(lever="L1", K=Ksub, N=10, method=method, n_subsets=len(aris),
                   hdb_ari_mean=round(float(aris.mean()),3), hdb_ari_std=round(float(aris.std()),3),
                   oracle_km_mean=round(float(np.nanmean(kms)),3),
                   frac_correctK=round(float(corr.mean()),3))
        row["frac_ari_ge0.6"] = round(float((aris>=0.6).mean()),3)
        lever1_rows.append(row); l1_detail[(Ksub, method)] = aris
        print(f"  K={Ksub} {method:>16}: HDB ARI={row['hdb_ari_mean']}+/-{row['hdb_ari_std']} "
              f"| frac>=0.6={row['frac_ari_ge0.6']} | oracle km={row['oracle_km_mean']} "
              f"| correctK={row['frac_correctK']} (n={len(aris)})")

# ============================================================================
# LEVER 2 — burst size N
# ============================================================================
print("\n" + "="*76 + "\nLEVER 2 — burst-size N sweep {10,50,200,per-segment}\n" + "="*76)
lever2_rows = []
def n_bursts(keep_af, Nspec):
    if Nspec == "perseg": return balance(perseg_index_bursts(keep_af)), "per-segment"
    b = balance(E1_index_bursts(keep_af, Nspec))
    # fallback if too few bursts formed (N exceeds per-condition counts)
    if len(b) < 2 * len(keep_af):
        return balance(perseg_index_bursts(keep_af)), f"N={Nspec}->per-seg(fallback)"
    return b, f"N={Nspec}"
for scope, keeps, Kc in (("full8", [tuple(MAVIC)], KFULL),
                         ("K4", list(itertools.combinations(MAVIC, 4)), 4)):
    for Nspec in (10, 50, 200, "perseg"):
        for method in ("classical-19", "hybrid-pca64+19"):
            aris, kms, kests, fbtag = [], [], [], None
            for sub in keeps:
                bursts, tag = n_bursts(sub, Nspec); fbtag = tag
                if len(bursts) < Kc: continue
                M, lbl = build_method(method, bursts)
                s = score_set(M, lbl, Kc)
                aris.append(s["hdb_ari"]); kms.append(s["oracle_km"]); kests.append(s["hdb_kest"])
            if not aris: continue
            row = dict(lever="L2", scope=scope, K=Kc, N=str(Nspec), tag=fbtag, method=method,
                       n=len(aris), hdb_ari_mean=round(float(np.mean(aris)),3),
                       oracle_km_mean=round(float(np.nanmean(kms)),3),
                       kest_mean=round(float(np.mean(kests)),2))
            lever2_rows.append(row)
            print(f"  {scope:>5} {method:>16} {str(Nspec):>7} ({fbtag}): HDB ARI={row['hdb_ari_mean']} "
                  f"oracle km={row['oracle_km_mean']} Kest={row['kest_mean']} (n={row['n']})")

# ============================================================================
# LEVER 3 — FULL-BAND NATIVE CLASSICAL (50 MS/s, per-segment). FIXED LIST (36 dims):
#  [cfo_global, cfo_traj_mean, cfo_traj_std, cfo_traj_slope]                              (4)
#  envelope: mean,std,skew,kurt,papr,onset_ratio,env_ac1                                  (7)
#  full-band PSD moments: centroid,spread,skew,kurt,flatness,rolloff85,rolloff95          (7)
#  band-energy ratios: outer_frac, q0,q1,q2,q3                                            (5)
#  IQ-imbalance: iq_pow_ratio,iq_corr,gain_imb,phase_skew                                 (4)
#  log-PSD polynomial shape coeffs (deg 5 -> 6) + spectral_entropy + kurt_I + kurt_Q      (9)
# ============================================================================
print("\n" + "="*76 + "\nLEVER 3 — full-band native classical (50 MS/s, per-segment, 36-D fixed)\n" + "="*76)
def rich_feats_native(iq):
    I, Q = iq[0].astype(np.float64), iq[1].astype(np.float64)
    x = I + 1j*Q; L = len(I); eps = 1e-12
    env = np.abs(x); e2 = env**2
    # CFO trajectory via sliding lag-1 autocorr phase
    lag1 = x[1:]*np.conj(x[:-1]); cfo_g = float(np.angle(lag1.mean()+0j))
    nseg = 20; step = max(1, (L-1)//nseg)
    traj = [np.angle(lag1[i:i+step].mean()+0j) for i in range(0, len(lag1)-step, step)] or [cfo_g]
    traj = np.array(traj); tt = np.arange(len(traj))
    slope = float(np.polyfit(tt, traj, 1)[0]) if len(traj) > 1 else 0.0
    em, es = float(env.mean()), float(env.std())
    esk = float(((env-em)**3).mean()/(es**3+eps)); eku = float(((env-em)**4).mean()/(es**4+eps))
    papr = float(e2.max()/(e2.mean()+eps)); onset = float(env[:max(1,L//10)].mean()/(em+eps))
    env_ac1 = float(np.corrcoef(env[:-1], env[1:])[0,1]) if L > 2 else 0.0
    f, P = welch(x, nperseg=min(4096, L), return_onesided=False)
    idx = np.argsort(f); f = f[idx]; P = np.abs(P[idx]); Ps = P.sum()+eps
    cent = float((f*P).sum()/Ps); spread = float(np.sqrt(((f-cent)**2*P).sum()/Ps))
    psk = float((((f-cent)/(spread+eps))**3*P).sum()/Ps); pku = float((((f-cent)/(spread+eps))**4*P).sum()/Ps)
    flat = float(np.exp(np.log(P+eps).mean())/(P.mean()+eps))
    cum = np.cumsum(P)/Ps; r85 = float(f[np.searchsorted(cum,0.85)]); r95 = float(f[np.searchsorted(cum,min(0.95,cum[-1]))])
    # band-energy: outer (|f|>0.25 of native band) fraction + quarters of the band
    outer = float(P[np.abs(f) > 0.25].sum()/Ps)
    q = np.array_split(P, 4); qf = [float(qq.sum()/Ps) for qq in q]
    iq_ratio = float(I.var()/(Q.var()+eps)); iq_corr = float(np.corrcoef(I,Q)[0,1]) if I.std()>0 and Q.std()>0 else 0.0
    gain_imb = float(np.sqrt(I.var())/(np.sqrt(Q.var())+eps)); phase_skew = float((I*Q).mean()/(np.sqrt(I.var()*Q.var())+eps))
    logP = np.log(P+eps); ff = (f-f.mean())/(f.std()+eps)
    poly = np.polyfit(ff, logP, 5).astype(float)                                   # 6 coeffs
    pn = P/Ps; spec_ent = float(-(pn*np.log(pn+eps)).sum())
    kI = float(((I-I.mean())**4).mean()/(I.var()**2+eps)); kQ = float(((Q-Q.mean())**4).mean()/(Q.var()**2+eps))
    return np.array([cfo_g, float(traj.mean()), float(traj.std()), slope,
                     em, es, esk, eku, papr, onset, env_ac1,
                     cent, spread, psk, pku, flat, r85, r95,
                     outer, qf[0], qf[1], qf[2], qf[3],
                     iq_ratio, iq_corr, gain_imb, phase_skew,
                     *poly, spec_ent, kI, kQ], dtype=np.float64)

print("[L3] computing 36-D native features per mavicAir2 segment ...")
seg_ids = sorted(SEG_NATIVE.keys()); iqcache = {}
seg_af = {}
for gs in np.unique(SEG):
    rows = np.where(SEG == gs)[0]; seg_af[gs] = AF[rows[0]]
NATIVE = {}; native_bad = 0
for gs in seg_ids:
    fn, off, ln = SEG_NATIVE[gs]
    if fn not in iqcache: iqcache[fn] = np.load(os.path.join(DRFF_DIR, fn))["iq"]
    seg = iqcache[fn][:, off:off+ln]
    try: NATIVE[gs] = rich_feats_native(seg)
    except Exception: native_bad += 1
seg_keys = [gs for gs in seg_ids if gs in NATIVE]
NAT = np.stack([NATIVE[gs] for gs in seg_keys]); NAT = StandardScaler().fit_transform(NAT).astype(np.float32)
nat_af = np.array([seg_af[gs] for gs in seg_keys])
# per-segment encoder (mean cached F512 over seg) for the combo row
ENC_SEG = np.stack([F512[SEG == gs].mean(0) for gs in seg_keys]).astype(np.float32)
print(f"    {len(seg_keys)} segments featurized ({native_bad} skipped); dims={NAT.shape[1]}")

def combo_block(nat, enc):
    Zc = StandardScaler().fit_transform(nat); Zc /= np.sqrt(nat.shape[1])
    Ze = StandardScaler().fit_transform(enc); Ze /= np.sqrt(enc.shape[1])
    return unit(np.concatenate([0.5*Zc, 0.5*Ze], axis=1))     # fixed 50/50 (equal block after equalization)

def perseg_score(keep_af, feat, af_arr, K):
    msk = np.isin(af_arr, keep_af)
    bp, bl = feat[msk], af_arr[msk]
    # balance segments per airframe
    r = np.random.default_rng(0); per = min(int((bl==a).sum()) for a in np.unique(bl)); keep=[]
    for a in np.unique(bl):
        ii = np.where(bl==a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    k = np.array(sorted(keep)); return score_set(bp[k], bl[k], K)

COMBO = combo_block(NAT, ENC_SEG)
lever3_rows = []
for name, feat in (("classical-19(25MS/s,per-seg)", None), ("fullband-native(50MS/s)", NAT),
                   ("fullband(+)encoder-block", COMBO)):
    # classical-19 per-segment baseline (recompute per-seg means of CLS19)
    if feat is None:
        cls_seg = np.stack([CLS19[SEG==gs].mean(0) for gs in seg_keys]).astype(np.float32); feat = cls_seg
    for scope, keeps, Kc in (("full8", [tuple(MAVIC)], KFULL),
                             ("K4", list(itertools.combinations(MAVIC, 4)), 4)):
        aris, kms, corr = [], [], []
        for sub in keeps:
            s = perseg_score(sub, feat, nat_af, Kc)
            aris.append(s["hdb_ari"]); kms.append(s["oracle_km"]); corr.append(s["correctK"])
        row = dict(lever="L3", feature=name, scope=scope, K=Kc, n=len(aris),
                   hdb_ari_mean=round(float(np.mean(aris)),3), oracle_km_mean=round(float(np.nanmean(kms)),3),
                   frac_correctK=round(float(np.mean(corr)),3))
        lever3_rows.append(row)
        print(f"  {name:>28} {scope:>5}: HDB ARI={row['hdb_ari_mean']} oracle km={row['oracle_km_mean']} "
              f"correctK={row['frac_correctK']} (n={row['n']})")

# ============================================================================
# GATE
# ============================================================================
def l1(K, method, col):
    return next(r for r in lever1_rows if r["K"]==K and r["method"]==method)[col]
gate_hits = []
for method in ("classical-19", "frozen-512", "hybrid-pca64+19"):
    k4a = l1(4, method, "hdb_ari_mean"); k4c = l1(4, method, "frac_correctK"); k3a = l1(3, method, "hdb_ari_mean")
    if k4a >= 0.6 and k4c >= 0.5 and k3a >= 0.7:
        gate_hits.append((method, "N10-E1", k4a, k4c, k3a))
# also consider best K4 config across levers (deployable HDBSCAN ARI)
best_k4 = max(lever1_rows + [r for r in lever2_rows if r["scope"]=="K4"] +
              [r for r in lever3_rows if r["scope"]=="K4"],
              key=lambda r: r.get("hdb_ari_mean", -1))
best_k3 = max([r for r in lever1_rows if r["K"]==3], key=lambda r: r["hdb_ari_mean"])
if gate_hits:
    m, cfg, a4, c4, a3 = gate_hits[0]
    GATE = (f"DEMO-VIABLE: {m} @ {cfg} — K4 mean ARI={a4} (>=0.6), correctK={c4} (>=0.5), "
            f"K3 mean ARI={a3} (>=0.7).")
else:
    GATE = (f"NOT gated. Best K4 deployable: {best_k4['method'] if 'method' in best_k4 else best_k4.get('feature')} "
            f"[{best_k4.get('scope','L1')}/N{best_k4.get('N','10')}] HDB ARI={best_k4['hdb_ari_mean']} "
            f"(correctK best {max(r['frac_correctK'] for r in lever1_rows if r['K']==4)}); "
            f"best K3 HDB ARI={best_k3['hdb_ari_mean']} ({best_k3['method']}).")
# N-curve shape (hybrid, full8)
ncurve = [(r["N"], r["hdb_ari_mean"]) for r in lever2_rows if r["scope"]=="full8" and r["method"]=="hybrid-pca64+19"]

print("\n=== GATE ===\n" + GATE)
print("N-curve (hybrid, full8):", ncurve)

# ============================================================================
# SAVE — separate DEMO ledger
# ============================================================================
LEDGER_HEADER = "DEMO-SIDE — NOT PAPER RESULTS"
def wledger(fn, rows, fields):
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        f.write(f"# {LEDGER_HEADER}\n")
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
wledger("demo_ledger.csv", lever1_rows + lever2_rows + lever3_rows,
        ["lever","K","scope","N","tag","method","feature","n_subsets","n","hdb_ari_mean","hdb_ari_std",
         "frac_ari_ge0.6","oracle_km_mean","oracle_sp","frac_correctK","kest_mean"])
wledger("lever1_smallK.csv", lever1_rows,
        ["lever","K","N","method","n_subsets","hdb_ari_mean","hdb_ari_std","frac_ari_ge0.6","oracle_km_mean","frac_correctK"])
wledger("lever2_burstN.csv", lever2_rows, ["lever","scope","K","N","tag","method","n","hdb_ari_mean","oracle_km_mean","kest_mean"])
wledger("lever3_fullband.csv", lever3_rows, ["lever","feature","scope","K","n","hdb_ari_mean","oracle_km_mean","frac_correctK"])

report = dict(header=LEDGER_HEADER, note="DEMO-side engineering numbers; NEVER merged into paper tables.",
    gpu_used=False, data="DRFF mavicAir2 (8) — cached seed-777 OPT-B F512 (features_B.npz) + reproduced "
        "windows for classical-19 (alignment VERIFIED) + native 50 MS/s segments for Lever 3",
    lever1=lever1_rows, lever2=lever2_rows, lever3=lever3_rows,
    lever3_feature_list="cfo(global/traj mean/std/slope); env(mean/std/skew/kurt/papr/onset/ac1); "
        "psd(centroid/spread/skew/kurt/flatness/rolloff85/95); band(outer_frac/q0-3); "
        "iq(pow_ratio/corr/gain_imb/phase_skew); logPSD-poly(6)+spec_entropy+kurtI+kurtQ = 36 dims (FIXED)",
    n_curve_hybrid_full8=ncurve, gate=GATE,
    guardrails="eval-only; NO GPU/training; cached F512 + CPU classical; fixed battery (no alpha/feature "
        "selection); paper tables untouched (separate ledger); TEST/board-18 CLOSED; M100 untouched.")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)
print(f"\nsaved -> results/demo_play0/ (demo_ledger.csv[{LEDGER_HEADER}], lever1/2/3 csv, report.json)")
print("CHECKPOINT — DEMO-side; no GPU; no training; paper tables untouched; TEST/M100 sealed.")
