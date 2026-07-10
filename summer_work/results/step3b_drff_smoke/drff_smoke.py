"""STEP-3b DRFF-R2 TRANSFER SMOKE TEST (eval-only, second cross-domain gate).
WiFi(WiSig)->OcuSync(DRFF-R2) 5.8 GHz. Does the FROZEN WiSig encoder retain same-model
airframe signal, and does the 50->25 decimation choice (A band-limited vs B bw-preserving)
matter? Encoder FROZEN. CosFace head reproduced NOT refit. DRFF-R2 eval-only (never trains
anything). Balanced per-airframe. Burst-disjoint (segment-level) splits. drff_r2_mixed NOT
touched. No M100/ManySig/ORACLE/TEST.

  python3 results/step3b_drff_smoke/drff_smoke.py
"""
import os, sys, json, csv, re, glob
from collections import defaultdict, Counter
import numpy as np
import torch
from scipy.signal import firwin, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import geometry_consolidate as GC
from burst_probe import embed_times
from integrity import score_locked, hdbscan_pred

# ---- frozen constants (mirror Step-2b / Step-3) ----
RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
DRFF_DIR  = os.path.expanduser("~/Desktop/processed/drff_r2")
COS_M, COS_S = 0.20, 32
WIN = 256
N = 10                       # discovery burst size
MCS_DISC = [5, 7]
CAP = 320                    # max windows sampled per airframe (small balanced subset)
NTAPS = 129
SEED = 777
OUT = _HERE
os.makedirs(OUT, exist_ok=True)

def unitrows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# FIR anti-alias filters (cutoff normalized to OLD Nyquist = 25 MHz).
# OPT-A: cutoff 0.5 -> 12.5 MHz = new Nyquist, proper anti-alias, keeps central 25 MHz band.
# OPT-B: cutoff 0.9 -> 22.5 MHz, passes wider content; energy 12.5-22.5 MHz aliases back on
#        the /2 decimation (mild edge aliasing) to RETAIN fingerprint energy in the outer band.
TAPS = {"A": firwin(NTAPS, 0.5), "B": firwin(NTAPS, 0.9)}

# =====================================================================================
# 0. FROZEN ENCODER + reproduce CosFace head on WiSig train (DRFF never touches this)
# =====================================================================================
print("[0] WiSig train -> frozen encoder + reproduced CosFace head ...")
TXD, _ = W.load_manytx(eq=0, verbose=False)
sp = json.load(open(SPLIT)); train_tx = sp["train_tx"]; n_cls = len(train_tx)
model = RFEncoder().cuda()
model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
model.eval()
for pth in model.parameters(): pth.requires_grad_(False)
tt = np.concatenate([GC.consec_windows(TXD, tx, GC.TRAIN_PER_DEV) for tx in train_tx])
ty = np.concatenate([np.full(min(GC.TRAIN_PER_DEV, TXD[tx]["iq"].shape[0]), i)
                     for i, tx in enumerate(train_tx)])
Ftr = GC.extract512(model, tt); del tt
scaler = StandardScaler().fit(Ftr)
head = GC.train_cosface(torch.from_numpy(scaler.transform(Ftr).astype(np.float32)).cuda(),
                        torch.from_numpy(ty).cuda(), n_cls, COS_M, COS_S)
del Ftr
print("    head reproduced (seed42, m=%.2f s=%d dim=%d)" % (COS_M, COS_S, GC.DIM))

# =====================================================================================
# STEP A/B — build balanced window subset for OPT-A and OPT-B (spec: 256 win @ 25 MS/s)
# transform chain per window = W.standardize([2,256]) -> compute_stft_batch -> encoder
# (identical to WiSig inference; reuses the exact loader funcs; diff below).
# =====================================================================================
manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
pat = re.compile(r'(.+?)_(\d+)_hover')
# map airframe(TD) -> list of npz files
af_files = defaultdict(list)
for r in manifest["clean"]:
    npz = r["file"].replace(".mat", ".npz")
    af_files[r["TD"]].append(npz)
all_af = sorted(af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
model_of = {af: pat.match(af + "_hover").group(1) for af in all_af}

def decimate_file(iq, opt):
    """iq [2,L] f16 @50MS/s -> [2, L//2] f32 @25MS/s via FIR(opt)+/2 downsample."""
    x = iq.astype(np.float32)
    pad = min(3 * NTAPS, x.shape[1] - 1)
    y = filtfilt(TAPS[opt], [1.0], x, axis=1, padlen=pad)
    return y[:, ::2]

def build_subset(opt, af_list, cap=CAP):
    """Returns dict of arrays: X_iq[Nw,2,256], af, model, seg_id(global), U,D,C,Height,State."""
    rng = np.random.default_rng(SEED)
    Xs, afs, mods, segs, Us, Ds, Cs, Hs, Ss = [], [], [], [], [], [], [], [], []
    gseg = 0
    for a in af_list:
        got = 0
        files = af_files[a][:]; rng.shuffle(files)
        # collect (file, seg) units, shuffle for diversity across conditions
        units = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]):
                units.append((fn, si))
        rng.shuffle(units)
        zcache = {}
        for fn, si in units:
            if got >= cap: break
            if fn not in zcache:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zcache[fn] = dict(dec=decimate_file(z["iq"], opt), sb=z["seg_bounds"],
                                  U=str(z["U"]), D=str(z["D"]), C=int(z["C"]),
                                  H=str(z["Height"]), St=str(z["State"]))
            zc = zcache[fn]
            off, ln = zc["sb"][si]
            o2, l2 = off // 2, ln // 2                      # seg bounds at 25 MS/s
            seg = zc["dec"][:, o2:o2 + l2]                  # [2, l2]
            nw = seg.shape[1] // WIN
            if nw < 1: continue
            take = min(nw, cap - got, 12)                   # <=12 win/segment for spread
            for k in range(take):
                w = seg[:, k * WIN:(k + 1) * WIN]
                Xs.append(W.standardize(w.astype(np.float32)))
                afs.append(a); mods.append(model_of[a]); segs.append(gseg)
                Us.append(zc["U"]); Ds.append(zc["D"]); Cs.append(zc["C"])
                Hs.append(zc["H"]); Ss.append(zc["St"])
            got += take; gseg += 1
    return dict(X=np.stack(Xs).astype(np.float32), af=np.array(afs), model=np.array(mods),
                seg=np.array(segs), U=np.array(Us), D=np.array(Ds), C=np.array(Cs),
                H=np.array(Hs), St=np.array(Ss))

print("[A/B] building OPT-A and OPT-B window subsets (all 23 airframes) ...")
SUB = {opt: build_subset(opt, all_af) for opt in ("A", "B")}
for opt in ("A", "B"):
    s = SUB[opt]
    print(f"    OPT-{opt}: {s['X'].shape[0]} windows, {len(np.unique(s['seg']))} segments, "
          f"{len(np.unique(s['af']))} airframes")

# =====================================================================================
# STEP A/B features (512-D + 128-D + CosFace) per option
# =====================================================================================
print("[feat] extracting frozen features for both options ...")
FEAT = {}
for opt in ("A", "B"):
    s = SUB[opt]
    F512 = GC.extract512(model, s["X"]); F128 = embed_times(model, s["X"])
    FCOS = GC.apply_head(head, {"m": scaler.transform(F512).astype(np.float32)})["m"]
    FEAT[opt] = dict(F512=F512, F128=F128, FCOS=FCOS)
    np.savez(os.path.join(OUT, f"features_{opt}.npz"), F512=F512, F128=F128, FCOS=FCOS,
             af=s["af"], model=s["model"], seg=s["seg"], U=s["U"], D=s["D"], C=s["C"],
             H=s["H"], St=s["St"])

# =====================================================================================
# G1 — window sanity (both options)
# =====================================================================================
def g1(opt, name, Fw, af):
    U = unitrows(Fw); intra, inter = [], []
    for a in np.unique(af):
        m = af == a
        if m.sum() < 2: continue
        C = U[m] @ U[m].T; iu = np.triu_indices(m.sum(), 1)
        intra.append(C[iu].mean()); inter.append((U[m] @ U[~m].T).mean())
    return dict(opt=opt, emb=name, std=float(Fw.std()), nan=bool(np.isnan(Fw).any()),
                intra=float(np.mean(intra)), inter=float(np.mean(inter)),
                sep=float(np.mean(intra) - np.mean(inter)))
g1_rows = []
print("[G1] window sanity:")
for opt in ("A", "B"):
    for name in ("F512", "F128", "FCOS"):
        r = g1(opt, name.replace("F", ""), FEAT[opt][name], SUB[opt]["af"])
        g1_rows.append(r)
        print(f"    OPT-{opt} {r['emb']:>4}: std={r['std']:.3f} nan={r['nan']} "
              f"intra={r['intra']:.3f} inter={r['inter']:.3f} sep={r['sep']:+.3f}")

# =====================================================================================
# STEP B — supervised probes (burst-disjoint at SEGMENT level)
# =====================================================================================
def balanced_mask(af, seg, sub_af, rng):
    """restrict to sub_af airframes, balance windows per airframe to the group min."""
    keep = np.isin(af, sub_af)
    idx = np.where(keep)[0]
    per = min((af[idx] == a).sum() for a in sub_af)
    sel = []
    for a in sub_af:
        ii = idx[af[idx] == a]; rng.shuffle(ii); sel += ii[:per].tolist()
    return np.array(sorted(sel)), per

def probe(Fw, y, seg, rng):
    """burst(segment)-disjoint probe -> (acc_win, acc_burst) for logreg & mlp."""
    y = np.unique(y, return_inverse=True)[1]        # integer-encode labels (sklearn 1.8 MLP)
    useg = np.unique(seg); rng.shuffle(useg)
    tr_s = set(useg[:int(0.65 * len(useg))])
    tr = np.array([s in tr_s for s in seg]); te = ~tr
    # guarantee every class appears in train; if not, this split is degenerate -> report nan
    if len(set(y[tr])) < len(set(y)) or te.sum() == 0 or tr.sum() == 0:
        return {"logreg": (float("nan"), float("nan")), "mlp": (float("nan"), float("nan"))}
    sc = StandardScaler().fit(Fw[tr]); Xtr, Xte = sc.transform(Fw[tr]), sc.transform(Fw[te])
    out = {}
    for pn, clf in [("logreg", LogisticRegression(max_iter=2000)),
                    ("mlp", MLPClassifier((256,), max_iter=300, early_stopping=True, random_state=0))]:
        clf.fit(Xtr, y[tr]); proba = clf.predict_proba(Xte)
        aw = float((clf.classes_[proba.argmax(1)] == y[te]).mean())
        ab = []
        for s in np.unique(seg[te]):
            m = seg[te] == s
            ab.append(clf.classes_[proba[m].mean(0).argmax()] == y[te][m][0])
        out[pn] = (aw, float(np.mean(ab)))
    return out

rng = np.random.default_rng(SEED)
probe_rows = []
scopes = [("all23", all_af, 1/23)]
for grp in ("mavicAir2", "mavicAir2s"):
    grp_af = [a for a in all_af if model_of[a] == grp]
    scopes.append((f"same_{grp}", grp_af, 1/len(grp_af)))
print("[B] supervised probes (segment-disjoint):")
for opt in ("A", "B"):
    s = SUB[opt]
    for sname, sub_af, chance in scopes:
        msk, per = balanced_mask(s["af"], s["seg"], sub_af, np.random.default_rng(SEED))
        for name in ("F512", "F128"):
            r = probe(FEAT[opt][name][msk], s["af"][msk], s["seg"][msk], np.random.default_rng(1))
            for pn, (aw, ab) in r.items():
                probe_rows.append(dict(opt=opt, scope=sname, emb=name.replace("F",""), probe=pn,
                                       n_af=len(sub_af), per_af=int(per), acc_win=aw,
                                       acc_burst=ab, chance=chance))
                print(f"    OPT-{opt} {sname:>16} {name[1:]:>3} {pn:>6}: "
                      f"win={aw:.3f} burst={ab:.3f} (chance {chance:.3f}, {len(sub_af)}af x{per})")

# G2 verdict: best same-model, 512-D, better A/B, burst-level
sm = [r for r in probe_rows if r["scope"].startswith("same_") and r["emb"]=="512"
      and not np.isnan(r["acc_burst"])]
best = max(sm, key=lambda r: r["acc_burst"]) if sm else None
if best:
    b = best["acc_burst"]
    g2 = "PASS" if b >= 0.50 else ("MARGINAL" if b >= 0.25 else "FAIL")
    g2_line = (f"{g2}: best same-model 512-D burst acc={b:.3f} "
               f"({best['scope']}, OPT-{best['opt']}, chance {best['chance']:.3f})")
else:
    g2, g2_line = "FAIL", "FAIL: no valid same-model split"
print(f"[G2] {g2_line}")

# =====================================================================================
# STEP C / G4 — confound: predict condition (D/C/U/Height/State) within same-model groups
# =====================================================================================
print("[C] confound probes within same-model groups (predict condition, not unit):")
conf_rows = []
for opt in ("A", "B"):
    s = SUB[opt]
    for grp in ("mavicAir2", "mavicAir2s"):
        grp_af = [a for a in all_af if model_of[a] == grp]
        msk, per = balanced_mask(s["af"], s["seg"], grp_af, np.random.default_rng(SEED))
        seg = s["seg"][msk]
        useg = np.unique(seg); r2 = np.random.default_rng(2); r2.shuffle(useg)
        tr_s = set(useg[:int(0.65*len(useg))]); tr = np.array([x in tr_s for x in seg]); te = ~tr
        F = FEAT[opt]["F512"][msk]
        sc = StandardScaler().fit(F[tr])
        for lab in ("D", "C", "U", "H", "St"):
            y = s[lab][msk].astype(str)
            if len(set(y)) < 2 or len(set(y[tr])) < len(set(y)):
                conf_rows.append(dict(opt=opt, group=grp, target=lab, acc=float("nan"),
                                      chance=float(max(Counter(y).values())/len(y)), nclass=len(set(y))))
                continue
            clf = LogisticRegression(max_iter=2000).fit(sc.transform(F[tr]), y[tr])
            acc = float((clf.predict(sc.transform(F[te])) == y[te]).mean())
            ch = float(max(Counter(y).values())/len(y))
            conf_rows.append(dict(opt=opt, group=grp, target=lab, acc=acc, chance=ch, nclass=len(set(y))))
            print(f"    OPT-{opt} {grp:>11} {lab:>2}: acc={acc:.3f} (chance {ch:.3f}, {len(set(y))}cls)")

# unit-vs-condition cross-tab (from metadata, OPT-A subset) for the flag
def crosstab_unit_cond(grp):
    s = SUB["A"]; grp_af = [a for a in all_af if model_of[a]==grp]
    msk,_ = balanced_mask(s["af"], s["seg"], grp_af, np.random.default_rng(SEED))
    tab = defaultdict(Counter)
    for a, d, c in zip(s["af"][msk], s["D"][msk], s["C"][msk]):
        tab[a][f"{d}/c{c}"] += 1
    return tab

# =====================================================================================
# STEP D — same-model discovery smoke (D0 coherent / D1 multi-condition), both options
# =====================================================================================
print("[D] same-model discovery (D0 coherent vs D1 multi-condition):")
def build_D0(emb, af, seg):
    bp, bl = [], []
    for sgid in np.unique(seg):
        idx = np.where(seg == sgid)[0]
        for k in range(len(idx)//N):
            ch = idx[k*N:(k+1)*N]; bp.append(emb[ch].mean(0)); bl.append(af[ch][0])
    return unitrows(np.array(bp)), np.array(bl)
def build_D1(emb, af, D, C, seed=0):
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
    return unitrows(np.array(bp)), np.array(bl)
def balance_bursts(bp, bl):
    r = np.random.default_rng(0); per = min(int((bl==a).sum()) for a in np.unique(bl))
    keep = []
    for a in np.unique(bl):
        ii = np.where(bl==a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    keep = np.array(sorted(keep)); return bp[keep], bl[keep], per

disc_rows = []
for opt in ("A", "B"):
    s = SUB[opt]
    for grp, ktrue in (("mavicAir2", 8), ("mavicAir2s", 7)):
        grp_af = [a for a in all_af if model_of[a]==grp]
        msk, per = balanced_mask(s["af"], s["seg"], grp_af, np.random.default_rng(SEED))
        af_m, seg_m, D_m, C_m = s["af"][msk], s["seg"][msk], s["D"][msk], s["C"][msk]
        for proto, fn in (("D0_coherent", lambda E: build_D0(E, af_m, seg_m)),
                          ("D1_multicond", lambda E: build_D1(E, af_m, D_m, C_m, SEED))):
            for ename, key in (("128D","F128"),("512D","F512"),("CosFace","FCOS")):
                bp, bl = fn(FEAT[opt][key][msk])
                if len(bl) < ktrue: continue
                bp, bl, pb = balance_bursts(bp, bl)
                for mcs in MCS_DISC:
                    sc = score_locked(hdbscan_pred(bp, mcs), bl)
                    disc_rows.append(dict(opt=opt, group=grp, ktrue=ktrue, proto=proto, emb=ename,
                                          mcs=mcs, n_bursts=len(bl), **sc))
                    print(f"    OPT-{opt} {grp:>11} {proto:>13} {ename:>8} mcs={mcs}: "
                          f"ARI={sc['ARI']:.3f} NMI={sc['NMI']:.3f} pur={sc['purity']:.3f} "
                          f"K={sc['K_est']}(t{ktrue}) noise={sc['noise']:.3f} nb={len(bl)}")

# =====================================================================================
# G3 (decimation A/B) verdict + save
# =====================================================================================
def best_scope(scope, opt, emb="512"):
    v = [r["acc_burst"] for r in probe_rows if r["scope"]==scope and r["opt"]==opt
         and r["emb"]==emb and not np.isnan(r["acc_burst"])]
    return max(v) if v else float("nan")
accA = np.nanmean([best_scope(sc[0], "A") for sc in scopes if sc[0].startswith("same_")])
accB = np.nanmean([best_scope(sc[0], "B") for sc in scopes if sc[0].startswith("same_")])
g3 = f"OPT-{'A' if accA>=accB else 'B'} retains more (same-model 512-D burst: A={accA:.3f} B={accB:.3f}, delta={accA-accB:+.3f})"
print(f"[G3] {g3}")

def wcsv(fn, rows):
    if not rows: return
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
wcsv("g1_sanity.csv", g1_rows)
wcsv("probe_supervised.csv", probe_rows)
wcsv("probe_confound.csv", conf_rows)
wcsv("discovery_samemodel.csv", disc_rows)
# unit counts + crosstab
unit_rows = []
for grp in ("mavicAir2", "mavicAir2s"):
    for a, cnt in sorted(crosstab_unit_cond(grp).items()):
        unit_rows.append(dict(group=grp, airframe=a, conditions=";".join(f"{k}:{v}" for k,v in sorted(cnt.items()))))
wcsv("unit_condition_crosstab.csv", unit_rows)

report = dict(domain="WiFi(WiSig25MS/s)->OcuSync DRFF-R2 5.8GHz 50MS/s",
              U_is_airframe=False, TD_is_airframe=True,
              airframes_per_model={m: sum(1 for a in all_af if model_of[a]==m)
                                   for m in sorted(set(model_of.values()))},
              n_airframes=len(all_af),
              decimation=dict(OPT_A="FIR-129 cutoff0.5(12.5MHz)+/2, central 25MHz band",
                              OPT_B="FIR-129 cutoff0.9(22.5MHz)+/2, mild edge-alias, retains outer band"),
              subset={opt: dict(windows=int(SUB[opt]["X"].shape[0]),
                                segments=int(len(np.unique(SUB[opt]["seg"])))) for opt in ("A","B")},
              g1=g1_rows, g2=g2_line, g3=g3, probe=probe_rows, confound=conf_rows, discovery=disc_rows)
json.dump(report, open(os.path.join(OUT, "drff_smoke_report.json"), "w"), indent=2, default=str)
print("\nSaved -> results/step3b_drff_smoke/")
print(f"\n=== VERDICTS ===\nG2 (same-model transfer): {g2_line}\nG3 (decimation): {g3}")
