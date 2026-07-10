"""STEP-3 M100 TRANSFER SMOKE TEST (eval-only).
Does the WiSig-trained FROZEN encoder transfer to M100 drone RF, and does the Step-2b
decoherence mechanism survive with only ONE receiver (distance as the within-device axis)?

Encoder FROZEN. CosFace head reproduced deterministically (seed 42, m=0.20 s=32, == Step-2b
provisional), NOT refit on drones. M100 NEVER enters training/head/scaler fit. Small balanced
subset only. DRFF-R2 out of scope. Window spec = A-b (resampled 25 MS/s, present on disk);
A-a (raw 10 MS/s 256-sample) requires raw .bin which are absent -> reported, not built.

  python3 results/step3_m100_smoke/m100_smoke.py
"""
import os, sys, json, csv, re
from collections import defaultdict, Counter
import numpy as np
import torch

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
from integrity import score_locked, hdbscan_pred, purity_std

# ---- constants (frozen; mirror Step-2b) ----
RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
M100_DIR  = os.path.expanduser("~/Desktop/processed/m100")
COS_M, COS_S = 0.20, 32
N = 10                          # discovery burst size (== Step-2b)
MCS_DISC = [5, 7]               # HDBSCAN min_cluster_size for M100 (7 devices, small)
BURSTS_SEL = 28                 # bursts sampled per airframe (small balanced subset)
SEED = 777
rng = np.random.default_rng(SEED)
OUT = _HERE
os.makedirs(OUT, exist_ok=True)

def unitrows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# =====================================================================================
# 0. FROZEN ENCODER + reproduce CosFace head on WiSig train (M100 never touches this)
# =====================================================================================
print("[0] loading WiSig train, freezing encoder, reproducing CosFace head ...")
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
scaler = StandardScaler().fit(Ftr)          # WiSig-train scaler, reused as-is on M100
head = GC.train_cosface(torch.from_numpy(scaler.transform(Ftr).astype(np.float32)).cuda(),
                        torch.from_numpy(ty).cuda(), n_cls, COS_M, COS_S)
del Ftr
print("    head reproduced (seed 42, m=%.2f s=%d, dim=%d)" % (COS_M, COS_S, GC.DIM))

# =====================================================================================
# STEP B — small balanced extraction from the 7 airframes (window spec A-b)
# =====================================================================================
print("[B] building small balanced M100 subset (spec A-b, resampled 25 MS/s) ...")
manifest = json.load(open(os.path.join(M100_DIR, "manifest.json")))
pat = re.compile(r'uav\d+_(\d+)ft_burst(\d+)_(\d+)\.bin')

def burst_session_map(dk):
    """burst_id -> session(1-4) via burst_files parse (verified aligned to distance)."""
    bf = manifest["devices"][dk]["burst_files"]
    return {b: int(pat.match(f).group(2)) for b, f in enumerate(bf)}

sub_iq, sub_af, sub_dist, sub_sess, sub_burst = [], [], [], [], []
airframes = [f"uav{i}" for i in range(1, 8)]
crosstab_af_dist = defaultdict(Counter)
for af_i, dk in enumerate(airframes):
    z = np.load(os.path.join(M100_DIR, f"{dk}.npz"))
    iq, bid, dist, widx = z["iq"], z["burst_id"], z["distance_ft"], z["window_idx"]
    sess_of = burst_session_map(dk)
    # group bursts by (distance, session) cell, pick evenly across cells
    cell_bursts = defaultdict(list)
    for b in np.unique(bid):
        cell_bursts[(int(dist[bid == b][0]), sess_of[int(b)])].append(int(b))
    cells = sorted(cell_bursts.keys())
    per_cell = max(1, BURSTS_SEL // len(cells))
    chosen = []
    for c in cells:
        bl = cell_bursts[c]; rng.shuffle(bl)
        chosen += bl[:per_cell]
    rng.shuffle(chosen); chosen = chosen[:BURSTS_SEL]
    for b in chosen:
        msk = bid == b
        w = iq[msk].astype(np.float32)          # [32,2,256] already channel-first
        d = int(dist[msk][0]); s = sess_of[int(b)]
        # standardize EXACTLY as WiSig inference (per-window zero-mean unit-std on [2,256])
        w = np.stack([W.standardize(w[k]) for k in range(w.shape[0])]).astype(np.float32)
        sub_iq.append(w)
        sub_af += [af_i] * w.shape[0]
        sub_dist += [d] * w.shape[0]
        sub_sess += [s] * w.shape[0]
        sub_burst += [f"{dk}_b{b}"] * w.shape[0]
        crosstab_af_dist[af_i][d] += w.shape[0]

X_iq = np.concatenate(sub_iq, axis=0)                     # [Nw,2,256] standardized
af   = np.array(sub_af); dist = np.array(sub_dist)
sess = np.array(sub_sess); burst = np.array(sub_burst)
print(f"    subset: {X_iq.shape[0]} windows, {len(np.unique(burst))} bursts, "
      f"{len(airframes)} airframes")

# airframe x distance cross-tab (windows)
dists_all = sorted(set(dist.tolist()))
print("    airframe x distance (windows):")
print("      af  " + "  ".join(f"{d:>4}ft" for d in dists_all) + "   total")
for a in range(7):
    row = [crosstab_af_dist[a][d] for d in dists_all]
    print(f"      uav{a+1} " + "  ".join(f"{r:>6}" for r in row) + f"   {sum(row)}")

# =====================================================================================
# STEP C.1 — frozen features (512-D pre-proj + 128-D proj + CosFace)
# =====================================================================================
print("[C1] extracting frozen 512-D / 128-D / CosFace features ...")
F512 = GC.extract512(model, X_iq)
F128 = embed_times(model, X_iq)
FCOS = GC.apply_head(head, {"m": scaler.transform(F512).astype(np.float32)})["m"]
np.savez(os.path.join(OUT, "features.npz"), F512=F512, F128=F128, FCOS=FCOS,
         af=af, dist=dist, sess=sess, burst=burst)
print(f"    F512 {F512.shape}  F128 {F128.shape}  FCOS {FCOS.shape}")

# =====================================================================================
# G1 — window sanity (non-degenerate embeddings)
# =====================================================================================
def g1_check(name, Fw):
    std = float(Fw.std())
    nan = bool(np.isnan(Fw).any())
    U = unitrows(Fw)
    intra, inter = [], []
    for a in range(7):
        m = af == a
        if m.sum() < 2: continue
        C = U[m] @ U[m].T
        iu = np.triu_indices(m.sum(), 1)
        intra.append(C[iu].mean())
        inter.append((U[m] @ U[~m].T).mean())
    return dict(emb=name, std=std, nan=nan,
                intra=float(np.mean(intra)), inter=float(np.mean(inter)),
                sep=float(np.mean(intra) - np.mean(inter)))
g1 = [g1_check("512D", F512), g1_check("128D", F128), g1_check("CosFace", FCOS)]
print("[G1] window sanity (spec A-b):")
for r in g1:
    print(f"     {r['emb']:>8}: std={r['std']:.3f} nan={r['nan']} "
          f"intra_cos={r['intra']:.3f} inter_cos={r['inter']:.3f} sep={r['sep']:+.3f}")

# =====================================================================================
# STEP C.2 — supervised probe (burst-disjoint), 7-way airframe   [G2 existential gate]
# =====================================================================================
print("[C2] supervised airframe probe (burst-disjoint) ...")
ub = np.unique(burst); rng.shuffle(ub)
n_tr = int(0.65 * len(ub)); tr_b = set(ub[:n_tr]); te_b = set(ub[n_tr:])
tr = np.array([b in tr_b for b in burst]); te = ~tr
CHANCE = 1.0 / 7

def probe(Fw, name):
    sc = StandardScaler().fit(Fw[tr])
    Xtr, Xte = sc.transform(Fw[tr]), sc.transform(Fw[te])
    out = {}
    for pname, clf in [("logreg", LogisticRegression(max_iter=2000, C=1.0)),
                       ("mlp", MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                                             early_stopping=True, random_state=0))]:
        clf.fit(Xtr, af[tr])
        pw = clf.predict(Xte)
        acc_w = float((pw == af[te]).mean())
        # burst-level: mean features per test burst -> predict
        bl_acc = []
        proba = clf.predict_proba(Xte)
        for b in np.unique(burst[te]):
            m = (burst[te] == b)
            pred = clf.classes_[proba[m].mean(0).argmax()]
            bl_acc.append(pred == af[te][m][0])
        acc_b = float(np.mean(bl_acc))
        out[pname] = (acc_w, acc_b)
    return out

probe_rows = []
for name, Fw in [("512D", F512), ("128D", F128), ("CosFace", FCOS)]:
    r = probe(Fw, name)
    for pname, (aw, ab) in r.items():
        probe_rows.append(dict(emb=name, probe=pname, acc_window=aw, acc_burst=ab, chance=CHANCE))
        print(f"     {name:>8} {pname:>6}: win={aw:.3f}  burst={ab:.3f}  (chance {CHANCE:.3f})")

best512 = max(x["acc_window"] for x in probe_rows if x["emb"] == "512D")
g2 = "PASS" if best512 >= 0.60 else ("MARGINAL" if best512 >= 0.30 else "FAIL")
print(f"[G2] 512-D best window acc = {best512:.3f} -> {g2}")

# =====================================================================================
# STEP C.3 / G3 — confound: predict DISTANCE from features (Fc metadata absent -> N/A)
# =====================================================================================
print("[C3] confound probe: predict DISTANCE (4-way) from features ...")
conf_rows = []
for name, Fw in [("512D", F512), ("128D", F128)]:
    sc = StandardScaler().fit(Fw[tr])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Fw[tr]), dist[tr])
    acc = float((clf.predict(sc.transform(Fw[te])) == dist[te]).mean())
    chance_d = float(max(np.bincount(dist).astype(float)) / len(dist))
    conf_rows.append(dict(target="distance", emb=name, acc=acc, chance=chance_d))
    print(f"     {name:>8}: distance acc={acc:.3f} (chance {chance_d:.3f})")
print("     Fc/center-frequency: NOT retained per-window in npz -> confound check N/A")

# =====================================================================================
# STEP D — decoherence axis (D0 coherent / D1 multi-condition; D2 N/A: 1 receiver)
# =====================================================================================
print("[D] decoherence axis (D0 coherent vs D1 multi-condition) ...")
def build_D0(emb):
    """10 consecutive windows within one (af,dist,sess) burst_file."""
    bp, bl = [], []
    for b in np.unique(burst):
        idx = np.where(burst == b)[0]           # consecutive within a burst_file
        for k in range(idx.shape[0] // N):
            ch = idx[k*N:(k+1)*N]
            bp.append(emb[ch].mean(0)); bl.append(af[ch][0])
    return unitrows(np.array(bp)), np.array(bl)

def build_D1(emb, seed=0):
    """10 windows of one airframe spread across DIFFERENT (dist,sess) cells, 1 receiver."""
    r = np.random.default_rng(seed)
    bp, bl = [], []
    for a in range(7):
        idx = np.where(af == a)[0]
        cells = defaultdict(list)
        for i in idx: cells[(dist[i], sess[i])].append(i)
        cell_keys = list(cells.keys())
        n_bursts = len(idx) // N
        for _ in range(n_bursts):
            # round-robin pick one window from distinct cells until 10 gathered
            r.shuffle(cell_keys); pick = []
            ci = 0
            while len(pick) < N:
                c = cell_keys[ci % len(cell_keys)]
                if cells[c]:
                    pick.append(cells[c][r.integers(len(cells[c]))])
                ci += 1
                if ci > 1000: break
            if len(pick) == N:
                bp.append(emb[pick].mean(0)); bl.append(a)
    return unitrows(np.array(bp)), np.array(bl)

def balance(bp, bl):
    """equal bursts per device (HDBSCAN is imbalance-sensitive; manifest warned)."""
    r = np.random.default_rng(0)
    per = min(int((bl == a).sum()) for a in np.unique(bl))
    keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    keep = np.array(sorted(keep))
    return bp[keep], bl[keep], per

disc_rows = []
embs = {"128D": F128, "512D": F512, "CosFace": FCOS}
for proto, builder in [("D0_coherent", build_D0), ("D1_multicond", lambda e: build_D1(e, SEED))]:
    for ename, E in embs.items():
        bp, bl = builder(E)
        bp, bl, per = balance(bp, bl)
        for mcs in MCS_DISC:
            pred = hdbscan_pred(bp, mcs)
            sc = score_locked(pred, bl)
            disc_rows.append(dict(proto=proto, emb=ename, mcs=mcs, n_bursts=len(bl),
                                  per_dev=per, **sc))
            print(f"     {proto:>13} {ename:>8} mcs={mcs}: ARI={sc['ARI']:.3f} "
                  f"NMI={sc['NMI']:.3f} pur={sc['purity']:.3f} K={sc['K_est']}(t7) "
                  f"noise={sc['noise']:.3f} nb={len(bl)}")

# ---- G4 mechanism verdict ----
def mean_ari(proto, emb):
    v = [r["ARI"] for r in disc_rows if r["proto"] == proto and r["emb"] == emb]
    return float(np.mean(v))
best_emb = max(embs, key=lambda e: mean_ari("D1_multicond", e))
d0 = mean_ari("D0_coherent", best_emb); d1 = mean_ari("D1_multicond", best_emb)
d1K = np.mean([r["K_est"] for r in disc_rows if r["proto"]=="D1_multicond" and r["emb"]==best_emb])
if max(d0, d1) <= 0.05:
    g4 = (f"BOTH LOW -> feature-level transfer failure (defer to Step-C/G2); neither protocol "
          f"discovers [{best_emb}] d0={d0:.3f} d1={d1:.3f} K~{d1K:.1f} (true=7)")
elif d1 > d0 + 0.05 and abs(d1K - 7) <= 3:
    g4 = f"D1 RECOVERS (distance substitutes for receiver) [{best_emb}] d1={d1:.3f}>d0={d0:.3f} K~{d1K:.1f}"
elif d1 <= d0 + 0.02:
    g4 = f"D1 COLLAPSES vs D0 (mechanism receiver-specific, like WiSig P2) [{best_emb}] d1={d1:.3f} d0={d0:.3f} K~{d1K:.1f}"
else:
    g4 = f"AMBIGUOUS [{best_emb}] d1={d1:.3f} d0={d0:.3f} K~{d1K:.1f}"
print(f"[G4] {g4}")

# =====================================================================================
# save tables
# =====================================================================================
def wcsv(fn, rows):
    if not rows: return
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# cross-tab rows
ct_rows = []
for a in range(7):
    row = {"airframe": f"uav{a+1}"}
    for d in dists_all: row[f"{d}ft"] = crosstab_af_dist[a][d]
    ct_rows.append(row)
wcsv("crosstab_af_distance.csv", ct_rows)
wcsv("g1_window_sanity.csv", g1)
wcsv("probe_supervised.csv", probe_rows)
wcsv("probe_confound.csv", conf_rows)
wcsv("discovery_decoherence.csv", disc_rows)
report = dict(window_spec="A-b resampled 25MS/s (A-a raw 10MS/s absent: raw .bin not on disk)",
              n_receivers=1, n_distances=len(dists_all), distances=dists_all,
              n_sessions=4, n_airframes=7, subset_windows=int(X_iq.shape[0]),
              subset_bursts=int(len(np.unique(burst))),
              g1=g1, g2_512_best_window=best512, g2_verdict=g2,
              probe=probe_rows, confound=conf_rows, discovery=disc_rows, g4=g4)
json.dump(report, open(os.path.join(OUT, "m100_smoke_report.json"), "w"), indent=2)
print("\nSaved -> results/step3_m100_smoke/  (features.npz + 5 CSVs + report.json)")
print(f"\n=== VERDICTS ===\nG2 (transfer): {g2} (512-D best win acc {best512:.3f}, chance {CHANCE:.3f})\nG4 (mechanism): {g4}")
