"""STEP-6 DOMAIN ADAPTATION (Days 1-3) — FIRST gradient steps since best_model.pt was frozen.
Question: does light adaptation on the OTHER 15 drone airframes unlock open-set discovery of the
8 UNSEEN mavicAir2 units (close the cross-domain wall: frozen HDBSCAN 0.075 / oracle-K 0.218)?

DATA DISCIPLINE (hard):
  EVAL GROUP  = all 8 mavicAir2 airframes — QUARANTINED from adaptation in every form (no windows,
                no norm stats, no early-stopping signal).
  ADAPT POOL  = the other 15 usable airframes (labels = airframe TD). OPT-B decimation, same
                256-window / STFT[2,33,13] / per-window standardize chain as step3b.
  POOL VAL    = 15% of pool SEGMENTS (segment-disjoint) — early-stopping / ckpt selection only.
  REPLAY      = WiSig 109 TRAIN devices (legal training set), ~30% of each batch (anti-forgetting).
  RETENTION   = WiSig DEV slices (seeds 201-202), eval-only.
  mavicAir2s (x7) is SPENT as adaptation data -> the adapted-discovery claim rests on mavicAir2
  alone (stated trade-off).

FIXED BATTERY — 3 recipes, NO iteration/sweeps beyond what is written here:
  R1 HEAD-ADAPT   : backbone frozen; train residual adapter (512->128->512, zero-init) + proj head. LR 1e-3
  R2 PARTIAL      : unfreeze last resblock of each branch (b4) + cross-attn + norm + proj head. LR 1e-4
  R3 FULL         : everything unfrozen. LR 5e-5
  Common: start from best_model.pt (fresh reload per recipe; NEVER overwritten). SupCon on airframe
  labels, tau=0.5 CONSTANT. AdamW, cosine sched, warmup 10%, grad-clip 1.0, AMP. Mixed-domain replay
  (~70% DRFF pool + ~30% WiSig train). 3000 steps, ckpt every 500, keep best by pool-val intra/inter
  cosine gap. Balanced per-airframe batch sampling. Seed 1234.

EVAL (identical for frozen baseline + R1/R2/R3, on the untouched mavicAir2 8, 512-D pre-projection):
  a. DISCOVERY: burst-mean N=10, E1 multi-cond + E0 coherent, HDBSCAN(locked noise) + oracle-K@8
     (CEILING) + kNN-1/5 + intra/inter gap.
  b. SUPERVISED session-disjoint probe (session=(TD,U,D,C)) — did the linear ceiling move?
  c. WiSig RETENTION: 2 DEV slices (201,202) P2 bursts K18, head-free 512-D, HDBSCAN + oracle-K@18.
  d. combined table {frozen,R1,R2,R3} x {E1 HDB, E1 oracleK@8, E0 HDB, kNN1, gap, probe, retention}.

PRE-REGISTERED GATE (mavicAir2 E1 oracle-K@8, best recipe): PASS>=0.50 / PARTIAL 0.35-0.50 / FAIL<0.35.
Also report HDBSCAN (deployable, non-oracle) movement. Do NOT tune past the gate.

  python3 results/step6_adaptation/adapt.py
"""
import os, sys, json, csv, re, copy, math
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin, filtfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity"),
          os.path.join(_SW, "results", "step2b_decoherence")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder, SupervisedContrastiveLoss
import wisig_manytx as W
import geometry_consolidate as GC
from integrity import score_locked, hdbscan_pred, unitrows
from decohere import bursts_p2_multidate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
PART      = os.path.join(RUN_DIR, "splits", "discovery_partition.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")     # FROZEN, never overwritten
DRFF_DIR  = os.path.expanduser("~/Desktop/processed/drff_r2")
OUT       = _HERE
ADAPT_ROOT = os.path.join(_SW, "runs", "drff_adapt")

# ---- FIXED CONFIG (stated before launch) ----
SEED = 1234
STEPS = 3000
CKPT_EVERY = 500
WARMUP = 300                       # 10%
N = 10                             # burst size
WIN = 256
NTAPS = 129
DRFF_CAP = 2000                    # pool windows/airframe (<=12/seg)
EVAL_CAP = 320                     # mavicAir2 eval windows/airframe (== step5)
VAL_FRAC = 0.15                    # pool segments held out (segment-disjoint)
REPLAY_PER_DEV = 250               # WiSig replay windows/train-device
BATCH_DRFF_AF = 7                  # per pool airframe -> 15*7 = 105
BATCH_WIS_DEV, BATCH_WIS_PER = 15, 3   # 15 devices x3 = 45  -> total 150, ~70% DRFF
TAU = 0.5
MCS_DRFF = [5, 7]
MCS_WISIG = [13, 15, 17]
KDRFF, KWISIG = 8, 18
RETENTION_BAR = 0.65
LR = {"R1": 1e-3, "R2": 1e-4, "R3": 5e-5}
RECIPES = ["baseline", "R1", "R2", "R3"]

torch.manual_seed(SEED); np.random.seed(SEED)
def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

# ============================================================================
# DATA
# ============================================================================
manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
pat = re.compile(r'(.+?)_(\d+)_hover')
af_files = defaultdict(list)
for r in manifest["clean"]:
    af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
all_af = sorted(af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
model_of = {a: pat.match(a + "_hover").group(1) for a in all_af}
POOL_AF = [a for a in all_af if model_of[a] != "mavicAir2"]        # 15
EVAL_AF = [a for a in all_af if model_of[a] == "mavicAir2"]        # 8 (quarantined)
assert len(POOL_AF) == 15 and len(EVAL_AF) == 8

TAPS_B = firwin(NTAPS, 0.9)
def decimate_B(iq):
    x = iq.astype(np.float32); pad = min(3 * NTAPS, x.shape[1] - 1)
    return filtfilt(TAPS_B, [1.0], x, axis=1, padlen=pad)[:, ::2]

def build_windows(af_list, cap, keep_meta=False, seed=SEED):
    """OPT-B decimate -> <=12 win/seg -> per-window standardize. Returns time[N,2,256], seg ids,
    airframe idx, and (U,D,C) if keep_meta."""
    rng = np.random.default_rng(seed)
    Xt, y, seg, Us, Ds, Cs = [], [], [], [], [], []
    gseg = 0
    for ai, a in enumerate(af_list):
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
                zc[fn] = dict(dec=decimate_B(z["iq"]), sb=z["seg_bounds"],
                              U=str(z["U"]), D=str(z["D"]), C=int(z["C"]))
            zz = zc[fn]; off, ln = zz["sb"][si]; o2, l2 = off // 2, ln // 2
            nw = l2 // WIN
            if nw < 1: continue
            take = min(nw, cap - got, 12)
            segw = zz["dec"][:, o2:o2 + l2]
            for k in range(take):
                Xt.append(W.standardize(segw[:, k * WIN:(k + 1) * WIN].astype(np.float32)))
                y.append(ai); seg.append(gseg)
                if keep_meta: Us.append(zz["U"]); Ds.append(zz["D"]); Cs.append(zz["C"])
            got += take; gseg += 1
    Xt = np.stack(Xt).astype(np.float32)
    out = dict(Xt=Xt, y=np.array(y), seg=np.array(seg))
    if keep_meta: out.update(U=np.array(Us), D=np.array(Ds), C=np.array(Cs))
    return out

print("[data] building DRFF adaptation pool (15 airframes, OPT-B) ...")
pool = build_windows(POOL_AF, DRFF_CAP)
# segment-disjoint 15% val per airframe
rng = np.random.default_rng(SEED)
val_mask = np.zeros(len(pool["y"]), bool)
for ai in range(15):
    segs = np.unique(pool["seg"][pool["y"] == ai]); rng.shuffle(segs)
    nval = min(max(0, int(round(VAL_FRAC * len(segs)))), max(0, len(segs) - 1))  # keep >=1 train seg
    if nval == 0 and len(segs) >= 2: nval = 1
    vs = set(segs[:nval].tolist())
    val_mask |= np.array([(pool["y"][i] == ai and pool["seg"][i] in vs) for i in range(len(pool["y"]))])
tr_mask = ~val_mask
print(f"    pool windows: {len(pool['y'])} ({tr_mask.sum()} train / {val_mask.sum()} val, "
      f"{len(np.unique(pool['seg']))} segments)")
for ai, a in enumerate(POOL_AF):
    print(f"      {a:14} idx{ai:2d}: {(pool['y']==ai).sum():5d} win "
          f"({(tr_mask&(pool['y']==ai)).sum()} tr / {(val_mask&(pool['y']==ai)).sum()} val)")

print("[data] STFT for pool ...")
pool_Xs = W.compute_stft_batch(pool["Xt"])

print("[data] building mavicAir2 EVAL group (quarantined, cap 320) ...")
ev = build_windows(EVAL_AF, EVAL_CAP, keep_meta=True)
ev_Xs = W.compute_stft_batch(ev["Xt"])
print(f"    eval windows: {len(ev['y'])}, {len(np.unique(ev['seg']))} segments, 8 airframes")

print("[data] loading WiSig (ManyTx) for replay + retention ...")
TXD, _ = W.load_manytx(eq=0, verbose=False)
sp = json.load(open(SPLIT)); part = json.load(open(PART))
train_tx = sp["train_tx"]; dev_tx = part["dev_tx"]
# replay windows from the 109 legal training devices
rep_Xt, rep_y = [], []
for di, tx in enumerate(train_tx):
    t = GC.consec_windows(TXD, tx, REPLAY_PER_DEV)
    rep_Xt.append(t); rep_y.append(np.full(t.shape[0], 1000 + di))   # labels disjoint from DRFF 0..14
rep_Xt = np.concatenate(rep_Xt).astype(np.float32); rep_y = np.concatenate(rep_y)
rep_Xs = W.compute_stft_batch(rep_Xt)
print(f"    replay windows: {len(rep_y)} from {len(train_tx)} WiSig train devices")

# DEV cache for retention (time+STFT+rx+date); embed per recipe
WIN_CACHE = 2400
dev_cache = {}
for tx in dev_tx:
    t = GC.consec_windows(TXD, tx, WIN_CACHE); Wn = t.shape[0]
    dev_cache[tx] = dict(Xt=t, Xs=W.compute_stft_batch(t),
                         rx=TXD[tx]["rx"][:Wn].copy(), date=TXD[tx]["date"][:Wn].copy())

# precompute torch tensors on CPU for training
pool_Xt_t = torch.from_numpy(pool["Xt"]); pool_Xs_t = torch.from_numpy(pool_Xs)
rep_Xt_t = torch.from_numpy(rep_Xt); rep_Xs_t = torch.from_numpy(rep_Xs)
# per-class training index lists
pool_tr_idx = {ai: np.where(tr_mask & (pool["y"] == ai))[0] for ai in range(15)}
rep_by_dev = {d: np.where(rep_y == 1000 + d)[0] for d in range(len(train_tx))}

# ============================================================================
# MODEL WRAPPER (adapter-aware, uniform 512-D eval path)
# ============================================================================
class Adapter(nn.Module):
    """residual bottleneck 512->128->512, zero-init output -> starts as identity."""
    def __init__(self, dim=512, bott=128):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bott); self.up = nn.Linear(bott, dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, h):
        return h + self.up(F.relu(self.down(self.ln(h))))

class AdaptedEncoder(nn.Module):
    def __init__(self, base, use_adapter):
        super().__init__()
        self.base = base
        self.adapter = Adapter().cuda() if use_adapter else nn.Identity()
    def enc512(self, xt, xs):
        return self.adapter(self.base.get_encoder_output(xt, xs))
    def forward(self, xt, xs):
        return F.normalize(self.base.projection_head(self.enc512(xt, xs)), dim=1, eps=1e-6)

def fresh_base():
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    return m

def configure(recipe):
    """returns (enc, trainable_params). Fresh base each time; best_model.pt untouched."""
    base = fresh_base()
    enc = AdaptedEncoder(base, use_adapter=(recipe == "R1")).cuda()
    for p in enc.parameters(): p.requires_grad_(False)
    if recipe == "baseline":
        return enc, []
    if recipe == "R1":
        train_mods = [enc.adapter, base.projection_head]
    elif recipe == "R2":
        train_mods = [base.time_branch.b4, base.spectral_branch.b4,
                      base.cross_attn, base.norm, base.projection_head]
    elif recipe == "R3":
        train_mods = [base]
    params = []
    for mod in train_mods:
        for p in mod.parameters(): p.requires_grad_(True);
    params = [p for p in enc.parameters() if p.requires_grad]
    return enc, params

# ============================================================================
# EMBED + SCORING helpers
# ============================================================================
@torch.no_grad()
def embed512(enc, Xt, Xs, batch=1024):
    enc.eval()
    f = np.empty((Xt.shape[0], 512), dtype=np.float32)
    xt = torch.from_numpy(Xt); xs = torch.from_numpy(Xs)
    for i in range(0, Xt.shape[0], batch):
        with torch.amp.autocast('cuda'):
            f[i:i+batch] = enc.enc512(xt[i:i+batch].cuda(), xs[i:i+batch].cuda()).float().cpu().numpy()
    return f

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
    km = float(adjusted_rand_score(yi, KMeans(K, n_init=10, random_state=0).fit_predict(bp)))
    try:
        sp_ = SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                 n_neighbors=15).fit_predict(bp)
        sp_a = float(adjusted_rand_score(yi, sp_))
    except Exception:
        sp_a = float("nan")
    return km, sp_a

def build_E0(emb, af, seg):
    bp, bl = [], []
    for sgid in np.unique(seg):
        idx = np.where(seg == sgid)[0]
        for k in range(len(idx) // N):
            ch = idx[k*N:(k+1)*N]; bp.append(emb[ch].mean(0)); bl.append(af[ch][0])
    return np.array(bp), np.array(bl)

def build_E1(emb, af, D, C, seed=SEED):
    r = np.random.default_rng(seed); bp, bl = [], []
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
            if len(pick) == N: bp.append(emb[pick].mean(0)); bl.append(a)
    return np.array(bp), np.array(bl)

def balance(bp, bl, seed=0):
    r = np.random.default_rng(seed); per = min(int((bl == a).sum()) for a in np.unique(bl))
    keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    return bp[np.array(sorted(keep))], bl[np.array(sorted(keep))]

def disc_block(emb, af, seg, D, C, proto):
    bp, bl = (build_E0(emb, af, seg) if proto == "E0" else build_E1(emb, af, D, C))
    bp, bl = balance(bp, bl); bpu = unit(bp)
    hb = hdbscan_best(bpu, bl, MCS_DRFF)
    km, spc = oracle_k(bpu, bl, KDRFF)
    kp = knn_purity(bp, bl); intra, inter = cos_gap(bp, bl)
    return dict(hdb_ARI=hb["ARI"], hdb_K=hb["K_est"], hdb_noise=hb["noise"],
                oracleK_kmeans=km, oracleK_spectral=spc, knn1=kp[1], knn5=kp[5],
                gap=intra - inter, n_bursts=int(len(bl)))

def session_probe(F512, af, U, D, C):
    """step3b protocol: session=(TD,U,D,C) held out disjointly; per-airframe >=1 tr & >=1 te session."""
    sess = np.array([f"{a}|{u}|{d}|{c}" for a, u, d, c in zip(af, U, D, C)])
    y = np.unique(af, return_inverse=True)[1]
    rng = np.random.default_rng(1); tr = np.zeros(len(af), bool)
    for a in np.unique(af):
        s_a = np.unique(sess[af == a]); rng.shuffle(s_a)
        ntr = max(1, int(round(0.6 * len(s_a))))
        if len(s_a) - ntr < 1: ntr = len(s_a) - 1
        trs = set(s_a[:ntr].tolist())
        tr |= np.array([(af[i] == a and sess[i] in trs) for i in range(len(af))])
    te = ~tr
    if len(set(y[tr])) < len(set(y)) or len(set(y[te])) < len(set(y)):
        return dict(logreg_win=float("nan"), logreg_burst=float("nan"),
                    mlp_win=float("nan"), mlp_burst=float("nan"))
    scaler = StandardScaler().fit(F512[tr]); Xtr, Xte = scaler.transform(F512[tr]), scaler.transform(F512[te])
    out = {}
    for pn, clf in [("logreg", LogisticRegression(max_iter=2000)),
                    ("mlp", MLPClassifier((256,), max_iter=300, early_stopping=True, random_state=0))]:
        clf.fit(Xtr, y[tr]); proba = clf.predict_proba(Xte)
        aw = float((clf.classes_[proba.argmax(1)] == y[te]).mean())
        ab = [clf.classes_[proba[sess[te] == s].mean(0).argmax()] == y[te][sess[te] == s][0]
              for s in np.unique(sess[te])]
        out[f"{pn}_win"] = aw; out[f"{pn}_burst"] = float(np.mean(ab))
    return out

def wisig_retention(enc, seeds=(201, 202)):
    """P2 bursts K18, head-free 512-D, HDBSCAN + oracle-K@18, averaged over 2 DEV slices."""
    ors, hbs = [], []
    dev512 = {tx: embed512(enc, dev_cache[tx]["Xt"], dev_cache[tx]["Xs"]) for tx in dev_tx}
    for s in seeds:
        sl = GC.scatter_slice(dev_tx, KWISIG, s)
        bp, bl = [], []
        for di, tx in enumerate(sl):
            b = bursts_p2_multidate(dev512[tx], dev_cache[tx]["rx"], dev_cache[tx]["date"], seed=s)[0]
            if b is None: continue
            bp.append(b); bl.append(np.full(len(b), di))
        bp = np.concatenate(bp); bl = np.concatenate(bl)
        # cap for tractable spectral
        r = np.random.default_rng(s); keep = []
        for a in np.unique(bl):
            ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:120].tolist()
        keep = np.array(sorted(keep)); bp, bl = bp[keep], bl[keep]
        hb = hdbscan_best(bp, bl, MCS_WISIG); km, spc = oracle_k(bp, bl, KWISIG)
        ors.append(max(km, spc)); hbs.append(hb["ARI"])
    return float(np.mean(ors)), float(np.mean(hbs))

@torch.no_grad()
def pool_val_gap(enc):
    F = embed512(enc, pool["Xt"][val_mask], pool_Xs[val_mask])
    y = pool["y"][val_mask]
    intra, inter = cos_gap(F, y)
    return intra - inter

# ============================================================================
# TRAINING
# ============================================================================
def sample_batch(rng):
    idx_t, idx_s, lab, dom = [], [], [], []
    # DRFF pool — balanced per airframe
    for ai in range(15):
        pool_ix = pool_tr_idx[ai]
        sel = rng.choice(pool_ix, size=BATCH_DRFF_AF, replace=len(pool_ix) < BATCH_DRFF_AF)
        for j in sel: idx_t.append(("p", j)); lab.append(ai)
    # WiSig replay — random devices
    devs = rng.choice(len(train_tx), size=BATCH_WIS_DEV, replace=False)
    for d in devs:
        rix = rep_by_dev[d]
        sel = rng.choice(rix, size=BATCH_WIS_PER, replace=len(rix) < BATCH_WIS_PER)
        for j in sel: idx_t.append(("w", j)); lab.append(1000 + d)
    pt = [j for s, j in idx_t if s == "p"]; wt = [j for s, j in idx_t if s == "w"]
    xt = torch.cat([pool_Xt_t[pt], rep_Xt_t[wt]], 0).cuda()
    xs = torch.cat([pool_Xs_t[pt], rep_Xs_t[wt]], 0).cuda()
    y = torch.tensor([l for (s, _), l in zip(idx_t, lab) if s == "p"] +
                     [l for (s, _), l in zip(idx_t, lab) if s == "w"], device="cuda")
    return xt, xs, y

def lr_lambda(step):
    if step < WARMUP: return step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * prog))

def train_recipe(recipe):
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    enc, params = configure(recipe)
    ckpt_dir = os.path.join(ADAPT_ROOT, recipe); os.makedirs(ckpt_dir, exist_ok=True)
    n_train = sum(p.numel() for p in params)
    print(f"\n=== TRAIN {recipe} — LR={LR[recipe]:.0e}, trainable params={n_train:,} "
          f"(steps={STEPS}, warmup={WARMUP}, batch={15*BATCH_DRFF_AF+BATCH_WIS_DEV*BATCH_WIS_PER}) ===")
    opt = torch.optim.AdamW(params, lr=LR[recipe], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    supcon = SupervisedContrastiveLoss()
    curve = []; best = dict(gap=-1e9, step=-1, state=None)
    for step in range(1, STEPS + 1):
        enc.train()
        xt, xs, y = sample_batch(rng)
        with torch.amp.autocast('cuda'):
            emb = enc(xt, xs)
            loss = supcon(emb, y, temperature=TAU)
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % CKPT_EVERY == 0:
            gap = pool_val_gap(enc)
            curve.append(dict(step=step, loss=float(loss.item()), val_gap=float(gap),
                              lr=float(sched.get_last_lr()[0])))
            state = dict(base=copy.deepcopy(enc.base.state_dict()),
                         adapter=(copy.deepcopy(enc.adapter.state_dict())
                                  if isinstance(enc.adapter, Adapter) else None))
            torch.save(state, os.path.join(ckpt_dir, f"ckpt_step{step}.pt"))
            if gap > best["gap"]: best = dict(gap=gap, step=step, state=state)
            print(f"    step {step:4d}: loss={loss.item():.4f} val_gap={gap:+.4f} "
                  f"lr={sched.get_last_lr()[0]:.2e}{'  <-best' if best['step']==step else ''}")
    # load best
    enc.base.load_state_dict(best["state"]["base"])
    if isinstance(enc.adapter, Adapter): enc.adapter.load_state_dict(best["state"]["adapter"])
    torch.save(best["state"], os.path.join(ckpt_dir, "best.pt"))
    json.dump(curve, open(os.path.join(ckpt_dir, "curve.json"), "w"), indent=2)
    return enc, best, curve, n_train

# ============================================================================
# RUN BATTERY
# ============================================================================
def evaluate(enc, tag):
    F = embed512(enc, ev["Xt"], ev_Xs)
    e1 = disc_block(F, ev["y"], ev["seg"], ev["D"], ev["C"], "E1")
    e0 = disc_block(F, ev["y"], ev["seg"], ev["D"], ev["C"], "E0")
    probe = session_probe(F, ev["y"], ev["U"], ev["D"], ev["C"])
    ret_or, ret_hb = wisig_retention(enc)
    row = dict(recipe=tag,
               E1_HDBSCAN=round(e1["hdb_ARI"], 3),
               E1_oracleK8=round(max(e1["oracleK_kmeans"], e1["oracleK_spectral"]), 3),
               E1_oracleK8_km=round(e1["oracleK_kmeans"], 3), E1_oracleK8_sp=round(e1["oracleK_spectral"], 3),
               E0_HDBSCAN=round(e0["hdb_ARI"], 3),
               E0_oracleK8=round(max(e0["oracleK_kmeans"], e0["oracleK_spectral"]), 3),
               kNN1=round(e1["knn1"], 3), kNN5=round(e1["knn5"], 3), gap=round(e1["gap"], 3),
               probe_burst=round(max(probe["logreg_burst"], probe["mlp_burst"]), 3),
               probe_win=round(max(probe["logreg_win"], probe["mlp_win"]), 3),
               WiSig_retention_oracleK18=round(ret_or, 3), WiSig_retention_HDBSCAN=round(ret_hb, 3),
               E1_nbursts=e1["n_bursts"], E0_nbursts=e0["n_bursts"])
    print(f"  [{tag}] E1 HDB={row['E1_HDBSCAN']} oracleK@8={row['E1_oracleK8']} "
          f"(km{row['E1_oracleK8_km']}/sp{row['E1_oracleK8_sp']}) | E0 HDB={row['E0_HDBSCAN']} | "
          f"kNN1={row['kNN1']} gap={row['gap']} | probe={row['probe_burst']} | "
          f"retention oracleK@18={row['WiSig_retention_oracleK18']} (HDB {row['WiSig_retention_HDBSCAN']})")
    return row, dict(E1=e1, E0=e0, probe=probe)

os.makedirs(ADAPT_ROOT, exist_ok=True)
table = []; details = {}; curves = {}; nparams = {}
print("\n" + "=" * 80)
print("FROZEN BASELINE (no gradient) — sanity vs known frozen numbers (E1 ~0.075/0.218)")
print("=" * 80)
enc0, _ = configure("baseline")
row0, det0 = evaluate(enc0, "baseline")
table.append(row0); details["baseline"] = det0
del enc0; torch.cuda.empty_cache()

for recipe in ["R1", "R2", "R3"]:
    enc, best, curve, npar = train_recipe(recipe)
    print(f"  best ckpt: step {best['step']} (val_gap {best['gap']:+.4f})")
    row, det = evaluate(enc, recipe)
    row["best_step"] = best["step"]; row["best_val_gap"] = round(float(best["gap"]), 4)
    table.append(row); details[recipe] = det; curves[recipe] = curve; nparams[recipe] = npar
    del enc; torch.cuda.empty_cache()

# ============================================================================
# GATE + SAVE
# ============================================================================
adapted = [r for r in table if r["recipe"] != "baseline"]
best_recipe = max(adapted, key=lambda r: r["E1_oracleK8"])
g = best_recipe["E1_oracleK8"]
base_e1_or = table[0]["E1_oracleK8"]; base_e1_hb = table[0]["E1_HDBSCAN"]
if g >= 0.50:   gate = f"PASS: best recipe {best_recipe['recipe']} mavicAir2 E1 oracle-K@8={g:.3f} (>=0.50)"
elif g >= 0.35: gate = f"PARTIAL: best recipe {best_recipe['recipe']} mavicAir2 E1 oracle-K@8={g:.3f} (0.35-0.50)"
else:           gate = f"FAIL: best recipe {best_recipe['recipe']} mavicAir2 E1 oracle-K@8={g:.3f} (<0.35)"
hb_move = best_recipe["E1_HDBSCAN"] - base_e1_hb
or_move = g - base_e1_or
deploy_flag = ("oracle-K moved (+%.3f) but HDBSCAN barely (+%.3f) -> separability up, density-cluster "
               "not yet -> deployable number lags" % (or_move, hb_move)) if (or_move >= 0.10 and hb_move < 0.05) \
              else ("HDBSCAN moved +%.3f, oracle-K +%.3f" % (hb_move, or_move))
retentions = {r["recipe"]: (r["WiSig_retention_oracleK18"],
                            "OK" if r["WiSig_retention_oracleK18"] >= RETENTION_BAR else "BELOW BAR")
              for r in table}

def wcsv(fn, rows, fields=None):
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys())); w.writeheader(); w.writerows(rows)
cols = ["recipe", "E1_HDBSCAN", "E1_oracleK8", "E0_HDBSCAN", "kNN1", "gap", "probe_burst",
        "WiSig_retention_oracleK18", "E1_oracleK8_km", "E1_oracleK8_sp", "E0_oracleK8", "kNN5",
        "probe_win", "WiSig_retention_HDBSCAN", "E1_nbursts", "E0_nbursts"]
for r in table:
    r.setdefault("best_step", ""); r.setdefault("best_val_gap", "")
wcsv("combined_table.csv", table, cols + ["best_step", "best_val_gap"])

report = dict(
    question="does light adaptation on 15 other drone airframes unlock open-set discovery of the "
             "8 unseen mavicAir2 units?",
    data=dict(pool_airframes=POOL_AF, eval_group=EVAL_AF,
              pool_windows=int(len(pool["y"])), pool_train=int(tr_mask.sum()), pool_val=int(val_mask.sum()),
              val_frac=VAL_FRAC, replay_devices=len(train_tx), replay_windows=int(len(rep_y)),
              trade_off="mavicAir2s (x7) spent as adaptation data; adapted-discovery claim rests on "
                        "mavicAir2 (8) alone — stated, not accidental."),
    config=dict(seed=SEED, steps=STEPS, ckpt_every=CKPT_EVERY, warmup=WARMUP,
                batch_total=15*BATCH_DRFF_AF+BATCH_WIS_DEV*BATCH_WIS_PER,
                batch_drff=15*BATCH_DRFF_AF, batch_wisig=BATCH_WIS_DEV*BATCH_WIS_PER,
                tau=TAU, lrs=LR, replay_frac=round(15*BATCH_DRFF_AF/(15*BATCH_DRFF_AF+BATCH_WIS_DEV*BATCH_WIS_PER), 3),
                optimizer="AdamW cosine warmup, grad-clip 1.0, AMP", trainable_params=nparams,
                recipes=dict(R1="head-adapt: residual adapter 512->128->512 (zero-init) + proj head",
                             R2="partial: b4 each branch + cross_attn + norm + proj head",
                             R3="full unfreeze")),
    best_checkpoints={r["recipe"]: dict(step=r.get("best_step"), val_gap=r.get("best_val_gap"))
                      for r in adapted},
    combined_table=table,
    retention=dict(bar=RETENTION_BAR, per_recipe=retentions),
    gate=gate, deployable_note=deploy_flag,
    frozen_baseline=dict(E1_oracleK8=base_e1_or, E1_HDBSCAN=base_e1_hb),
    guardrails="mavicAir2 quarantined (data/stats/stopping); best_model.pt never overwritten; fixed "
               "3-recipe battery, no extra sweeps; replay mandatory R2/R3; oracle-K=CEILING; locked "
               "noise rule; TEST/board-18 + M100 untouched.")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)
for r, c in curves.items():
    wcsv(f"curve_{r}.csv", c)

print("\n" + "=" * 80)
print("COMBINED TABLE (mavicAir2 EVAL, 512-D pre-projection)")
print("=" * 80)
print(f"{'recipe':>9} {'E1_HDB':>7} {'E1_orK8':>8} {'E0_HDB':>7} {'kNN1':>6} {'gap':>7} "
      f"{'probe':>6} {'retain_orK18':>12}")
for r in table:
    print(f"{r['recipe']:>9} {r['E1_HDBSCAN']:>7} {r['E1_oracleK8']:>8} {r['E0_HDBSCAN']:>7} "
          f"{r['kNN1']:>6} {r['gap']:>7} {r['probe_burst']:>6} {r['WiSig_retention_oracleK18']:>12}")
print("\nRetention (bar %.2f):" % RETENTION_BAR, retentions)
print("Deployable movement:", deploy_flag)
print("\n=== PRE-REGISTERED GATE ===")
print(gate)
print("\nsaved -> results/step6_adaptation/ (combined_table.csv, curve_*.csv, report.json); "
      "ckpts -> runs/drff_adapt/<recipe>/")
print("CHECKPOINT — best_model.pt untouched; mavicAir2 quarantined; TEST/M100 sealed; battery complete.")
