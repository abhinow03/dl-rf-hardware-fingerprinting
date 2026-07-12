"""PLAY 1 — DRONE-NATIVE ENCODER FROM SCRATCH (GPU-exclusive training, DEMO-side).

Question: does the proven WiSig recipe (train metric on many identities -> discover held-out
identities) work NATIVELY on drone RF? Train SupCon on 13 non-mavicAir2 airframes (2 pool
airframes held out ENTIRELY for val), discover the untouched mavicAir2 8.

DEMO-SIDE — NOT PAPER RESULTS. best_model.pt is NEVER loaded or written (from scratch).
Writes ONLY to runs/drone_native/ + results/demo_play1/. mavicAir2 quarantined until final eval.

PRE-REGISTERED DECISIONS (stated before launch):
  WINDOW SPEC : 4096 samples @ NATIVE 50 MS/s (81.9 us) from RAW iq (NO OPT-B, NO decimation).
                STFT n_fft=256 hop=64 Hann -> [2,129,61]. Per-window UNIT-POWER standardize.
  M100        : EXCLUDED. M100 cache = 32x256 @ 25 MS/s bursts; building 4096-sample windows
                from it means a DIFFERENT sample rate AND time-span than native 50 MS/s DRFF,
                so STFT bins map to different physical frequencies and windows span different
                durations -> not commensurable in ONE SupCon metric space without a rate/time
                normalization not validated here. DRFF-native (13 train airframes) is the
                primary plan and has ample identities/data (140k+ candidate windows).

DATA DISCIPLINE:
  EVAL GROUP  = all 8 mavicAir2 airframes. No windows/stats/stopping/selection. Touched ONLY at
                final eval of selected checkpoints.
  TRAIN POOL  = 15 non-mavicAir2 airframes (== step6 list). VAL AIRFRAMES held out ENTIRELY:
                mini3pro_2, mini4PRO_2 (model-mates stay in train -> same-model discrimination is
                trainable). => SupCon trains on the remaining 13 airframes (13-way).
  VAL SEG     = 15% segment-disjoint of the 13 train airframes (checkpoint selection only).
  SELECTION COMPOSITE (exact, stated before launch): every 500 steps embed 512-D and compute
     (a) knn1  = kNN-1 airframe purity on a 6-way slice = [2 val airframes] + [4 FIXED train
                 airframes], using ONLY held-out (val) segments of every airframe in the slice
                 (segment-disjoint from gradient steps);
     (b) gap6  = intra-inter cosine gap on that same 6-way slice;
     (c) vgap  = intra-inter cosine gap on ALL 13 train airframes' held-out val segments.
     composite = knn1 + gap6 + vgap  (higher better). Checkpoint = argmax(composite). mavicAir2
     is NEVER in this metric.

MODEL/TRAIN: parent dual-branch RFEncoder (1D+2D ResNet + cross-attn, GroupNorm, 512-D pre-proj,
  128-D L2 head), FRESH init. SupCon on airframe labels, tau=0.5 constant. Balanced per-airframe
  batches: per train airframe sample 4 segments x 2 windows = 8 windows -> same-segment pairs are
  multi-view positives; SupCon positive set = every same-airframe sample in the batch (both
  multi-view and cross-segment). Augment (on standardized time, GPU): AWGN SNR U(10,40)dB p=0.5,
  amp-scale U(0.7,1.4) p=0.5, phase-rot U(-pi,pi) p=0.5. NO CFO aug (CFO is fingerprint). AdamW
  5e-4, cosine + 10% warmup, grad-clip 1.0, AMP. 6000 steps, ckpt every 500. Two seeds.

FINAL EVAL (once per seed, selected ckpt) on mavicAir2 (8, untouched), 512-D head-free:
  feature modes: PER-SEGMENT mean-pool AND N=50 multi-condition bursts (Play-0 Lever-2 framing).
  Locked noise rule. Report per mode: 8-way HDBSCAN(mean over mcs{5,7}) + oracle-K@8 km/sp +
  kNN-1 + gap; AND K=4 subset battery (all 70 subsets): mean ARI, frac>=0.6, correct-K frac.
  Also report the 2 val-airframe numbers (generalization-vs-memorization).

GATE (pre-registered): PASS = oracle-K@8 >= 0.5 AND K=4 mean HDBSCAN ARI >= 0.6.
  PARTIAL = oracle-K@8 in [0.35,0.5) OR K=4 mean in [0.45,0.6). FAIL = below both.

  python3 results/demo_play1/drone_native.py
"""
import os, sys, json, csv, re, math, copy, itertools
from collections import defaultdict, Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder, SupervisedContrastiveLoss
from integrity import score_locked, hdbscan_pred
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

DRFF_DIR   = os.path.expanduser("~/Desktop/processed/drff_r2")
OUT        = _HERE
CKPT_ROOT  = os.path.join(_SW, "runs", "drone_native")
os.makedirs(OUT, exist_ok=True); os.makedirs(CKPT_ROOT, exist_ok=True)

# ---- FIXED CONFIG ----
WIN       = 4096                 # native 50 MS/s window (81.9 us)
NFFT, HOP = 256, 64              # STFT -> [2,129,61]
WPSEG     = 8                    # max windows/segment (spread)
TRAIN_CAP = 2500                 # train windows/airframe
VALAF_CAP = 1000                 # val-airframe windows/airframe
EVAL_CAP  = 1500                 # mavicAir2 windows/airframe (final eval)
STEPS     = 6000
CKPT_EVERY= 500
WARMUP    = 600                  # 10%
LR        = 5e-4
TAU       = 0.5
SEGS_PER_AF_BATCH = 4            # 4 segments x 2 windows = 8 win/af
WPS_BATCH = 2
MCS_DRFF  = [5, 7]
KTRUE     = 8
NBURST    = 50                   # Play-0 Lever-2 burst size
SEEDS     = [1234, 2024]
VAL_AIRFRAMES = ["mini3pro_2", "mini4PRO_2"]   # held out ENTIRELY

FROZEN_HANN = torch.hann_window(NFFT)

def unit_power(w):
    return (w / (np.sqrt((w.astype(np.float32) ** 2).mean()) + 1e-8)).astype(np.float32)

def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

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
POOL_AF = [a for a in all_af if model_of[a] != "mavicAir2"]          # 15
EVAL_AF = [a for a in all_af if model_of[a] == "mavicAir2"]          # 8 quarantined
assert len(POOL_AF) == 15 and len(EVAL_AF) == 8
for v in VAL_AIRFRAMES:
    assert v in POOL_AF, f"{v} not in pool"
TRAIN_AF = [a for a in POOL_AF if a not in VAL_AIRFRAMES]            # 13 trained
# 4 FIXED train airframes for the 6-way selection slice: model-mates of the two val
# airframes (so same-model generalization is probed) + first 2 remaining, deterministic.
mates = [a for a in TRAIN_AF if model_of[a] in {model_of[v] for v in VAL_AIRFRAMES}]
others = [a for a in TRAIN_AF if a not in mates]
FIXED4 = (mates + others)[:4]
print(f"[cfg] TRAIN_AF(13)={TRAIN_AF}")
print(f"[cfg] VAL_AIRFRAMES={VAL_AIRFRAMES}  FIXED4_slice={FIXED4}")
print(f"[cfg] EVAL(quarantined)={EVAL_AF}")

def build_windows(af_list, cap, keep_meta=False, seed=0):
    """NATIVE 50 MS/s 4096-win from RAW iq (NO decimation). <=WPSEG win/seg, per-window
    unit-power standardize. Returns Xt[N,2,4096] f32, y(af idx), seg(global), (U,D,C) if meta."""
    rng = np.random.default_rng(seed)
    Xt, y, seg, Us, Ds, Cs = [], [], [], [], [], []
    gseg = 0
    for ai, a in enumerate(af_list):
        files = af_files[a][:]; rng.shuffle(files)
        units = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]):
                units.append((fn, si))
        rng.shuffle(units)
        got = 0; zc = {}
        for fn, si in units:
            if got >= cap: break
            if fn not in zc:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zc[fn] = dict(iq=z["iq"].astype(np.float32), sb=z["seg_bounds"],
                              U=str(z["U"]), D=str(z["D"]), C=int(z["C"]))
            zz = zc[fn]; off, ln = zz["sb"][si]
            nw = int(ln) // WIN
            if nw < 1: continue
            take = min(nw, cap - got, WPSEG)
            base = int(off)
            for k in range(take):
                w = zz["iq"][:, base + k * WIN: base + (k + 1) * WIN]
                Xt.append(unit_power(w)); y.append(ai); seg.append(gseg)
                if keep_meta: Us.append(zz["U"]); Ds.append(zz["D"]); Cs.append(zz["C"])
            got += take; gseg += 1
    out = dict(Xt=np.stack(Xt).astype(np.float32), y=np.array(y), seg=np.array(seg))
    if keep_meta: out.update(U=np.array(Us), D=np.array(Ds), C=np.array(Cs))
    return out

print("[data] building 13 train airframes (native 4096, cap %d) ..." % TRAIN_CAP)
tr = build_windows(TRAIN_AF, TRAIN_CAP, seed=777)
# 15% segment-disjoint val within the 13 train airframes
rng = np.random.default_rng(777)
val_seg_mask = np.zeros(len(tr["y"]), bool)
for ai in range(len(TRAIN_AF)):
    segs = np.unique(tr["seg"][tr["y"] == ai]); rng.shuffle(segs)
    nval = min(max(0, int(round(0.15 * len(segs)))), max(0, len(segs) - 1))
    if nval == 0 and len(segs) >= 2: nval = 1
    vs = set(segs[:nval].tolist())
    val_seg_mask |= np.array([(tr["y"][i] == ai and tr["seg"][i] in vs) for i in range(len(tr["y"]))])
tr_mask = ~val_seg_mask
print("    train windows: %d (%d grad / %d val-seg), %d segments" %
      (len(tr["y"]), tr_mask.sum(), val_seg_mask.sum(), len(np.unique(tr["seg"]))))
for ai, a in enumerate(TRAIN_AF):
    print("      %-14s idx%2d: %5d win (%d gr / %d vseg)" %
          (a, ai, (tr["y"] == ai).sum(), (tr_mask & (tr["y"] == ai)).sum(),
           (val_seg_mask & (tr["y"] == ai)).sum()))

print("[data] building 2 VAL airframes (held out entirely, cap %d) ..." % VALAF_CAP)
va = build_windows(VAL_AIRFRAMES, VALAF_CAP, seed=778)
print("    val-airframe windows: %d, %d segments" % (len(va["y"]), len(np.unique(va["seg"]))))

# per-(af) grad-segment -> window index lists for the batch sampler
tr_seg_of_af = {ai: {} for ai in range(len(TRAIN_AF))}
for i in np.where(tr_mask)[0]:
    tr_seg_of_af[tr["y"][i]].setdefault(tr["seg"][i], []).append(i)
tr_seg_of_af = {ai: {s: np.array(v) for s, v in d.items()} for ai, d in tr_seg_of_af.items()}

tr_Xt_t = torch.from_numpy(tr["Xt"])

# ============================================================================
# GPU STFT + AUGMENT
# ============================================================================
def stft_gpu(xt):                         # xt [B,2,4096] cuda f32 -> [B,2,129,61]
    B = xt.shape[0]
    x = xt.reshape(B * 2, WIN)
    sp = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                    window=FROZEN_HANN.to(xt.device), center=False, return_complex=True)
    return sp.abs().reshape(B, 2, NFFT // 2 + 1, -1)

def augment(xt):                          # xt [B,2,4096] cuda f32
    B, dev = xt.shape[0], xt.device
    I, Q = xt[:, 0], xt[:, 1]
    # phase rotation
    do = (torch.rand(B, device=dev) < 0.5)[:, None]
    th = (torch.rand(B, device=dev) * 2 - 1) * math.pi
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    I2, Q2 = I * c - Q * s, I * s + Q * c
    I, Q = torch.where(do, I2, I), torch.where(do, Q2, Q)
    # amp scale
    doa = (torch.rand(B, device=dev) < 0.5)[:, None]
    a = (0.7 + torch.rand(B, device=dev) * 0.7)[:, None]
    I, Q = torch.where(doa, I * a, I), torch.where(doa, Q * a, Q)
    # AWGN
    don = (torch.rand(B, device=dev) < 0.5)[:, None]
    snr = (10 + torch.rand(B, device=dev) * 30)[:, None]
    sp = (I ** 2 + Q ** 2).mean(1, keepdim=True)
    npow = sp / (10 ** (snr / 10))
    nI, nQ = torch.randn_like(I) * torch.sqrt(npow / 2), torch.randn_like(Q) * torch.sqrt(npow / 2)
    I, Q = torch.where(don, I + nI, I), torch.where(don, Q + nQ, Q)
    return torch.stack([I, Q], 1)

# ============================================================================
# EMBED + SCORING
# ============================================================================
@torch.no_grad()
def embed512(model, Xt, batch=384):
    model.eval()
    f = np.empty((Xt.shape[0], 512), dtype=np.float32)
    xt = torch.from_numpy(Xt)
    for i in range(0, Xt.shape[0], batch):
        xb = xt[i:i + batch].cuda()
        xs = stft_gpu(xb)
        with torch.amp.autocast('cuda'):
            f[i:i + batch] = model.get_encoder_output(xb, xs).float().cpu().numpy()
    return f

def knn1_purity(X, y):
    Xn = unit(X); k = min(2, len(Xn))
    nn = NearestNeighbors(n_neighbors=k).fit(Xn); _, ind = nn.kneighbors(Xn)
    yi = np.unique(y, return_inverse=True)[1]
    return float((yi[ind[:, 1]] == yi).mean())

def cos_gap(X, y):
    Xn = unit(X); yi = np.unique(y, return_inverse=True)[1]; intra, inter = [], []
    for a in np.unique(yi):
        m = yi == a
        if m.sum() < 2: continue
        C = Xn[m] @ Xn[m].T; iu = np.triu_indices(m.sum(), 1)
        intra.append(C[iu].mean()); inter.append((Xn[m] @ Xn[~m].T).mean())
    return float(np.mean(intra) - np.mean(inter))

def hdb_mean_ari(bp, bl, mcs_list):
    aris, ks = [], []
    for mcs in mcs_list:
        sc = score_locked(hdbscan_pred(bp, mcs), bl)
        aris.append(sc["ARI"]); ks.append(sc["K_est"])
    return float(np.mean(aris)), float(np.median(ks))

def oracle_km_sp(bp, bl, K):
    yi = np.unique(bl, return_inverse=True)[1]
    km = float(adjusted_rand_score(yi, KMeans(K, n_init=10, random_state=0).fit_predict(bp)))
    try:
        sp = SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                n_neighbors=15).fit_predict(bp)
        spa = float(adjusted_rand_score(yi, sp))
    except Exception:
        spa = float("nan")
    return km, spa

def build_perseg(emb, af, seg):
    bp, bl = [], []
    for sg in np.unique(seg):
        idx = np.where(seg == sg)[0]
        bp.append(emb[idx].mean(0)); bl.append(af[idx][0])
    return np.array(bp), np.array(bl)

def build_N(emb, af, D, C, N, seed=0):
    r = np.random.default_rng(seed); bp, bl = [], []
    for a in np.unique(af):
        idx = np.where(af == a)[0]; cells = defaultdict(list)
        for i in idx: cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx) // N
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < N and ci < 4000:
                cc = ck[ci % len(ck)]
                if cells[cc]: pick.append(cells[cc][r.integers(len(cells[cc]))])
                ci += 1
            if len(pick) == N: bp.append(emb[pick].mean(0)); bl.append(a)
    return np.array(bp), np.array(bl)

def balance(bp, bl, seed=0):
    r = np.random.default_rng(seed); per = min(int((bl == a).sum()) for a in np.unique(bl))
    keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    keep = np.array(sorted(keep)); return bp[keep], bl[keep], per

def score_full8(bp_raw, bl):
    bp, bl, per = balance(bp_raw, bl); bpu = unit(bp)
    hdb, kest = hdb_mean_ari(bpu, bl, MCS_DRFF)
    km, sp = oracle_km_sp(bpu, bl, KTRUE)
    return dict(hdb_ari=round(hdb, 3), hdb_kest=round(kest, 2),
                oracle_km=round(km, 3), oracle_sp=round(sp, 3),
                knn1=round(knn1_purity(bp, bl), 3), gap=round(cos_gap(bp, bl), 3),
                n_bursts=int(len(bl)), per_af=int(per))

def score_k4_battery(emb, af, D, C, seg, mode):
    """all 70 C(8,4) subsets: mean HDBSCAN ARI (mean over mcs), frac>=0.6, correct-K frac."""
    uaf = sorted(np.unique(af))
    aris, correct = [], []
    for sub in itertools.combinations(uaf, 4):
        m = np.isin(af, sub)
        if mode == "perseg":
            bp, bl = build_perseg(emb[m], af[m], seg[m])
        else:
            bp, bl = build_N(emb[m], af[m], D[m], C[m], NBURST, seed=0)
        if len(bl) < 4 or len(np.unique(bl)) < 4: continue
        bp, bl, _ = balance(bp, bl); bpu = unit(bp)
        ari, kest = hdb_mean_ari(bpu, bl, MCS_DRFF)
        aris.append(ari); correct.append(abs(kest - 4) < 0.5)
    aris = np.array(aris)
    return dict(mean_ari=round(float(aris.mean()), 3), std_ari=round(float(aris.std()), 3),
                frac_ge0p6=round(float((aris >= 0.6).mean()), 3),
                frac_correctK=round(float(np.mean(correct)), 3), n_subsets=int(len(aris)))

# ============================================================================
# SELECTION COMPOSITE (val only; mavicAir2 never seen)
# ============================================================================
slice6_af = FIXED4 + VAL_AIRFRAMES        # 4 train + 2 val = 6
def val_composite(model):
    # (a)(b) 6-way slice on held-out (val) segments only
    Xs, ys = [], []
    for a in slice6_af:
        if a in VAL_AIRFRAMES:
            ai = VAL_AIRFRAMES.index(a); m = va["y"] == ai
            Xs.append(va["Xt"][m]); ys.append(np.full(m.sum(), a))
        else:
            ai = TRAIN_AF.index(a); m = val_seg_mask & (tr["y"] == ai)
            Xs.append(tr["Xt"][m]); ys.append(np.full(m.sum(), a))
    Xs = np.concatenate(Xs); ys = np.concatenate(ys)
    F6 = embed512(model, Xs)
    knn1 = knn1_purity(F6, ys); gap6 = cos_gap(F6, ys)
    # (c) val-segment gap over all 13 train airframes
    Fv = embed512(model, tr["Xt"][val_seg_mask])
    vgap = cos_gap(Fv, tr["y"][val_seg_mask])
    comp = knn1 + gap6 + vgap
    return dict(knn1=round(knn1, 4), gap6=round(gap6, 4), vgap=round(vgap, 4),
                composite=round(comp, 4))

# ============================================================================
# BATCH SAMPLER  (13 af x 4 seg x 2 win = 104; same-seg pairs = multi-view positives)
# ============================================================================
def sample_batch(rng):
    idx, lab = [], []
    for ai in range(len(TRAIN_AF)):
        segd = tr_seg_of_af[ai]; segs = list(segd.keys())
        chosen = rng.choice(len(segs), size=min(SEGS_PER_AF_BATCH, len(segs)), replace=False)
        picks = []
        for si in chosen:
            pool = segd[segs[si]]
            take = rng.choice(pool, size=WPS_BATCH, replace=len(pool) < WPS_BATCH)
            picks += take.tolist()
        # top up to 8 if fewer segments than SEGS_PER_AF_BATCH
        while len(picks) < SEGS_PER_AF_BATCH * WPS_BATCH:
            s = segs[rng.integers(len(segs))]; pool = segd[s]
            picks.append(int(rng.choice(pool)))
        for j in picks: idx.append(j); lab.append(ai)
    xt = tr_Xt_t[idx].cuda()
    y = torch.tensor(lab, device="cuda")
    return xt, y

def lr_lambda(step):
    if step < WARMUP: return step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * prog))

# ============================================================================
# TRAIN ONE SEED
# ============================================================================
def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    ckpt_dir = os.path.join(CKPT_ROOT, f"seed{seed}"); os.makedirs(ckpt_dir, exist_ok=True)
    model = RFEncoder().cuda()                     # FRESH init — best_model.pt NEVER loaded
    params = [p for p in model.parameters()]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    supcon = SupervisedContrastiveLoss()
    curve = []; best = dict(composite=-1e9, step=-1, state=None)
    # STFT shape sanity (printed once, before training)
    _s = stft_gpu(tr_Xt_t[:2].cuda())
    print(f"    [sanity] STFT tensor shape={tuple(_s.shape)} (expect (2,2,129,61)); "
          f"time window={WIN}@50MS/s")
    for step in range(1, STEPS + 1):
        model.train()
        xt, y = sample_batch(rng)
        xt_aug = augment(xt.float())
        xs = stft_gpu(xt_aug)
        with torch.amp.autocast('cuda'):
            emb = model(xt_aug, xs)
            loss = supcon(emb, y, temperature=TAU)
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if step % CKPT_EVERY == 0:
            vc = val_composite(model)
            curve.append(dict(step=step, loss=round(float(loss.item()), 4),
                              lr=round(float(sched.get_last_lr()[0]), 6), **vc))
            state = copy.deepcopy(model.state_dict())
            torch.save(state, os.path.join(ckpt_dir, f"ckpt_step{step}.pt"))
            flag = ""
            if vc["composite"] > best["composite"]:
                best = dict(composite=vc["composite"], step=step, state=state); flag = "  <-best"
            print(f"    step {step:4d}: loss={loss.item():.4f} knn1={vc['knn1']:.3f} "
                  f"gap6={vc['gap6']:+.3f} vgap={vc['vgap']:+.3f} comp={vc['composite']:+.3f} "
                  f"lr={sched.get_last_lr()[0]:.2e}{flag}")
    model.load_state_dict(best["state"])
    torch.save(best["state"], os.path.join(ckpt_dir, "best.pt"))
    json.dump(curve, open(os.path.join(ckpt_dir, "curve.json"), "w"), indent=2)
    return model, best, curve

# ============================================================================
# FINAL EVAL ON mavicAir2 (built ONCE, after all selection)
# ============================================================================
def final_eval(model, seed):
    F = embed512(model, ev["Xt"])
    out = {}
    for mode in ("perseg", "N50"):
        if mode == "perseg":
            bp, bl = build_perseg(F, ev["y"], ev["seg"])
        else:
            bp, bl = build_N(F, ev["y"], ev["D"], ev["C"], NBURST, seed=0)
        full8 = score_full8(bp, bl)
        k4 = score_k4_battery(F, ev["y"], ev["D"], ev["C"], ev["seg"], mode)
        out[mode] = dict(full8=full8, k4=k4)
        print(f"  [seed{seed}] {mode:>7}: 8way HDB={full8['hdb_ari']} "
              f"orK@8 km={full8['oracle_km']}/sp={full8['oracle_sp']} kNN1={full8['knn1']} "
              f"gap={full8['gap']} | K4 mean={k4['mean_ari']} frac>=.6={k4['frac_ge0p6']} "
              f"correctK={k4['frac_correctK']} (n={k4['n_subsets']})")
    # val-airframe reference (2 val af, generalization-vs-memorization) — per-segment
    Fv = embed512(model, va["Xt"])
    vbp, vbl = build_perseg(Fv, va["y"], va["seg"])
    vbp, vbl, _ = balance(vbp, vbl); vbpu = unit(vbp)
    vhdb, _ = hdb_mean_ari(vbpu, vbl, MCS_DRFF)
    vkm, vsp = oracle_km_sp(vbpu, vbl, 2)
    valref = dict(hdb_ari=round(vhdb, 3), oracle_km=round(vkm, 3), oracle_sp=round(vsp, 3),
                  knn1=round(knn1_purity(Fv, va["y"]), 3), gap=round(cos_gap(Fv, va["y"]), 3),
                  n_bursts=int(len(vbl)))
    print(f"  [seed{seed}] VAL-airframes(2, perseg): HDB={valref['hdb_ari']} "
          f"orK@2 km={valref['oracle_km']}/sp={valref['oracle_sp']} kNN1={valref['knn1']} "
          f"gap={valref['gap']}")
    out["val_airframes"] = valref
    return out

# ============================================================================
# RUN
# ============================================================================
print("\n[eval-data] building mavicAir2 EVAL group (quarantined, cap %d) ..." % EVAL_CAP)
ev = build_windows(EVAL_AF, EVAL_CAP, keep_meta=True, seed=779)
print("    eval windows: %d, %d segments, 8 airframes" % (len(ev["y"]), len(np.unique(ev["seg"]))))

results = {}; curves = {}; selected = {}
for seed in SEEDS:
    print("\n" + "=" * 78)
    print(f"TRAIN seed={seed}  (fresh RFEncoder, {STEPS} steps, native 4096 @ 50 MS/s)")
    print("=" * 78)
    model, best, curve = train_seed(seed)
    print(f"  selected ckpt: step {best['step']} (composite {best['composite']:+.4f})")
    print(f"  --- FINAL EVAL seed={seed} (mavicAir2, untouched) ---")
    res = final_eval(model, seed)
    results[seed] = res; curves[seed] = curve
    selected[seed] = dict(step=best["step"], composite=best["composite"])
    del model; torch.cuda.empty_cache()

# ---- GATE (best over seeds/modes) ----
def best_metric(key_or, key_k4):
    best_or = max(max(results[s][m]["full8"][key_or] for m in ("perseg", "N50")) for s in SEEDS)
    best_k4 = max(max(results[s][m]["k4"][key_k4] for m in ("perseg", "N50")) for s in SEEDS)
    return best_or, best_k4
best_orK8 = max(max(max(results[s][m]["full8"]["oracle_km"], results[s][m]["full8"]["oracle_sp"])
                    for m in ("perseg", "N50")) for s in SEEDS)
best_k4mean = max(max(results[s][m]["k4"]["mean_ari"] for m in ("perseg", "N50")) for s in SEEDS)
if best_orK8 >= 0.5 and best_k4mean >= 0.6:
    gate = f"PASS: best oracle-K@8={best_orK8:.3f} (>=0.5) AND K=4 mean HDB ARI={best_k4mean:.3f} (>=0.6) -> demo-grade native encoder; Tier-2 alive."
elif (0.35 <= best_orK8 < 0.5) or (0.45 <= best_k4mean < 0.6):
    gate = f"PARTIAL: best oracle-K@8={best_orK8:.3f}, K=4 mean HDB ARI={best_k4mean:.3f} -> real signal; human decides on Play 2."
else:
    gate = f"FAIL: best oracle-K@8={best_orK8:.3f}, K=4 mean HDB ARI={best_k4mean:.3f} -> identity scarcity is the binding constraint; Play 2/3 with human."

# ============================================================================
# SAVE
# ============================================================================
def wcsv(fn, rows, header=None):
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        if header: f.write(header + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# demo ledger (flat)
ledger = []
for s in SEEDS:
    for m in ("perseg", "N50"):
        r = dict(seed=s, feat_mode=m, sel_step=selected[s]["step"],
                 sel_composite=selected[s]["composite"])
        r.update({f"full8_{k}": v for k, v in results[s][m]["full8"].items()})
        r.update({f"k4_{k}": v for k, v in results[s][m]["k4"].items()})
        ledger.append(r)
wcsv("demo_ledger.csv", ledger, header="# DEMO-SIDE — NOT PAPER RESULTS")

for s in SEEDS:
    wcsv(f"curve_seed{s}.csv", curves[s])

report = dict(
    header="DEMO-SIDE — NOT PAPER RESULTS",
    note="Play 1 drone-native encoder from scratch; a separate CAPSTONE artifact; NEVER merged "
         "into paper tables. best_model.pt never loaded/written.",
    question="does the WiSig recipe (metric on many identities -> discover held-out identities) "
             "work NATIVELY on drone RF (train 13 non-mavicAir2 airframes, discover mavicAir2 8)?",
    window_spec=dict(win=WIN, rate_MSps=50, dur_us=round(WIN / 50.0, 1), decimation="NONE (raw 50 MS/s)",
                     stft=dict(n_fft=NFFT, hop=HOP, window="hann", shape=[2, NFFT // 2 + 1, 61]),
                     standardize="per-window unit-power"),
    m100_decision="EXCLUDED — 32x256 @ 25 MS/s cache is a different sample-rate and time-span; not "
                  "commensurable with native 50 MS/s 4096-windows in one SupCon space without an "
                  "unvalidated rate/time normalization. DRFF-native (13 train af) is the primary plan.",
    data=dict(pool_airframes=POOL_AF, train_airframes=TRAIN_AF, val_airframes=VAL_AIRFRAMES,
              eval_group=EVAL_AF, fixed4_slice=FIXED4, slice6=slice6_af,
              train_windows=int(len(tr["y"])), train_grad=int(tr_mask.sum()),
              train_valseg=int(val_seg_mask.sum()), valaf_windows=int(len(va["y"])),
              eval_windows=int(len(ev["y"]))),
    selection_composite="composite = knn1(6-way slice, val segs) + gap6(6-way slice) + "
                        "vgap(13 train af val segs); argmax over ckpts; mavicAir2 never in metric.",
    train_config=dict(seeds=SEEDS, steps=STEPS, ckpt_every=CKPT_EVERY, warmup=WARMUP, lr=LR,
                      tau=TAU, batch=len(TRAIN_AF) * SEGS_PER_AF_BATCH * WPS_BATCH,
                      positives="same-airframe (incl. same-segment multi-view pairs)",
                      augment="AWGN SNR U(10,40)dB p=.5, amp-scale U(0.7,1.4) p=.5, phase-rot U(-pi,pi) p=.5; NO CFO",
                      optimizer="AdamW 5e-4, cosine+10% warmup, grad-clip 1.0, AMP"),
    selected_checkpoints=selected,
    curves=curves,
    final_eval=results,
    gate=gate,
    guardrails="from scratch (best_model.pt never loaded/written); mavicAir2 quarantined until final "
               "eval (no windows/stats/stopping/selection); fixed recipe (no arch/loss sweeps); "
               "writes only runs/drone_native/ + results/demo_play1/; separate demo ledger; "
               "TEST/board-18 + M100 untouched.")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)

print("\n" + "=" * 78)
print("FINAL mavicAir2 TABLE (512-D head-free; best-of per cell shown in ledger)")
print("=" * 78)
print(f"{'seed':>5} {'mode':>7} {'8wHDB':>6} {'orKm':>6} {'orSp':>6} {'kNN1':>6} "
      f"{'K4mean':>7} {'K4>=.6':>7} {'corrK':>6}")
for s in SEEDS:
    for m in ("perseg", "N50"):
        f8 = results[s][m]["full8"]; k4 = results[s][m]["k4"]
        print(f"{s:>5} {m:>7} {f8['hdb_ari']:>6} {f8['oracle_km']:>6} {f8['oracle_sp']:>6} "
              f"{f8['knn1']:>6} {k4['mean_ari']:>7} {k4['frac_ge0p6']:>7} {k4['frac_correctK']:>6}")
print("\nVAL-airframe reference (generalization signal):")
for s in SEEDS:
    v = results[s]["val_airframes"]
    print(f"  seed{s}: HDB={v['hdb_ari']} orKm={v['oracle_km']} orSp={v['oracle_sp']} kNN1={v['knn1']} gap={v['gap']}")
print("\n=== PRE-REGISTERED GATE ===")
print(gate)
print("\nsaved -> results/demo_play1/ (demo_ledger.csv, curve_seed*.csv, report.json); "
      "ckpts -> runs/drone_native/seed*/")
print("CHECKPOINT — DEMO-side; from scratch; best_model.pt untouched; mavicAir2 quarantined until "
      "final eval; TEST/M100 sealed.")
