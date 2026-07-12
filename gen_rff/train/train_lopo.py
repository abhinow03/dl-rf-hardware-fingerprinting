"""GEN-RFF PHASE 2 — first multi-domain training run (LOPO holdout=DRFF, the claim cell).

Train on WiSig 109 + ORACLE 12 (121 identities, NO drone data); discover the untouched
mavicAir2 8. Single FIXED config — no arch/loss sweeps. Build+train+select+final-eval in one
run. Writes ONLY to runs_gen/ + results_gen/. Frozen assets read-only.

  OMP_NUM_THREADS=1 ... python3 -m gen_rff.train.train_lopo

STOP at checkpoint (no DANN, no ablations, no second LOPO cell).
"""
import os
import sys
import json
import math
import time
import copy
import itertools
from collections import defaultdict
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DLM not in sys.path:
    sys.path.insert(0, _DLM)
_SW = os.path.join(_DLM, "summer_work")
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2b_decoherence")):
    if p not in sys.path:
        sys.path.insert(0, p)

from gen_rff.data import loaders, registry
from gen_rff.data.unified import UnifiedRFDataset
from gen_rff.model.gen_encoder import GenRFEncoder, param_count
from gen_rff.physics.features import classical_matrix, residual_batch
from gen_rff.bench import harness as H
from gen_rff.bench import lopo
from shared import SupervisedContrastiveLoss
import geometry_consolidate as GC
from decohere import bursts_p2_multidate
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score
from scipy.sparse.csgraph import laplacian

RUNS_GEN = os.path.join(_DLM, "runs_gen", "lopo_drff")
RESULTS_GEN = os.path.join(_DLM, "results_gen")
os.makedirs(RUNS_GEN, exist_ok=True)
os.makedirs(RESULTS_GEN, exist_ok=True)

# ---------------- FIXED CONFIG ----------------
SEED = 1234
STEPS = 10000
CKPT_EVERY = 500
VAL_EVERY = 1000
WARMUP = 1000
LR = 5e-4
TAU = 0.5
P_DEV, V_VIEW = 8, 4                 # per-device identities x views (where domain allows)
WISIG_CAP = 250
ORACLE_CAP = 200
WISIG_VALA_CAP = 1500
ORACLE_DISTS = ("8ft", "14ft")
NBURST_PAPER = 10                    # N=10 paper-comparable
NBURST_DEMO = 120                    # N=120 demo operating point
VALA_SLICE_SEED = 201
KWISIG, KORACLE, KDRFF = 18, 4, 8
MCS_DRFF = [5, 7]
SPECIALIST_REF = 0.72                # WiSig DEV P2 specialist oracle-km (normalizer)

# CONTEXT ROWS (cited, NOT recomputed)
CTX_FROZEN_WISIG = (0.297, 0.298)    # frozen WiSig-only encoder, mavicAir2 E1 oracle km/sp
CTX_NATIVE_DRONE = (0.733, 0.756)    # native drone-trained (in-domain), N=50 oracle km/sp

# SELECTION COMPOSITE (stated BEFORE training; frozen):
#   composite = mean( VAL-A_oracle_km / 0.72 , VAL-B_oracle_km )  +  mean( VAL-A_knn1 , VAL-B_knn1 )
SELECTION_FORMULA = ("composite = mean(VAL-A_oracleKm/0.72, VAL-B_oracleKm) + "
                     "mean(VAL-A_knn1, VAL-B_knn1)  [argmax; frozen before training]")


def composite(va_km, va_knn, vb_km, vb_knn):
    return 0.5 * (va_km / SPECIALIST_REF + vb_km) + 0.5 * (va_knn + vb_knn)


def unit(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


# ============================================================================
# GPU augment (phase/amp shared on iq+res [linear -> exact for the residual view];
# AWGN on iq only). NO CFO. physics is left augment-invariant (a stable identity feature).
# ============================================================================
def augment_4ch(xt):                 # xt [B,4,L] = [I,Q,Ires,Qres]
    B, dev = xt.shape[0], xt.device
    I, Q, Ir, Qr = xt[:, 0], xt[:, 1], xt[:, 2], xt[:, 3]
    # phase rotation (same theta on iq and res — exact for LPC residual)
    do = (torch.rand(B, device=dev) < 0.5)[:, None]
    th = (torch.rand(B, device=dev) * 2 - 1) * math.pi
    c, s = torch.cos(th)[:, None], torch.sin(th)[:, None]
    I2, Q2 = I * c - Q * s, I * s + Q * c
    Ir2, Qr2 = Ir * c - Qr * s, Ir * s + Qr * c
    I, Q = torch.where(do, I2, I), torch.where(do, Q2, Q)
    Ir, Qr = torch.where(do, Ir2, Ir), torch.where(do, Qr2, Qr)
    # amp scale (same a on iq and res)
    doa = (torch.rand(B, device=dev) < 0.5)[:, None]
    a = (0.7 + torch.rand(B, device=dev) * 0.7)[:, None]
    I, Q = torch.where(doa, I * a, I), torch.where(doa, Q * a, Q)
    Ir, Qr = torch.where(doa, Ir * a, Ir), torch.where(doa, Qr * a, Qr)
    # AWGN on iq only
    don = (torch.rand(B, device=dev) < 0.5)[:, None]
    snr = (10 + torch.rand(B, device=dev) * 30)[:, None]
    sp = (I ** 2 + Q ** 2).mean(1, keepdim=True)
    npow = sp / (10 ** (snr / 10))
    nI = torch.randn_like(I) * torch.sqrt(npow / 2); nQ = torch.randn_like(Q) * torch.sqrt(npow / 2)
    I, Q = torch.where(don, I + nI, I), torch.where(don, Q + nQ, Q)
    return torch.stack([I, Q, Ir, Qr], 1)


def make_spec4(iq, res, n_fft, hop):
    return torch.cat([registry.stft_mag(iq, n_fft, hop), registry.stft_mag(res, n_fft, hop)], 1)


# ============================================================================
# EMBED (precomputed iq/res/phys -> 512-D via GenRFEncoder.get_encoder_output)
# ============================================================================
@torch.no_grad()
def embed512(model, iq, res, phys, n_fft, hop, batch=256):
    model.eval()
    n = iq.shape[0]; out = np.empty((n, 512), np.float32)
    for i in range(0, n, batch):
        xt = torch.from_numpy(iq[i:i + batch]).cuda()
        rs = torch.from_numpy(res[i:i + batch]).cuda()
        ph = torch.from_numpy(phys[i:i + batch]).cuda()
        x_time = torch.cat([xt, rs], 1)
        x_spec = make_spec4(xt, rs, n_fft, hop)
        with torch.amp.autocast('cuda'):
            out[i:i + batch] = model.get_encoder_output(x_time, x_spec, ph).float().cpu().numpy()
    return out


# ============================================================================
# DATA
# ============================================================================
def build_train():
    wtrain, _ = registry.domain_device_pool("WISIG")
    otrain, _ = registry.domain_device_pool("ORACLE")
    spec = {"WISIG": [g.split(":", 1)[1] for g in wtrain],
            "ORACLE": [g.split(":", 1)[1] for g in otrain]}
    caps = {"WISIG": WISIG_CAP, "ORACLE": ORACLE_CAP}
    ds = UnifiedRFDataset(spec, caps=caps, verbose=True)
    # iq/res stay as ragged lists (shapes differ across domains); phys is homogeneous [19].
    # Batches are domain-homogeneous, so we stack per-batch at fetch time (fetch_batch).
    ds._iql = ds.iq
    ds._resl = ds.res
    ds._phys = np.stack(ds.phys).astype(np.float32)
    return ds


def fetch_batch(ds, idxs):
    """idxs all from one domain -> homogeneous stack (iq, res float32; phys)."""
    iq = np.stack([ds._iql[i] for i in idxs]).astype(np.float32)
    res = np.stack([ds._resl[i] for i in idxs]).astype(np.float32)
    ph = ds._phys[idxs]
    return iq, res, ph


def domain_index(ds):
    """domain -> device_label -> condition_key -> [idx] for the P x V sampler."""
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, r in enumerate(ds.records):
        idx[r["domain"]][r["device_label"]][r["condition_key"]].append(i)
    return idx


def build_val_a():
    """WiSig DEV 18-slice (seed 201): iq/res/phys per device + rx/date for P2 bursts."""
    TXD = loaders._wisig_txd()
    _, dev_tx, _ = loaders.wisig_devices()
    slice18 = GC.scatter_slice(dev_tx, KWISIG, VALA_SLICE_SEED)
    cache = {}
    for tx in slice18:
        t = GC.consec_windows(TXD, tx, WISIG_VALA_CAP)          # [W,2,256] standardized
        Wn = t.shape[0]
        res, _ = residual_batch(t, order=32)
        phys = classical_matrix(t, fs=registry.REGISTRY["WISIG"].native_rate)
        cache[tx] = dict(iq=t.astype(np.float32), res=res.astype(np.float32), phys=phys,
                         rx=TXD[tx]["rx"][:Wn].copy(), date=TXD[tx]["date"][:Wn].copy())
    return slice18, cache


def build_val_b():
    """ORACLE held-out 4: iq/res/phys per device + distance label for cross-distance bursts."""
    _, oeval = registry.domain_device_pool("ORACLE")
    serials = [g.split(":", 1)[1] for g in oeval]
    units = loaders.build_oracle_cache(serials, distances=ORACLE_DISTS, cap=ORACLE_CAP)
    d = registry.REGISTRY["ORACLE"]
    cache = defaultdict(lambda: dict(iq=[], dist=[]))
    for u in units:
        cache[u["device_gid"]]["iq"].append(u["X"])
        cache[u["device_gid"]]["dist"].append(np.full(u["X"].shape[0], u["condition_key"]))
    out = {}
    for gid, c in cache.items():
        X = np.concatenate(c["iq"]).astype(np.float32)
        res, _ = residual_batch(X, order=32)
        phys = classical_matrix(X, fs=d.native_rate)
        out[gid] = dict(iq=X, res=res.astype(np.float32), phys=phys, dist=np.concatenate(c["dist"]))
    return list(out.keys()), out


# ---- burst builders ----
def p2_bursts(emb, rx, date, seed):
    return bursts_p2_multidate(emb, rx, date, seed=seed)[0]


def cross_cond_bursts(emb, cond, N, seed=0):
    """N-window bursts drawn across distinct condition values (cross-distance analog of E1)."""
    r = np.random.default_rng(seed); cells = defaultdict(list)
    for i, cc in enumerate(cond):
        cells[cc].append(i)
    ck = list(cells.keys()); nb = len(emb) // N; out = []
    if not ck:
        return None
    for _ in range(nb):
        r.shuffle(ck); pick = []; ci = 0
        while len(pick) < N and ci < 4000:
            c = ck[ci % len(ck)]
            if cells[c]:
                pick.append(cells[c][r.integers(len(cells[c]))])
            ci += 1
        if len(pick) == N:
            out.append(emb[pick].mean(0))
    return unit(np.stack(out)) if out else None


def balance(bp, bl, seed=0):
    r = np.random.default_rng(seed); per = min(int((bl == a).sum()) for a in np.unique(bl)); keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    k = np.array(sorted(keep)); return bp[k], bl[k]


# ============================================================================
# VALIDATION
# ============================================================================
def val_a(model, slice18, cache):
    nfft, hop = registry.REGISTRY["WISIG"].n_fft, registry.REGISTRY["WISIG"].hop
    bp, bl = [], []
    for di, tx in enumerate(slice18):
        c = cache[tx]
        e = embed512(model, c["iq"], c["res"], c["phys"], nfft, hop)
        b = p2_bursts(e, c["rx"], c["date"], VALA_SLICE_SEED)
        if b is None:
            continue
        bp.append(b); bl.append(np.full(len(b), di))
    bp = np.concatenate(bp); bl = np.concatenate(bl)
    km, sp = H.oracle_km_sp(unit(bp), bl, KWISIG)
    return dict(oracle_km=km, oracle_sp=sp, knn1=H.knn_purity(bp, bl)[1], n=int(len(bl)))


def val_b(model, gids, cache, N=NBURST_PAPER):
    nfft, hop = registry.REGISTRY["ORACLE"].n_fft, registry.REGISTRY["ORACLE"].hop
    bp, bl = [], []
    for di, gid in enumerate(gids):
        c = cache[gid]
        e = embed512(model, c["iq"], c["res"], c["phys"], nfft, hop)
        b = cross_cond_bursts(e, c["dist"], N, seed=0)
        if b is None:
            continue
        bp.append(b); bl.append(np.full(len(b), di))
    bp = np.concatenate(bp); bl = np.concatenate(bl)
    bp, bl = balance(bp, bl)
    km, sp = H.oracle_km_sp(unit(bp), bl, KORACLE)
    return dict(oracle_km=km, oracle_sp=sp, knn1=H.knn_purity(bp, bl)[1], n=int(len(bl)))


def phys_grad_share(model, batch_iq, batch_res, batch_phys, y, nfft, hop, supcon):
    model.train(); model.zero_grad()
    xt = torch.from_numpy(batch_iq).cuda(); rs = torch.from_numpy(batch_res).cuda()
    ph = torch.from_numpy(batch_phys).cuda(); yy = torch.from_numpy(y).cuda()
    x_time = torch.cat([xt, rs], 1); x_spec = make_spec4(xt, rs, nfft, hop)
    emb = model(x_time, x_spec, ph)
    loss = supcon(emb, yy, temperature=TAU)
    loss.backward()
    g = dict(time=0.0, spectral=0.0, phys=0.0, other=0.0)
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        gg = float(p.grad.norm())
        if n.startswith("time_branch"): g["time"] += gg
        elif n.startswith("spectral_branch"): g["spectral"] += gg
        elif n.startswith("phys"): g["phys"] += gg
        else: g["other"] += gg
    model.zero_grad()
    tot = sum(g.values()) + 1e-9
    return round(g["phys"] / tot, 4)


# ============================================================================
# EIGENGAP (Play-1b stack, copied small)
# ============================================================================
def _nn(n):
    return int(min(15, max(3, n // 4)))


def eigengap_k(X, kmax=10):
    n = len(X); K = min(kmax, n - 1)
    A = kneighbors_graph(X, n_neighbors=min(_nn(n), n - 1), mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T); L = laplacian(A, normed=True).toarray()
    vals = np.sort(np.linalg.eigvalsh(L)); gaps = {k: vals[k] - vals[k - 1] for k in range(2, K + 1)}
    return int(max(gaps, key=gaps.get))


def partition_at(X, K):
    try:
        return SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                  n_neighbors=min(_nn(len(X)), len(X) - 1)).fit_predict(X)
    except Exception:
        return KMeans(K, n_init=10, random_state=0).fit_predict(X)


# ============================================================================
# TRAIN
# ============================================================================
def sample_batch(dom_idx, dom, rng):
    devs = list(dom_idx[dom].keys())
    chosen = rng.choice(len(devs), size=min(P_DEV, len(devs)), replace=False)
    idxs, ys = [], []
    for di in chosen:
        dl = devs[di]; conds = dom_idx[dom][dl]; cks = list(conds.keys())
        picks = []
        if len(cks) >= 2:
            rng.shuffle(cks)
            for ck in cks[:2]:
                picks.append(int(conds[ck][rng.integers(len(conds[ck]))]))
        allw = [w for ck in cks for w in conds[ck]]
        while len(picks) < V_VIEW:
            picks.append(int(allw[rng.integers(len(allw))]))
        idxs.extend(picks[:V_VIEW]); ys.extend([dl] * V_VIEW)
    return np.array(idxs), np.array(ys)


def lr_lambda(step):
    if step < WARMUP:
        return step / max(1, WARMUP)
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return 0.5 * (1 + math.cos(math.pi * prog))


def train(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    ckdir = os.path.join(RUNS_GEN, f"seed{seed}"); os.makedirs(ckdir, exist_ok=True)
    print(f"\n=== TRAIN seed={seed} (LOPO holdout=DRFF; train=WiSig109+ORACLE12) ===")
    print(f"SELECTION: {SELECTION_FORMULA}")
    ds = build_train()
    dom_idx = domain_index(ds)
    doms = [d for d in dom_idx if len(dom_idx[d]) >= P_DEV]
    sizes = {d: len(dom_idx[d]) for d in doms}
    tot = sum(sizes.values()); dweights = np.array([sizes[d] for d in doms], float); dweights /= dweights.sum()
    for d in doms:
        percond = np.mean([len(dom_idx[d][dl]) for dl in dom_idx[d]])
        print(f"  domain {d}: {sizes[d]} identities, P={min(P_DEV,sizes[d])} V={V_VIEW}, "
              f"~{percond:.1f} conditions/identity, weight={sizes[d]/tot:.3f}")
    slice18, cacheA = build_val_a()
    gidsB, cacheB = build_val_b()
    print(f"  VAL-A WiSig DEV slice: {len(slice18)} devices | VAL-B ORACLE eval: {len(gidsB)} devices")

    nfft = {d: registry.REGISTRY[d].n_fft for d in doms}
    hop = {d: registry.REGISTRY[d].hop for d in doms}

    model = GenRFEncoder(branch_dropout=True, p=0.15).cuda()
    tot_p, _ = param_count(model)
    print(f"  GenRFEncoder params: {tot_p:,} ({tot_p/1e6:.2f}M)")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler('cuda')
    supcon = SupervisedContrastiveLoss()

    dom_sched = list(rng.choice(doms, size=STEPS + 1, p=dweights))
    curve = []; dom_loss = defaultdict(list)
    best = dict(comp=-1e9, step=-1, state=None); t0 = time.time()
    for step in range(1, STEPS + 1):
        dom = dom_sched[step]
        model.train()
        idxs, y = sample_batch(dom_idx, dom, rng)
        iq, res, ph = fetch_batch(ds, idxs)
        xt = torch.from_numpy(iq).cuda(); rs = torch.from_numpy(res).cuda()
        x4 = augment_4ch(torch.cat([xt, rs], 1))
        x_spec = make_spec4(x4[:, :2], x4[:, 2:], nfft[dom], hop[dom])
        phcu = torch.from_numpy(ph).cuda(); ycu = torch.from_numpy(y).cuda()
        with torch.amp.autocast('cuda'):
            emb = model(x4, x_spec, phcu)
            loss = supcon(emb, ycu, temperature=TAU)
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        dom_loss[dom].append(float(loss))
        if step % 100 == 0:
            dl = {d: round(float(np.mean(dom_loss[d][-50:])) if dom_loss[d] else float('nan'), 3) for d in doms}
            print(f"  step {step:5d}: loss={loss:.3f} per-dom(last50)={dl} lr={sched.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)")
        if step % VAL_EVERY == 0:
            ra = val_a(model, slice18, cacheA); rb = val_b(model, gidsB, cacheB)
            # physics grad share on a fresh WiSig batch
            ii, yy = sample_batch(dom_idx, "WISIG", rng)
            iqi, resi, phi = fetch_batch(ds, ii)
            pshare = phys_grad_share(model, iqi, resi, phi, yy,
                                     nfft["WISIG"], hop["WISIG"], supcon)
            comp = composite(ra["oracle_km"], ra["knn1"], rb["oracle_km"], rb["knn1"])
            row = dict(step=step, VALA_km=round(ra["oracle_km"], 3), VALA_knn1=round(ra["knn1"], 3),
                       VALB_km=round(rb["oracle_km"], 3), VALB_knn1=round(rb["knn1"], 3),
                       phys_grad_share=pshare, composite=round(comp, 4),
                       loss=round(float(loss), 3))
            curve.append(row)
            flag = ""
            if comp > best["comp"]:
                best = dict(comp=comp, step=step, state=copy.deepcopy(model.state_dict())); flag = "  <-best"
            print(f"    [VAL {step}] A:km={ra['oracle_km']:.3f}/knn1={ra['knn1']:.3f}  "
                  f"B:km={rb['oracle_km']:.3f}/knn1={rb['knn1']:.3f}  physShare={pshare}  comp={comp:.4f}{flag}")
        if step % CKPT_EVERY == 0:
            torch.save(model.state_dict(), os.path.join(ckdir, f"ckpt_step{step}.pt"))
    model.load_state_dict(best["state"])
    torch.save(best["state"], os.path.join(ckdir, "best.pt"))
    json.dump(curve, open(os.path.join(ckdir, "curve.json"), "w"), indent=2)
    print(f"  selected step {best['step']} (composite {best['comp']:.4f})")
    return model, best, curve, ds, (slice18, cacheA), (gidsB, cacheB)


# ============================================================================
# FINAL EVAL — mavicAir2 native, E1 bursts N=10 and N=120
# ============================================================================
MAVIC_CAP = 1500


def build_mavicair2_native():
    _, ev = loaders.drff_airframes()
    units = loaders.load_drff_native(ev, cap=MAVIC_CAP)     # native 4096
    d = registry.REGISTRY["DRFF"]
    iq, res, phys, af, D, C = [], [], [], [], [], []
    for u in units:
        X = u["X"].astype(np.float32)
        rr, _ = residual_batch(X, order=32)
        pp = classical_matrix(X, fs=d.native_rate)
        for j in range(X.shape[0]):
            iq.append(X[j]); res.append(rr[j]); phys.append(pp[j])
            af.append(u["meta"]["airframe"]); D.append(u["meta"]["D"]); C.append(u["meta"]["C"])
    return (np.stack(iq).astype(np.float32), np.stack(res).astype(np.float32),
            np.stack(phys).astype(np.float32), np.array(af), np.array(D), np.array(C))


def E1_bursts(emb, af, D, C, N, seed=777):
    r = np.random.default_rng(seed); bp, bl = [], []
    for a in np.unique(af):
        idx = np.where(af == a)[0]; cells = defaultdict(list)
        for i in idx:
            cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx) // N
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < N and ci < 4000:
                c = ck[ci % len(ck)]
                if cells[c]:
                    pick.append(cells[c][r.integers(len(cells[c]))])
                ci += 1
            if len(pick) == N:
                bp.append(emb[pick].mean(0)); bl.append(a)
    return np.array(bp), np.array(bl)


def final_eval(model):
    d = registry.REGISTRY["DRFF"]
    iq, res, phys, af, D, C = build_mavicair2_native()
    emb = embed512(model, iq, res, phys, d.n_fft, d.hop)
    out = {}
    for N in (NBURST_PAPER, NBURST_DEMO):
        bp, bl = E1_bursts(emb, af, D, C, N)
        bp, bl = balance(bp, bl); bpu = unit(bp)
        per, hdbmean = H.hdbscan_grid(bpu, bl, MCS_DRFF)
        km, sp = H.oracle_km_sp(bpu, bl, KDRFF)
        intra, inter = H.cos_gap(bp, bl)
        cell = dict(N=N, n_bursts=int(len(bl)), hdb_mean=round(hdbmean, 3),
                    oracle_km=round(km, 3), oracle_sp=round(sp, 3),
                    knn1=round(H.knn_purity(bp, bl)[1], 3), gap=round(intra - inter, 3))
        if N == NBURST_DEMO:
            kest = eigengap_k(bpu); lab = partition_at(bpu, kest)
            yi = np.unique(bl, return_inverse=True)[1]
            cell["eigengap_Kest"] = int(kest)
            cell["ARI_at_Kest"] = round(float(adjusted_rand_score(yi, lab)), 3)
        out[f"N{N}"] = cell
        print(f"  mavicAir2 E1 N={N}: HDBmean={cell['hdb_mean']} oracle km={cell['oracle_km']}/sp={cell['oracle_sp']} "
              f"kNN1={cell['knn1']} gap={cell['gap']}" +
              (f" | eigengap K={cell.get('eigengap_Kest')} ARI@K={cell.get('ARI_at_Kest')}" if N == NBURST_DEMO else ""))
    # K=4 subset battery (70 subsets) at N=120
    uaf = sorted(np.unique(af)); aris, correct = [], []
    for sub in itertools.combinations(uaf, 4):
        m = np.isin(af, sub)
        bp, bl = E1_bursts(emb[m], af[m], D[m], C[m], NBURST_DEMO)
        if len(bl) < 4 or len(np.unique(bl)) < 4:
            continue
        bp, bl = balance(bp, bl); bpu = unit(bp)
        if len(bl) <= max(MCS_DRFF):     # too few bursts for HDBSCAN grid
            continue
        _, hm = H.hdbscan_grid(bpu, bl, MCS_DRFF)
        # correct-K via HDBSCAN K_est (median over grid)
        ks = [H.score_locked(H.hdbscan_pred(bpu, mcs), bl)["K_est"] for mcs in MCS_DRFF]
        aris.append(hm); correct.append(abs(np.median(ks) - 4) < 0.5)
    k4 = dict(n_subsets=len(aris), mean_ari=round(float(np.mean(aris)), 3),
              frac_correctK=round(float(np.mean(correct)), 3))
    print(f"  K=4 battery ({k4['n_subsets']} subsets): mean HDB ARI={k4['mean_ari']} correct-K frac={k4['frac_correctK']}")
    out["k4_battery"] = k4
    return out


def self_cells(model, va, vb):
    slice18, cacheA = va; gidsB, cacheB = vb
    ra = val_a(model, slice18, cacheA); rb = val_b(model, gidsB, cacheB, N=NBURST_PAPER)
    print(f"  self-cell WiSig-DEV P2 oracle-km@18={ra['oracle_km']:.3f} (specialist ref {SPECIALIST_REF}) kNN1={ra['knn1']:.3f}")
    print(f"  self-cell ORACLE-4 oracle-km@4={rb['oracle_km']:.3f} kNN1={rb['knn1']:.3f}")
    return dict(wisig_dev=dict(oracle_km=round(ra["oracle_km"], 3), knn1=round(ra["knn1"], 3),
                               specialist_ref=SPECIALIST_REF),
                oracle4=dict(oracle_km=round(rb["oracle_km"], 3), knn1=round(rb["knn1"], 3)))


def band(km):
    if km >= 0.50:
        return f"STRONG (oracle-K@8 km={km:.3f} >= 0.50): multi-domain + physics beats the transfer wall."
    if km >= 0.40:
        return f"MEANINGFUL (oracle-K@8 km={km:.3f} in [0.40,0.50)): real lift over the 0.30 wall."
    return f"NULL (oracle-K@8 km={km:.3f} <= 0.40): recipe additions bought little over the 0.30 wall."


def main():
    model, best, curve, ds, va, vb = train(SEED)
    print("\n=== FINAL EVAL — mavicAir2 (untouched until now) ===")
    fe = final_eval(model)
    sc = self_cells(model, va, vb)
    km_head = max(fe["N10"]["oracle_km"], fe["N120"]["oracle_km"])
    verdict = band(km_head)
    print(f"\nBAND: {verdict}")
    report = dict(header="GEN-RFF PHASE 2 — LOPO holdout=DRFF (DEMO-SIDE, R&D branch)",
                  config=dict(seed=SEED, steps=STEPS, P=P_DEV, V=V_VIEW, tau=TAU, lr=LR,
                              wisig_cap=WISIG_CAP, oracle_cap=ORACLE_CAP, oracle_dists=ORACLE_DISTS),
                  selection_formula=SELECTION_FORMULA, selected_step=best["step"],
                  selected_composite=round(best["comp"], 4), curve=curve,
                  final_eval=fe, self_cells=sc,
                  context_rows=dict(frozen_wisig_only=dict(oracle_km=CTX_FROZEN_WISIG[0], oracle_sp=CTX_FROZEN_WISIG[1]),
                                    native_drone_trained_N50=dict(oracle_km=CTX_NATIVE_DRONE[0], oracle_sp=CTX_NATIVE_DRONE[1])),
                  band_verdict=verdict,
                  guardrails="LOPO holdout=DRFF; mavicAir2 touched only in final eval; single fixed "
                             "config (no sweeps); frozen assets read-only; gen_rff/results_gen/runs_gen only.")
    json.dump(report, open(os.path.join(RESULTS_GEN, "phase2_lopo_drff_report.json"), "w"), indent=2, default=str)
    print(f"\nsaved -> results_gen/phase2_lopo_drff_report.json ; ckpts -> runs_gen/lopo_drff/seed{SEED}/")
    print("CHECKPOINT — R&D; mavicAir2 discovered once; no DANN/ablations/second-cell.")


if __name__ == "__main__":
    main()
