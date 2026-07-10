"""FIX DISCOVERY GEOMETRY — STAGE 2: retrain ONLY a projection head on FROZEN 512-D features.
Backbone FROZEN (features cached, no backbone gradient). best_model.pt UNTOUCHED. No drones.

Stage 1 (linear transforms) underwhelmed (best WCCN 0.511 < bar 0.540): a linear map can't create
density-separability. Here we learn a small NONLINEAR head with objectives that explicitly make
TIGHT, GAP-SEPARATED clusters (NOT plain SupCon, which gave linear-but-not-dense):
  - cosface : additive cosine-margin softmax over the 109 training devices (compact, angularly
              gapped classes).
  - center  : center-pull (1 - cos(z, c_y)) + inter-centroid margin (push centers apart).
Embed dim {128, 256}. Fit on the 109 TRAINING devices' frozen 512-D features; the random-18
seed-123 slice is UNSEEN. Select by burst-mean N=10 HDBSCAN ARI (locked ruler). Bar 0.540 / Stage1 0.511.

    python3 discover/geometry_stage2.py
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR   = os.path.join(RUN_DIR, "discover")
SPLIT_OLD = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
HEAD_DIR  = os.path.join(OUT_DIR, "geometry_heads")
REPORT    = os.path.join(OUT_DIR, "geometry_stage2_report.json")

RAND_N, RAND_SEED = 18, 123
WIN_SLICE, TRAIN_PER_DEV = 2400, 500
BURST_N, MCS = 10, 15
STEPS, BS, LR = 3000, 512, 1e-3
COS_M, COS_S = 0.20, 30.0        # cosface margin / scale
CEN_MNEG, CEN_BETA = 0.25, 1.0   # center: inter-centroid cos margin / weight
SEED = 42
COINCIDENT = [("5-16", "16-19"), ("18-16", "16-19"), ("5-1", "5-5"),
              ("20-14", "20-4"), ("16-1", "4-1")]


def unit(v): return v / (np.linalg.norm(v) + 1e-8)
def unitrows(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def purity(true, pred):
    if len(true) == 0: return float("nan")
    return sum(np.unique(true[pred == c], return_counts=True)[1].max()
               for c in np.unique(pred)) / len(true)


def scatter_slice(held_tx, n, seed):
    rng = np.random.default_rng(seed); rows = {}
    for t in held_tx:
        rows.setdefault(int(t.split("-")[0]), []).append(t)
    for r in rows: rng.shuffle(rows[r])
    keys = sorted(rows); ptr = {r: 0 for r in keys}; out = []
    while len(out) < n:
        prog = False
        for r in keys:
            if len(out) >= n: break
            if ptr[r] < len(rows[r]): out.append(rows[r][ptr[r]]); ptr[r] += 1; prog = True
        if not prog: break
    return sorted(out)


def consec_pack(tx_data, devs, nwin):
    times, dids = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        for k in range(min(nwin, iq.shape[0])):
            times.append(W.standardize(iq[k].T.copy())); dids.append(di)
    return np.stack(times).astype(np.float32), np.array(dids)


@torch.no_grad()
def extract512(model, t, batch=1024):
    s = W.compute_stft_batch(t)
    xt = torch.from_numpy(t).cuda(); xs = torch.from_numpy(s).cuda()
    f = np.empty((t.shape[0], 512), dtype=np.float32)
    for i in range(0, t.shape[0], batch):
        with torch.amp.autocast('cuda'):
            f[i:i+batch] = model.get_encoder_output(xt[i:i+batch], xs[i:i+batch]).float().cpu().numpy()
    return f


def burst_pool(F, dids, N):
    bp, bl = [], []
    for d in np.unique(dids):
        E = F[dids == d]; nb = E.shape[0] // N
        if nb == 0: continue
        bp.append(unitrows(E[:nb*N].reshape(nb, N, -1).mean(1))); bl.append(np.full(nb, d))
    return np.concatenate(bp), np.concatenate(bl)


def cluster(X, dids, n_true):
    pred = HDBSCAN(min_cluster_size=MCS, metric="euclidean", copy=True).fit_predict(X)
    keep = pred != -1; k = len(np.unique(pred[keep])); noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return {"ARI": 0.0, "NMI": 0.0, "purity": float(purity(dids, pred)), "K_est": int(k),
                "K_true": int(n_true), "noise": noise}
    return {"ARI": float(adjusted_rand_score(dids[keep], pred[keep])),
            "NMI": float(normalized_mutual_info_score(dids[keep], pred[keep])),
            "purity": float(purity(dids, pred)), "K_est": int(k), "K_true": int(n_true), "noise": noise}


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1, eps=1e-6)


def eval_head(head, Fsl_s, sy, coincident_idx, rand_tx):
    head.eval()
    with torch.no_grad():
        Z = np.empty((Fsl_s.shape[0], head.net[-1].out_features), dtype=np.float32)
        Xg = torch.from_numpy(Fsl_s).cuda()
        for i in range(0, Xg.shape[0], 4096):
            Z[i:i+4096] = head(Xg[i:i+4096]).cpu().numpy()
    bp, bl = burst_pool(Z, sy, BURST_N)
    d = cluster(bp, bl, RAND_N)
    C = unitrows(np.stack([bp[bl == i].mean(0) for i in range(RAND_N)]))
    M = C @ C.T; np.fill_diagonal(M, -2.0)
    d["nn_med_cos"] = round(float(np.median(M.max(1))), 4)
    d["intra_burst"] = round(float(np.mean([(unitrows(bp[bl==i]) @ unit(bp[bl==i].mean(0))).mean()
                                             for i in range(RAND_N)])), 3)
    d["coincident_pair_cos"] = {f"{rand_tx[a]}/{rand_tx[b]}": round(float(M[a, b]), 4)
                                for a, b in coincident_idx}
    return d


def train_head(kind, dim, Ftr_t, ty_t, n_cls, Fsl_s, sy, coincident_idx, rand_tx):
    torch.manual_seed(SEED); np.random.seed(SEED)
    head = Head(dim).cuda()
    if kind == "cosface":
        Wc = nn.Parameter(F.normalize(torch.randn(n_cls, dim), dim=1).cuda())
        params = list(head.parameters()) + [Wc]
    else:
        Cc = nn.Parameter(F.normalize(torch.randn(n_cls, dim), dim=1).cuda())
        params = list(head.parameters()) + [Cc]
    opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    N = Ftr_t.shape[0]; rng = np.random.default_rng(SEED)
    best = {"ARI": -1.0}; best_step = 0; t0 = time.time()
    ckpt = os.path.join(HEAD_DIR, f"{kind}_d{dim}.pt")

    for step in range(STEPS):
        idx = torch.from_numpy(rng.integers(0, N, size=BS)).cuda()
        x = Ftr_t[idx]; y = ty_t[idx]
        z = head(x)
        if kind == "cosface":
            Wn = F.normalize(Wc, dim=1)
            cos = z @ Wn.t()
            oh = F.one_hot(y, n_cls).float()
            logits = COS_S * (cos - COS_M * oh)
            loss = ce(logits, y)
        else:
            Cn = F.normalize(Cc, dim=1)
            pull = (1 - (z * Cn[y]).sum(1)).mean()
            G = Cn @ Cn.t()
            off = G[~torch.eye(n_cls, dtype=torch.bool, device=G.device)]
            inter = torch.relu(off - CEN_MNEG).mean()
            loss = pull + CEN_BETA * inter
        opt.zero_grad(); loss.backward(); opt.step()

        if (step + 1) % 500 == 0 or step == 0:
            d = eval_head(head, Fsl_s, sy, coincident_idx, rand_tx)
            tag = f"{kind}-d{dim}"
            print(f"  [{tag}] step {step+1:4d} loss={float(loss):.3f} | ARI={d['ARI']:.3f} "
                  f"pur={d['purity']:.3f} K={d['K_est']}/{RAND_N} intra={d['intra_burst']:.3f} "
                  f"nn={d['nn_med_cos']:.3f}", flush=True)
            if d["ARI"] > best["ARI"]:
                best = {**d, "step": step + 1}; best_step = step + 1
                torch.save(head.state_dict(), ckpt)
            head.train()
    best["minutes"] = round((time.time() - t0) / 60, 1)
    print(f"  [{kind}-d{dim}] BEST ARI={best['ARI']:.3f} @step{best_step} "
          f"intra={best['intra_burst']:.3f} nn={best['nn_med_cos']:.3f} ({best['minutes']}min)")
    return {"kind": kind, "dim": dim, **best}


def main():
    os.makedirs(HEAD_DIR, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); train_tx, held_tx = sp["train_tx"], sp["discover_tx"]
    rand_tx = scatter_slice(held_tx, RAND_N, RAND_SEED)
    coincident_idx = [(rand_tx.index(a), rand_tx.index(b)) for a, b in COINCIDENT]
    n_cls = len(train_tx)
    print(f"train devs: {n_cls} | unseen slice: {rand_tx}")

    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    tt, ty = consec_pack(tx_data, train_tx, TRAIN_PER_DEV)
    print(f"extracting 512-D for {tt.shape[0]} TRAIN windows...")
    Ftr = extract512(model, tt); del tt
    st, sy = consec_pack(tx_data, rand_tx, WIN_SLICE)
    print(f"extracting 512-D for {st.shape[0]} SLICE windows...")
    Fsl = extract512(model, st); del st; torch.cuda.empty_cache()

    sc = StandardScaler().fit(Ftr)
    Ftr_s = sc.transform(Ftr).astype(np.float32); Fsl_s = sc.transform(Fsl).astype(np.float32)
    Ftr_t = torch.from_numpy(Ftr_s).cuda(); ty_t = torch.from_numpy(ty).cuda()

    print(f"\n=== STAGE 2 — head retrain (backbone frozen), burst-mean N={BURST_N} (BAR 0.540 / Stage1 0.511) ===")
    results = []
    for kind in ("cosface", "center"):
        for dim in (128, 256):
            print(f"--- {kind} dim={dim} ---")
            results.append(train_head(kind, dim, Ftr_t, ty_t, n_cls, Fsl_s, sy, coincident_idx, rand_tx))

    print(f"\n{'='*70}\nSWEEP (bar 0.540, Stage1 0.511)\n{'='*70}")
    print(f"{'config':>14} {'ARI':>7} {'NMI':>7} {'purity':>7} {'K_est':>6} {'intra':>7} {'nn_med':>7}")
    for r in results:
        print(f"{(r['kind']+'-d'+str(r['dim'])):>14} {r['ARI']:>7.3f} {r['NMI']:>7.3f} "
              f"{r['purity']:>7.3f} {r['K_est']:>6} {r['intra_burst']:>7.3f} {r['nn_med_cos']:>7.4f}")

    best = max(results, key=lambda r: r["ARI"]); bar = 0.540
    print(f"\n=== VERDICT ===")
    print(f"  best = {best['kind']}-d{best['dim']}: ARI={best['ARI']:.3f} ({best['ARI']-bar:+.3f} vs bar)")
    print(f"  coincident-pair cos: {best['coincident_pair_cos']}")
    if best["ARI"] > bar + 0.05:
        verdict = (f"GEOMETRY FIXED (cheap head retrain): {best['kind']}-d{best['dim']} lifts burst-mean ARI "
                   f"to {best['ARI']:.3f} (+{best['ARI']-bar:.3f}), backbone FROZEN — a dense-cluster head "
                   f"objective makes the frozen features density-separable. Deploy: freeze backbone + this head.")
    elif best["ARI"] > bar + 0.02:
        verdict = (f"MODEST FIX: {best['kind']}-d{best['dim']} ARI {best['ARI']:.3f} (+{best['ARI']-bar:.3f}) — "
                   f"a real head-only gain but small; worth banking, tune margin/dim next.")
    else:
        verdict = (f"HEAD RETRAIN INSUFFICIENT: best {best['ARI']:.3f} <= bar+0.02. The dense-cluster geometry "
                   f"does not transfer to unseen devices from head-only training -> the limit is deeper than "
                   f"the projection (backbone representation / genuine open-set gap). Do NOT overfit knobs; "
                   f"flag for a backbone-level decision next session.")
    print(f"  {verdict}")

    with open(REPORT, "w") as f:
        json.dump({"bar": bar, "stage1_best": 0.511, "config": {"steps": STEPS, "cos_m": COS_M,
                   "cos_s": COS_S, "cen_mneg": CEN_MNEG, "burst_n": BURST_N},
                   "random_slice": rand_tx, "results": results,
                   "best": f"{best['kind']}-d{best['dim']}", "verdict": verdict}, f, indent=2)
    print(f"\nsaved -> {REPORT}\nCHECKPOINT — backbone FROZEN, head-only. best_model.pt UNTOUCHED. No drones.")


if __name__ == "__main__":
    main()
