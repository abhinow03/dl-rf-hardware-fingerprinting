"""FIX DISCOVERY GEOMETRY — STAGE 1: no-gradient linear discriminative transform.
Backbone FROZEN. Fit transform on 109 TRAINING devices' frozen 512-D features (labels there),
apply to the UNSEEN random-18 seed-123 slice, cluster burst-mean (N=10) HDBSCAN. Bar 0.540.

The features are LINEARLY separable (readout 0.92) but not DENSITY-separable (raw 512-D HDBSCAN
0.45). Q: can a learned linear transform (LDA / within-class whitening) make them cluster-friendly
with NO gradient training? Target dims {64,128,256}.

    python3 discover/geometry_stage1.py
Backbone frozen; transform fit on training devices only; the 18 are unseen. No drones.
"""
import os, sys, json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rffp.models import RFEncoder
from rffp.data import wisig_manytx as W
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR   = os.path.join(RUN_DIR, "discover")
SPLIT_OLD = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
REPORT    = os.path.join(OUT_DIR, "geometry_stage1_report.json")

RAND_N, RAND_SEED = 18, 123
WIN_SLICE = 2400          # consec windows/dev for the 18 (matches 0.540 baseline budget)
TRAIN_PER_DEV = 300       # consec windows/dev for fitting the transform (109 devs)
BURST_N, MCS = 10, 15
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


def burst_pool(F, dids, N, renorm):
    bp, bl = [], []
    for d in np.unique(dids):
        E = F[dids == d]; nb = E.shape[0] // N
        if nb == 0: continue
        ch = E[:nb*N].reshape(nb, N, -1).mean(1)
        bp.append(unitrows(ch) if renorm else ch); bl.append(np.full(nb, d))
    return np.concatenate(bp), np.concatenate(bl)


def cluster(X, dids, n_true):
    pred = HDBSCAN(min_cluster_size=MCS, metric="euclidean", copy=True).fit_predict(X)
    keep = pred != -1; k = len(np.unique(pred[keep])); noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return {"ARI": 0.0, "NMI": 0.0, "purity": float(purity(dids, pred)), "K_est": int(k),
                "K_true": int(n_true), "noise": noise}, pred
    return {"ARI": float(adjusted_rand_score(dids[keep], pred[keep])),
            "NMI": float(normalized_mutual_info_score(dids[keep], pred[keep])),
            "purity": float(purity(dids, pred)), "K_est": int(k), "K_true": int(n_true),
            "noise": noise}, pred


def wccn_whiten(Xtr, ytr, shrink=0.1):
    """within-class whitening: W = Sw^{-1/2} (shrinkage-regularized). Returns (mean, W)."""
    d = Xtr.shape[1]; mu = Xtr.mean(0)
    Sw = np.zeros((d, d), dtype=np.float64)
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]; Xc = Xc - Xc.mean(0)
        Sw += Xc.T @ Xc
    Sw /= len(Xtr)
    Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / d * np.eye(d)
    ev, V = np.linalg.eigh(Sw)
    ev = np.clip(ev, 1e-6, None)
    Wm = (V / np.sqrt(ev)) @ V.T
    return mu.astype(np.float64), Wm


def eval_transform(name, Xslice_win, dids18, transform, results, coincident_idx, rand_tx):
    """apply per-window transform, burst-mean (both renorm & raw), keep best; record."""
    Z = transform(Xslice_win)
    best = None
    for rn in (True, False):
        bp, bl = burst_pool(Z, dids18, BURST_N, renorm=rn)
        d, pred = cluster(bp, bl, RAND_N)
        d["renorm"] = rn
        if best is None or d["ARI"] > best[0]["ARI"]:
            best = (d, bp, bl, pred)
    d, bp, bl, pred = best
    # centroid geometry on transformed burst points
    C = unitrows(np.stack([bp[bl == i].mean(0) for i in range(RAND_N)]))
    M = C @ C.T; np.fill_diagonal(M, -2.0)
    nn_med = float(np.median(M.max(1)))
    coinc = {f"{rand_tx[a]}/{rand_tx[b]}": round(float(M[a, b]), 4) for a, b in coincident_idx}
    intra = float(np.mean([ (unitrows(bp[bl==i]) @ unit(bp[bl==i].mean(0))).mean() for i in range(RAND_N)]))
    d.update({"nn_med_cos": round(nn_med, 4), "intra_burst": round(intra, 3),
              "coincident_pair_cos": coinc, "dim": int(Z.shape[1])})
    results[name] = d
    print(f"{name:>16} dim{Z.shape[1]:>4} renorm={str(d['renorm']):>5} | ARI={d['ARI']:>6.3f} "
          f"NMI={d['NMI']:.3f} pur={d['purity']:.3f} K={d['K_est']}/{RAND_N} "
          f"nn={nn_med:.3f} intra={intra:.3f}")
    return d


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); train_tx, held_tx = sp["train_tx"], sp["discover_tx"]
    rand_tx = scatter_slice(held_tx, RAND_N, RAND_SEED)
    coincident_idx = [(rand_tx.index(a), rand_tx.index(b)) for a, b in COINCIDENT]
    print(f"train devs: {len(train_tx)} | unseen slice: {rand_tx}")

    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # ── frozen 512-D features: training (fit) + unseen slice (eval) ──
    tt, ty = consec_pack(tx_data, train_tx, TRAIN_PER_DEV)
    print(f"extracting 512-D for {tt.shape[0]} TRAIN windows...")
    Ftr = extract512(model, tt); del tt
    st, sy = consec_pack(tx_data, rand_tx, WIN_SLICE)
    print(f"extracting 512-D for {st.shape[0]} SLICE windows...")
    Fsl = extract512(model, st); del st; torch.cuda.empty_cache()

    # standardize on training
    sc = StandardScaler().fit(Ftr)
    Ftr_s = sc.transform(Ftr).astype(np.float32)
    Fsl_s = sc.transform(Fsl).astype(np.float32)

    results = {}
    print(f"\n=== STAGE 1 — no-gradient transforms, burst-mean N={BURST_N} HDBSCAN (BAR 0.540) ===")

    # baselines (no transform)
    eval_transform("raw512", Fsl_s, sy, lambda X: X, results, coincident_idx, rand_tx)

    # LDA at several component counts (max 108 = n_classes-1)
    for k in (32, 64, 108):
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=k)
        lda.fit(Ftr_s, ty)
        eval_transform(f"LDA-{k}", Fsl_s, sy, lambda X, l=lda: l.transform(X).astype(np.float32),
                       results, coincident_idx, rand_tx)

    # within-class whitening (WCCN), full 512-D
    mu, Wm = wccn_whiten(Ftr_s, ty, shrink=0.1)
    wccn = lambda X: ((X.astype(np.float64) - mu) @ Wm).astype(np.float32)
    eval_transform("WCCN-512", Fsl_s, sy, wccn, results, coincident_idx, rand_tx)

    # WCCN + PCA to {128,256} (discriminative directions after whitening)
    Ztr_w = wccn(Ftr_s)
    for k in (128, 256):
        pca = PCA(n_components=k, whiten=False).fit(Ztr_w)
        eval_transform(f"WCCN+PCA-{k}", Fsl_s, sy,
                       lambda X, p=pca: p.transform(wccn(X)).astype(np.float32),
                       results, coincident_idx, rand_tx)

    # ── verdict ──
    bar = 0.540
    best_name = max(results, key=lambda n: results[n]["ARI"])
    best = results[best_name]
    print(f"\n=== VERDICT (Stage 1) ===")
    print(f"  bar (128-D burst-mean) = {bar:.3f}")
    print(f"  best transform = {best_name}: ARI={best['ARI']:.3f} ({best['ARI']-bar:+.3f}) "
          f"pur={best['purity']:.3f} K={best['K_est']} nn_med={best['nn_med_cos']}")
    print(f"  coincident-pair cos under {best_name}: {best['coincident_pair_cos']}")
    if best["ARI"] > bar + 0.05:
        verdict = (f"STAGE 1 WINS (FREE): {best_name} lifts burst-mean ARI to {best['ARI']:.3f} "
                   f"(+{best['ARI']-bar:.3f}) with NO gradient training — a fit-on-training linear "
                   f"discriminative transform makes the frozen features density-clusterable. "
                   f"Skip Stage 2. Deploy: freeze this transform after the encoder.")
    elif best["ARI"] > bar + 0.02:
        verdict = (f"STAGE 1 MARGINAL: {best_name} ARI {best['ARI']:.3f} (+{best['ARI']-bar:.3f}) — small "
                   f"free gain; run STAGE 2 (head retrain) to see if a learned dense-cluster objective does better.")
    else:
        verdict = (f"STAGE 1 UNDERWHELMS: best {best_name} ARI {best['ARI']:.3f} <= bar+0.02 — a linear "
                   f"transform can't create density-separability (confirms linear!=dense). PROCEED TO STAGE 2 "
                   f"(freeze backbone, retrain head with a dense-cluster objective).")
    print(f"  {verdict}")

    with open(REPORT, "w") as f:
        json.dump({"bar": bar, "config": {"win_slice": WIN_SLICE, "train_per_dev": TRAIN_PER_DEV,
                   "burst_n": BURST_N, "mcs": MCS}, "random_slice": rand_tx,
                   "results": results, "best": best_name, "verdict": verdict}, f, indent=2)
    print(f"\nsaved -> {REPORT}\nCHECKPOINT — backbone FROZEN, no gradient training. No drones.")


if __name__ == "__main__":
    main()
