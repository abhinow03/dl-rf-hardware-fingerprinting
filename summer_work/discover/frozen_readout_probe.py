"""FROZEN-FEATURE READOUT PROBE — split "encoder capacity/projection" from "features don't
generalize open-set", BEFORE any architecture retrain. Encoder FROZEN. Cache + fresh shallow
readouts only. No retrain, no drones. STOP at checkpoint.

The multi-class probe (closed-set, trained on the 18) proved the 256-window carries the full
fingerprint. This asks the open-set question with the ENCODER NEVER TRAINED ON THESE 18:
  - fresh readout on frozen 512-D (get_encoder_output) separates the unseen 18 WELL  -> generalizable
    fingerprint IS in the representation, lost in the 128-D metric geometry -> Diagnosis A
    (capacity/projection, CHEAP).
  - 512-D readout ALSO low -> features don't generalize open-set -> Diagnosis B (HARD).
  - 512-D >> 128-D -> the PROJECTION HEAD destroys usable structure -> cheapest fix (cluster 512-D).

Steps: (1) extract frozen 512-D pre-proj + 128-D proj for the random seed-123 18-slice;
(2) fresh LogisticRegression + small MLP readouts on each (per-signal balanced split);
(3) burst-level (N=10) linear separability on 512-D; (4) burst-mean HDBSCAN discovery on 512-D
vs 128-D (does 512-D beat the locked 0.54 @128-D,N=10?).

    python3 discover/frozen_readout_probe.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR   = os.path.join(RUN_DIR, "discover")
SPLIT_OLD = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
REPORT    = os.path.join(OUT_DIR, "frozen_readout_report.json")

RAND_N, RAND_SEED = 18, 123
WIN_BUDGET = 2400          # consecutive windows/dev (matches burst_probe's 0.54 baseline budget)
BURST_N = 10               # locked deployment unit
MCS = 15
READOUT_PER_DEV = 1200     # subsample for the per-signal readouts (speed)
VAL_FRAC = 0.2
SEED = 42
MLP_EPOCHS, BS = 40, 256


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def unitrows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def purity(true, pred):
    if len(true) == 0:
        return float("nan")
    return sum(np.unique(true[pred == c], return_counts=True)[1].max()
               for c in np.unique(pred)) / len(true)


def scatter_slice(held_tx, n, seed):
    rng = np.random.default_rng(seed)
    rows = {}
    for t in held_tx:
        rows.setdefault(int(t.split("-")[0]), []).append(t)
    for r in rows:
        rng.shuffle(rows[r])
    keys = sorted(rows); ptr = {r: 0 for r in keys}; out = []
    while len(out) < n:
        prog = False
        for r in keys:
            if len(out) >= n:
                break
            if ptr[r] < len(rows[r]):
                out.append(rows[r][ptr[r]]); ptr[r] += 1; prog = True
        if not prog:
            break
    return sorted(out)


def consec_pack(tx_data, devs, nwin):
    times, dids = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        for k in range(min(nwin, iq.shape[0])):
            times.append(W.standardize(iq[k].T.copy())); dids.append(di)
    return np.stack(times).astype(np.float32), np.array(dids)


@torch.no_grad()
def extract(model, t, batch=1024):
    """returns (512-D pre-proj, 128-D proj) for time windows t."""
    s = W.compute_stft_batch(t)
    xt = torch.from_numpy(t).cuda(); xs = torch.from_numpy(s).cuda()
    f512 = np.empty((t.shape[0], 512), dtype=np.float32)
    f128 = np.empty((t.shape[0], 128), dtype=np.float32)
    for i in range(0, t.shape[0], batch):
        with torch.amp.autocast('cuda'):
            f512[i:i+batch] = model.get_encoder_output(xt[i:i+batch], xs[i:i+batch]).float().cpu().numpy()
            f128[i:i+batch] = model(xt[i:i+batch], xs[i:i+batch]).float().cpu().numpy()
    return f512, f128


def strat_split(y, ncls, frac, seed):
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in range(ncls):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        nv = int(len(ci) * frac); va.extend(ci[:nv]); tr.extend(ci[nv:])
    return np.array(tr), np.array(va)


def lr_readout(X, y, ncls, seed=SEED):
    tr, va = strat_split(y, ncls, VAL_FRAC, seed)
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(X[tr]), y[tr])
    return float(clf.score(sc.transform(X[va]), y[va]))


class MLP(nn.Module):
    def __init__(self, d, ncls):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, ncls))

    def forward(self, x):
        return self.net(x)


def mlp_readout(X, y, ncls, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, va = strat_split(y, ncls, VAL_FRAC, seed)
    sc = StandardScaler().fit(X[tr])
    Xt = torch.from_numpy(sc.transform(X).astype(np.float32)).cuda()
    Y = torch.from_numpy(y).cuda()
    model = MLP(X.shape[1], ncls).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(MLP_EPOCHS):
        model.train(); perm = tr[np.random.permutation(len(tr))]
        for i in range(0, len(perm), BS):
            b = torch.from_numpy(perm[i:i+BS]).cuda()
            opt.zero_grad(); loss = lossf(model(Xt[b]), Y[b]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(Xt[torch.from_numpy(va).cuda()]).argmax(1) == Y[torch.from_numpy(va).cuda()]).float().mean())
        best = max(best, acc)
    del Xt, Y, model; torch.cuda.empty_cache()
    return best


def burst_pool(F, dids, N, renorm):
    bp, bl = [], []
    for d in np.unique(dids):
        E = F[dids == d]; nb = E.shape[0] // N
        if nb == 0:
            continue
        ch = E[:nb*N].reshape(nb, N, -1).mean(1)
        bp.append(unitrows(ch) if renorm else ch); bl.append(np.full(nb, d))
    return np.concatenate(bp), np.concatenate(bl)


def cluster(X, dids, n_true):
    pred = HDBSCAN(min_cluster_size=MCS, metric="euclidean", n_jobs=-1, copy=True).fit_predict(X)
    keep = pred != -1; k = len(np.unique(pred[keep])); noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return {"ARI": 0.0, "NMI": 0.0, "purity": float(purity(dids, pred)), "K_est": int(k),
                "K_true": int(n_true), "noise": noise, "n_pts": int(len(dids))}
    return {"ARI": float(adjusted_rand_score(dids[keep], pred[keep])),
            "NMI": float(normalized_mutual_info_score(dids[keep], pred[keep])),
            "purity": float(purity(dids, pred)), "K_est": int(k), "K_true": int(n_true),
            "noise": noise, "n_pts": int(len(dids))}


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); held_tx = sp["discover_tx"]
    rand_tx = scatter_slice(held_tx, RAND_N, RAND_SEED)
    print(f"random slice (unseen by encoder): {rand_tx}")

    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t, dids = consec_pack(tx_data, rand_tx, WIN_BUDGET)
    print(f"extracting frozen features for {t.shape[0]} windows (512-D pre-proj + 128-D proj)...")
    f512, f128 = extract(model, t)
    del t; torch.cuda.empty_cache()

    # ── per-signal readouts (subsample for speed, balanced) ──
    rng = np.random.default_rng(SEED); sub = []
    for d in range(RAND_N):
        idx = np.where(dids == d)[0]; sub.extend(rng.choice(idx, min(READOUT_PER_DEV, len(idx)), replace=False))
    sub = np.array(sub)
    Xs512, Xs128, ys = f512[sub], f128[sub], dids[sub]

    print("\n=== FRESH SHALLOW READOUTS on FROZEN features (per-signal, balanced 80/20) ===")
    lr512 = lr_readout(Xs512, ys, RAND_N); lr128 = lr_readout(Xs128, ys, RAND_N)
    mlp512 = mlp_readout(Xs512, ys, RAND_N); mlp128 = mlp_readout(Xs128, ys, RAND_N)
    print(f"{'feature':>14} {'LogReg':>8} {'MLP':>8}")
    print(f"{'512-D preproj':>14} {lr512:>8.3f} {mlp512:>8.3f}")
    print(f"{'128-D proj':>14} {lr128:>8.3f} {mlp128:>8.3f}")

    # ── burst-level (N=10) linear separability on 512-D ──
    bp512, bl512 = burst_pool(f512, dids, BURST_N, renorm=False)
    lr_burst512 = lr_readout(bp512, bl512, RAND_N)
    print(f"\nburst-level (N={BURST_N}) LogReg on 512-D: {lr_burst512:.3f}  ({len(bl512)} bursts)")

    # ── discovery: burst-mean HDBSCAN, 512-D vs 128-D ──
    print(f"\n=== BURST-MEAN DISCOVERY (N={BURST_N}, HDBSCAN mcs={MCS}) — 512-D vs 128-D ===")
    d128 = cluster(*burst_pool(f128, dids, BURST_N, renorm=True), RAND_N)
    d512n = cluster(*burst_pool(f512, dids, BURST_N, renorm=True), RAND_N)
    d512r = cluster(*burst_pool(f512, dids, BURST_N, renorm=False), RAND_N)
    print(f"{'features':>18} {'ARI':>7} {'NMI':>7} {'purity':>7} {'K_est':>6} {'noise':>6} {'n_pts':>6}")
    for name, d in [("128-D renorm (bar)", d128), ("512-D L2norm", d512n), ("512-D raw", d512r)]:
        print(f"{name:>18} {d['ARI']:>7.3f} {d['NMI']:>7.3f} {d['purity']:>7.3f} "
              f"{d['K_est']:>6} {d['noise']*100:>5.0f}% {d['n_pts']:>6}")

    # ── interpretation ──
    best512 = max(lr512, mlp512); best128 = max(lr128, mlp128)
    best_disc512 = max(d512n["ARI"], d512r["ARI"])
    print(f"\n=== INTERPRETATION ===")
    print(f"  readout: 512-D best={best512:.3f}  128-D best={best128:.3f}  (delta {best512-best128:+.3f})")
    print(f"  discovery ARI: 128-D={d128['ARI']:.3f}  512-D best={best_disc512:.3f}  (bar 0.54)")
    parts = []
    if best512 > 0.80:
        parts.append(f"Diagnosis A (CHEAP): frozen 512-D readout separates the UNSEEN 18 at {best512:.3f}>0.80 "
                     f"-> open-set-generalizable fingerprint IS in the representation.")
        if best512 - best128 > 0.05:
            parts.append(f"PROJECTION HEAD destroys structure: 512-D {best512:.3f} >> 128-D {best128:.3f} "
                         f"(-{best512-best128:.3f}). CHEAPEST fix = cluster/retrain on pre-proj features.")
        else:
            parts.append(f"128-D retains most of it ({best128:.3f}); projection isn't the main loss "
                         f"-> capacity/embedding-dim, not head-destruction.")
    elif best512 < 0.55:
        parts.append(f"Diagnosis B (HARD): even frozen 512-D readout caps at {best512:.3f} -> features do NOT "
                     f"generalize open-set. Needs a different training signal, not just capacity.")
    else:
        parts.append(f"MIXED: 512-D readout {best512:.3f} — partial open-set generalization; capacity helps "
                     f"but won't fully close it.")
    if best_disc512 > d128["ARI"] + 0.03:
        parts.append(f"NEAR-FREE WIN: clustering 512-D lifts discovery ARI {d128['ARI']:.3f}->{best_disc512:.3f} "
                     f"without retraining.")
    elif best_disc512 <= d128["ARI"] + 0.03 and best512 - best128 > 0.05:
        parts.append(f"BUT unsupervised clustering of 512-D does NOT auto-improve ARI ({best_disc512:.3f}) even "
                     f"though a SUPERVISED readout does -> the generalizable info needs a LEARNED projection to "
                     f"become cluster-friendly (cheap retrain of head), not raw 512-D clustering.")
    verdict = " ".join(parts)
    print(f"  VERDICT: {verdict}")

    with open(REPORT, "w") as f:
        json.dump({"config": {"win_budget": WIN_BUDGET, "burst_n": BURST_N, "mcs": MCS,
                              "readout_per_dev": READOUT_PER_DEV, "rand_seed": RAND_SEED},
                   "random_slice": rand_tx,
                   "readout": {"lr_512": lr512, "lr_128": lr128, "mlp_512": mlp512, "mlp_128": mlp128,
                               "burst512_lr": lr_burst512},
                   "discovery": {"d128_renorm": d128, "d512_l2norm": d512n, "d512_raw": d512r},
                   "verdict": verdict}, f, indent=2)
    print(f"\nsaved -> {REPORT}\nCHECKPOINT — encoder FROZEN, fresh readouts only. best_model.pt UNTOUCHED. No drones.")


if __name__ == "__main__":
    main()
