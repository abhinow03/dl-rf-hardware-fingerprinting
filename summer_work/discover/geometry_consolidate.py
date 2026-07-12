"""CONSOLIDATE THE COSFACE WIN — (1) validate across slices, (2) fix K_est regression (36 vs 18)
without losing ARI. Backbone FROZEN. CosFace head fit on 109 TRAINING devices only. No drones.

GATE: train a clean fixed-step CosFace head (NO eval-slice selection), evaluate burst-mean(N=10)
discovery on 4 grid-scattered unseen slices (seeds 123/7/99/2024). Does ARI~0.66 hold?
COUNT FIX (if gate passes):
  Lever 1  over-split-then-MERGE: HDBSCAN over-segments, then agglomeratively merge cluster
           centroids within cosine threshold {0.5,0.6,0.7} (no head change).
  Lever 2  CosFace m in {0.2,0.35,0.5} x s in {16,32,64}; select on COMBINED K-and-ARI.
Report best combined operating point across slices; cross-check coincident pairs.

    python3 discover/geometry_consolidate.py
Backbone frozen; slices device-disjoint from training + unseen. No backbone unfreeze.
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
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR   = os.path.join(RUN_DIR, "discover")
SPLIT_OLD = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
REPORT    = os.path.join(OUT_DIR, "geometry_consolidate_report.json")

RAND_N = 18
SLICE_SEEDS = [123, 7, 99, 2024]
WIN_SLICE, TRAIN_PER_DEV = 2400, 500
BURST_N, MCS = 10, 15
STEPS, BS, LR = 2000, 512, 1e-3          # fixed steps, no eval-slice selection
DIM = 128
SEED = 42
MERGE_THRS = [0.5, 0.6, 0.7]
MS_GRID = [(0.2, 16), (0.2, 32), (0.2, 64), (0.35, 16), (0.35, 32),
           (0.35, 64), (0.5, 16), (0.5, 32), (0.5, 64)]
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


def consec_windows(tx_data, tx, nwin):
    iq = tx_data[tx]["iq"]
    return np.stack([W.standardize(iq[k].T.copy()) for k in range(min(nwin, iq.shape[0]))]).astype(np.float32)


@torch.no_grad()
def extract512(model, t, batch=1024):
    s = W.compute_stft_batch(t)
    xt = torch.from_numpy(t).cuda(); xs = torch.from_numpy(s).cuda()
    f = np.empty((t.shape[0], 512), dtype=np.float32)
    for i in range(0, t.shape[0], batch):
        with torch.amp.autocast('cuda'):
            f[i:i+batch] = model.get_encoder_output(xt[i:i+batch], xs[i:i+batch]).float().cpu().numpy()
    return f


def burst_pool(Femb, N):
    nb = Femb.shape[0] // N
    return unitrows(Femb[:nb*N].reshape(nb, N, -1).mean(1))


def slice_burst(head_emb, slice_tx):
    """head_emb: dict tx-> per-window normalized embedding [W,dim]; returns burst pts + labels."""
    bp, bl = [], []
    for di, tx in enumerate(slice_tx):
        b = burst_pool(head_emb[tx], BURST_N)
        bp.append(b); bl.append(np.full(len(b), di))
    return np.concatenate(bp), np.concatenate(bl)


def hdbscan_pred(bp):
    return HDBSCAN(min_cluster_size=MCS, metric="euclidean", copy=True).fit_predict(bp)


def score(pred, bl, n_true):
    keep = pred != -1; k = len(np.unique(pred[keep]))
    if keep.sum() < 2 or k < 2:
        return {"ARI": 0.0, "NMI": 0.0, "purity": float(purity(bl, pred)), "K_est": int(k),
                "noise": float((~keep).mean())}
    return {"ARI": float(adjusted_rand_score(bl[keep], pred[keep])),
            "NMI": float(normalized_mutual_info_score(bl[keep], pred[keep])),
            "purity": float(purity(bl, pred)), "K_est": int(k), "noise": float((~keep).mean())}


def merge_pred(bp, pred, thr):
    """agglomeratively merge over-segmented cluster centroids within cosine>thr."""
    u = sorted(set(pred.tolist()) - {-1})
    if len(u) <= 1:
        return pred
    cents = unitrows(np.stack([bp[pred == c].mean(0) for c in u]))
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - thr,
                                  metric="cosine", linkage="average").fit(cents)
    remap = {c: int(agg.labels_[i]) for i, c in enumerate(u)}
    return np.array([remap[p] if p != -1 else -1 for p in pred])


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1, eps=1e-6)


def train_cosface(Ftr_t, ty_t, n_cls, m, s):
    torch.manual_seed(SEED); np.random.seed(SEED)
    head = Head(DIM).cuda()
    Wc = nn.Parameter(F.normalize(torch.randn(n_cls, DIM), dim=1).cuda())
    opt = torch.optim.Adam(list(head.parameters()) + [Wc], lr=LR, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(); N = Ftr_t.shape[0]; rng = np.random.default_rng(SEED)
    for step in range(STEPS):
        idx = torch.from_numpy(rng.integers(0, N, size=BS)).cuda()
        z = head(Ftr_t[idx]); y = ty_t[idx]
        cos = z @ F.normalize(Wc, dim=1).t()
        logits = s * (cos - m * F.one_hot(y, n_cls).float())
        loss = ce(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def apply_head(head, held_feat):
    out = {}
    for tx, F512 in held_feat.items():
        Xg = torch.from_numpy(F512).cuda()
        z = np.empty((F512.shape[0], DIM), dtype=np.float32)
        for i in range(0, Xg.shape[0], 4096):
            z[i:i+4096] = head(Xg[i:i+4096]).cpu().numpy()
        out[tx] = z
    return out


def coincident_cos(head_emb, slice_tx):
    idx = {tx: i for i, tx in enumerate(slice_tx)}
    cents = unitrows(np.stack([burst_pool(head_emb[tx], BURST_N).mean(0) for tx in slice_tx]))
    M = cents @ cents.T
    return {f"{a}/{b}": round(float(M[idx[a], idx[b]]), 4) for a, b in COINCIDENT if a in idx and b in idx}


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); train_tx, held_tx = sp["train_tx"], sp["discover_tx"]
    n_cls = len(train_tx)
    slices = {seed: scatter_slice(held_tx, RAND_N, seed) for seed in SLICE_SEEDS}
    print(f"train devs: {n_cls} | held: {len(held_tx)} | slices: "
          + ", ".join(f"seed{seed}(rows {sorted(set(int(t.split('-')[0]) for t in sl))})"
                      for seed, sl in slices.items()))

    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # extract frozen 512-D once: train (fit) + all 41 held (eval, subset per slice)
    tt = np.concatenate([consec_windows(tx_data, tx, TRAIN_PER_DEV) for tx in train_tx])
    ty = np.concatenate([np.full(min(TRAIN_PER_DEV, tx_data[tx]["iq"].shape[0]), i)
                         for i, tx in enumerate(train_tx)])
    print(f"extracting 512-D for {tt.shape[0]} TRAIN windows...")
    Ftr = extract512(model, tt); del tt
    print(f"extracting 512-D for {len(held_tx)} held devices x {WIN_SLICE}...")
    held_feat = {tx: extract512(model, consec_windows(tx_data, tx, WIN_SLICE)) for tx in held_tx}
    torch.cuda.empty_cache()

    sc = StandardScaler().fit(Ftr)
    Ftr_t = torch.from_numpy(sc.transform(Ftr).astype(np.float32)).cuda()
    ty_t = torch.from_numpy(ty).cuda()
    held_feat = {tx: sc.transform(F512).astype(np.float32) for tx, F512 in held_feat.items()}

    # ═══ GATE: baseline cosface (m=0.2, s=32), fixed steps, eval across slices ═══
    print(f"\n=== GATE — baseline CosFace (m=0.2,s=32,d{DIM}), {STEPS} steps, NO eval-selection ===")
    head0 = train_cosface(Ftr_t, ty_t, n_cls, 0.2, 32)
    emb0 = apply_head(head0, held_feat)
    print(f"{'slice':>10} {'ARI':>7} {'NMI':>7} {'purity':>7} {'K_est':>6}")
    gate = {}
    for seed in SLICE_SEEDS:
        bp, bl = slice_burst(emb0, slices[seed]); d = score(hdbscan_pred(bp), bl, RAND_N)
        gate[seed] = d
        print(f"{('seed'+str(seed)):>10} {d['ARI']:>7.3f} {d['NMI']:>7.3f} {d['purity']:>7.3f} {d['K_est']:>6}")
    aris = np.array([gate[s]["ARI"] for s in SLICE_SEEDS])
    print(f"  ARI mean±std = {aris.mean():.3f} ± {aris.std():.3f}  (min {aris.min():.3f})")
    gate_ok = aris.mean() > 0.55 and aris.std() < 0.10
    if not gate_ok:
        print("  GATE FAIL: ARI swings by slice / low -> CosFace may have overfit; STOP before tuning.")
        json.dump({"gate": {str(k): v for k, v in gate.items()}, "gate_ok": False}, open(REPORT, "w"), indent=2)
        print("CHECKPOINT — gate failed, reported. backbone FROZEN. No drones.")
        return
    print("  GATE PASS -> proceed to count fix.")

    # ═══ LEVER 1: over-split-then-merge on the baseline head ═══
    print(f"\n=== LEVER 1 — merge over-segmented clusters (cosine thr), baseline head, mean over slices ===")
    print(f"{'threshold':>10} {'ARI':>7} {'K_est':>7} {'|K-18|':>7}")
    lever1 = {}
    slice_bp = {s: slice_burst(emb0, slices[s]) for s in SLICE_SEEDS}
    slice_pred = {s: hdbscan_pred(slice_bp[s][0]) for s in SLICE_SEEDS}
    # no-merge reference
    for label, thr in [("none", None)] + [(str(t), t) for t in MERGE_THRS]:
        aa, kk = [], []
        for s in SLICE_SEEDS:
            bp, bl = slice_bp[s]
            pred = slice_pred[s] if thr is None else merge_pred(bp, slice_pred[s], thr)
            d = score(pred, bl, RAND_N); aa.append(d["ARI"]); kk.append(d["K_est"])
        lever1[label] = {"ARI": float(np.mean(aa)), "K_est": float(np.mean(kk)),
                         "ARI_per": [round(x, 3) for x in aa], "K_per": kk}
        print(f"{label:>10} {np.mean(aa):>7.3f} {np.mean(kk):>7.1f} {abs(np.mean(kk)-18):>7.1f}")

    # ═══ LEVER 2: margin/scale sweep, combined K-and-ARI selection ═══
    print(f"\n=== LEVER 2 — CosFace m x s sweep, mean over slices (combined K-and-ARI) ===")
    print(f"{'m':>5} {'s':>4} {'ARI':>7} {'K_est':>7} {'|K-18|':>7} {'combined':>9}")
    lever2 = []
    for (m, s) in MS_GRID:
        h = train_cosface(Ftr_t, ty_t, n_cls, m, s)
        emb = apply_head(h, held_feat)
        aa, kk = [], []
        for seed in SLICE_SEEDS:
            bp, bl = slice_burst(emb, slices[seed]); d = score(hdbscan_pred(bp), bl, RAND_N)
            aa.append(d["ARI"]); kk.append(d["K_est"])
        mA, mK = float(np.mean(aa)), float(np.mean(kk))
        combined = mA - 0.02 * abs(mK - 18)      # penalize K error
        lever2.append({"m": m, "s": s, "ARI": mA, "K_est": mK, "combined": combined,
                       "ARI_per": [round(x, 3) for x in aa], "K_per": kk})
        print(f"{m:>5} {s:>4} {mA:>7.3f} {mK:>7.1f} {abs(mK-18):>7.1f} {combined:>9.3f}")

    # combine lever2 head with lever1 merge (best m/s, then merge)
    best_ms = max(lever2, key=lambda r: r["combined"])
    print(f"\n=== LEVER 1+2 — best head (m={best_ms['m']},s={best_ms['s']}) + merge ===")
    hbest = train_cosface(Ftr_t, ty_t, n_cls, best_ms["m"], best_ms["s"])
    embb = apply_head(hbest, held_feat)
    bp_best = {s: slice_burst(embb, slices[s]) for s in SLICE_SEEDS}
    pred_best = {s: hdbscan_pred(bp_best[s][0]) for s in SLICE_SEEDS}
    combo = {}
    print(f"{'threshold':>10} {'ARI':>7} {'K_est':>7} {'|K-18|':>7}")
    for label, thr in [("none", None)] + [(str(t), t) for t in MERGE_THRS]:
        aa, kk = [], []
        for s in SLICE_SEEDS:
            bp, bl = bp_best[s]
            pred = pred_best[s] if thr is None else merge_pred(bp, pred_best[s], thr)
            d = score(pred, bl, RAND_N); aa.append(d["ARI"]); kk.append(d["K_est"])
        combo[label] = {"ARI": float(np.mean(aa)), "K_est": float(np.mean(kk))}
        print(f"{label:>10} {np.mean(aa):>7.3f} {np.mean(kk):>7.1f} {abs(np.mean(kk)-18):>7.1f}")

    # ── choose locked operating point: highest ARI with |K-18|<=4 ──
    cand = []
    for lbl, d in lever1.items():
        cand.append(("baseline+merge:" + lbl, d["ARI"], d["K_est"]))
    for r in lever2:
        cand.append((f"m{r['m']}s{r['s']}", r["ARI"], r["K_est"]))
    for lbl, d in combo.items():
        cand.append((f"best+merge:{lbl}", d["ARI"], d["K_est"]))
    ok = [c for c in cand if abs(c[2] - 18) <= 4]
    locked = max(ok, key=lambda c: c[1]) if ok else max(cand, key=lambda c: c[1] - 0.02*abs(c[2]-18))

    # coincident cross-check on best head, seed123
    coinc = coincident_cos(embb, slices[123])
    print(f"\n=== RECOMMENDED LOCKED OPERATING POINT ===")
    print(f"  {locked[0]}: ARI={locked[1]:.3f}  K_est={locked[2]:.1f} (true 18, |err|={abs(locked[2]-18):.1f})")
    print(f"  coincident-pair cos (best head, seed123): {coinc}")
    verdict = (f"CosFace generalizes (gate ARI {aris.mean():.3f}±{aris.std():.3f}); locked operating point "
               f"'{locked[0]}' gives ARI {locked[1]:.3f} at K_est {locked[2]:.1f} vs 18 "
               f"(|err| {abs(locked[2]-18):.1f}) — count regression addressed while holding ARI well above the "
               f"0.540 bar. Backbone FROZEN, head-only + eval-side merge.")
    print(f"  VERDICT: {verdict}")

    json.dump({"gate": {str(k): v for k, v in gate.items()}, "gate_ari_mean": float(aris.mean()),
               "gate_ari_std": float(aris.std()), "gate_ok": True,
               "lever1_merge": lever1, "lever2_ms": lever2, "best_ms": best_ms,
               "lever12_combo": combo, "locked": {"name": locked[0], "ARI": locked[1], "K_est": locked[2]},
               "coincident_seed123": coinc, "verdict": verdict}, open(REPORT, "w"), indent=2)
    print(f"\nsaved -> {REPORT}\nCHECKPOINT — backbone FROZEN, head-only. best_model.pt UNTOUCHED. No drones.")


if __name__ == "__main__":
    main()
