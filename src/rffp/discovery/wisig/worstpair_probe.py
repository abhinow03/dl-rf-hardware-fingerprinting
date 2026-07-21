"""Worst-pair separability probe — does the hardware fingerprint SIGNAL exist for the
collapsed pairs, before committing a full retrain?  Cache + tiny fresh probes only.

NO full retrain, NO touching best_model.pt, eq=0, M100/DRFF/drive untouched.

  STEP 1  worst pairs   — highest centroid-cosine pairs among the 41 held-out nodes.
  STEP 2  probe A       — logistic regression on the CACHED 128-D embeddings (residual
                          linear signal already in the frozen encoder; expected low).
  STEP 3  probe B       — a SMALL fresh CNN on RAW IQ [2,256] (eq=0) for the ~3 worst
                          pairs. The decisive test: is the signal in the INPUT at all?
  STEP 4  control       — probe B on one EASY (well-separated) pair: sanity that low
                          accuracy on hard pairs means "no signal," not "broken probe".

Interpretation:
  probe B >> chance on worst pairs  -> signal IS in the input -> margin/hard-neg retrain warranted.
  probe B ~ chance on worst pairs   -> signal NOT recoverable at this window/preproc -> re-scope.

    python3 discover/worstpair_probe.py
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rffp.data import wisig_manytx as W

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
EMB_CACHE = os.path.join(RUN_DIR, "discover", "discover_embeddings.npz")
OUT_JSON  = os.path.join(RUN_DIR, "discover", "worstpair_probe_report.json")

SEED      = 42
N_WORST   = 10        # worst pairs to list
N_RAW     = 3         # worst pairs to run the raw-IQ CNN on
DEV       = "cuda" if torch.cuda.is_available() else "cpu"


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


# ───────────────────────── STEP 1 — worst pairs ─────────────────────────
def worst_pairs(emb, true, n_tx, k):
    cents = np.stack([unit(emb[true == i].mean(0)) for i in range(n_tx)])
    C = cents @ cents.T
    iu, ju = np.triu_indices(n_tx, k=1)
    cos = C[iu, ju]
    order = np.argsort(cos)[::-1]
    worst = [(int(iu[o]), int(ju[o]), float(cos[o])) for o in order[:k]]
    easy = [(int(iu[o]), int(ju[o]), float(cos[o])) for o in order[::-1][:k]]
    return worst, easy, C


# ───────────────────────── STEP 2 — probe A (linear, cached emb) ─────────────────────────
def probe_A(emb, true, i, j, seed=SEED):
    """Balanced logistic-regression test accuracy separating cached embeddings of tx i vs j."""
    ai = np.where(true == i)[0]; aj = np.where(true == j)[0]
    rng = np.random.default_rng(seed)
    m = min(len(ai), len(aj))
    ai = rng.permutation(ai)[:m]; aj = rng.permutation(aj)[:m]      # balance -> chance 0.5
    X = np.concatenate([emb[ai], emb[aj]]); y = np.r_[np.zeros(m), np.ones(m)]
    cut = int(0.7 * m)
    tr = np.r_[np.arange(cut), np.arange(m, m + cut)]
    te = np.r_[np.arange(cut, m), np.arange(m + cut, 2 * m)]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(X[tr]), y[tr])
    return float(clf.score(sc.transform(X[te]), y[te])), m


# ───────────────────────── STEP 3/4 — probe B (raw-IQ CNN) ─────────────────────────
class TinyCNN(nn.Module):
    """Small 1-D CNN on standardized raw IQ [2,256] -> 2-class logit. GroupNorm (locked)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3),  nn.GroupNorm(8, 32),  nn.ReLU(),  nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.GroupNorm(8, 64),  nn.ReLU(),  nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),nn.GroupNorm(8, 128), nn.ReLU(),  nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        return self.fc(self.net(x).squeeze(-1))


def build_raw(tx_data, tx_id):
    """All signals for one tx -> standardized [n,2,256] (same preproc the encoder sees)."""
    iq = tx_data[tx_id]["iq"]                                  # [n,256,2]
    return np.stack([W.standardize(iq[k].T.copy()) for k in range(iq.shape[0])]).astype(np.float32)


def probe_B(tx_data, tx_a, tx_b, seed=SEED, epochs=40, bs=64, lr=1e-3):
    """Fresh TinyCNN telling tx_a from tx_b on raw IQ. Device-balanced split. Returns metrics."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xa, Xb = build_raw(tx_data, tx_a), build_raw(tx_data, tx_b)
    rng = np.random.default_rng(seed)
    m = min(len(Xa), len(Xb))
    Xa = Xa[rng.permutation(len(Xa))[:m]]; Xb = Xb[rng.permutation(len(Xb))[:m]]   # balance
    cut = int(0.7 * m)
    Xtr = np.concatenate([Xa[:cut], Xb[:cut]]); ytr = np.r_[np.zeros(cut), np.ones(cut)].astype(np.int64)
    Xte = np.concatenate([Xa[cut:], Xb[cut:]]); yte = np.r_[np.zeros(m - cut), np.ones(m - cut)].astype(np.int64)

    Xtr_t = torch.from_numpy(Xtr).to(DEV); ytr_t = torch.from_numpy(ytr).to(DEV)
    Xte_t = torch.from_numpy(Xte).to(DEV); yte_t = torch.from_numpy(yte).to(DEV)

    net = TinyCNN().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    best, ep_70, curve = 0.0, None, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xtr_t), device=DEV)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            lossf(net(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(Xte_t).argmax(1) == yte_t).float().mean())
        curve.append(round(acc, 3))
        best = max(best, acc)
        if ep_70 is None and acc >= 0.70:
            ep_70 = ep
    return {"test_acc_best": round(best, 3), "test_acc_final": round(curve[-1], 3),
            "epoch_reach_0.70": ep_70, "n_per_class": int(m), "curve": curve}


# ─────────────────────────────── main ───────────────────────────────
def main():
    z = np.load(EMB_CACHE)
    emb, true, tx = z["emb"].astype(np.float32), z["true"], z["tx"]
    n_tx = len(tx)
    print(f"cache: {len(emb)} signals, {n_tx} held-out nodes  | device={DEV}")

    # STEP 1
    worst, easy, C = worst_pairs(emb, true, n_tx, N_WORST)
    print(f"\n=== STEP 1 — {N_WORST} WORST confuser pairs (highest centroid cosine) ===")
    print(f"{'rank':>4}  {'pair':>13}  {'centroid_cos':>12}")
    for r, (i, j, c) in enumerate(worst):
        print(f"{r+1:>4}  {tx[i]+'/'+tx[j]:>13}  {c:>12.4f}")
    print(f"\n  easiest pair (for control): {tx[easy[0][0]]}/{tx[easy[0][1]]}  cos={easy[0][2]:+.4f}")

    # STEP 2 — probe A on ALL worst pairs
    print(f"\n=== STEP 2 — probe A: linear separability of CACHED 128-D embeddings ===")
    print(f"{'pair':>13}  {'centroid_cos':>12}  {'logreg_test_acc':>15}  {'n/class':>8}  (chance 0.50)")
    pA = []
    for i, j, c in worst:
        acc, m = probe_A(emb, true, i, j)
        pA.append({"pair": f"{tx[i]}/{tx[j]}", "centroid_cos": round(c, 4),
                   "logreg_acc": round(acc, 3), "n_per_class": m})
        print(f"{tx[i]+'/'+tx[j]:>13}  {c:>12.4f}  {acc:>15.3f}  {m:>8}")

    # STEP 3 — probe B on the N_RAW worst pairs (raw IQ, fresh CNN)
    print(f"\n=== STEP 3 — probe B: fresh TinyCNN on RAW IQ [2,256] (eq=0) — {N_RAW} worst pairs ===")
    tx_data, _ = W.load_manytx(eq=0)
    pB = []
    for i, j, c in worst[:N_RAW]:
        t0 = time.time()
        r = probe_B(tx_data, tx[i], tx[j])
        r.update({"pair": f"{tx[i]}/{tx[j]}", "centroid_cos": round(c, 4), "kind": "worst"})
        pB.append(r)
        print(f"  {tx[i]+'/'+tx[j]:>13} cos={c:.4f} | test_acc best={r['test_acc_best']:.3f} "
              f"final={r['test_acc_final']:.3f}  reach0.70@ep={r['epoch_reach_0.70']}  "
              f"n/class={r['n_per_class']}  ({time.time()-t0:.0f}s)", flush=True)

    # STEP 4 — control: probe B on the EASIEST pair
    print(f"\n=== STEP 4 — control: probe B on the EASIEST pair (signal clearly present) ===")
    ei, ej, ec = easy[0]
    t0 = time.time()
    rc = probe_B(tx_data, tx[ei], tx[ej])
    rc.update({"pair": f"{tx[ei]}/{tx[ej]}", "centroid_cos": round(ec, 4), "kind": "easy_control"})
    print(f"  {tx[ei]+'/'+tx[ej]:>13} cos={ec:+.4f} | test_acc best={rc['test_acc_best']:.3f} "
          f"final={rc['test_acc_final']:.3f}  reach0.70@ep={rc['epoch_reach_0.70']}  "
          f"n/class={rc['n_per_class']}  ({time.time()-t0:.0f}s)", flush=True)

    # verdict
    worst_b = np.mean([r["test_acc_best"] for r in pB])
    verdict = ("SIGNAL PRESENT in raw input (worst-pair CNN >> chance) -> hard-neg/margin RETRAIN WARRANTED"
               if worst_b >= 0.70 else
               "NO recoverable signal at this window/preproc (worst-pair CNN ~chance) -> RE-SCOPE, margin won't help")
    print(f"\n=== VERDICT ===")
    print(f"  worst-pair raw-IQ CNN mean best test acc = {worst_b:.3f}")
    print(f"  easy control best test acc               = {rc['test_acc_best']:.3f}")
    print(f"  -> {verdict}")

    out = {"n_tx": int(n_tx), "n_signals": int(len(emb)), "device": DEV,
           "worst_pairs": [{"pair": f"{tx[i]}/{tx[j]}", "centroid_cos": round(c, 4)} for i, j, c in worst],
           "easy_pair": {"pair": f"{tx[ei]}/{tx[ej]}", "centroid_cos": round(ec, 4)},
           "probe_A_linear_cached_emb": pA,
           "probe_B_rawiq_cnn": pB + [rc],
           "verdict": {"worst_mean_best_acc": round(float(worst_b), 3),
                       "easy_control_best_acc": rc["test_acc_best"],
                       "signal_present": bool(worst_b >= 0.70), "text": verdict}}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")
    print("\nCHECKPOINT — probes only, no retrain, best_model.pt untouched.")


if __name__ == "__main__":
    main()
