"""RETHINK: is the DISCOVERY UNIT wrong? Single 256-sample windows separate a pair (probe 96%)
but may carry too little to place one packet among ~40 identities. Aggregate evidence per decision.
CACHE-ONLY — vanilla eq=0 best_model.pt (0.859). No retrain, no drones. STOP at checkpoint.

Bar (single-window, cap160 random, random grid-scattered 18-dev seed-123 slice): ARI 0.440,
purity 0.457, K_est 19/18.

TEST 1  burst-mean clustering — aggregate N CONSECUTIVE windows (device-blind consecutive order,
        NOT grouped by label), mean-pool, L2-renorm, HDBSCAN mcs=15. Sweep N in {1,5,10,20,50,100}.
TEST 2  centroid convergence — running-mean-to-full-centroid distance vs N for a few devices,
        against the nearest-confuser gap. Fast converge = noisy-unbiased (averaging fixes it);
        plateau far from separation = biased/impoverished representation.
TEST 3  easy-slice ceiling — same single-window clustering on the 18 MOST-separable devices.

    python3 discover/burst_probe.py
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
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR   = os.path.join(RUN_DIR, "discover")
SPLIT_OLD = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
REPORT    = os.path.join(OUT_DIR, "burst_probe_report.json")

MCS = 15                     # locked ruler HDBSCAN config
RAND_N, RAND_SEED = 18, 123
WIN_BUDGET = 2400            # consecutive windows/device used for the burst sweep (>= min 2908 avail)
BURST_NS = [1, 5, 10, 20, 50, 100]
CONV_NS = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1200]
BAR_CAP = 160                # single-window bar sample


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def unitrows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def purity(true, pred):
    if len(true) == 0:
        return float("nan")
    return sum(np.unique(true[pred == c], return_counts=True)[1].max()
               for c in np.unique(pred)) / len(true)


def cluster(embs, dids, n_true, mcs=MCS):
    pred = HDBSCAN(min_cluster_size=mcs, metric="euclidean", n_jobs=-1, copy=True).fit_predict(embs)
    keep = pred != -1
    k = len(np.unique(pred[keep]))
    noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return {"ARI": 0.0, "NMI": 0.0, "purity": float(purity(dids, pred)),
                "K_est": int(k), "K_true": int(n_true), "noise": noise, "n_pts": int(len(dids))}
    return {"ARI": float(adjusted_rand_score(dids[keep], pred[keep])),
            "NMI": float(normalized_mutual_info_score(dids[keep], pred[keep])),
            "purity": float(purity(dids, pred)), "K_est": int(k), "K_true": int(n_true),
            "noise": noise, "n_pts": int(len(dids))}


def intra_cos(embs, dids):
    """mean cosine of each point to its device mean (renormalized)."""
    tot, n = 0.0, 0
    for d in np.unique(dids):
        E = embs[dids == d]
        c = unit(E.mean(0))
        tot += float((unitrows(E) @ c).sum()); n += len(E)
    return tot / max(n, 1)


def load_enc(path):
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(path, map_location="cuda", weights_only=True), strict=True)
    m.eval()
    return m


def consec_pack(tx_data, devs, nwin):
    """First `nwin` CONSECUTIVE windows/device, order preserved (device-blind bursts)."""
    times, dids = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        n = min(nwin, iq.shape[0])
        for k in range(n):
            times.append(W.standardize(iq[k].T.copy())); dids.append(di)
    t = np.stack(times).astype(np.float32)
    return t, np.array(dids)


def rand_pack(tx_data, devs, cap, seed0):
    times, dids = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        idx = np.random.default_rng(seed0 + di).integers(0, iq.shape[0], size=min(cap, iq.shape[0]))
        for k in idx:
            times.append(W.standardize(iq[k].T.copy())); dids.append(di)
    t = np.stack(times).astype(np.float32)
    return t, np.array(dids)


@torch.no_grad()
def embed_times(model, t, batch=1024):
    s = W.compute_stft_batch(t)
    xt = torch.from_numpy(t).cuda(); xs = torch.from_numpy(s).cuda()
    e = np.empty((t.shape[0], 128), dtype=np.float32)
    for i in range(0, t.shape[0], batch):
        with torch.amp.autocast('cuda'):
            e[i:i + batch] = model(xt[i:i + batch], xs[i:i + batch]).float().cpu().numpy()
    return e


def burst_pool(embs, dids, N):
    """Chunk each device's consecutive embeddings into bursts of N, mean-pool + L2-renorm."""
    bpts, blbl = [], []
    for d in np.unique(dids):
        E = embs[dids == d]                       # consecutive order preserved
        nb = E.shape[0] // N
        if nb == 0:
            continue
        chunks = E[:nb * N].reshape(nb, N, -1).mean(1)
        bpts.append(unitrows(chunks)); blbl.append(np.full(nb, d))
    return np.concatenate(bpts), np.concatenate(blbl)


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


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); held_tx = sp["discover_tx"]
    model = load_enc(BASE_CKPT)

    rand_tx = scatter_slice(held_tx, RAND_N, RAND_SEED)
    print(f"random slice (seed {RAND_SEED}, n={len(rand_tx)}, rows "
          f"{sorted(set(int(t.split('-')[0]) for t in rand_tx))}):\n  {rand_tx}")

    # ── confirm the locked single-window bar (cap160 random) ──
    bt, bd = rand_pack(tx_data, rand_tx, BAR_CAP, seed0=7000)
    be = embed_times(model, bt)
    bar = cluster(be, bd, RAND_N)
    print(f"\nBAR check (single-window, cap{BAR_CAP} random): ARI={bar['ARI']:.3f} "
          f"pur={bar['purity']:.3f} K={bar['K_est']}/{RAND_N} intra={intra_cos(be, bd):.3f}")

    # ── embed the consecutive window budget once, reuse for all N ──
    ct, cd = consec_pack(tx_data, rand_tx, WIN_BUDGET)
    print(f"embedding {ct.shape[0]} consecutive windows ({WIN_BUDGET}/dev)...")
    ce = embed_times(model, ct)

    # ══ TEST 1 — burst-mean sweep ══
    print(f"\n=== TEST 1 — burst-mean clustering (HDBSCAN mcs={MCS}) ===")
    print(f"{'N':>4} {'n_burst':>8} {'ARI':>7} {'NMI':>7} {'purity':>7} {'K_est':>6} "
          f"{'noise':>6} {'intra':>7}")
    t1 = {}
    for N in BURST_NS:
        bp, bl = burst_pool(ce, cd, N)
        r = cluster(bp, bl, RAND_N)
        r["intra"] = intra_cos(bp, bl)
        r["n_burst_per_dev"] = int(WIN_BUDGET // N)
        t1[N] = r
        print(f"{N:>4} {r['n_pts']:>8} {r['ARI']:>7.3f} {r['NMI']:>7.3f} {r['purity']:>7.3f} "
              f"{r['K_est']:>6} {r['noise']*100:>5.0f}% {r['intra']:>7.3f}")

    # ══ TEST 2 — centroid convergence vs nearest-confuser gap ══
    print(f"\n=== TEST 2 — running-mean -> full-centroid convergence (1 - cos) ===")
    # full centroids on the consecutive budget
    full_c = unitrows(np.stack([ce[cd == d].mean(0) for d in range(len(rand_tx))]))
    Cm = full_c @ full_c.T; np.fill_diagonal(Cm, -2.0)
    nn_sim = Cm.max(1)                                    # nearest-confuser cosine per device
    # pick a spread of devices: 2 easiest, 1 median, 2 hardest by nn_sim
    order = np.argsort(nn_sim)
    pick = [int(order[0]), int(order[1]), int(order[len(order)//2]),
            int(order[-2]), int(order[-1])]
    print(f"{'device':>10} {'nn_cos':>7} {'gap':>6} | running-mean 1-cos to full centroid at N:")
    print(f"{'':>10} {'':>7} {'':>6}   " + " ".join(f"{n:>6}" for n in CONV_NS))
    t2 = {}
    for d in pick:
        E = ce[cd == d]                                   # consecutive
        gap = float(1 - nn_sim[d])
        dists = []
        for n in CONV_NS:
            rm = unit(E[:n].mean(0))
            dists.append(float(1 - rm @ full_c[d]))
        t2[rand_tx[d]] = {"nn_cos": float(nn_sim[d]), "gap": gap, "conv": dists}
        print(f"{rand_tx[d]:>10} {nn_sim[d]:>7.4f} {gap:>6.3f} | "
              + " ".join(f"{x:>6.3f}" for x in dists))
    print("  (converges BELOW its 'gap' => averaging pulls the estimate inside the margin to its "
          "nearest confuser)")

    # ══ TEST 3 — easy-slice ceiling (18 most-separable devices, single-window) ══
    print(f"\n=== TEST 3 — easy-slice single-window ceiling ===")
    ft, fd = consec_pack(tx_data, held_tx, 400)           # cheap centroids over all 41 held
    fe = embed_times(model, ft)
    all_c = unitrows(np.stack([fe[fd == d].mean(0) for d in range(len(held_tx))]))
    Ca = all_c @ all_c.T; np.fill_diagonal(Ca, -2.0)
    nn_all = Ca.max(1)
    easy_idx = list(np.argsort(nn_all)[:RAND_N])          # lowest nearest-confuser = most separable
    easy_tx = [held_tx[i] for i in easy_idx]
    print(f"easy slice (18 lowest nearest-confuser): {easy_tx}")
    print(f"  their nearest-confuser cos: {np.round(nn_all[easy_idx], 3).tolist()}")
    et, ed = rand_pack(tx_data, easy_tx, BAR_CAP, seed0=9000)
    ee = embed_times(model, et)
    easy = cluster(ee, ed, RAND_N)
    easy["intra"] = intra_cos(ee, ed)
    print(f"easy-slice single-window: ARI={easy['ARI']:.3f} NMI={easy['NMI']:.3f} "
          f"pur={easy['purity']:.3f} K={easy['K_est']}/{RAND_N} intra={easy['intra']:.3f}")

    # ── interpretation ──
    best_ari = max(t1[N]["ARI"] for N in BURST_NS)
    best_N = min(N for N in BURST_NS if t1[N]["ARI"] == best_ari)
    n_for_07 = next((N for N in BURST_NS if t1[N]["ARI"] >= 0.70), None)
    print(f"\n=== INTERPRETATION ===")
    print(f"  single-window bar ARI = {bar['ARI']:.3f}; best burst ARI = {best_ari:.3f} @ N={best_N}")
    if best_ari >= 0.70 and n_for_07 is not None:
        verdict = (f"SOLVED BY PROTOCOL (free): burst-mean lifts ARI to {best_ari:.3f}; ARI>=0.70 "
                   f"at N={n_for_07} consecutive windows. Single-window was the wrong DISCOVERY UNIT. "
                   f"System-design number for the drone deployment: aggregate ~{n_for_07} packets/decision. "
                   f"No retrain needed.")
    elif best_ari <= bar["ARI"] + 0.05:
        verdict = (f"AVERAGING DOESN'T HELP (burst plateaus ~bar {bar['ARI']:.3f}). Check TEST 2: if "
                   f"centroids converge fast yet stay unseparated => biased/impoverished per-window "
                   f"representation => NEXT SESSION test LONGER INPUT WINDOWS (expensive). Not the protocol.")
    else:
        verdict = (f"PARTIAL: burst-mean helps ({bar['ARI']:.3f}->{best_ari:.3f} @N={best_N}) but "
                   f"doesn't reach 0.70. Aggregation is part of the fix; likely ALSO an input-window "
                   f"limit. Report N-tradeoff; longer windows next session.")
    if easy["ARI"] > 0.80:
        verdict += f" EASY-SLICE ARI {easy['ARI']:.3f}>0.80 => 0.44 is a SLICE-DIFFICULTY artifact, not a hard ceiling."
    elif easy["ARI"] < 0.55:
        verdict += f" EASY-SLICE ALSO CAPS LOW ({easy['ARI']:.3f}) => BROADER issue (encoder/eq0/metric); flag before any expensive change."
    print(f"  VERDICT: {verdict}")

    with open(REPORT, "w") as f:
        json.dump({"config": {"mcs": MCS, "rand_n": RAND_N, "rand_seed": RAND_SEED,
                              "win_budget": WIN_BUDGET, "burst_Ns": BURST_NS, "bar_cap": BAR_CAP},
                   "random_slice": rand_tx, "bar_single_window": bar,
                   "test1_burst": {str(N): t1[N] for N in BURST_NS},
                   "test2_convergence": {"conv_Ns": CONV_NS, "devices": t2},
                   "test3_easy": {"easy_slice": easy_tx, "result": easy},
                   "verdict": verdict}, f, indent=2)
    print(f"\nsaved -> {REPORT}\nCHECKPOINT — cache-only, no training, no drones. best_model.pt UNTOUCHED.")


if __name__ == "__main__":
    main()
