"""STEP-2 INTEGRITY — validate ARI 0.540 (128-D) / 0.667 (CosFace) measure HARDWARE, not
session/channel structure or metric artifacts. Backbone FROZEN. CosFace head reproduced
DETERMINISTICALLY from the locked recipe (seed 42, fixed steps == "reused as-is", no tuning/
selection). DEV devices only for Tests 2-3 (Test 1 audits the recorded seed-123 slice).

  python3 results/step2_integrity/integrity.py
"""
import os, sys, json, csv
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# reuse the EXACT locked pipelines (no drift)
import geometry_consolidate as GC          # extract512, consec_windows, train_cosface, apply_head, Head
from burst_probe import embed_times        # 128-D projection embedding

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
PART      = os.path.join(RUN_DIR, "splits", "discovery_partition.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
OUT       = _HERE

BURST_N = 10
WIN_CACHE = 2400          # matches the recorded 0.540/0.667 budget (48 rx-date cells present)
MCS_JITTER = [13, 15, 17] # the 3 "clustering seeds" = min_cluster_size jitter (HDBSCAN is deterministic)
COS_M, COS_S = 0.20, 32   # locked CosFace gate head (== recorded ~0.667); NOT tuned here


def unitrows(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def purity_std(true, pred):
    """standard purity over the SUPPORT of (true,pred) as given (each distinct pred label = a cluster)."""
    N = len(true)
    if N == 0: return float("nan")
    return sum(np.unique(true[pred == c], return_counts=True)[1].max()
               for c in np.unique(pred)) / N


def relabel_noise_as_singletons(pred):
    p = pred.copy()
    nm = p == -1
    if nm.any():
        start = (p.max() + 1) if (p >= 0).any() else 0
        p[nm] = np.arange(start, start + nm.sum())
    return p


def score_current(pred, true):
    """CURRENT code behavior: ARI/NMI exclude -1; purity over full (noise = one cluster)."""
    keep = pred != -1
    k = len(np.unique(pred[keep]))
    noise = float((~keep).mean())
    if keep.sum() < 2 or k < 2:
        return dict(ARI=0.0, NMI=0.0, purity=float(purity_std(true, pred)), K_est=int(k), noise=noise)
    return dict(ARI=float(adjusted_rand_score(true[keep], pred[keep])),
                NMI=float(normalized_mutual_info_score(true[keep], pred[keep])),
                purity=float(purity_std(true, pred)), K_est=int(k), noise=noise)


def score_locked(pred, true):
    """LOCKED rule: each -1 point = its own singleton cluster; ARI/NMI/purity over ALL points."""
    p = relabel_noise_as_singletons(pred)
    k = len(np.unique(pred[pred != -1]))          # real (non-noise) cluster count, reported as-is
    noise = float((pred == -1).mean())
    return dict(ARI=float(adjusted_rand_score(true, p)),
                NMI=float(normalized_mutual_info_score(true, p)),
                purity=float(purity_std(true, p)), K_est=int(k), noise=noise)


def hdbscan_pred(bp, mcs=15):
    return HDBSCAN(min_cluster_size=mcs, metric="euclidean", copy=True).fit_predict(bp)


# ── burst construction (coherent = consecutive; scattered = distinct rx/date cells) ──
def bursts_coherent(emb, N=BURST_N):
    nb = emb.shape[0] // N
    return unitrows(emb[:nb * N].reshape(nb, N, -1).mean(1))


def bursts_scattered(emb, rx, date, N=BURST_N, n_bursts=None, seed=0):
    """each burst = N windows from N DISTINCT (rx,date) cells (max channel decoherence)."""
    cells = {}
    for i, (r, d) in enumerate(zip(rx.tolist(), date.tolist())):
        cells.setdefault((int(r), int(d)), []).append(i)
    keys = list(cells.keys())
    rng = np.random.default_rng(seed)
    if n_bursts is None:
        n_bursts = emb.shape[0] // N
    if len(keys) < N:
        return None, len(keys)
    out = []
    for _ in range(n_bursts):
        ck = rng.choice(len(keys), size=N, replace=False)
        idx = [cells[keys[c]][rng.integers(0, len(cells[keys[c]]))] for c in ck]
        out.append(emb[idx].mean(0))
    return unitrows(np.stack(out)), len(keys)


def build_slice_bursts(cache, slice_tx, kind, N=BURST_N, seed=0):
    bp, bl = [], []
    for di, tx in enumerate(slice_tx):
        c = cache[tx]
        if kind == "coherent":
            b = bursts_coherent(c["emb"], N)
        else:
            b, _ = bursts_scattered(c["emb"], c["rx"], c["date"], N, seed=seed)
        bp.append(b); bl.append(np.full(len(b), di))
    return np.concatenate(bp), np.concatenate(bl)


def csv_write(path, rows, header):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(r)
    print(f"  saved -> {path}")


# ─────────────────────────────────────────────────────────────
def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    sp = json.load(open(SPLIT)); part = json.load(open(PART))
    train_tx, held_tx = sp["train_tx"], sp["discover_tx"]
    dev_tx = part["dev_tx"]
    n_cls = len(train_tx)

    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # ── train CosFace head deterministically (locked recipe, seed 42) ──
    print("extracting 512-D train features (109 devs) + fitting CosFace head (m=%.2f s=%d)..." % (COS_M, COS_S))
    tt = np.concatenate([GC.consec_windows(TXD, tx, GC.TRAIN_PER_DEV) for tx in train_tx])
    ty = np.concatenate([np.full(min(GC.TRAIN_PER_DEV, TXD[tx]["iq"].shape[0]), i)
                         for i, tx in enumerate(train_tx)])
    Ftr = GC.extract512(model, tt); del tt
    scaler = StandardScaler().fit(Ftr)
    Ftr_t = torch.from_numpy(scaler.transform(Ftr).astype(np.float32)).cuda()
    ty_t = torch.from_numpy(ty).cuda()
    head = GC.train_cosface(Ftr_t, ty_t, n_cls, COS_M, COS_S)   # deterministic; reused for all tests

    # ── build embedding caches for every device we will need ──
    seed123 = GC.scatter_slice(held_tx, 18, 123)
    need = sorted(set(dev_tx) | set(seed123))
    print(f"caching embeddings for {len(need)} devices (WIN_CACHE={WIN_CACHE})...")
    cache128, cache512, cacheCF = {}, {}, {}
    for tx in need:
        t = GC.consec_windows(TXD, tx, WIN_CACHE)               # [W,2,256] first WIN_CACHE consec
        # rx/date aligned to those first W windows (storage order)
        rx = TXD[tx]["rx"][:t.shape[0]].copy(); date = TXD[tx]["date"][:t.shape[0]].copy()
        e128 = embed_times(model, t)                            # 128-D projection
        e512 = GC.extract512(model, t)                          # raw 512-D
        ecf = GC.apply_head(head, {tx: scaler.transform(e512).astype(np.float32)})[tx]  # CosFace-128
        cache128[tx] = dict(emb=e128, rx=rx, date=date)
        cache512[tx] = dict(emb=e512, rx=rx, date=date)
        cacheCF[tx]  = dict(emb=ecf,  rx=rx, date=date)
    torch.cuda.empty_cache()
    caches = {"128D": cache128, "raw512": cache512, "CosFace": cacheCF}

    report = {}

    # ═══════════════ TEST 1 — NOISE + PURITY AUDIT (seed-123) ═══════════════
    print("\n" + "=" * 70 + "\nTEST 1 — HDBSCAN NOISE + PURITY AUDIT (seed-123 slice)\n" + "=" * 70)
    t1_rows = []
    for emb_name in ("128D", "CosFace"):
        bp, bl = build_slice_bursts(caches[emb_name], seed123, "coherent")
        pred = hdbscan_pred(bp, 15)
        cur = score_current(pred, bl); lok = score_locked(pred, bl)
        # hand purity over the SAME support the code uses (full, noise=-1 as its own cluster)
        hand_full = purity_std(bl, pred)
        keep = pred != -1
        hand_keptonly = purity_std(bl[keep], pred[keep]) if keep.sum() else float("nan")
        print(f"\n[{emb_name}] noise_frac={cur['noise']:.3f}  K_est={cur['K_est']}")
        print(f"  CURRENT rule : ARI={cur['ARI']:.3f} NMI={cur['NMI']:.3f} purity={cur['purity']:.3f}"
              f"   (ARI/NMI exclude -1; purity over full incl -1)")
        print(f"  LOCKED  rule : ARI={lok['ARI']:.3f} NMI={lok['NMI']:.3f} purity={lok['purity']:.3f}"
              f"   (-1 = singletons, all metrics over all points)")
        print(f"  HAND purity  : full(incl -1 cluster)={hand_full:.4f}  kept-only={hand_keptonly:.4f}"
              f"   | code purity={cur['purity']:.4f}  match_full={abs(hand_full-cur['purity'])<1e-9}")
        report[f"test1_{emb_name}"] = dict(current=cur, locked=lok,
                                           hand_purity_full=hand_full, hand_purity_keptonly=hand_keptonly)
        t1_rows.append([emb_name, cur['noise'], cur['K_est'],
                        cur['ARI'], cur['NMI'], cur['purity'],
                        lok['ARI'], lok['NMI'], lok['purity'],
                        round(hand_full, 4), round(hand_keptonly, 4)])
    csv_write(os.path.join(OUT, "test1_noise_purity_audit.csv"), t1_rows,
              ["emb", "noise_frac", "K_est", "cur_ARI", "cur_NMI", "cur_purity",
               "lock_ARI", "lock_NMI", "lock_purity", "hand_purity_full", "hand_purity_keptonly"])

    # ═══════════════ TEST 2 — COHERENT vs SCATTERED (channel leakage) ═══════════════
    print("\n" + "=" * 70 + "\nTEST 2 — BURST SHUFFLE CONTROL (coherent vs scattered)\n" + "=" * 70)
    dev_slice = GC.scatter_slice(dev_tx, 18, 201)     # DEV-only slice
    print(f"DEV slice (seed201, from {len(dev_tx)} DEV devs): {dev_slice}")
    ncells = min(len(set(zip(caches['128D'][tx]['rx'].tolist(),
                             caches['128D'][tx]['date'].tolist()))) for tx in dev_slice)
    print(f"max channel separation available in cache: {ncells} distinct (rx,date) sessions/device")
    t2_rows = []; t2 = {}
    for emb_name in ("128D", "CosFace"):
        row = {"emb": emb_name}
        for kind in ("coherent", "scattered"):
            bp, bl = build_slice_bursts(caches[emb_name], dev_slice, kind, seed=201)
            d = score_locked(hdbscan_pred(bp, 15), bl)
            row[kind] = d
            print(f"[{emb_name:8s} {kind:9s}] ARI={d['ARI']:.3f} NMI={d['NMI']:.3f} "
                  f"purity={d['purity']:.3f} K_est={d['K_est']} noise={d['noise']:.3f}")
        drop = row["coherent"]["ARI"] - row["scattered"]["ARI"]
        row["ARI_drop"] = drop
        print(f"   -> {emb_name} ARI drop (coherent - scattered) = {drop:+.3f}")
        t2[emb_name] = row
        t2_rows.append([emb_name, row["coherent"]["ARI"], row["scattered"]["ARI"], drop,
                        row["coherent"]["K_est"], row["scattered"]["K_est"]])
    csv_write(os.path.join(OUT, "test2_coherent_vs_scattered.csv"), t2_rows,
              ["emb", "coherent_ARI", "scattered_ARI", "ARI_drop", "coh_K", "scat_K"])
    max_drop = max(t2["128D"]["ARI_drop"], t2["CosFace"]["ARI_drop"])
    if max_drop < 0.05:
        g2, verdict2 = True, f"PASS: max ARI drop {max_drop:+.3f} < 0.05 — burst-mean DENOISES, not channel leakage."
    elif max_drop > 0.15:
        g2, verdict2 = False, f"FAIL: max ARI drop {max_drop:+.3f} > 0.15 — protocol exploits within-session channel coherence."
    else:
        g2, verdict2 = None, f"IN-BETWEEN: max ARI drop {max_drop:+.3f} in [0.05,0.15] — report + discuss."
    print(f"\nG2 VERDICT: {verdict2}")
    report["test2"] = {"slice": dev_slice, "max_sep_cells": ncells, "results": t2,
                       "max_ARI_drop": max_drop, "G2": g2, "verdict": verdict2}

    if g2 is not True:
        report["stopped_at"] = "G2"
        json.dump(report, open(os.path.join(OUT, "integrity_report.json"), "w"), indent=2, default=float)
        print("\nG2 not a clean PASS — STOP before Test 3 (do not recompute ladder). Report saved.")
        return

    # ═══════════════ TEST 3 — VARIANCE-HONEST LADDER (5 DEV slices x 3 mcs) ═══════════════
    print("\n" + "=" * 70 + "\nTEST 3 — VARIANCE-HONEST LADDER (seeds 201-205 x mcs{13,15,17})\n" + "=" * 70)
    slice_seeds = [201, 202, 203, 204, 205]
    methods = ["128D", "raw512", "CosFace"]
    t3_rows = []; ladder = {m: {k: [] for k in ("ARI", "NMI", "purity", "K_est", "noise")} for m in methods}
    kspread = {m: [] for m in methods}
    for m in methods:
        for s in slice_seeds:
            sl = GC.scatter_slice(dev_tx, 18, s)
            bp, bl = build_slice_bursts(caches[m], sl, "coherent")
            for mcs in MCS_JITTER:
                d = score_locked(hdbscan_pred(bp, mcs), bl)
                for k in ("ARI", "NMI", "purity", "K_est", "noise"):
                    ladder[m][k].append(d[k])
                kspread[m].append(d["K_est"])
    for m in methods:
        a = {k: (float(np.mean(ladder[m][k])), float(np.std(ladder[m][k]))) for k in ladder[m]}
        report[f"test3_{m}"] = {k: {"mean": v[0], "std": v[1]} for k, v in a.items()}
        report[f"test3_{m}"]["K_spread"] = [int(min(kspread[m])), int(max(kspread[m]))]
        print(f"[{m:8s}] ARI {a['ARI'][0]:.3f}±{a['ARI'][1]:.3f}  NMI {a['NMI'][0]:.3f}±{a['NMI'][1]:.3f}"
              f"  pur {a['purity'][0]:.3f}±{a['purity'][1]:.3f}  K {a['K_est'][0]:.1f}±{a['K_est'][1]:.1f}"
              f"  (range {min(kspread[m])}-{max(kspread[m])})  noise {a['noise'][0]:.3f}")
        t3_rows.append([m, a['ARI'][0], a['ARI'][1], a['NMI'][0], a['NMI'][1],
                        a['purity'][0], a['purity'][1], a['K_est'][0], a['K_est'][1],
                        min(kspread[m]), max(kspread[m]), a['noise'][0]])
    csv_write(os.path.join(OUT, "test3_ladder.csv"), t3_rows,
              ["method", "ARI_mean", "ARI_std", "NMI_mean", "NMI_std", "purity_mean", "purity_std",
               "K_mean", "K_std", "K_min", "K_max", "noise_mean"])

    # G3: CosFace mean ARI > 128D mean ARI by more than one COMBINED std
    cf_m, cf_s = report["test3_CosFace"]["ARI"]["mean"], report["test3_CosFace"]["ARI"]["std"]
    bl_m, bl_s = report["test3_128D"]["ARI"]["mean"], report["test3_128D"]["ARI"]["std"]
    comb = (cf_s ** 2 + bl_s ** 2) ** 0.5
    g3 = (cf_m - bl_m) > comb
    verdict3 = (f"G3 {'PASS' if g3 else 'FAIL'}: CosFace {cf_m:.3f}±{cf_s:.3f} vs 128-D {bl_m:.3f}±{bl_s:.3f}; "
                f"gap {cf_m-bl_m:+.3f} vs combined std {comb:.3f} -> "
                f"{'real (> 1 combined std)' if g3 else 'NOT separated — 0.667 may be slice luck'}.")
    print(f"\n{verdict3}")
    report["test3_G3"] = {"CosFace_ARI": [cf_m, cf_s], "baseline_ARI": [bl_m, bl_s],
                          "gap": cf_m - bl_m, "combined_std": comb, "G3": bool(g3), "verdict": verdict3}
    json.dump(report, open(os.path.join(OUT, "integrity_report.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(OUT, 'integrity_report.json')}\nCHECKPOINT — backbone FROZEN, head reused. No count fix.")


if __name__ == "__main__":
    print("loading ManyTx (eq=0)...")
    TXD, _ = W.load_manytx(eq=0, verbose=False)
    main()
