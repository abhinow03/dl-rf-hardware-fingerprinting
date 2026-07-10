"""STEP-2b DECOHERENCE AXIS — find the MINIMUM deployable decoherence that recovers the scattered
gain. Encoder FROZEN. CosFace head reproduced deterministically (seed 42, m=0.2 s=32) == Step-2,
NOT refit. DEV devices only. Locked noise rule (singletons). Reuses Step-2 slices (seeds 201-205).

Protocols (burst = N=10 windows of one device, grouped by rx/date metadata only):
  P0 COHERENT      : 10 consecutive windows, same (rx,date) cell.            [deployable]
  P1 MULTI-RX      : same date, 10 windows from 10 DIFFERENT receivers.      [deployable: swarm]
  P2 MULTI-DATE    : same receiver, 10 windows spread across all 4 dates.    [semi-deployable]
  P3 FULLY SCATTER : 10 windows from 10 distinct (rx,date) cells.            [diagnostic ceiling]

  python3 results/step2b_decoherence/decohere.py
"""
import os, sys, json, csv
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
from sklearn.preprocessing import StandardScaler
import geometry_consolidate as GC
from burst_probe import embed_times
# reuse Step-2 locked scoring + reference builders (guaranteed identical)
from integrity import score_locked, hdbscan_pred, unitrows, bursts_coherent, bursts_scattered

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
SPLIT     = os.path.join(RUN_DIR, "splits", "split_manytx.json")
PART      = os.path.join(RUN_DIR, "splits", "discovery_partition.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
OUT       = _HERE

N = 10
WIN_CACHE = 2400                 # SAME cache as Step-2 (48 rx-date cells) -> reproduces P3
MCS_JITTER = [13, 15, 17]
SLICE_SEEDS = [201, 202, 203, 204, 205]
COS_M, COS_S = 0.20, 32
N_BURSTS = WIN_CACHE // N        # 240; every protocol targets this count/device


# ── deployable burst builders (group by rx/date metadata; never by device id) ──
def bursts_p1_multirx(emb, rx, date, n_bursts=N_BURSTS, seed=0):
    """same date, N windows from N distinct receivers."""
    rng = np.random.default_rng(seed)
    by_date = {}
    for i, (r, d) in enumerate(zip(rx.tolist(), date.tolist())):
        by_date.setdefault(int(d), {}).setdefault(int(r), []).append(i)
    usable = {d: rr for d, rr in by_date.items() if len(rr) >= N}
    if not usable:
        return None, 0, 0
    dates = list(usable.keys())
    out = []
    for _ in range(n_bursts):
        d = dates[rng.integers(0, len(dates))]
        rxs = list(usable[d].keys())
        ch = rng.choice(len(rxs), size=N, replace=False)
        idx = [usable[d][rxs[c]][rng.integers(0, len(usable[d][rxs[c]]))] for c in ch]
        out.append(emb[idx].mean(0))
    return unitrows(np.stack(out)), len(out), min(len(rr) for rr in usable.values())


def bursts_p2_multidate(emb, rx, date, n_bursts=N_BURSTS, seed=0):
    """same receiver, N windows spread across all available dates (max 4-way date decoherence)."""
    rng = np.random.default_rng(seed)
    by_rx = {}
    for i, (r, d) in enumerate(zip(rx.tolist(), date.tolist())):
        by_rx.setdefault(int(r), {}).setdefault(int(d), []).append(i)
    usable = {r: dd for r, dd in by_rx.items()
              if len(dd) >= 2 and sum(len(v) for v in dd.values()) >= N}
    if not usable:
        return None, 0, 0
    rxs = list(usable.keys())
    out = []; dates_per_burst = []
    for _ in range(n_bursts):
        r = rxs[rng.integers(0, len(rxs))]
        dd = usable[r]; dts = list(dd.keys())
        alloc = np.zeros(len(dts), dtype=int)
        for k in range(N):
            alloc[k % len(dts)] += 1           # balanced across dates: e.g. 3,3,2,2
        idx = []
        for j, dt in enumerate(dts):
            pool = dd[dt]
            sel = rng.choice(len(pool), size=min(alloc[j], len(pool)),
                             replace=(len(pool) < alloc[j]))
            idx += [pool[s] for s in sel]
        allpool = [i for v in dd.values() for i in v]
        while len(idx) < N:
            idx.append(allpool[rng.integers(0, len(allpool))])
        out.append(emb[idx[:N]].mean(0)); dates_per_burst.append(len(dts))
    return unitrows(np.stack(out)), len(out), float(np.mean(dates_per_burst))


def build_bursts(cache, slice_tx, proto, seed):
    bp, bl = [], []; shortfalls = []
    for di, tx in enumerate(slice_tx):
        c = cache[tx]
        if proto == "P0":
            b = bursts_coherent(c["emb"], N); cnt = len(b)
        elif proto == "P1":
            b, cnt, _ = bursts_p1_multirx(c["emb"], c["rx"], c["date"], seed=seed)
        elif proto == "P2":
            b, cnt, _ = bursts_p2_multidate(c["emb"], c["rx"], c["date"], seed=seed)
        elif proto == "P3":
            b, _ = bursts_scattered(c["emb"], c["rx"], c["date"], N, seed=seed); cnt = len(b)
        if b is None:
            shortfalls.append((tx, 0)); continue
        if cnt < N_BURSTS:
            shortfalls.append((tx, cnt))
        bp.append(b); bl.append(np.full(len(b), di))
    return np.concatenate(bp), np.concatenate(bl), shortfalls


def agg(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    sp = json.load(open(SPLIT)); part = json.load(open(PART))
    train_tx = sp["train_tx"]; dev_tx = part["dev_tx"]; n_cls = len(train_tx)

    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    # confirm cell layout in code before building bursts
    d0 = TXD[dev_tx[0]]
    cells = {}
    for r, dt in zip(d0["rx"].tolist(), d0["date"].tolist()):
        cells[(int(r), int(dt))] = cells.get((int(r), int(dt)), 0) + 1
    n_rx = len(set(d0["rx"].tolist())); n_dt = len(set(d0["date"].tolist()))
    print(f"LAYOUT CHECK {dev_tx[0]}: rx={n_rx} dates={n_dt} cells={len(cells)} "
          f"cell_size(min/max)={min(cells.values())}/{max(cells.values())}")
    if not (n_rx == 18 and n_dt == 4 and max(cells.values()) == 50):
        print("LAYOUT DIFFERS from 18x4x50 — STOP."); return

    # ── deterministic CosFace head (== Step-2) ──
    print(f"fitting CosFace head (m={COS_M} s={COS_S}, seed 42)...")
    tt = np.concatenate([GC.consec_windows(TXD, tx, GC.TRAIN_PER_DEV) for tx in train_tx])
    ty = np.concatenate([np.full(min(GC.TRAIN_PER_DEV, TXD[tx]["iq"].shape[0]), i)
                         for i, tx in enumerate(train_tx)])
    Ftr = GC.extract512(model, tt); del tt
    scaler = StandardScaler().fit(Ftr)
    Ftr_t = torch.from_numpy(scaler.transform(Ftr).astype(np.float32)).cuda()
    ty_t = torch.from_numpy(ty).cuda()
    head = GC.train_cosface(Ftr_t, ty_t, n_cls, COS_M, COS_S)

    # ── cache embeddings for the 25 DEV devices (first WIN_CACHE windows) ──
    print(f"caching embeddings for {len(dev_tx)} DEV devices (WIN_CACHE={WIN_CACHE})...")
    cache128, cacheCF = {}, {}
    for tx in dev_tx:
        t = GC.consec_windows(TXD, tx, WIN_CACHE)
        rx = TXD[tx]["rx"][:t.shape[0]].copy(); date = TXD[tx]["date"][:t.shape[0]].copy()
        e128 = embed_times(model, t)
        e512 = GC.extract512(model, t)
        ecf = GC.apply_head(head, {tx: scaler.transform(e512).astype(np.float32)})[tx]
        cache128[tx] = dict(emb=e128, rx=rx, date=date)
        cacheCF[tx]  = dict(emb=ecf,  rx=rx, date=date)
    torch.cuda.empty_cache()
    caches = {"128D": cache128, "CosFace": cacheCF}

    # confirm seeds reproduce Step-2 slices
    slices = {s: GC.scatter_slice(dev_tx, 18, s) for s in SLICE_SEEDS}
    print(f"seed201 slice: {slices[201]}")

    # ── G1: reproduce Step-2 Test-2 single-slice (seed201, mcs15) P0 & P3 ──
    print("\n=== G1 — reproduce Step-2 (seed201, mcs15) ===")
    g1 = {}
    for emb_name in ("128D", "CosFace"):
        for proto, ref in (("P0", None), ("P3", None)):
            bp, bl, _ = build_bursts(caches[emb_name], slices[201], proto, seed=201)
            d = score_locked(hdbscan_pred(bp, 15), bl)
            g1[f"{emb_name}_{proto}"] = d["ARI"]
            print(f"  {emb_name:8s} {proto}: ARI={d['ARI']:.3f} noise={d['noise']:.3f}")
    # Step-2 Test-2 refs: 128D coh 0.569 / scat 0.664 ; CosFace coh 0.396 / scat 0.802
    refs = {"128D_P0": 0.569, "128D_P3": 0.664, "CosFace_P0": 0.396, "CosFace_P3": 0.802}
    g1_ok = all(abs(g1[k] - refs[k]) <= 0.03 for k in refs)
    for k in refs:
        print(f"    {k}: got {g1[k]:.3f} vs Step-2 {refs[k]:.3f} (|d|={abs(g1[k]-refs[k]):.3f})"
              f" {'OK' if abs(g1[k]-refs[k])<=0.03 else 'MISMATCH'}")
    print(f"  G1 {'PASS' if g1_ok else 'FAIL'}")

    # ── full grid: 4 protocols x 2 embeddings x 5 slices x 3 mcs ──
    print("\n=== FULL GRID (5 slices x mcs{13,15,17}, locked noise rule) ===")
    protos = ["P0", "P1", "P2", "P3"]
    results = {}; all_shortfalls = {}
    for proto in protos:
        for emb_name in ("128D", "CosFace"):
            met = {k: [] for k in ("ARI", "NMI", "purity", "K_est", "noise")}
            ksp = []; sfs = set()
            for s in SLICE_SEEDS:
                bp, bl, short = build_bursts(caches[emb_name], slices[s], proto, seed=s)
                for (tx, c) in short: sfs.add((tx, c))
                for mcs in MCS_JITTER:
                    d = score_locked(hdbscan_pred(bp, mcs), bl)
                    for k in met: met[k].append(d[k]);
                    ksp.append(d["K_est"])
            results[(proto, emb_name)] = {k: agg(met[k]) for k in met}
            results[(proto, emb_name)]["K_range"] = (int(min(ksp)), int(max(ksp)))
            if sfs: all_shortfalls[(proto, emb_name)] = sorted(sfs)

    # print per-protocol/embedding detail
    print(f"\n{'proto':>5} {'emb':>8} {'ARI':>13} {'NMI':>13} {'purity':>13} {'K_est':>13} {'noise':>7}")
    for proto in protos:
        for emb_name in ("128D", "CosFace"):
            r = results[(proto, emb_name)]
            print(f"{proto:>5} {emb_name:>8} "
                  f"{r['ARI'][0]:>6.3f}±{r['ARI'][1]:<5.3f} "
                  f"{r['NMI'][0]:>6.3f}±{r['NMI'][1]:<5.3f} "
                  f"{r['purity'][0]:>6.3f}±{r['purity'][1]:<5.3f} "
                  f"{r['K_est'][0]:>5.1f}±{r['K_est'][1]:<4.1f}({r['K_range'][0]}-{r['K_range'][1]}) "
                  f"{r['noise'][0]:>6.3f}")

    # ── THE DECIDING TABLE ──
    deploy = {"P0": "yes", "P1": "yes", "P2": "semi", "P3": "no"}
    print(f"\n=== DECIDING TABLE ===")
    print(f"{'protocol':>8} {'deployable':>11} {'128-D ARI':>18} {'CosFace ARI':>18} {'noise(128/CF)':>16}")
    dec_rows = []
    for proto in protos:
        b = results[(proto, "128D")]; c = results[(proto, "CosFace")]
        print(f"{proto:>8} {deploy[proto]:>11} {b['ARI'][0]:>10.3f}±{b['ARI'][1]:<6.3f} "
              f"{c['ARI'][0]:>10.3f}±{c['ARI'][1]:<6.3f} "
              f"{b['noise'][0]:>6.3f}/{c['noise'][0]:<6.3f}")
        dec_rows.append([proto, deploy[proto], b['ARI'][0], b['ARI'][1],
                         c['ARI'][0], c['ARI'][1], b['noise'][0], c['noise'][0],
                         f"{b['K_range'][0]}-{b['K_range'][1]}", f"{c['K_range'][0]}-{c['K_range'][1]}"])

    with open(os.path.join(OUT, "decision_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "deployable", "ARI_128D_mean", "ARI_128D_std",
                    "ARI_CosFace_mean", "ARI_CosFace_std", "noise_128D", "noise_CosFace",
                    "K_range_128D", "K_range_CosFace"])
        for r in dec_rows: w.writerow(r)

    # full grid csv
    with open(os.path.join(OUT, "full_grid.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "emb", "ARI_mean", "ARI_std", "NMI_mean", "NMI_std",
                    "purity_mean", "purity_std", "K_mean", "K_std", "K_min", "K_max",
                    "noise_mean", "noise_std"])
        for proto in protos:
            for emb_name in ("128D", "CosFace"):
                r = results[(proto, emb_name)]
                w.writerow([proto, emb_name, r['ARI'][0], r['ARI'][1], r['NMI'][0], r['NMI'][1],
                            r['purity'][0], r['purity'][1], r['K_est'][0], r['K_est'][1],
                            r['K_range'][0], r['K_range'][1], r['noise'][0], r['noise'][1]])

    # shortfalls
    print(f"\n=== BURST-COUNT SHORTFALLS (target {N_BURSTS}/device) ===")
    if not all_shortfalls:
        print("  none — every protocol reached the target count on every DEV device.")
    else:
        for k, v in all_shortfalls.items():
            print(f"  {k}: {v}")

    # ── decision ──
    bar = 0.558
    best_deployable = max([("P0", "128D"), ("P0", "CosFace"), ("P1", "128D"), ("P1", "CosFace")],
                          key=lambda pe: results[pe]["ARI"][0])
    bdp, bde = best_deployable
    bd_ari = results[best_deployable]["ARI"]
    beats = bd_ari[0] > bar
    print(f"\n=== DECISION ===")
    print(f"  honest bar (P0 128-D, Step-2): {bar:.3f}")
    print(f"  best DEPLOYABLE (P0/P1): {bdp} {bde} ARI {bd_ari[0]:.3f}±{bd_ari[1]:.3f} "
          f"-> {'BEATS' if beats else 'does NOT beat'} bar")

    json.dump({"g1": {"values": g1, "refs": refs, "ok": bool(g1_ok)},
               "results": {f"{p}|{e}": {k: (v if not isinstance(v, tuple) else list(v))
                                        for k, v in results[(p, e)].items()}
                           for p in protos for e in ("128D", "CosFace")},
               "shortfalls": {f"{k[0]}|{k[1]}": v for k, v in all_shortfalls.items()},
               "bar": bar, "best_deployable": [bdp, bde, bd_ari[0], bd_ari[1]], "beats_bar": bool(beats)},
              open(os.path.join(OUT, "decohere_report.json"), "w"), indent=2, default=float)
    print(f"\nsaved -> {os.path.join(OUT, 'decohere_report.json')}")
    print("CHECKPOINT — encoder frozen, head reused, DEV only, TEST sealed. No M100.")


if __name__ == "__main__":
    print("loading ManyTx (eq=0)...")
    TXD, _ = W.load_manytx(eq=0, verbose=False)
    main()
