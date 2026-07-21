"""GEN-RFF PHASE 5 — protocol-router demo orchestrator (BLE tier integrated).

Raw IQ -> tracker -> protocol router (rate-aware, 3-class + OOD) -> tier encoder (config-driven
TIER_REGISTRY incl. T-D BLE native-128) -> accumulator -> per-protocol-group base-station
discovery -> namespaced global IDs + JSON. Scenarios: S1/S2/S3 (regression) + S4 ble×K + S6
grand-mixed. Ground truth confined to scoring.py. NO training here — frozen tiers only.

  OMP_NUM_THREADS=1 ... python3 -m gen_rff.demo_router.run_demo
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import csv
from collections import defaultdict
import numpy as np

from rffp.discovery.router import replay, router as R, tiers as TI, accumulate, base_station as BS, scoring, geometry, contract
from rffp.data import loaders
from rffp.config import RESULTS_DIR, RUNS_DIR

OUT_DIR = os.path.join(RESULTS_DIR, "demo_router", "out")
RES_DIR = os.path.join(RESULTS_DIR, "demo_router")
os.makedirs(OUT_DIR, exist_ok=True)
R_RECV, NSTAR, REPEATS = 4, 120, 5

GROUP_SUCCESS = {}     # (name, mode) -> {protocol: success_rate}  (report/gate side; not in CSV)


def build_messages(tracks, gt, tiers, router, forced_unknown=False):
    """tracker -> router -> tier -> accumulator -> contract messages (+ aux side map)."""
    msgs, aux_by_tid = [], {}
    for t in tracks:
        feat = R.track_features(t["Xt"], t["sample_rate"])
        if forced_unknown:
            proto, conf = "unknown", 0.0
        else:
            proto, conf, _ = router.route(feat)
        deep, aux = tiers.encode(proto, t["Xt"])
        emb, aux_mean, n_used, partial = accumulate.accumulate(deep, aux, NSTAR)
        r, s = t["receiver_id"], t["scene_idx"]
        msgs.append(dict(receiver_id=f"rx{r}", track_id=t["track_id"], timestamp=t["timestamp"],
                         embedding=emb.tolist(), rssi=geometry.rssi_dbm(r, s),
                         aoa_deg=geometry.aoa_deg(r, s), tdoa_ns=geometry.tdoa_ns(r, s),
                         n_windows=n_used, partial=partial, **{"class": "unknown"},
                         protocol=proto, protocol_conf=round(float(conf), 3)))
        aux_by_tid[t["track_id"]] = aux_mean
    return msgs, aux_by_tid


def true_K_by_group(full_msgs, gt):
    g = defaultdict(set)
    for m in full_msgs:
        g[m["protocol"]].add(gt[m["track_id"]][1])
    return {p: len(v) for p, v in g.items()}


def run_scenario(name, emitters, tiers, router, forced_unknown=False, modes=("assisted", "unassisted"),
                 ble_collections=None):
    kt_proto = defaultdict(set)
    rows = []; example = None; assoc_dump = []
    per_mode = {m: [] for m in modes}
    grp_succ = {m: defaultdict(list) for m in modes}
    for rep in range(REPEATS):
        tracks, gt = replay.stream_tracks(emitters, R=R_RECV, N=NSTAR, repeat_seed=rep,
                                          ble_collections=ble_collections)
        msgs, aux = build_messages(tracks, gt, tiers, router, forced_unknown=forced_unknown)
        route_acc = scoring.routing_accuracy(msgs, gt, forced_unknown=forced_unknown)
        full = [m for m in msgs if not m["partial"]]
        n_partial = len(msgs) - len(full)
        Kt = true_K_by_group(full, gt)
        for mode in modes:
            aK = {p: Kt[p] for p in Kt} if mode == "assisted" else None
            assoc, info = BS.associate(full, aux, assisted_K=aK)
            # per-group scoring
            grp_msgs = defaultdict(list)
            for m in assoc:
                grp_msgs[m["protocol"]].append(m)
            group_scores = {}
            for p, gm in grp_msgs.items():
                sc = scoring.score_group(gm, gt, info[p]["K_est"], Kt[p])
                group_scores[p] = sc
            # scenario success
            if forced_unknown:
                succ = None                                  # S3: report only, no bar
            elif mode == "assisted":
                succ = route_acc >= 0.95 and all(s["ari"] >= 0.6 for s in group_scores.values())
            else:
                succ = route_acc >= 0.95 and all(s["correct_k"] and s["ari"] >= 0.6 for s in group_scores.values())
            # per-group success (route ok AND that group's ari>=0.6) for per-protocol gates
            for p, s in group_scores.items():
                grp_succ[mode][p].append(bool(route_acc >= 0.95 and s["ari"] >= 0.6))
            per_mode[mode].append(dict(route_acc=route_acc, n_partial=n_partial,
                                       groups={p: dict(ari=group_scores[p]["ari"], K_est=group_scores[p]["K_est"],
                                                       K_true=group_scores[p]["K_true"]) for p in group_scores},
                                       success=succ))
            if mode == "assisted" and rep == 0:
                example = assoc[0]
                for m in assoc:
                    mm = dict(m); mm["_run"] = f"{name}:{mode}:rep0"; assoc_dump.append(mm)
                run_scenario._last_table = {p: scoring.format_association_table(group_scores[p]["association_table"])
                                            for p in group_scores}
    # aggregate
    for mode in modes:
        recs = per_mode[mode]
        succs = [r["success"] for r in recs if r["success"] is not None]
        rate = float(np.mean(succs)) if succs else float("nan")
        mean_route = float(np.mean([r["route_acc"] for r in recs]))
        arivals = defaultdict(list)
        for r in recs:
            for p, gs in r["groups"].items():
                arivals[p].append(gs["ari"])
        mean_ari = {p: round(float(np.mean(v)), 3) for p, v in arivals.items()}
        rows.append(dict(scenario=name, mode=mode, repeats=REPEATS, route_acc=round(mean_route, 3),
                         success_rate=(round(rate, 3) if succs else "n/a (report-only)"),
                         mean_group_ari=mean_ari, n_partial=recs[0]["n_partial"]))
        GROUP_SUCCESS[(name, mode)] = {p: round(float(np.mean(v)), 3) for p, v in grp_succ[mode].items()}
    return rows, example, assoc_dump


# ---------------------------------------------------------------------------------------------
def rate_fix_evidence(router):
    """3a regression evidence: legacy 25 MS/s feature vector numerically UNCHANGED; drone 50 MS/s
    scales the frequency-dimensioned features by exactly 2.0; BLE 6 MS/s by 6/25."""
    tx = loaders.wisig_devices()[0][0]
    wifi_win = replay.get_pool("wifi", tx)["Xt"][:60]
    af = list(loaders.drff_airframes()[0])[0]
    drone_win = replay.get_pool("drone", af)["Xt"][:60]

    def norm_only(Xt):     # legacy bin-relative feature vector (rf forced to 1.0 == 25 MS/s)
        return R.track_features(Xt, R.REF_RATE)
    fdims = [0, 1, 3, 4, 8]   # cent, spread, roll, occ, hop  (frequency-dimensioned)
    wifi_new = R.track_features(wifi_win, replay.RATE["wifi"])
    wifi_leg = norm_only(wifi_win)
    drone_new = R.track_features(drone_win, replay.RATE["drone"])
    drone_leg = norm_only(drone_win)
    ratio = []
    for i in fdims:
        denom = drone_leg[i] if abs(drone_leg[i]) > 1e-9 else 1e-9
        ratio.append(round(float(drone_new[i] / denom), 4))
    return dict(
        wifi_unchanged_max_abs_delta=float(np.max(np.abs(wifi_new - wifi_leg))),
        wifi_bitwise_identical=bool(np.array_equal(wifi_new, wifi_leg)),
        drone_freqdim_ratio_new_over_legacy=ratio,                 # expect ~[2,2,2,2,2]
        drone_expected_ratio=round(replay.RATE["drone"] / R.REF_RATE, 4),
        ble_freq_scale=round(replay.RATE["ble"] / R.REF_RATE, 4))


def g_fm(router):
    """3c G-FM: canonical narrowband FM sweep at drone rate must route UNKNOWN. Report per-class
    Mahalanobis distances (named risk: BLE may now be the nearest class)."""
    fm = R.make_fm_sweep(replay.RATE["drone"], 4096, n_win=60)
    feat = R.track_features(fm, replay.RATE["drone"])
    proto, conf, dmin = router.route(feat)
    dists = router.distances(feat)
    nearest = min(dists, key=dists.get)
    return dict(proto=proto, dmin=round(float(dmin), 3), thresh=round(float(router.ood_thresh), 3),
                per_class=dists, nearest_class=nearest, pass_=bool(proto == "unknown"))


def g_rt(router, tiers):
    """3c G-RT: 4-class held-out routing accuracy (wifi / drone / ble / unknown-synthetic).
    Labels used for SCORING ONLY. BLE side = eval_units tracks (never used to fit the router)."""
    _, dev_tx, test_tx = loaders.wisig_devices()
    rows = []   # (feat, true_label)
    for tx in list(dev_tx)[:8]:
        p = replay.get_pool("wifi", tx)["Xt"]
        for t in range(min(6, len(p) // 60)):
            rows.append((R.track_features(p[t * 60:(t + 1) * 60], replay.RATE["wifi"]), "wifi"))
    for af in [f"mavicAir2_{i}" for i in (1, 2, 3, 4)]:
        p = replay.get_pool("drone", af)["Xt"]
        for t in range(min(8, len(p) // 60)):
            rows.append((R.track_features(p[t * 60:(t + 1) * 60], replay.RATE["drone"]), "drone_ocusync"))
    for u in R.BLE_EVAL_UNITS:
        p = replay.get_ble_pool(u, R.BLE_EVAL_COLLECTIONS)["Xt"]
        for t in range(min(6, len(p) // 60)):
            rows.append((R.track_features(p[t * 60:(t + 1) * 60], replay.RATE["ble"]), "ble"))
    for seed in range(6):
        fm = R.make_fm_sweep(replay.RATE["drone"], 4096, n_win=60, seed=10 + seed)
        rows.append((R.track_features(fm, replay.RATE["drone"]), "unknown"))
    ok = 0; conf = defaultdict(lambda: defaultdict(int))
    for feat, lab in rows:
        pr = router.route(feat)[0]
        ok += int(pr == lab)
        conf[lab][pr] += 1
    acc = ok / len(rows)
    return dict(acc=round(float(acc), 4), n=len(rows), pass_=bool(acc >= 0.95),
                confusion={k: dict(v) for k, v in conf.items()})


def main():
    print("=" * 84 + "\nGEN-RFF PHASE 5 — PROTOCOL-ROUTER DEMO (BLE tier integrated)\n" + "=" * 84)
    # ---- fit router (3-class rate-aware) + held-out R1 ----
    print("[router] building training tracks (WiSig / DRFF non-mavicAir2 / BLE train_units) ...")
    X, y = R.build_training_tracks()
    print(f"[router] train rows: wifi={int((y==0).sum())} drone={int((y==1).sum())} ble={int((y==2).sum())}")
    rng = np.random.default_rng(0); idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X)); tr, te = idx[:ntr], idx[ntr:]
    router = R.ProtocolRouter().fit(X[tr], y[tr])

    def _routed_label_idx(i):
        pr = router.route(X[i])[0]
        return R.LABELS.index(pr) if pr in R.LABELS else -1
    r1_acc = float(np.mean([_routed_label_idx(i) == y[i] for i in te]))

    ratefix = rate_fix_evidence(router)
    print(f"[3a rate-fix] wifi bitwise-identical={ratefix['wifi_bitwise_identical']} "
          f"(max|Δ|={ratefix['wifi_unchanged_max_abs_delta']:.2e}); drone freq-dim ratio="
          f"{ratefix['drone_freqdim_ratio_new_over_legacy']} (expect {ratefix['drone_expected_ratio']})")

    gfm = g_fm(router)
    print(f"[3c G-FM] FM-sweep -> '{gfm['proto']}' dmin={gfm['dmin']} thresh={gfm['thresh']} "
          f"nearest={gfm['nearest_class']} per-class={gfm['per_class']}  => {'PASS' if gfm['pass_'] else 'FAIL'}")
    if not gfm["pass_"]:
        print("\n*** G-FM FAILED — narrowband FM routed to a real class. Design checkpoint: reporting "
              "distance table and STOPPING (no threshold hacking). ***")
        json.dump(dict(gfm=gfm, ratefix=ratefix), open(os.path.join(RES_DIR, "phase5_GFM_FAIL.json"), "w"), indent=2)
        return

    grt = g_rt(router, None)
    print(f"[3c G-RT] 4-class held-out routing acc={grt['acc']} (n={grt['n']}) => {'PASS' if grt['pass_'] else 'FAIL'}")
    print(f"          confusion={grt['confusion']}")

    tiers = TI.Tiers()
    dev_tx = loaders.wisig_devices()[1]
    ble_pool_cols = R.BLE_EVAL_COLLECTIONS          # pooled (demo-honest) variant
    kU = R.BLE_EVAL_UNITS

    # ---- S1–S3 REGRESSION (unchanged emitters/seeds) ----
    S_reg = {
        "S1_drone3": [("drone", f"mavicAir2_{i}") for i in (1, 2, 3)],
        "S1_drone4": [("drone", f"mavicAir2_{i}") for i in (1, 2, 3, 4)],
        "S2_mixed": [("drone", "mavicAir2_1"), ("drone", "mavicAir2_2"), ("wifi", dev_tx[0]), ("wifi", dev_tx[1])],
        "S3_unknown": [("drone", "mavicAir2_1"), ("drone", "mavicAir2_2")],
    }
    all_rows = []; example_msg = None; dumps = []; tables = {}
    for name, emitters in S_reg.items():
        fu = (name == "S3_unknown")
        rows, ex, dump = run_scenario(name, emitters, tiers, router, forced_unknown=fu)
        all_rows += rows
        if example_msg is None and ex is not None:
            example_msg = ex
        dumps += dump
        tables[name] = getattr(run_scenario, "_last_table", {})
        for r in rows:
            print(f"  {name:<12} {r['mode']:<11} route={r['route_acc']} success={r['success_rate']} "
                  f"grpARI={r['mean_group_ari']} partial={r['n_partial']}")

    # ---- S4 ble×K (single-collection canonical-like + pooled-collection demo-honest) ----
    ble_example = None; s4_rows = []
    for variant, cols in [("single", [R.BLE_EVAL_COLLECTIONS[0]]), ("pooled", ble_pool_cols)]:
        for K in (2, 3, 4):
            name = f"S4_ble_{variant}_K{K}"
            emitters = [("ble", u) for u in kU[:K]]
            rows, ex, dump = run_scenario(name, emitters, tiers, router, ble_collections=cols)
            s4_rows += rows; all_rows += rows; dumps += dump
            if variant == "pooled" and ble_example is None:
                ble_example = ex
            for r in rows:
                print(f"  {name:<20} {r['mode']:<11} route={r['route_acc']} success={r['success_rate']} "
                      f"grpARI={r['mean_group_ari']} partial={r['n_partial']}")

    # ---- S6 grand-mixed: 2 drone + 2 wifi + 2 ble ----
    s6_emitters = [("drone", "mavicAir2_1"), ("drone", "mavicAir2_2"),
                   ("wifi", dev_tx[0]), ("wifi", dev_tx[1]),
                   ("ble", kU[0]), ("ble", kU[1])]
    s6_rows, s6_ex, s6_dump = run_scenario("S6_grand", s6_emitters, tiers, router, ble_collections=ble_pool_cols)
    all_rows += s6_rows; dumps += s6_dump
    for r in s6_rows:
        print(f"  {'S6_grand':<20} {r['mode']:<11} route={r['route_acc']} success={r['success_rate']} "
              f"grpARI={r['mean_group_ari']} partial={r['n_partial']}")
    # cross-protocol ID collisions on the S6 assisted rep0 dump
    s6_assoc = [m for m in s6_dump if m.get("_run", "").startswith("S6_grand:assisted")]
    fids_by_proto = defaultdict(set)
    for m in s6_assoc:
        fids_by_proto[m["protocol"]].add(m["fingerprint_id"])
    protos = list(fids_by_proto)
    collisions = 0
    for i in range(len(protos)):
        for j in range(i + 1, len(protos)):
            collisions += len(fids_by_proto[protos[i]] & fids_by_proto[protos[j]])
    s6_table = getattr(run_scenario, "_last_table", {})

    # ---- REGRESSION CHECK vs committed baseline (S1–S3) ----
    import io
    base_csv = os.path.join(RES_DIR, "scenario_results.csv")
    prev = {}
    if os.path.exists(base_csv):
        with open(base_csv) as f:
            for r in csv.DictReader(filter(lambda l: not l.startswith("#"), f)):
                prev[(r["scenario"], r["mode"])] = r
    reg = []
    for r in all_rows:
        if not r["scenario"].startswith(("S1", "S2", "S3")):
            continue
        key = (r["scenario"], r["mode"]); p = prev.get(key)
        new_ari = r["mean_group_ari"]
        old_ari = json.loads(p["mean_group_ari"]) if p else None
        ari_ok = old_ari is not None and set(new_ari) == set(old_ari) and all(
            abs(new_ari[k] - old_ari[k]) <= 0.02 for k in new_ari)
        route_ok = p is not None and abs(float(p["route_acc"]) - r["route_acc"]) <= 0.02
        succ_ok = p is not None and str(p["success_rate"]) == str(r["success_rate"])
        reg.append(dict(cell=f"{key[0]}/{key[1]}", route_ok=route_ok, ari_ok=ari_ok, succ_ok=succ_ok,
                        old_route=(p["route_acc"] if p else None), new_route=r["route_acc"],
                        old_ari=old_ari, new_ari=new_ari,
                        old_succ=(p["success_rate"] if p else None), new_succ=r["success_rate"]))
    reg_pass = all(x["route_ok"] and x["ari_ok"] and x["succ_ok"] for x in reg)

    # ---- gates ----
    def get_rate(name, mode):
        r = next(x for x in all_rows if x["scenario"] == name and x["mode"] == mode)
        return r["success_rate"], r["route_acc"]
    g = {}
    g["R1"] = dict(pass_=bool(r1_acc >= 0.95), detail=f"router held-out acc={r1_acc:.3f} (>=0.95)")
    g["G-FM"] = dict(pass_=gfm["pass_"], detail=f"FM->{gfm['proto']} nearest={gfm['nearest_class']} (dmin {gfm['dmin']}>thresh {gfm['thresh']})")
    g["G-RT"] = dict(pass_=grt["pass_"], detail=f"4-class routing acc={grt['acc']} (>=0.95)")
    g["G-REG"] = dict(pass_=bool(reg_pass), detail=f"S1–S3 reproduced ({sum(x['route_ok'] and x['ari_ok'] and x['succ_ok'] for x in reg)}/{len(reg)} cells)")
    # S4 pooled gate (the gated variant): routing>=0.95 AND assisted success>=0.8 across K
    s4_pool = [get_rate(f"S4_ble_pooled_K{K}", "assisted") for K in (2, 3, 4)]
    s4_route_min = min(rt for _, rt in s4_pool); s4_succ_min = min(sc for sc, _ in s4_pool)
    g["G-S4"] = dict(pass_=bool(s4_route_min >= 0.95 and s4_succ_min >= 0.8),
                     detail=f"S4 pooled: min routing={s4_route_min} (>=0.95), min assisted success={s4_succ_min} (>=0.8)")
    # S6 gate: routing>=0.95, per-group assisted success>=0.8, 0 cross-protocol collisions
    s6_route = get_rate("S6_grand", "assisted")[1]
    s6_grp = GROUP_SUCCESS[("S6_grand", "assisted")]
    s6_grp_min = min(s6_grp.values()) if s6_grp else 0.0
    g["G-S6"] = dict(pass_=bool(s6_route >= 0.95 and s6_grp_min >= 0.8 and collisions == 0),
                     detail=f"S6 routing={s6_route} (>=0.95), per-group assisted success={s6_grp} (min>=0.8), collisions={collisions}")

    # ---- contract validation on dumped stream (mixed 512/128 dims) ----
    pre_errs = sum(len(contract.validate_associated_message(m)) for m in dumps)

    # ---- save ----
    with open(base_csv, "w", newline="") as f:
        f.write("# DEMO-SIDE — NOT PAPER RESULTS (gen_rff sandbox)\n")
        w = csv.DictWriter(f, fieldnames=["scenario", "mode", "repeats", "route_acc", "success_rate",
                                          "mean_group_ari", "n_partial"])
        w.writeheader()
        for r in all_rows:
            w.writerow({**r, "mean_group_ari": json.dumps(r["mean_group_ari"])})
    with open(os.path.join(OUT_DIR, "associated_stream.jsonl"), "w") as f:
        for m in dumps:
            f.write(json.dumps(m) + "\n")

    report = dict(header="GEN-RFF PHASE 5 — protocol-router demo, BLE tier (DEMO-SIDE, R&D sandbox)",
                  R=R_RECV, Nstar=NSTAR, repeats=REPEATS,
                  tier_registry={p: {k: (v[k] if k != "ckpt" else os.path.relpath(v[k], RUNS_DIR))
                                     for k in ("tier", "backend", "dim", "aux", "ckpt")}
                                 for p, v in TI.TIER_REGISTRY.items()},
                  router=dict(r1_acc=r1_acc, classes=[R.LABELS[c] for c in router.classes_],
                              ood_thresh=router.ood_thresh),
                  rate_fix=ratefix, g_fm=gfm, g_rt=grt,
                  scenarios=all_rows, group_success={f"{k[0]}/{k[1]}": v for k, v in GROUP_SUCCESS.items()},
                  regression=dict(pass_=reg_pass, cells=reg),
                  s6_collisions=collisions, gates=g, contract_assoc_errs=pre_errs,
                  example_deep_message=example_msg, example_ble_message=ble_example,
                  s6_association_tables=s6_table)
    json.dump(report, open(os.path.join(RES_DIR, "phase5_router_report.json"), "w"), indent=2, default=str)

    print("\n=== REGRESSION (S1–S3 vs committed) ===", "PASS" if reg_pass else "FAIL")
    for x in reg:
        print(f"  {x['cell']:<24} route {x['old_route']}->{x['new_route']} ari {x['old_ari']}->{x['new_ari']} "
              f"succ {x['old_succ']}->{x['new_succ']}  [{'ok' if x['route_ok'] and x['ari_ok'] and x['succ_ok'] else 'DIFF'}]")
    print("\n=== GATES ===")
    for k, v in g.items():
        print(f"  {k}: {'PASS' if v['pass_'] else 'FAIL'} — {v['detail']}")
    print(f"  contract assoc errors (mixed 512/128): {pre_errs}")
    if ble_example is not None:
        em = dict(ble_example); em["embedding"] = f"[128 floats: {ble_example['embedding'][0]:.4f}, ...]"
        print("\nEXAMPLE BLE ASSOCIATED MESSAGE (S4 pooled):")
        print(json.dumps(em, indent=2))
    print("\nsaved -> results_gen/demo_router/{scenario_results.csv, phase5_router_report.json}, out/associated_stream.jsonl")


if __name__ == "__main__":
    main()
