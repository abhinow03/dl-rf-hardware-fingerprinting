"""GEN-RFF PHASE 4 — protocol-router demo orchestrator.

Raw IQ -> tracker -> protocol router -> tier encoder -> accumulator -> per-protocol-group
base-station discovery -> namespaced global IDs + JSON. Scenarios S1/S2/S3 x modes x 5 repeats.
Ground truth confined to scoring.py. BUILD + SCENARIO VALIDATION ONLY — no training.

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

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DLM not in sys.path:
    sys.path.insert(0, _DLM)

from gen_rff.demo_router import replay, router as R, tiers as TI, accumulate, base_station as BS, scoring, geometry, contract
from gen_rff.data import loaders

OUT_DIR = os.path.join(_DLM, "results_gen", "demo_router", "out")
RES_DIR = os.path.join(_DLM, "results_gen", "demo_router")
os.makedirs(OUT_DIR, exist_ok=True)
R_RECV, NSTAR, REPEATS = 4, 120, 5


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


def run_scenario(name, emitters, tiers, router, forced_unknown=False, modes=("assisted", "unassisted")):
    kt_proto = defaultdict(set)
    rows = []; example = None; assoc_dump = []
    per_mode = {m: [] for m in modes}
    for rep in range(REPEATS):
        tracks, gt = replay.stream_tracks(emitters, R=R_RECV, N=NSTAR, repeat_seed=rep)
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
            per_mode[mode].append(dict(route_acc=route_acc, n_partial=n_partial,
                                       groups={p: dict(ari=group_scores[p]["ari"], K_est=group_scores[p]["K_est"],
                                                       K_true=group_scores[p]["K_true"]) for p in group_scores},
                                       success=succ))
            if mode == "assisted" and rep == 0:
                example = assoc[0]
                for m in assoc:
                    mm = dict(m); mm["_run"] = f"{name}:{mode}:rep0"; assoc_dump.append(mm)
                # capture one association table for the report
                run_scenario._last_table = {p: scoring.format_association_table(group_scores[p]["association_table"])
                                            for p in group_scores}
    # aggregate
    for mode in modes:
        recs = per_mode[mode]
        succs = [r["success"] for r in recs if r["success"] is not None]
        rate = float(np.mean(succs)) if succs else float("nan")
        mean_route = float(np.mean([r["route_acc"] for r in recs]))
        # mean per-group ARI (across repeats), per protocol
        arivals = defaultdict(list)
        for r in recs:
            for p, gs in r["groups"].items():
                arivals[p].append(gs["ari"])
        mean_ari = {p: round(float(np.mean(v)), 3) for p, v in arivals.items()}
        rows.append(dict(scenario=name, mode=mode, repeats=REPEATS, route_acc=round(mean_route, 3),
                         success_rate=(round(rate, 3) if succs else "n/a (report-only)"),
                         mean_group_ari=mean_ari, n_partial=recs[0]["n_partial"]))
    return rows, example, assoc_dump


def main():
    print("=" * 80 + "\nGEN-RFF PHASE 4 — PROTOCOL-ROUTER DEMO\n" + "=" * 80)
    # ---- fit router + R1 ----
    print("[router] building training tracks (WiSig train vs DRFF non-mavicAir2) ...")
    X, y = R.build_training_tracks()
    rng = np.random.default_rng(0); idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X)); tr, te = idx[:ntr], idx[ntr:]
    router = R.ProtocolRouter().fit(X[tr], y[tr])
    pred = np.array([R.LABELS.index(router.route(X[i])[0]) if router.route(X[i])[0] in R.LABELS else -1 for i in te])
    r1_acc = float((pred == y[te]).mean())
    # UNKNOWN smoke: synthetic FM sweep at drone rate (rate alone would misroute -> OOD must save it)
    fm = R.make_fm_sweep(replay.RATE["drone"], 4096, n_win=60)
    fm_proto, fm_conf, fm_d = router.route(R.track_features(fm, replay.RATE["drone"]))
    r1_pass = r1_acc >= 0.95 and fm_proto == "unknown"
    print(f"[R1] router held-out accuracy={r1_acc:.3f} (n={len(te)}); FM-sweep -> '{fm_proto}' "
          f"(dist {fm_d:.2f} vs thresh {router.ood_thresh:.2f})  => {'PASS' if r1_pass else 'FAIL'}")

    tiers = TI.Tiers()
    dev_tx = loaders.wisig_devices()[1]
    S = {
        "S1_drone3": [("drone", f"mavicAir2_{i}") for i in (1, 2, 3)],
        "S1_drone4": [("drone", f"mavicAir2_{i}") for i in (1, 2, 3, 4)],
        "S2_mixed": [("drone", "mavicAir2_1"), ("drone", "mavicAir2_2"), ("wifi", dev_tx[0]), ("wifi", dev_tx[1])],
        "S3_unknown": [("drone", "mavicAir2_1"), ("drone", "mavicAir2_2")],
    }
    all_rows = []; example_msg = None; dumps = []; tables = {}
    for name, emitters in S.items():
        fu = (name == "S3_unknown")
        modes = ("assisted", "unassisted")
        rows, ex, dump = run_scenario(name, emitters, tiers, router, forced_unknown=fu, modes=modes)
        all_rows += rows
        if example_msg is None and ex is not None:
            example_msg = ex
        dumps += dump
        tables[name] = getattr(run_scenario, "_last_table", {})
        for r in rows:
            print(f"  {name:<12} {r['mode']:<11} route={r['route_acc']} success={r['success_rate']} "
                  f"grpARI={r['mean_group_ari']} partial={r['n_partial']}")

    # ---- label-confinement audit (runtime): base-station partition invariant to geo/id scramble ----
    tr0, gt0 = replay.stream_tracks(S["S2_mixed"], R=R_RECV, N=NSTAR, repeat_seed=0)
    m0, aux0 = build_messages(tr0, gt0, tiers, router)
    full0 = [m for m in m0 if not m["partial"]]
    a1, _ = BS.associate(full0, aux0, assisted_K={"drone_ocusync": 2, "wifi": 2})
    scr = []
    for i, m in enumerate(full0):
        mm = dict(m); mm["rssi"] = 0.0; mm["aoa_deg"] = 0.0; mm["tdoa_ns"] = 0.0
        mm["receiver_id"] = f"x{i}"; mm["track_id"] = f"x{i}"; scr.append(mm)
    aux_scr = {f"x{i}": aux0.get(m["track_id"]) for i, m in enumerate(full0)}
    a2, _ = BS.associate(scr, aux_scr, assisted_K={"drone_ocusync": 2, "wifi": 2})
    from sklearn.metrics import adjusted_rand_score
    lf = float(adjusted_rand_score(np.unique([m["fingerprint_id"] for m in a1], return_inverse=True)[1],
                                   np.unique([m["fingerprint_id"] for m in a2], return_inverse=True)[1])) == 1.0
    audit_line = ("LABEL-CONFINEMENT: ground_truth dict flows ONLY into scoring.* ; router.route takes "
                  "features only; base_station.associate takes embeddings only (no gt arg). Runtime proof: "
                  f"zeroing geo + scrambling ids leaves the partition identical (ARI==1: {lf}).")
    print("\n" + audit_line)

    # ---- gates ----
    def rate(name, mode):
        r = next(x for x in all_rows if x["scenario"] == name and x["mode"] == mode)
        return r["success_rate"], r["route_acc"]
    g = {}
    s1_min = min(rate("S1_drone3", "assisted")[0], rate("S1_drone4", "assisted")[0])
    g["G-S1"] = dict(pass_=bool(s1_min >= 0.8), detail=f"S1 assisted success min(3,4)={s1_min} (>=0.8)")
    s2s, s2r = rate("S2_mixed", "assisted")
    g["G-S2"] = dict(pass_=bool(s2r >= 0.95 and s2s >= 0.8), detail=f"S2 routing={s2r} (>=0.95) assisted success={s2s} (>=0.8)")
    g["G-S3"] = dict(pass_=True, detail="reported (honesty tier, no bar)")
    g["R1"] = dict(pass_=bool(r1_pass), detail=f"router acc={r1_acc:.3f} (>=0.95), FM->{fm_proto}")

    # ---- contract validation on the dumped stream ----
    pre_errs = sum(len(contract.validate_associated_message(m)) for m in dumps)

    # ---- save ----
    with open(os.path.join(RES_DIR, "scenario_results.csv"), "w", newline="") as f:
        f.write("# DEMO-SIDE — NOT PAPER RESULTS (gen_rff sandbox)\n")
        w = csv.DictWriter(f, fieldnames=["scenario", "mode", "repeats", "route_acc", "success_rate",
                                          "mean_group_ari", "n_partial"])
        w.writeheader()
        for r in all_rows:
            w.writerow({**r, "mean_group_ari": json.dumps(r["mean_group_ari"])})
    with open(os.path.join(OUT_DIR, "associated_stream.jsonl"), "w") as f:
        for m in dumps:
            f.write(json.dumps(m) + "\n")
    report = dict(header="GEN-RFF PHASE 4 — protocol-router demo (DEMO-SIDE, R&D sandbox)",
                  R=R_RECV, Nstar=NSTAR, repeats=REPEATS, router_R1=dict(acc=r1_acc, fm=fm_proto, pass_=r1_pass),
                  scenarios=all_rows, gates=g, label_confinement=audit_line,
                  contract_assoc_errs=pre_errs, example_associated_message=example_msg,
                  association_tables=tables)
    json.dump(report, open(os.path.join(RES_DIR, "phase4_router_report.json"), "w"), indent=2, default=str)

    print("\n=== GATES ===")
    for k, v in g.items():
        print(f"  {k}: {'PASS' if v['pass_'] else 'FAIL'} — {v['detail']}")
    print(f"  contract assoc errors: {pre_errs}")
    print("\nEXAMPLE ASSOCIATED MESSAGE:")
    em = dict(example_msg); em["embedding"] = f"[512 floats: {example_msg['embedding'][0]:.4f}, ...]"
    print(json.dumps(em, indent=2))
    if "S2_mixed" in tables:
        print("\nS2 ASSOCIATION TABLES (assisted rep0):")
        for p, t in tables["S2_mixed"].items():
            print(f" [{p}]\n{t}")
    print("\nsaved -> results_gen/demo_router/{scenario_results.csv, phase4_router_report.json}, out/associated_stream.jsonl")


if __name__ == "__main__":
    main()
