"""CAPSTONE DEMO — offline replay pipeline (Design-2 base-station association).

Raw IQ streams -> track accumulation (replay) -> drone-side encoder + contract JSON
(receiver) -> base-station discovery (eigengap / assisted K -> spectral partition ->
fingerprint_id) -> global ids + JSON out. Scoring joins ground truth (labels used for
SCORING ONLY) to report ARI / correct-K / association tables.

Implements the LOCKED operating point: encoder runs/drone_native/seed2024/best.pt (frozen),
native 4096 @ 50 MS/s windows, N*=120 windows/track, eigengap K + spectral partition.

Usage:  python run_demo.py [T1|T2|all]     (default all)

DEMO-SIDE — NOT PAPER RESULTS. best_model.pt / TEST / M100 untouched.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import csv
from collections import OrderedDict
import numpy as np

import replay
import receiver
import base_station
import scoring
import contract
from encoder_frontend import load_encoder

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", "summer_work"))
RESULTS_DIR = os.path.join(_SW, "results", "demo_build")
OUT_DIR = os.path.join(_HERE, "out")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ENCODER_SEED = 2024          # locked primary encoder
R_DEFAULT = 4
N_STAR = 120
N_REPEATS = 5

# ---- fixed scenarios (fixed emitter / unit sets) ----
SCENARIOS = OrderedDict([
    ("T1", dict(tier="mixed-model",
                emitters=["mini3pro_2", "mini4PRO_2", "mavicAir2_1", "mavicAir2_2"])),
    ("T2a", dict(tier="same-model", emitters=["mavicAir2_1", "mavicAir2_2", "mavicAir2_3"])),
    ("T2b", dict(tier="same-model",
                 emitters=["mavicAir2_1", "mavicAir2_2", "mavicAir2_3", "mavicAir2_4"])),
])
T2_SCENARIOS = ["T2a", "T2b"]


def run_repeat(model, emitters, R, N, repeat_seed):
    """One replay->receiver pass; returns (messages, ground_truth)."""
    tracks, gt = replay.stream_tracks(emitters, R=R, N=N, repeat_seed=repeat_seed)
    msgs = receiver.process_tracks(model, tracks)
    return msgs, gt


def audit_label_free(messages, assisted_K):
    """G3 runtime proof: clustering is invariant to geometry/receiver ids -> depends ONLY
    on embeddings. Zero out rssi/aoa/tdoa + scramble ids, re-associate, require ARI==1."""
    base, _ = base_station.associate(messages, assisted_K=assisted_K)
    scrub = []
    for i, m in enumerate(messages):
        mm = dict(m)
        mm["rssi"] = 0.0; mm["aoa_deg"] = 0.0; mm["tdoa_ns"] = 0.0
        mm["receiver_id"] = f"scrambled{i}"; mm["track_id"] = f"scrambled{i}"
        scrub.append(mm)
    scr, _ = base_station.associate(scrub, assisted_K=assisted_K)
    from sklearn.metrics import adjusted_rand_score
    a = np.unique([m["fingerprint_id"] for m in base], return_inverse=True)[1]
    b = np.unique([m["fingerprint_id"] for m in scr], return_inverse=True)[1]
    return float(adjusted_rand_score(a, b)) == 1.0


def audit_contract(pre_msgs, assoc_msgs):
    """G4: validate pre-association (receiver) + post-association (base station) contracts."""
    pre_errs = sum(len(contract.validate_receiver_message(m)) for m in pre_msgs)
    post_errs = sum(len(contract.validate_associated_message(m)) for m in assoc_msgs)
    emb_ok = all(len(m["embedding"]) == contract.EMB_DIM for m in pre_msgs)
    fid_only_base = (all(contract.BASE_STATION_FIELD not in m for m in pre_msgs)
                     and all(contract.BASE_STATION_FIELD in m for m in assoc_msgs))
    return dict(pre_errs=pre_errs, post_errs=post_errs, emb_ok=emb_ok,
                fid_only_base=fid_only_base)


def main(which="all"):
    which = which.lower()
    if which == "t1":
        run_names = ["T1"]
    elif which == "t2":
        run_names = ["T2a", "T2b"]
    else:
        run_names = list(SCENARIOS.keys())

    print("=" * 80)
    print(f"CAPSTONE DEMO BUILD — encoder seed{ENCODER_SEED} (frozen), R={R_DEFAULT}, "
          f"N*={N_STAR}, repeats={N_REPEATS}")
    print("  operating point: native 4096@50MS/s -> 512-D head-free -> mean-pool+L2 track"
          " -> eigengap/assisted-K -> spectral partition")
    print("=" * 80)
    model = load_encoder(ENCODER_SEED)

    rows = []                       # scenario_results.csv
    example_msg = None
    t2_assoc_table_text = None
    stream_dump = []                # associated_stream.jsonl (representative runs)
    receiver_dump = []              # receiver_stream.jsonl (pre-association, one run)
    audit_g3 = []                   # per representative run label-free proof
    audit_g4 = dict(pre_errs=0, post_errs=0, emb_ok=True, fid_only_base=True)

    for name in run_names:
        sc = SCENARIOS[name]
        emitters = sc["emitters"]
        K_true = len(emitters)
        # cache each repeat's receiver messages (reused across both modes)
        per_repeat_msgs = []
        for rep in range(N_REPEATS):
            msgs, gt = run_repeat(model, emitters, R_DEFAULT, N_STAR, repeat_seed=rep)
            per_repeat_msgs.append((msgs, gt))
        # contract audit on repeat 0 (pre + post from assisted association)
        a0, _ = base_station.associate(per_repeat_msgs[0][0], assisted_K=K_true)
        ca = audit_contract(per_repeat_msgs[0][0], a0)
        audit_g4["pre_errs"] += ca["pre_errs"]; audit_g4["post_errs"] += ca["post_errs"]
        audit_g4["emb_ok"] &= ca["emb_ok"]; audit_g4["fid_only_base"] &= ca["fid_only_base"]
        # label-free runtime proof on repeat 0 (assisted)
        audit_g3.append(audit_label_free(per_repeat_msgs[0][0], K_true))

        for mode in ("unassisted", "assisted"):
            aris, kests, successes = [], [], []
            for rep, (msgs, gt) in enumerate(per_repeat_msgs):
                aK = K_true if mode == "assisted" else None
                assoc, info = base_station.associate(msgs, assisted_K=aK)
                res = scoring.score_run(assoc, gt, info["K_est"], K_true)
                aris.append(res["ari"]); kests.append(res["K_est"])
                if mode == "unassisted":
                    succ = res["correct_k"] and res["ari"] >= 0.6
                else:
                    succ = res["ari"] >= 0.6
                successes.append(bool(succ))
                # capture representative artifacts (assisted repeat 0)
                if mode == "assisted" and rep == 0:
                    if example_msg is None:
                        example_msg = assoc[0]
                    for m in assoc:
                        mm = dict(m); mm["_run"] = f"{name}:{mode}:rep0"
                        stream_dump.append(mm)
                    if name in T2_SCENARIOS and t2_assoc_table_text is None:
                        t2_assoc_table_text = (
                            f"[{name} | assisted | rep0 | K_true={K_true}]  ARI={res['ari']}\n"
                            + scoring.format_association_table(res["association_table"]))
                if mode == "assisted" and rep == 0 and not receiver_dump:
                    receiver_dump = [dict(m) for m in msgs]
            rate = float(np.mean(successes))
            rows.append(dict(scenario=name, tier=sc["tier"], mode=mode, K_true=K_true,
                             R=R_DEFAULT, N=N_STAR, n_repeats=N_REPEATS,
                             success_rate=round(rate, 3), mean_ari=round(float(np.mean(aris)), 3),
                             aris=";".join(f"{a:.3f}" for a in aris),
                             k_ests=";".join(str(k) for k in kests)))
            print(f"  {name:<4s} {sc['tier']:<12s} {mode:<11s} K={K_true}: "
                  f"success={rate:.3f}  mean_ARI={np.mean(aris):.3f}  "
                  f"K_est={kests}  ARI={[round(a,2) for a in aris]}")

    # ---------- write deliverables ----------
    with open(os.path.join(RESULTS_DIR, "scenario_results.csv"), "w", newline="") as f:
        f.write("# DEMO-SIDE — NOT PAPER RESULTS\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT_DIR, "associated_stream.jsonl"), "w") as f:
        for m in stream_dump:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(OUT_DIR, "receiver_stream.jsonl"), "w") as f:
        for m in receiver_dump:
            f.write(json.dumps(m) + "\n")

    # ---------- gates ----------
    def rate_of(name, mode):
        return next(r["success_rate"] for r in rows if r["scenario"] == name and r["mode"] == mode)

    gates = {}
    if "T1" in run_names:
        t1u, t1a = rate_of("T1", "unassisted"), rate_of("T1", "assisted")
        gates["G1"] = dict(pass_=bool(t1u >= 0.8 and t1a >= 0.8),
                           detail=f"T1 unassisted={t1u:.3f} assisted={t1a:.3f} (need both >=0.8)")
    if any(n in run_names for n in T2_SCENARIOS):
        t2a = [rate_of(n, "assisted") for n in T2_SCENARIOS if n in run_names]
        gates["G2"] = dict(pass_=bool(min(t2a) >= 0.8),
                           detail=f"T2 assisted={[round(x,3) for x in t2a]} (need min >=0.8)")
    gates["G3"] = dict(pass_=bool(all(audit_g3)),
                       detail=f"label-free runtime proof (geo/id-invariant clustering) "
                              f"passed {sum(audit_g3)}/{len(audit_g3)} representative runs")
    gates["G4"] = dict(pass_=bool(audit_g4["pre_errs"] == 0 and audit_g4["post_errs"] == 0
                                  and audit_g4["emb_ok"] and audit_g4["fid_only_base"]),
                       detail=f"contract: pre_errs={audit_g4['pre_errs']} post_errs={audit_g4['post_errs']} "
                              f"emb512={audit_g4['emb_ok']} fingerprint_id_only_at_base={audit_g4['fid_only_base']}")

    report = dict(header="DEMO-SIDE — NOT PAPER RESULTS", encoder_seed=ENCODER_SEED,
                  R=R_DEFAULT, N_star=N_STAR, n_repeats=N_REPEATS,
                  scenarios={n: SCENARIOS[n] for n in run_names},
                  results=rows, gates=gates, example_associated_message=example_msg)
    json.dump(report, open(os.path.join(RESULTS_DIR, "report.json"), "w"), indent=2, default=str)

    # ---------- checkpoint print ----------
    print("\n" + "=" * 80 + "\nCHECKPOINT — DEMO BUILD\n" + "=" * 80)
    print("\nSCENARIO RESULTS (success = correctK & ARI>=0.6 [unassisted] / ARI>=0.6 [assisted]):")
    print(f"  {'scen':<5}{'tier':<13}{'mode':<12}{'K':<3}{'success':<9}{'meanARI':<9}")
    for r in rows:
        print(f"  {r['scenario']:<5}{r['tier']:<13}{r['mode']:<12}{r['K_true']:<3}"
              f"{r['success_rate']:<9}{r['mean_ari']:<9}")
    if example_msg is not None:
        em = dict(example_msg); em["embedding"] = f"[{len(example_msg['embedding'])} floats: " \
            f"{example_msg['embedding'][0]:.4f}, {example_msg['embedding'][1]:.4f}, ...]"
        print("\nEXAMPLE ASSOCIATED JSON MESSAGE:")
        print(json.dumps(em, indent=2))
    if t2_assoc_table_text:
        print("\nASSOCIATION TABLE (one T2 run):")
        print(t2_assoc_table_text)
    print("\nGATES:")
    for g, v in gates.items():
        print(f"  {g}: {'PASS' if v['pass_'] else 'FAIL'} — {v['detail']}")
    print(f"\nsaved -> {os.path.relpath(RESULTS_DIR)}/scenario_results.csv, report.json")
    print(f"saved -> demo/out/associated_stream.jsonl, receiver_stream.jsonl")
    return rows, gates


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(arg)
