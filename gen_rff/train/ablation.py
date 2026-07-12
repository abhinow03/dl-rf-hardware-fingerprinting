"""GEN-RFF PHASE 2b — PART 2: ablation battery (A1-A5).

Same config as Phase 2 except ONE stated toggle per arm. Same selection formula. No new
sweeps. Each arm: train + val-in-loop + select + final eval (mavicAir2 N=10 AND N=120 +
WiSig-DEV self-cell). Full-recipe row is CITED from the Phase-2 report (not re-run).

  OMP_NUM_THREADS=1 ... python3 -m gen_rff.train.ablation

mavicAir2 touched only in final evals; frozen assets read-only; artifacts in runs_gen/results_gen.
"""
import os
import sys
import json
import csv
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DLM not in sys.path:
    sys.path.insert(0, _DLM)

import gen_rff.train.train_lopo as T
from gen_rff.data import registry
from gen_rff.bench import harness as H

RESULTS_GEN = os.path.join(_DLM, "results_gen")
_MAV = None


def mav_data():
    global _MAV
    if _MAV is None:
        print("[eval] building mavicAir2 native eval data once (cap %d) ..." % T.MAVIC_CAP)
        _MAV = T.build_mavicair2_native()
    return _MAV


def eval_arm(model):
    iq, res, phys, af, D, C = mav_data()
    d = registry.REGISTRY["DRFF"]
    emb = T.embed512(model, iq, res, phys, d.n_fft, d.hop)
    out = {}
    for N in (10, 120):
        bp, bl = T.E1_bursts(emb, af, D, C, N)
        bp, bl = T.balance(bp, bl); bpu = H.unit(bp)
        _, hdb = H.hdbscan_grid(bpu, bl, [5, 7]); km, sp = H.oracle_km_sp(bpu, bl, 8)
        out[f"N{N}"] = dict(km=round(km, 3), sp=round(sp, 3), hdb=round(hdb, 3),
                            knn1=round(H.knn_purity(bp, bl)[1], 3), nb=int(len(bl)))
    return out


ARMS = {
    "A1_no_physics":   dict(use_physics=False),
    "A2_no_residual":  dict(use_residual=False),
    "A3_no_crosscond": dict(cross_condition=False),
    "A4_no_oracle":    dict(domains=("WISIG",)),
    "A5_long30k":      dict(steps=30000, warmup=3000),
}


def run():
    print("=" * 80 + "\nPHASE 2b PART 2 — ABLATION BATTERY (A1-A5)\n" + "=" * 80)
    results = {}
    # full-recipe row cited from Phase 2
    gp = json.load(open(os.path.join(RESULTS_GEN, "phase2_lopo_drff_report.json")))
    fe = gp["final_eval"]
    results["full_recipe_CITED"] = dict(
        selected_step=gp["selected_step"],
        mav_N10=dict(km=fe["N10"]["oracle_km"], sp=fe["N10"]["oracle_sp"]),
        mav_N120=dict(km=fe["N120"]["oracle_km"], sp=fe["N120"]["oracle_sp"]),
        wisig_self=gp["self_cells"]["wisig_dev"]["oracle_km"],
        oracle_loss_note="WiSig fell to ~2.4; ORACLE stagnated ~3.42 (Phase-2)")

    for name, over in ARMS.items():
        cfg = T.default_cfg(tag=name, **over)
        model, best, curve, ds, va, vb, dloss = T.train(cfg)
        cells = eval_arm(model)
        selfc = T.val_a(model, *va)["oracle_km"]
        # ORACLE loss behavior note
        wl = dloss.get("WISIG", []); ol = dloss.get("ORACLE", [])
        oracle_note = ("no ORACLE (A4)" if "ORACLE" not in cfg["domains"]
                       else f"WiSig {wl[0] if wl else '-'}->{wl[-1] if wl else '-'}, "
                            f"ORACLE {ol[0] if ol else '-'}->{ol[-1] if ol else '-'}")
        results[name] = dict(selected_step=best["step"], mav_N10=cells["N10"], mav_N120=cells["N120"],
                             wisig_self=round(float(selfc), 3), oracle_loss_note=oracle_note,
                             dloss=dloss)
        print(f"  >> {name}: mav@N10 km={cells['N10']['km']} | mav@N120 km={cells['N120']['km']} | "
              f"WiSig-self={selfc:.3f} | {oracle_note}")
        del model
        torch.cuda.empty_cache()

    # ---------- consolidated table ----------
    order = ["full_recipe_CITED", "A1_no_physics", "A2_no_residual", "A3_no_crosscond",
             "A4_no_oracle", "A5_long30k"]
    print("\n" + "=" * 80)
    print(f"{'arm':<20}{'mav@N10 km':>12}{'mav@N120 km':>13}{'WiSig-self':>12}   ORACLE-loss")
    for k in order:
        r = results[k]
        print(f"{k:<20}{r['mav_N10']['km']:>12}{r['mav_N120']['km']:>13}{r['wisig_self']:>12}   "
              f"{r.get('oracle_loss_note', '')}")

    # ---------- VERDICT LINE 2 ----------
    base = results["full_recipe_CITED"]
    b10, bself = base["mav_N10"]["km"], base["wisig_self"]
    deltas = {}
    for k in ["A1_no_physics", "A2_no_residual", "A3_no_crosscond", "A4_no_oracle", "A5_long30k"]:
        deltas[k] = dict(d_mav10=round(results[k]["mav_N10"]["km"] - b10, 3),
                         d_self=round(results[k]["wisig_self"] - bself, 3))
    a4_self = results["A4_no_oracle"]["wisig_self"]; a5_self = results["A5_long30k"]["wisig_self"]
    ingr = max(deltas, key=lambda k: abs(deltas[k]["d_mav10"]))
    recover = []
    if a4_self >= bself + 0.05:
        recover.append(f"A4(no-ORACLE) recovers WiSig self {bself:.3f}->{a4_self:.3f} (NEGATIVE TRANSFER from ORACLE)")
    if a5_self >= bself + 0.05:
        recover.append(f"A5(30k) recovers WiSig self {bself:.3f}->{a5_self:.3f} (UNDERTRAINING)")
    if not recover:
        recover.append(f"neither A4({a4_self:.3f}) nor A5({a5_self:.3f}) recovers WiSig self toward 0.72 "
                       f"(base {bself:.3f}) -> in-domain cost is NOT simply ORACLE-poisoning or undertraining")
    v2 = (f"largest mav@N10 delta: {ingr} ({deltas[ingr]['d_mav10']:+.3f}); per-arm deltas vs full "
          f"recipe (d_mav10 / d_self): "
          + "; ".join(f"{k.split('_',1)[1]} {deltas[k]['d_mav10']:+.3f}/{deltas[k]['d_self']:+.3f}"
                      for k in deltas) + ". " + " | ".join(recover))
    print("\nVERDICT LINE 2:", v2)

    out = dict(header="PHASE 2b PART 2 — ablation battery (DEMO-SIDE, R&D)",
               arms={k: {kk: vv for kk, vv in v.items() if kk != "dloss"} for k, v in results.items()},
               dloss_trajectories={k: results[k].get("dloss", {}) for k in results},
               deltas_vs_full=deltas, verdict_line_2=v2)
    json.dump(out, open(os.path.join(RESULTS_GEN, "phase2b_part2_ablation.json"), "w"), indent=2, default=str)
    with open(os.path.join(RESULTS_GEN, "phase2b_part2_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "mav_N10_km", "mav_N120_km", "wisig_self", "oracle_loss_note"])
        for k in order:
            r = results[k]
            w.writerow([k, r["mav_N10"]["km"], r["mav_N120"]["km"], r["wisig_self"], r.get("oracle_loss_note", "")])
    print("\nsaved -> results_gen/phase2b_part2_ablation.{json,csv}")


if __name__ == "__main__":
    run()
