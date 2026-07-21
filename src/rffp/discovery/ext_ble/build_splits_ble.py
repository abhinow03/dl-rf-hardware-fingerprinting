#!/usr/bin/env python3
"""
Phase 1 STEP 3 — D2 BLE splits. Seed=2026. Unit-disjoint eval group + val units;
condition axis over the 12 collections. Emits splits_ext_ble.json (+ prints sha256)
and a human-readable summary table. NO training. Authors' intra-collection train/test
split is IGNORED (pooled) — we re-split on our unit/collection axes (Phase-0b flag #3).
"""
import os, json, hashlib, glob
import numpy as np

ROOT = "/home/docker/pw26_akp_01/ext_data/ble_xiao"
OUT  = "/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols"
SEED = 2026
N_EVAL, N_VAL = 8, 2

# ---- collection groups (condition axis) ----
TRAIN_COLL = ["Wired (indoors)/Ch1_R1", "Wired (indoors)/Ch1_R2", "Wired (indoors)/Ch2_R1",
              "Wired (indoors)/Ch14_R1", "Wired (indoors)/Ch32_R1",
              "Wireless (Indoors)/Ch2", "Wireless (Indoors)/R1", "Wireless (Indoors)/R2"]
EVAL_COLL  = ["Wireless (outdoors)/Loc1", "Wireless (outdoors)/Loc2",
              "Wireless (outdoors)/Loc3", "Wireless (outdoors)/Loc4"]
# receiver-disjoint side arm (channel-matched Rx pairs)
RX_FIT_COLL  = ["Wired (indoors)/Ch1_R1", "Wireless (Indoors)/R1"]
RX_EVAL_COLL = ["Wired (indoors)/Ch1_R2", "Wireless (Indoors)/R2"]

def labels(coll):
    """pooled (train+test) per-segment unit index for a collection."""
    ytr = np.asarray(np.load(os.path.join(ROOT, coll, "Y_train.npy"), mmap_mode="r")).argmax(1)
    yte = np.asarray(np.load(os.path.join(ROOT, coll, "Y_test.npy"),  mmap_mode="r")).argmax(1)
    return np.concatenate([ytr, yte])

def main():
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(31)
    eval_units  = sorted(perm[:N_EVAL].tolist())
    val_units   = sorted(perm[N_EVAL:N_EVAL+N_VAL].tolist())
    train_units = sorted(perm[N_EVAL+N_VAL:].tolist())
    print(f"seed={SEED}  perm={perm.tolist()}")
    print(f"eval_units({len(eval_units)})={eval_units}")
    print(f"val_units({len(val_units)})={val_units}")
    print(f"train_units({len(train_units)})={train_units}")

    # per-collection pooled labels (cache in memory)
    all_coll = TRAIN_COLL + EVAL_COLL
    lab = {c: labels(c) for c in all_coll}

    def count(units, colls):
        u = set(units); return int(sum(np.isin(lab[c], list(u)).sum() for c in colls))

    splits = {
        "canonical":  {"desc": "strict cross-condition: train wired+wireless-indoor, eval outdoor",
                       "train": {"units": train_units, "collections": TRAIN_COLL},
                       "val":   {"units": val_units,   "collections": TRAIN_COLL},
                       "eval":  {"units": eval_units,  "collections": EVAL_COLL}},
        "diagnostic": {"desc": "matched-condition upper bound (report as diagnostic, never headline)",
                       "eval": {"units": eval_units,   "collections": TRAIN_COLL}},
        "receiver_disjoint": {"desc": "pre-registered Rx arm: fit R1-only, eval R2-only, eval_units",
                       "fit":  {"units": eval_units,   "collections": RX_FIT_COLL},
                       "eval": {"units": eval_units,   "collections": RX_EVAL_COLL}},
    }

    # ---- segment counts ----
    rows = []
    def add(tag, units, colls):
        rows.append((tag, len(units), len(colls), count(units, colls)))
    add("canonical/train",  train_units, TRAIN_COLL)
    add("canonical/val",    val_units,   TRAIN_COLL)
    add("canonical/eval",   eval_units,  EVAL_COLL)
    add("diagnostic/eval",  eval_units,  TRAIN_COLL)
    add("recv/fit(R1)",     eval_units,  RX_FIT_COLL)
    add("recv/eval(R2)",    eval_units,  RX_EVAL_COLL)

    print("\n=== SPLIT SUMMARY (units x collections x segments) ===")
    md = ["| split | #units | #collections | #segments |", "|---|---|---|---|"]
    for tag, nu, nc, ns in rows:
        print(f"  {tag:20s} units={nu:2d}  colls={nc}  segs={ns}")
        md.append(f"| {tag} | {nu} | {nc} | {ns} |")

    # ---- ASSERTIONS ----
    print("\n=== ASSERTIONS ===")
    seteval, setval, settrain = set(eval_units), set(val_units), set(train_units)
    a1 = bool(seteval.isdisjoint(settrain) and seteval.isdisjoint(setval))
    print(f"[A1] eval_units disjoint from train_units and val_units .......... {'PASS' if a1 else 'FAIL'}")
    # A2: no train/val ROW carries an eval_unit label (enumerate filtered rows)
    def rows_labels(units, colls):
        u=set(units); out=[]
        for c in colls: out.append(lab[c][np.isin(lab[c], list(u))])
        return np.concatenate(out) if out else np.array([],dtype=int)
    tr = rows_labels(train_units, TRAIN_COLL); va = rows_labels(val_units, TRAIN_COLL)
    a2 = bool((np.isin(tr, eval_units).sum()==0) and (np.isin(va, eval_units).sum()==0))
    print(f"[A2] zero eval_unit rows reachable from canonical train/val ...... {'PASS' if a2 else 'FAIL'} "
          f"(train rows={tr.size}, val rows={va.size}, eval-labelled among them={int(np.isin(tr,eval_units).sum()+np.isin(va,eval_units).sum())})")
    # A3: no outdoor collection in canonical train/val collection sets
    outdoor=set(EVAL_COLL)
    a3 = bool(outdoor.isdisjoint(set(TRAIN_COLL)))
    print(f"[A3] zero outdoor collections in canonical train/val ............. {'PASS' if a3 else 'FAIL'}")
    # A4: val_units disjoint from train_units (checkpoint-selection isolation)
    a4 = bool(setval.isdisjoint(settrain))
    print(f"[A4] val_units disjoint from train_units ......................... {'PASS' if a4 else 'FAIL'}")
    assert a1 and a2 and a3 and a4, "SPLIT ASSERTIONS FAILED"

    out = {"seed": SEED, "perm": perm.tolist(),
           "eval_units": eval_units, "val_units": val_units, "train_units": train_units,
           "train_collections": TRAIN_COLL, "eval_collections": EVAL_COLL,
           "receiver_arm": {"fit_collections": RX_FIT_COLL, "eval_collections": RX_EVAL_COLL},
           "splits": splits,
           "segment_counts": {tag: ns for tag,_,_,ns in rows},
           "assertions": {"A1_eval_unit_disjoint": a1, "A2_no_eval_rows_in_trainval": a2,
                          "A3_no_outdoor_in_trainval": a3, "A4_val_train_disjoint": a4},
           "note": "authors' intra-collection train/test pooled then re-split on unit/collection axes"}
    path = os.path.join(OUT, "splits_ext_ble.json")
    with open(path, "w") as f: json.dump(out, f, indent=2)
    h = hashlib.sha256(open(path,"rb").read()).hexdigest()
    print(f"\nWROTE {path}\nsha256 {h}")
    # write the md table for pasting into DATA_AUDIT
    with open(os.path.join(OUT,"audit_out","splits_summary.md"),"w") as f:
        f.write("\n".join(md)+"\n")
    print("MD_TABLE_START"); print("\n".join(md)); print("MD_TABLE_END")

if __name__ == "__main__":
    main()
