"""STEP-3b confound completion: SESSION-DISJOINT same-model probe.
The segment-disjoint probe splits WITHIN a file, so session/channel can leak (mavicAir2s_4,_5
are single-condition -> perfect confound). Here the atomic split unit is the SESSION =
(airframe, U, D, C): each airframe's held-out test windows come from a condition the probe
never trained on. Airframes with <2 sessions are excluded (can't be cross-session tested).
Reuses cached features (features_A/B.npz); no re-extraction, no training of the encoder.

  python3 results/step3b_drff_smoke/session_disjoint_probe.py
"""
import os, sys, csv, json
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def load(opt):
    z = np.load(os.path.join(_HERE, f"features_{opt}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}

def session_probe(F, y_af, sess, rng):
    """split SESSIONS disjointly; each airframe must contribute >=1 session to train & test."""
    y = np.unique(y_af, return_inverse=True)[1]
    # assign sessions to train/test, stratified so every airframe appears in both
    tr = np.zeros(len(y_af), bool)
    for a in np.unique(y_af):
        s_a = np.unique(sess[y_af == a]); rng.shuffle(s_a)
        ntr = max(1, int(round(0.6 * len(s_a))))
        if len(s_a) - ntr < 1: ntr = len(s_a) - 1     # guarantee >=1 test session
        tr_sess = set(s_a[:ntr].tolist())
        tr |= np.array([(y_af[i] == a and sess[i] in tr_sess) for i in range(len(y_af))])
    te = ~tr
    if len(set(y[tr])) < len(set(y)) or len(set(y[te])) < len(set(y)):
        return None
    sc = StandardScaler().fit(F[tr]); Xtr, Xte = sc.transform(F[tr]), sc.transform(F[te])
    out = {}
    for pn, clf in [("logreg", LogisticRegression(max_iter=2000)),
                    ("mlp", MLPClassifier((256,), max_iter=300, early_stopping=True, random_state=0))]:
        clf.fit(Xtr, y[tr]); proba = clf.predict_proba(Xte)
        aw = float((clf.classes_[proba.argmax(1)] == y[te]).mean())
        ab = []
        for s in np.unique(sess[te]):
            m = sess[te] == s
            ab.append(clf.classes_[proba[m].mean(0).argmax()] == y[te][m][0])
        out[pn] = (aw, float(np.mean(ab)))
    return out

rows = []
print("SESSION-DISJOINT same-model probe (512-D). session = (airframe,U,D,C).")
print(f"{'opt':>3} {'group':>11} {'n_af':>4} {'chance':>7} {'logreg_win':>11} {'logreg_burst':>13} {'mlp_win':>8} {'mlp_burst':>10}")
for opt in ("A", "B"):
    d = load(opt)
    af = d["af"].astype(str)
    sess = np.array([f"{a}|{u}|{dd}|{c}" for a, u, dd, c in
                     zip(af, d["U"].astype(str), d["D"].astype(str), d["C"].astype(str))])
    for grp in ("mavicAir2", "mavicAir2s"):
        model = d["model"].astype(str)
        gm = model == grp
        # keep airframes with >=2 sessions
        keep_af = [a for a in np.unique(af[gm]) if len(np.unique(sess[af == a])) >= 2]
        msk = np.isin(af, keep_af) & gm
        # balance windows per airframe to group min
        r = np.random.default_rng(777)
        per = min((af[msk] == a).sum() for a in keep_af)
        sel = []
        for a in keep_af:
            ii = np.where(msk & (af == a))[0]; r.shuffle(ii); sel += ii[:per].tolist()
        sel = np.array(sorted(sel))
        res = session_probe(d["F512"][sel], af[sel], sess[sel], np.random.default_rng(1))
        chance = 1.0 / len(keep_af)
        excluded = sorted(set(np.unique(af[gm])) - set(keep_af))
        if res is None:
            print(f"{opt:>3} {grp:>11} {len(keep_af):>4}  degenerate split")
            continue
        lw, lb = res["logreg"]; mw, mb = res["mlp"]
        print(f"{opt:>3} {grp:>11} {len(keep_af):>4} {chance:>7.3f} {lw:>11.3f} {lb:>13.3f} {mw:>8.3f} {mb:>10.3f}"
              + (f"   excl={excluded}" if excluded else ""))
        rows.append(dict(opt=opt, group=grp, n_af=len(keep_af), excluded=";".join(excluded),
                         chance=chance, logreg_win=lw, logreg_burst=lb, mlp_win=mw, mlp_burst=mb,
                         per_af=int(per)))

with open(os.path.join(_HERE, "probe_session_disjoint.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

best = max(rows, key=lambda r: r["mlp_burst"])
print(f"\nBest session-disjoint same-model 512-D burst acc = {best['mlp_burst']:.3f} "
      f"({best['group']}, OPT-{best['opt']}, chance {best['chance']:.3f})")
print("Compare vs segment-disjoint (leaky) best 0.871 — the drop is the session-leak magnitude.")
json.dump(rows, open(os.path.join(_HERE, "probe_session_disjoint.json"), "w"), indent=2, default=str)
