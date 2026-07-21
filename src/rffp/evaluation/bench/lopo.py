"""rffp.evaluation.bench.lopo — LOPO splits + pairwise matrix + baseline reproduction (P0-G1).

LOPOSplit(holdout): leave-one-DOMAIN-out. Train identities = all non-quarantined domains
except the holdout; eval = the holdout's designated discovery group. Hard disjointness
asserts; WiSig TEST (board 18) excluded everywhere; DRFF mavicAir2 never in a train pool.
Deterministic (seed 777) -> results_gen/splits_lopo.json.

Baseline rows reproduce the locked EVAL_PROTOCOL numbers to validate the harness port:
  DRFF frozen-512 E1 oracle-K@8  km/sp ~ 0.297/0.298
  DRFF classical-19 E1 HDBSCAN(mean over mcs{5,7}) ~ 0.245
  WiSig DEV P2 (single-rx multi-date) 512-D oracle-K@18 km ~ 0.72
All under rffp.evaluation.bench.harness (the ported §4.3 harness).
"""
import os
import re
import json
from collections import defaultdict
import numpy as np

from rffp.evaluation.bench import harness as H
from rffp.data import loaders, registry
from rffp.config import RESULTS_DIR as RESULTS_GEN, DRFF_DIR, RUNS_DIR, WIFI_CKPT as BASE_CKPT

RUN_DIR = os.path.join(RUNS_DIR, "wisig_supcon_fft64")
LOPO_SEED = 777
N = 10
WIN = 256
NTAPS = 129
DRFF_CAP = 320


# ============================================================================
# LOPO SPLITS
# ============================================================================
class LOPOSplit:
    def __init__(self, holdout, seed=LOPO_SEED):
        assert holdout in registry.DEFAULT_DOMAINS, f"{holdout} not a non-quarantined domain"
        self.holdout = holdout
        self.seed = seed
        train, evalg = {}, {}
        for dom in registry.DEFAULT_DOMAINS:
            pool, ev = registry.domain_device_pool(dom)
            if dom == holdout:
                evalg[dom] = ev
            else:
                train[dom] = pool
        self.train = train            # {domain: [train gids]}
        self.eval = {holdout: evalg[holdout]}
        self._assert_disjoint()

    def _assert_disjoint(self):
        train_gids = set(g for gs in self.train.values() for g in gs)
        eval_gids = set(g for gs in self.eval.values() for g in gs)
        assert train_gids.isdisjoint(eval_gids), "train/eval device_gid overlap!"
        # WiSig TEST (board 18) excluded everywhere
        _, _, test = loaders.wisig_devices()
        test_gids = set(f"WISIG:{t}" for t in test)
        assert test_gids.isdisjoint(train_gids | eval_gids), "WiSig TEST (board 18) leaked!"
        # DRFF mavicAir2 (the exact eval group; NOT the distinct mavicAir2s variant) never
        # in any train pool
        _, drff_eval = loaders.drff_airframes()
        mav_gids = set(f"DRFF:{a}" for a in drff_eval)
        assert mav_gids.isdisjoint(train_gids), "DRFF mavicAir2 (eval group) in a train pool!"

    def to_dict(self):
        return dict(holdout=self.holdout, seed=self.seed,
                    train={d: gs for d, gs in self.train.items()},
                    eval={d: gs for d, gs in self.eval.items()},
                    n_train=sum(len(g) for g in self.train.values()),
                    n_eval=sum(len(g) for g in self.eval.values()))


def write_all_lopo_splits():
    splits = {}
    for holdout in registry.DEFAULT_DOMAINS:
        splits[holdout] = LOPOSplit(holdout).to_dict()
    os.makedirs(RESULTS_GEN, exist_ok=True)
    path = os.path.join(RESULTS_GEN, "splits_lopo.json")
    json.dump(splits, open(path, "w"), indent=2)
    return splits, path


# ============================================================================
# DRFF OPT-B baseline path (copied verbatim from step7 test_and_harness.py)
# ============================================================================
def _decimate_B(iq):
    from scipy.signal import firwin, filtfilt
    taps = firwin(NTAPS, 0.9)
    x = iq.astype(np.float32); pad = min(3 * NTAPS, x.shape[1] - 1)
    return filtfilt(taps, [1.0], x, axis=1, padlen=pad)[:, ::2]


def _build_mavicAir2_optb(cap=DRFF_CAP, seed=LOPO_SEED):
    from rffp.data import wisig_manytx as W
    manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
    pat = re.compile(r'(.+?)_(\d+)_hover')
    af_files = defaultdict(list)
    for r in manifest["clean"]:
        af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
    all_af = sorted(af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
    model_of = {a: pat.match(a + "_hover").group(1) for a in all_af}
    eval_af = [a for a in all_af if model_of[a] == "mavicAir2"]
    rng = np.random.default_rng(seed)
    Xt, af, seg, Ds, Cs = [], [], [], [], []
    gseg = 0
    for ai, a in enumerate(eval_af):
        files = af_files[a][:]; rng.shuffle(files)
        units = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]):
                units.append((fn, si))
        rng.shuffle(units)
        got = 0; zc = {}
        for fn, si in units:
            if got >= cap:
                break
            if fn not in zc:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zc[fn] = dict(dec=_decimate_B(z["iq"]), sb=z["seg_bounds"], D=str(z["D"]), C=int(z["C"]))
            zz = zc[fn]; off, ln = zz["sb"][si]; o2, l2 = off // 2, ln // 2
            nw = l2 // WIN
            if nw < 1:
                continue
            take = min(nw, cap - got, 12); segw = zz["dec"][:, o2:o2 + l2]
            for k in range(take):
                Xt.append(W.standardize(segw[:, k * WIN:(k + 1) * WIN].astype(np.float32)))
                af.append(ai); seg.append(gseg); Ds.append(zz["D"]); Cs.append(zz["C"])
            got += take; gseg += 1
    return (np.stack(Xt).astype(np.float32), np.array(af), np.array(seg), np.array(Ds), np.array(Cs))


def _E1_index_bursts(afd, Dd, Cd, seed=LOPO_SEED):
    r = np.random.default_rng(seed); out = []
    for a in np.unique(afd):
        idx = np.where(afd == a)[0]; cells = defaultdict(list)
        for i in idx:
            cells[(Dd[i], Cd[i])].append(i)
        ck = list(cells.keys()); nb = len(idx) // N
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < N and ci < 2000:
                c = ck[ci % len(ck)]
                if cells[c]:
                    pick.append(cells[c][r.integers(len(cells[c]))])
                ci += 1
            if len(pick) == N:
                out.append((np.array(pick), a))
    return out


def _balance_idx(bursts, seed=0):
    r = np.random.default_rng(seed); lbl = np.array([b[1] for b in bursts])
    per = min(int((lbl == a).sum()) for a in np.unique(lbl)); keep = []
    for a in np.unique(lbl):
        ii = np.where(lbl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    return [bursts[i] for i in sorted(keep)]


def _mod_bursts(feat, bursts):
    return np.stack([feat[idx].mean(0) for idx, _ in bursts])


def reproduce_drff_baselines():
    """Frozen-512 + classical-19 on mavicAir2 OPT-B E1 bursts (locked harness). Returns cells."""
    import torch
    from rffp.models import RFEncoder
    from rffp.discovery.wisig import geometry_consolidate as GC
    from sklearn.preprocessing import StandardScaler
    Xt, afd, segd, Dd, Cd = _build_mavicAir2_optb()
    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    F512 = GC.extract512(model, Xt)
    CLS19 = StandardScaler().fit_transform(_classical(Xt)).astype(np.float32)
    E1 = _balance_idx(_E1_index_bursts(afd, Dd, Cd))
    labels = np.array([b[1] for b in E1])
    enc_b = _mod_bursts(F512, E1)
    cls_b = _mod_bursts(CLS19, E1)
    fr = H.full_score(enc_b, labels, 8, H.MCS_DRFF, with_oracle=True)
    cl = H.full_score(cls_b, labels, 8, H.MCS_DRFF, with_oracle=True)
    del model
    torch.cuda.empty_cache()
    return dict(
        frozen512_E1=dict(oracleK_km=fr["oracleK_km"], oracleK_sp=fr["oracleK_sp"],
                          hdb_mean=fr["hdb_mean_ARI"], knn1=fr["knn1"], n_bursts=fr["n_bursts"]),
        classical19_E1=dict(oracleK_km=cl["oracleK_km"], oracleK_sp=cl["oracleK_sp"],
                            hdb_mean=cl["hdb_mean_ARI"], knn1=cl["knn1"], n_bursts=cl["n_bursts"]),
        n_windows=int(len(Xt)))


def _classical(Xt):
    from rffp.physics.features import classical_matrix
    return classical_matrix(Xt)


def reproduce_wisig_p2(slice_seeds=(201, 202, 203, 204, 205), K=18, win_full=100000):
    """WiSig DEV P2 (single-rx multi-date) 512-D oracle-K@18 km/sp, mean over 5 DEV slices."""
    import torch
    from rffp.models import RFEncoder
    from rffp.discovery.wisig import geometry_consolidate as GC
    from rffp.evaluation.scoring import bursts_p2_multidate
    TXD = loaders._wisig_txd()
    _, dev_tx, _ = loaders.wisig_devices()
    model = RFEncoder().cuda()
    model.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # cache 512-D per DEV device
    cache = {}
    for tx in dev_tx:
        t = GC.consec_windows(TXD, tx, win_full); Wn = t.shape[0]
        cache[tx] = dict(f512=GC.extract512(model, t),
                         rx=TXD[tx]["rx"][:Wn].copy(), date=TXD[tx]["date"][:Wn].copy())
    kms, sps = [], []
    for s in slice_seeds:
        sl = GC.scatter_slice(dev_tx, K, s)
        bp, bl = [], []
        for di, tx in enumerate(sl):
            c = cache[tx]
            b = bursts_p2_multidate(c["f512"], c["rx"], c["date"], seed=s)[0]
            if b is None:
                continue
            bp.append(b); bl.append(np.full(len(b), di))
        bp = np.concatenate(bp); bl = np.concatenate(bl)
        km, sp = H.oracle_km_sp(H.unit(bp), bl, K)
        kms.append(km); sps.append(sp)
    del model
    torch.cuda.empty_cache()
    return dict(oracleK_km=float(np.mean(kms)), oracleK_sp=float(np.nanmean(sps)),
                per_seed_km=[round(x, 3) for x in kms], K=K, n_slices=len(slice_seeds))


# ============================================================================
# P0-G1 GATE
# ============================================================================
LOCKED = {
    "DRFF frozen-512 E1 oracle-km": (0.297, "drff", "frozen512_E1", "oracleK_km"),
    "DRFF frozen-512 E1 oracle-sp": (0.298, "drff", "frozen512_E1", "oracleK_sp"),
    "DRFF classical-19 E1 HDB-mean": (0.245, "drff", "classical19_E1", "hdb_mean"),
    "WiSig DEV P2 512-D oracle-km": (0.72, "wisig", None, "oracleK_km"),
}


def p0g1_check(drff_cells, wisig_cell, tol=0.02):
    rows = []
    for name, (locked, dom, sub, key) in LOCKED.items():
        if dom == "drff":
            got = drff_cells[sub][key]
        else:
            got = wisig_cell[key]
        ok = abs(got - locked) <= tol
        rows.append(dict(cell=name, locked=locked, reproduced=round(got, 3),
                         diff=round(got - locked, 3), within_tol=bool(ok)))
    return rows, all(r["within_tol"] for r in rows)
