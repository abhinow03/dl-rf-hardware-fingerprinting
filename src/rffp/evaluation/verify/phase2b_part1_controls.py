"""GEN-RFF PHASE 2b — PART 1: N-matched controls on mavicAir2 (8-way, E1 bursts).

Is the generalist's N=120 result (0.625) a per-fingerprint gain, or burst-averaging any
feature gets? Score frozen WiSig-only (OPT-B), classical-19 (OPT-B), native-drone-trained
seed2024 (native), and cite the generalist — all at N in {10, 120} under the locked harness.

  OMP_NUM_THREADS=1 ... python3 -m gen_rff.verify.phase2b_part1_controls
"""
import os
import sys
import json
import csv
from collections import defaultdict
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLM = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _DLM not in sys.path:
    sys.path.insert(0, _DLM)
_SW = os.path.join(_DLM, "summer_work")
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rffp.data import loaders, registry
from rffp.evaluation.bench import harness as H
from rffp.evaluation.bench import lopo
from rffp.physics.features import classical_matrix
from rffp.training.train_lopo import eigengap_k, partition_at, balance
from rffp.models import RFEncoder
from rffp.discovery.wisig import geometry_consolidate as GC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

RESULTS_GEN = os.path.join(_DLM, "results_gen")
DRONE_CKPT = os.path.join(_SW, "runs", "drone_native", "seed2024", "best.pt")
BASE_CKPT = os.path.join(_SW, "runs", "wisig_supcon_fft64", "retrain_best", "best_model.pt")
MCS_DRFF = [5, 7]
CAP = 1500
NS = (10, 120)


def build_E1(emb, af, D, C, N, seed=777):
    r = np.random.default_rng(seed); bp, bl = [], []
    for a in np.unique(af):
        idx = np.where(af == a)[0]; cells = defaultdict(list)
        for i in idx:
            cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx) // N
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < N and ci < 4000:
                c = ck[ci % len(ck)]
                if cells[c]:
                    pick.append(cells[c][r.integers(len(cells[c]))])
                ci += 1
            if len(pick) == N:
                bp.append(emb[pick].mean(0)); bl.append(a)
    return np.array(bp), np.array(bl)


def score(emb, af, D, C):
    row = {}
    for N in NS:
        bp, bl = build_E1(emb, af, D, C, N)
        bp, bl = balance(bp, bl); bpu = H.unit(bp)
        per, hdbmean = H.hdbscan_grid(bpu, bl, MCS_DRFF)
        km, sp = H.oracle_km_sp(bpu, bl, 8)
        intra, inter = H.cos_gap(bp, bl)
        cell = dict(km=round(km, 3), sp=round(sp, 3), hdb=round(hdbmean, 3),
                    knn1=round(H.knn_purity(bp, bl)[1], 3), gap=round(intra - inter, 3),
                    nb=int(len(bl)))
        if N == 120:
            kest = eigengap_k(bpu); lab = partition_at(bpu, kest)
            yi = np.unique(bl, return_inverse=True)[1]
            cell["eigengapK"] = int(kest); cell["ARI_at_K"] = round(float(adjusted_rand_score(yi, lab)), 3)
        row[f"N{N}"] = cell
    return row


@torch.no_grad()
def embed_plain(ckpt, Xt, stft_iq):
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(ckpt, map_location="cuda", weights_only=True), strict=True)
    m.eval()
    f = np.empty((Xt.shape[0], 512), np.float32)
    xt = torch.from_numpy(Xt)
    for i in range(0, Xt.shape[0], 256):
        xb = xt[i:i + 256].cuda(); xs = stft_iq(xb)
        with torch.amp.autocast('cuda'):
            f[i:i + 256] = m.get_encoder_output(xb, xs).float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return f


def main():
    print("=" * 78 + "\nPART 1 — N-MATCHED CONTROLS (mavicAir2 8-way, E1 bursts)\n" + "=" * 78)
    table = {}

    # --- OPT-B path (frozen WiSig-only 512-D + classical-19) ---
    print("[opt-b] building mavicAir2 OPT-B windows (cap %d) ..." % CAP)
    Xt, afd, segd, Dd, Cd = lopo._build_mavicAir2_optb(cap=CAP)
    afd_names = np.array([f"a{a}" for a in afd])
    # frozen WiSig encoder handles its own STFT via GC.extract512
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    F512 = GC.extract512(m, Xt); del m; torch.cuda.empty_cache()
    print("  frozen WiSig-only 512-D ...")
    table["frozen_wisig512_optb"] = score(F512, afd_names, Dd, Cd)
    print("  classical-19 ...")
    CLS = StandardScaler().fit_transform(classical_matrix(Xt)).astype(np.float32)
    table["classical19_optb"] = score(CLS, afd_names, Dd, Cd)

    # --- NATIVE path (native drone-trained seed2024) ---
    print("[native] building mavicAir2 native 4096 windows (cap %d) ..." % CAP)
    _, ev = loaders.drff_airframes()
    units = loaders.load_drff_native(ev, cap=CAP)
    iq, af2, D2, C2 = [], [], [], []
    for u in units:
        for j in range(u["X"].shape[0]):
            iq.append(u["X"][j].astype(np.float32)); af2.append(u["meta"]["airframe"])
            D2.append(u["meta"]["D"]); C2.append(u["meta"]["C"])
    iq = np.stack(iq); af2 = np.array(af2); D2 = np.array(D2); C2 = np.array(C2)
    d = registry.REGISTRY["DRFF"]
    stft_native = lambda xb: registry.stft_mag(xb, d.n_fft, d.hop)
    print("  native drone-trained seed2024 512-D ...")
    Fnat = embed_plain(DRONE_CKPT, iq, stft_native)
    table["native_drone_seed2024"] = score(Fnat, af2, D2, C2)

    # --- generalist step-5000 (cited from Phase 2 report) ---
    gp = json.load(open(os.path.join(RESULTS_GEN, "phase2_lopo_drff_report.json")))
    fe = gp["final_eval"]
    table["generalist_step5000_CITED"] = {
        "N10": dict(km=fe["N10"]["oracle_km"], sp=fe["N10"]["oracle_sp"], hdb=fe["N10"]["hdb_mean"],
                    knn1=fe["N10"]["knn1"], gap=fe["N10"]["gap"], nb=fe["N10"]["n_bursts"]),
        "N120": dict(km=fe["N120"]["oracle_km"], sp=fe["N120"]["oracle_sp"], hdb=fe["N120"]["hdb_mean"],
                     knn1=fe["N120"]["knn1"], gap=fe["N120"]["gap"], nb=fe["N120"]["n_bursts"],
                     eigengapK=fe["N120"]["eigengap_Kest"], ARI_at_K=fe["N120"]["ARI_at_Kest"])}

    # --- print table ---
    print("\n" + "=" * 78)
    print(f"{'method':<26}{'N10 km/sp':>14}{'N120 km/sp':>14}{'N120 hdb':>10}{'N120 knn1':>11}{'eigK/ARI':>12}")
    order = ["frozen_wisig512_optb", "classical19_optb", "native_drone_seed2024", "generalist_step5000_CITED"]
    for k in order:
        r = table[k]
        eig = f"{r['N120'].get('eigengapK','-')}/{r['N120'].get('ARI_at_K','-')}"
        print(f"{k:<26}{r['N10']['km']:>7.3f}/{r['N10']['sp']:<6.3f}"
              f"{r['N120']['km']:>7.3f}/{r['N120']['sp']:<6.3f}{r['N120']['hdb']:>10.3f}"
              f"{r['N120']['knn1']:>11.3f}{eig:>12}")

    # --- VERDICT LINE 1 ---
    fz = table["frozen_wisig512_optb"]["N120"]["km"]
    gen = table["generalist_step5000_CITED"]["N120"]["km"]
    if fz >= gen - 0.05:
        v1 = (f"AVERAGING-DOMINATED / RECIPE FAILED: frozen@N120 km={fz:.3f} >= generalist@N120 "
              f"km={gen:.3f} - 0.05. The generalist added ~nothing beyond burst-averaging; the "
              f"N=120 gain is available to the frozen features too. Focus: why multi-domain "
              f"training HURT at matched N (self-cell + N=10 both below baseline).")
    elif gen >= fz + 0.10:
        v1 = (f"REAL PER-FINGERPRINT GAIN AT SCALE: generalist@N120 km={gen:.3f} >= frozen@N120 "
              f"km={fz:.3f} + 0.10. A genuine representation gain exists at N=120; ablation should "
              f"find which ingredient carries it.")
    else:
        v1 = (f"MARGINAL: frozen@N120 km={fz:.3f} vs generalist@N120 km={gen:.3f} "
              f"(delta {gen-fz:+.3f}, within [-0.05,+0.10]).")
    print("\nVERDICT LINE 1:", v1)

    out = dict(header="PHASE 2b PART 1 — N-matched controls (DEMO-SIDE, R&D)",
               cap=CAP, mcs=MCS_DRFF, note="frozen/classical on OPT-B 256 windows; native/generalist "
               "on native 4096; all E1 bursts, locked harness. Locked frozen wall = 0.297 (cap 320, N=10).",
               table=table, verdict_line_1=v1)
    json.dump(out, open(os.path.join(RESULTS_GEN, "phase2b_part1_controls.json"), "w"), indent=2, default=str)
    with open(os.path.join(RESULTS_GEN, "phase2b_part1_controls.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "N", "km", "sp", "hdb", "knn1", "gap", "nb", "eigengapK", "ARI_at_K"])
        for k in order:
            for N in ("N10", "N120"):
                c = table[k][N]
                w.writerow([k, N, c["km"], c["sp"], c["hdb"], c["knn1"], c["gap"], c["nb"],
                            c.get("eigengapK", ""), c.get("ARI_at_K", "")])
    print(f"\nsaved -> results_gen/phase2b_part1_controls.{{json,csv}}")


if __name__ == "__main__":
    main()
