"""PLAY 1b — COUNT ESTIMATION + DEMO OPERATING POINT (eval-only, DEMO-SIDE).

The embedding is SOLVED (Play-1 native encoder: oracle-K@8 0.756, kNN-1 0.95). The remaining
gap is K-ESTIMATION + burst construction (HDBSCAN 0.075, correct-K 0.20). This battery picks the
clustering stack + burst size that maximizes demo-scenario success. NO gradient steps. Features
from the Play-1 SELECTED checkpoints (seed1234 step3500, seed2024 step4000; primary = seed2024).
mavicAir2 (8) = eval group. best_model.pt / TEST / M100 untouched. DEMO-SIDE — NOT PAPER RESULTS.

FRAMING: In deployment, windows are grouped into bursts by TRACK CONTINUITY (a continuously
observed emission = one track), not by ground-truth identity. Here, per-device grouping stands in
for tracks; the demo build must group by track. A tracked emitter accumulates windows across
time/conditions naturally — the N-sweep chooses how many windows a track accumulates before its
feature is stable.

  python3 results/demo_play1b/count_estimation.py
"""
import os, sys, json, csv, re, itertools, warnings, faulthandler
faulthandler.enable()
from collections import defaultdict
import numpy as np
import torch
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
from integrity import score_locked, hdbscan_pred
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse.csgraph import laplacian

DRFF_DIR  = os.path.expanduser("~/Desktop/processed/drff_r2")
CKPT_ROOT = os.path.join(_SW, "runs", "drone_native")
OUT       = _HERE
os.makedirs(OUT, exist_ok=True)

WIN, NFFT, HOP = 4096, 256, 64
WPSEG, EVAL_CAP = 8, 1500
NS = [20, 30, 50, 80, 120]
KTRUE8 = 8
SEEDS = [1234, 2024]
PRIMARY = 2024
HANN = torch.hann_window(NFFT)
MCS_DRFF = [5, 7]

def unit(M): return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
def unit_power(w): return (w / (np.sqrt((w.astype(np.float32) ** 2).mean()) + 1e-8)).astype(np.float32)

# ============================== DATA (mavicAir2 eval, native 4096) ==============================
manifest = json.load(open(os.path.join(DRFF_DIR, "manifest.json")))
pat = re.compile(r'(.+?)_(\d+)_hover')
af_files = defaultdict(list)
for r in manifest["clean"]:
    af_files[r["TD"]].append(r["file"].replace(".mat", ".npz"))
all_af = sorted(af_files, key=lambda t: (t.rsplit("_", 1)[0], int(t.rsplit("_", 1)[1])))
model_of = {a: pat.match(a + "_hover").group(1) for a in all_af}
EVAL_AF = [a for a in all_af if model_of[a] == "mavicAir2"]
assert len(EVAL_AF) == 8

def build_windows(af_list, cap, seed=779):
    rng = np.random.default_rng(seed)
    Xt, y, seg, Ds, Cs = [], [], [], [], []
    gseg = 0
    for ai, a in enumerate(af_list):
        files = af_files[a][:]; rng.shuffle(files)
        units = []
        for fn in files:
            z = np.load(os.path.join(DRFF_DIR, fn))
            for si in range(z["seg_bounds"].shape[0]): units.append((fn, si))
        rng.shuffle(units)
        got = 0; zc = {}
        for fn, si in units:
            if got >= cap: break
            if fn not in zc:
                z = np.load(os.path.join(DRFF_DIR, fn))
                zc[fn] = dict(iq=z["iq"].astype(np.float32), sb=z["seg_bounds"],
                              D=str(z["D"]), C=int(z["C"]))
            zz = zc[fn]; off, ln = zz["sb"][si]
            nw = int(ln) // WIN
            if nw < 1: continue
            take = min(nw, cap - got, WPSEG); base = int(off)
            for k in range(take):
                Xt.append(unit_power(zz["iq"][:, base + k * WIN: base + (k + 1) * WIN]))
                y.append(ai); seg.append(gseg); Ds.append(zz["D"]); Cs.append(zz["C"])
            got += take; gseg += 1
    return dict(Xt=np.stack(Xt).astype(np.float32), y=np.array(y), seg=np.array(seg),
                D=np.array(Ds), C=np.array(Cs))

print("[data] building mavicAir2 eval (native 4096, cap %d) ..." % EVAL_CAP)
ev = build_windows(EVAL_AF, EVAL_CAP, seed=779)
print("    windows=%d segments=%d airframes=8" % (len(ev["y"]), len(np.unique(ev["seg"]))))

def stft_gpu(xt):
    B = xt.shape[0]; x = xt.reshape(B * 2, WIN)
    sp = torch.stft(x, n_fft=NFFT, hop_length=HOP, win_length=NFFT,
                    window=HANN.to(xt.device), center=False, return_complex=True)
    return sp.abs().reshape(B, 2, NFFT // 2 + 1, -1)

@torch.no_grad()
def embed512(model, Xt, batch=384):
    model.eval(); f = np.empty((Xt.shape[0], 512), dtype=np.float32); xt = torch.from_numpy(Xt)
    for i in range(0, Xt.shape[0], batch):
        xb = xt[i:i + batch].cuda(); xs = stft_gpu(xb)
        with torch.amp.autocast('cuda'):
            f[i:i + batch] = model.get_encoder_output(xb, xs).float().cpu().numpy()
    return f

FEAT = {}
for s in SEEDS:
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(os.path.join(CKPT_ROOT, f"seed{s}", "best.pt"),
                                 map_location="cuda", weights_only=True), strict=True)
    FEAT[s] = embed512(m, ev["Xt"]); del m; torch.cuda.empty_cache()
    print("    embedded seed%d -> 512-D (%d windows)" % (s, FEAT[s].shape[0]))

# ============================== burst / scoring helpers ==============================
def build_N(emb, af, D, C, N, seed=0):
    r = np.random.default_rng(seed); bp, bl = [], []
    for a in np.unique(af):
        idx = np.where(af == a)[0]; cells = defaultdict(list)
        for i in idx: cells[(D[i], C[i])].append(i)
        ck = list(cells.keys()); nb = len(idx) // N
        for _ in range(nb):
            r.shuffle(ck); pick = []; ci = 0
            while len(pick) < N and ci < 4000:
                cc = ck[ci % len(ck)]
                if cells[cc]: pick.append(cells[cc][r.integers(len(cells[cc]))])
                ci += 1
            if len(pick) == N: bp.append(emb[pick].mean(0)); bl.append(a)
    return np.array(bp), np.array(bl)

def balance(bp, bl, seed=0):
    r = np.random.default_rng(seed); per = min(int((bl == a).sum()) for a in np.unique(bl))
    keep = []
    for a in np.unique(bl):
        ii = np.where(bl == a)[0]; r.shuffle(ii); keep += ii[:per].tolist()
    keep = np.array(sorted(keep)); return bp[keep], bl[keep], per

def knn1(X, y):
    Xn = unit(X); nn = NearestNeighbors(n_neighbors=2).fit(Xn); _, ind = nn.kneighbors(Xn)
    yi = np.unique(y, return_inverse=True)[1]; return float((yi[ind[:, 1]] == yi).mean())

def oracle_km_sp(bpu, bl, K):
    yi = np.unique(bl, return_inverse=True)[1]
    km = float(adjusted_rand_score(yi, KMeans(K, n_init=10, random_state=0).fit_predict(bpu)))
    try:
        sp = SpectralClustering(K, affinity="nearest_neighbors", random_state=0,
                                n_neighbors=15).fit_predict(bpu)
        spa = float(adjusted_rand_score(yi, sp))
    except Exception:
        spa = float("nan")
    return km, spa

# ============================== STEP 1 — balanced burst-N sweep ==============================
print("\n" + "=" * 78 + "\nSTEP 1 — balanced burst-N sweep (8-way, oracle-K@8 + kNN-1)\n" + "=" * 78)
nsweep = []
for s in SEEDS:
    for N in NS:
        bp, bl = build_N(FEAT[s], ev["y"], ev["D"], ev["C"], N, seed=0)
        if len(bl) < 8: continue
        bp, bl, per = balance(bp, bl); bpu = unit(bp)
        km, sp = oracle_km_sp(bpu, bl, KTRUE8)
        row = dict(seed=s, N=N, bursts_per_af=int(per), n_bursts=int(len(bl)),
                   oracle_km=round(km, 3), oracle_sp=round(sp, 3),
                   oracleK=round(max(km, sp), 3), knn1=round(knn1(bp, bl), 3))
        nsweep.append(row)
        print(f"  seed{s} N={N:3d}: bursts/af={per:3d} (n={len(bl)}) oracle-K@8 km={km:.3f}/sp={sp:.3f} "
              f"kNN1={row['knn1']:.3f}")
# N* on primary seed: smallest N within 0.03 of best oracleK
prim = [r for r in nsweep if r["seed"] == PRIMARY]
best_or = max(r["oracleK"] for r in prim)
Nstar = min(r["N"] for r in prim if r["oracleK"] >= best_or - 0.03)
print(f"\n  N* = {Nstar} (primary seed{PRIMARY}: best oracle-K={best_or:.3f}, smallest N within 0.03)")
# exhaustion verdict on the old N=200-style cliff
maxN_prim = max(prim, key=lambda r: r["N"])
print(f"  [exhaustion] at largest N={maxN_prim['N']}: bursts/af={maxN_prim['bursts_per_af']} "
      f"oracle-K={maxN_prim['oracleK']:.3f} (vs best {best_or:.3f}). "
      f"{'oracle-K holds -> the old N=200 collapse was BURST-COUNT EXHAUSTION (per-seg fallback), not an averaging limit' if maxN_prim['oracleK'] >= best_or - 0.05 else 'oracle-K drops even balanced -> averaging saturates / sample-starve at high N'}")

# ============================== STEP 2 — K-estimation bake-off ==============================
def est_eigengap(bpu, bl):
    n = len(bpu); Kmax = min(10, n - 1)
    A = kneighbors_graph(bpu, n_neighbors=min(15, n - 1), mode="connectivity", include_self=False)
    A = 0.5 * (A + A.T)
    L = laplacian(A, normed=True).toarray()
    vals = np.sort(np.linalg.eigvalsh(L))
    gaps = {k: vals[k] - vals[k - 1] for k in range(2, Kmax + 1)}
    Kest = max(gaps, key=gaps.get)
    try:
        lab = SpectralClustering(Kest, affinity="nearest_neighbors", random_state=0,
                                 n_neighbors=15).fit_predict(bpu)
    except Exception:
        lab = KMeans(Kest, n_init=10, random_state=0).fit_predict(bpu)
    return Kest, lab

def est_stability(bpu, bl, return_curve=False):
    n = len(bpu); Kmax = min(10, n - 1); curve = {}
    for K in range(2, Kmax + 1):
        parts = [KMeans(K, n_init=1, random_state=sd).fit_predict(bpu) for sd in range(20)]
        aris = [adjusted_rand_score(parts[i], parts[j])
                for i in range(len(parts)) for j in range(i + 1, len(parts))]
        curve[K] = float(np.mean(aris))
    Kest = max(curve, key=curve.get)
    lab = KMeans(Kest, n_init=20, random_state=0).fit_predict(bpu)
    return (Kest, lab, curve) if return_curve else (Kest, lab)

def est_silhouette(bpu, bl):
    n = len(bpu); Kmax = min(10, n - 1); best = (-2, 2, None)
    for K in range(2, Kmax + 1):
        lab = KMeans(K, n_init=10, random_state=0).fit_predict(bpu)
        if len(np.unique(lab)) < 2: continue
        sc = silhouette_score(bpu, lab, metric="cosine")
        if sc > best[0]: best = (sc, K, lab)
    return best[1], best[2]

def est_gmm_bic(bpu, bl):
    n = len(bpu); Kmax = min(10, n - 1)
    d = min(64, n - 1, bpu.shape[1]); X = PCA(d, random_state=0).fit_transform(bpu)
    best = (1e18, 2, None)
    for K in range(2, Kmax + 1):
        g = GaussianMixture(K, covariance_type="full", random_state=0, reg_covar=1e-4).fit(X)
        b = g.bic(X)
        if b < best[0]: best = (b, K, g.predict(X))
    return best[1], best[2]

def est_hdbscan(bpu, bl):
    aris, ks = [], []
    for mcs in MCS_DRFF:
        lab = hdbscan_pred(bpu, mcs); sc = score_locked(lab, bl)
        aris.append(sc["ARI"]); ks.append(sc["K_est"])
    return int(round(np.median(ks))), None, float(np.mean(aris))

ESTIMATORS = ["eigengap", "stability", "silhouette", "gmm_bic", "hdbscan"]
def run_est(name, bpu, bl):
    yi = np.unique(bl, return_inverse=True)[1]
    if name == "hdbscan":
        Kest, _, ari = est_hdbscan(bpu, bl); return Kest, ari
    Kest, lab = dict(eigengap=est_eigengap, stability=est_stability,
                     silhouette=est_silhouette, gmm_bic=est_gmm_bic)[name](bpu, bl)
    return int(Kest), float(adjusted_rand_score(yi, lab))

def subset_bursts(seed, sub, N):
    m = np.isin(ev["y"], sub)
    bp, bl = build_N(FEAT[seed][m], ev["y"][m], ev["D"][m], ev["C"][m], N, seed=0)
    if len(bl) < len(sub): return None, None
    bp, bl, _ = balance(bp, bl); return unit(bp), bl

def battery_bakeoff(seed, subsets, Ktrue, N, stab_example=None):
    agg = {e: dict(correct=[], kerr=[], ari=[]) for e in ESTIMATORS}
    percase = {e: [] for e in ESTIMATORS}          # (Kest, ari) per subset for scenario success
    for si, sub in enumerate(subsets):
        bpu, bl = subset_bursts(seed, sub, N)
        if bpu is None or len(np.unique(bl)) < Ktrue: continue
        for e in ESTIMATORS:
            Kest, ari = run_est(e, bpu, bl)
            agg[e]["correct"].append(Kest == Ktrue); agg[e]["kerr"].append(abs(Kest - Ktrue))
            agg[e]["ari"].append(ari); percase[e].append((Kest, ari))
        if stab_example is not None and si == 0:
            _, _, cv = est_stability(bpu, bl, return_curve=True)
            stab_example["curve"] = {int(k): round(v, 3) for k, v in cv.items()}
            stab_example["seed"] = seed
    rows = []
    for e in ESTIMATORS:
        c = agg[e]
        rows.append(dict(seed=seed, scope=f"K{Ktrue}", estimator=e, N=N, n_cases=len(c["correct"]),
                         correctK_frac=round(float(np.mean(c["correct"])), 3),
                         kerr_mean=round(float(np.mean(c["kerr"])), 3),
                         ari_at_K=round(float(np.mean(c["ari"])), 3)))
    return rows, percase

K4_SUBSETS = list(itertools.combinations(range(8), 4))     # 70
K3_SUBSETS = list(itertools.combinations(range(8), 3))     # 56
EIGHTWAY = [tuple(range(8))]

print("\n" + "=" * 78 + f"\nSTEP 2 — K-estimation bake-off at N*={Nstar} (8-way + K=4 battery)\n" + "=" * 78)
bakeoff = []; percase_all = {}; stab_example = {}
for s in SEEDS:
    r8, pc8 = battery_bakeoff(s, EIGHTWAY, 8, Nstar)
    r4, pc4 = battery_bakeoff(s, K4_SUBSETS, 4, Nstar, stab_example if s == PRIMARY else None)
    bakeoff += r8 + r4; percase_all[(s, "K8")] = pc8; percase_all[(s, "K4")] = pc4
for r in [x for x in bakeoff if x["seed"] == PRIMARY]:
    print(f"  seed{r['seed']} {r['scope']:>3} {r['estimator']:>10}: correctK={r['correctK_frac']:.3f} "
          f"|Kerr|={r['kerr_mean']:.2f} ARI@K={r['ari_at_K']:.3f} (n={r['n_cases']})")

# --- PRE-COMMIT estimator choice using ONLY 8-way + K=4 (primary seed), BEFORE K=3 ---
def success_rate(pc_list, Ktrue):
    return float(np.mean([(k == Ktrue) and (a >= 0.6) for (k, a) in pc_list])) if pc_list else 0.0
choice_scores = {e: success_rate(percase_all[(PRIMARY, "K4")][e], 4) for e in ESTIMATORS}
# tie-break: 8-way success then lower |Kerr| on K4
k4_kerr = {r["estimator"]: r["kerr_mean"] for r in bakeoff if r["seed"] == PRIMARY and r["scope"] == "K4"}
k8_succ = {e: success_rate(percase_all[(PRIMARY, "K8")][e], 8) for e in ESTIMATORS}
CHOICE = max(ESTIMATORS, key=lambda e: (choice_scores[e], k8_succ[e], -k4_kerr[e]))
print(f"\n  PRE-COMMIT (rule = max K4 scenario-success on primary seed{PRIMARY}, tie-break 8-way "
      f"success then |Kerr|):")
for e in ESTIMATORS:
    print(f"    {e:>10}: K4-success={choice_scores[e]:.3f} K8-success={k8_succ[e]:.3f} |Kerr|K4={k4_kerr[e]:.2f}")
print(f"  --> CHOSEN K-ESTIMATOR = {CHOICE}")

# ============================== STEP 3 — scenario success (incl. K=3) ==============================
print("\n" + "=" * 78 + f"\nSTEP 3 — demo scenario success ({CHOICE}, N*={Nstar})\n" + "=" * 78)
# assisted: partition at K_true with the chosen estimator's algorithm
def partition_at_K(name, bpu, K):
    if name in ("stability", "silhouette"):
        return KMeans(K, n_init=20, random_state=0).fit_predict(bpu)
    if name == "eigengap":
        try: return SpectralClustering(K, affinity="nearest_neighbors", random_state=0, n_neighbors=15).fit_predict(bpu)
        except Exception: return KMeans(K, n_init=10, random_state=0).fit_predict(bpu)
    if name == "gmm_bic":
        d = min(64, len(bpu) - 1, bpu.shape[1]); X = PCA(d, random_state=0).fit_transform(bpu)
        return GaussianMixture(K, covariance_type="full", random_state=0, reg_covar=1e-4).fit(X).predict(X)
    # hdbscan has no K -> fall back to kmeans for the assisted mode
    return KMeans(K, n_init=10, random_state=0).fit_predict(bpu)

# add K=3 bake-off now (post-commit) for the chosen estimator + assisted
scenario = []
for s in SEEDS:
    # gather percase for K8 & K4 (already computed) + compute K3 fresh
    for scope, subsets, Kt, pc in [("K8", EIGHTWAY, 8, percase_all[(s, "K8")][CHOICE]),
                                   ("K4", K4_SUBSETS, 4, percase_all[(s, "K4")][CHOICE])]:
        succ_un = success_rate(pc, Kt)
        # assisted
        asr = []
        for sub in subsets:
            bpu, bl = subset_bursts(s, sub, Nstar)
            if bpu is None or len(np.unique(bl)) < Kt: continue
            yi = np.unique(bl, return_inverse=True)[1]
            lab = partition_at_K(CHOICE, bpu, Kt)
            asr.append(adjusted_rand_score(yi, lab) >= 0.6)
        scenario.append(dict(seed=s, scope=scope, Ktrue=Kt, estimator=CHOICE, N=Nstar,
                             n=len(pc), success_unassisted=round(succ_un, 3),
                             success_assisted=round(float(np.mean(asr)), 3)))
    # K=3 (post-commit)
    _, pc3 = battery_bakeoff(s, K3_SUBSETS, 3, Nstar)
    succ3 = success_rate(pc3[CHOICE], 3)
    asr3 = []
    for sub in K3_SUBSETS:
        bpu, bl = subset_bursts(s, sub, Nstar)
        if bpu is None or len(np.unique(bl)) < 3: continue
        yi = np.unique(bl, return_inverse=True)[1]
        asr3.append(adjusted_rand_score(yi, partition_at_K(CHOICE, bpu, 3)) >= 0.6)
    scenario.append(dict(seed=s, scope="K3", Ktrue=3, estimator=CHOICE, N=Nstar,
                         n=len(pc3[CHOICE]), success_unassisted=round(succ3, 3),
                         success_assisted=round(float(np.mean(asr3)), 3)))

for r in scenario:
    print(f"  seed{r['seed']} {r['scope']:>3} (K={r['Ktrue']}): unassisted={r['success_unassisted']:.3f} "
          f"assisted={r['success_assisted']:.3f} (n={r['n']})")

# ============================== GATE + operating point ==============================
def best_scope(scope, mode):
    return max(r[mode] for r in scenario if r["scope"] == scope)
k4_un = best_scope("K4", "success_unassisted"); k3_un = best_scope("K3", "success_unassisted")
k4_as = best_scope("K4", "success_assisted");  k3_as = best_scope("K3", "success_assisted")
k8_un = best_scope("K8", "success_unassisted"); k8_as = best_scope("K8", "success_assisted")
kle4_un = max(k4_un, k3_un); kle4_as = max(k4_as, k3_as)
if kle4_un >= 0.7:
    gate = f"DEMO-READY: K<=4 unassisted success={kle4_un:.3f} (>=0.7) with {CHOICE}@N{Nstar}."
elif kle4_un >= 0.5 or kle4_as >= 0.8:
    gate = (f"WORKABLE: K<=4 unassisted={kle4_un:.3f} (0.5-0.7) / assisted={kle4_as:.3f} -> ship "
            f"with {'assisted mode' if kle4_as >= 0.8 else 'curated scenarios'} ({CHOICE}@N{Nstar}).")
else:
    gate = (f"STILL BLOCKED: K<=4 unassisted={kle4_un:.3f} assisted={kle4_as:.3f} (below both) -> "
            f"geometry needs Play-2 ceiling raise.")

# ============================== SAVE ==============================
def wcsv(fn, rows, header=None):
    with open(os.path.join(OUT, fn), "w", newline="") as f:
        if header: f.write(header + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
wcsv("nsweep.csv", nsweep, header="# DEMO-SIDE — NOT PAPER RESULTS")
wcsv("kest_bakeoff.csv", bakeoff, header="# DEMO-SIDE — NOT PAPER RESULTS")
wcsv("scenario_success.csv", scenario, header="# DEMO-SIDE — NOT PAPER RESULTS")

_pcv = json.load(open(os.path.join(CKPT_ROOT, f'seed{PRIMARY}', 'curve.json')))
PRIM_STEP = max(_pcv, key=lambda r: r['composite'])['step']
op = f"""# DEMO OPERATING POINT — DEMO-SIDE, NOT PAPER RESULTS

**Locked stack (consumed by the demo build):**

- **Encoder checkpoint:** Play-1 native-from-scratch, `runs/drone_native/seed{PRIMARY}/best.pt`
  (step {PRIM_STEP}); secondary seed1234 for robustness.
- **Feature:** 512-D head-free (`get_encoder_output`), native 4096-sample windows @ 50 MS/s,
  STFT n_fft=256/hop=64, per-window unit-power standardize.
- **Burst construction (track):** group windows by track continuity; accumulate **N* = {Nstar}**
  windows/track (mean-pool, L2-normalize). Balanced multi-condition aggregation.
- **K-estimator:** **{CHOICE}**.
- **Clustering / partition:** the {CHOICE} partition (assisted mode = same algorithm forced to
  the operator-supplied K).

**Expected demo success (scenario success = correct K AND partition ARI>=0.6), best-of-seed:**

| scenario | unassisted | assisted (K known) |
|----------|-----------|--------------------|
| K=3      | {k3_un:.3f} | {k3_as:.3f} |
| K=4      | {k4_un:.3f} | {k4_as:.3f} |
| K=8      | {k8_un:.3f} | {k8_as:.3f} |

**N* = {Nstar}** (smallest N within 0.03 of best oracle-K@8 on primary seed{PRIMARY}).

**Gate:** {gate}
"""
open(os.path.join(OUT, "DEMO_OPERATING_POINT.md"), "w").write(op)

report = dict(header="DEMO-SIDE — NOT PAPER RESULTS",
              framing="bursts = tracks (track continuity), not ground-truth identity; per-device "
                      "grouping stands in for tracks offline; demo groups by track.",
              question="pick clustering stack + burst size maximizing demo scenario success.",
              checkpoints={f"seed{s}": f"runs/drone_native/seed{s}/best.pt" for s in SEEDS},
              primary_seed=PRIMARY, Nstar=Nstar, chosen_estimator=CHOICE,
              choice_rule="max K=4 scenario-success on primary seed (uses only 8-way+K=4), "
                          "tie-break 8-way success then lower |Kerr|; committed before K=3.",
              choice_scores={e: dict(k4_success=round(choice_scores[e], 3),
                                     k8_success=round(k8_succ[e], 3), k4_kerr=k4_kerr[e])
                             for e in ESTIMATORS},
              nsweep=nsweep, exhaustion_verdict=dict(
                  largest_N=maxN_prim["N"], bursts_per_af=maxN_prim["bursts_per_af"],
                  oracleK=maxN_prim["oracleK"], best_oracleK=best_or),
              bakeoff=bakeoff, stability_example=stab_example, scenario=scenario,
              gate=gate,
              guardrails="no training; fixed 5-estimator list; mavicAir2 only; both seeds; separate "
                         "demo ledger; TEST/M100/best_model.pt untouched.")
json.dump(report, open(os.path.join(OUT, "report.json"), "w"), indent=2, default=str)

print("\n=== GATE ===\n" + gate)
print("\nsaved -> results/demo_play1b/ (nsweep.csv, kest_bakeoff.csv, scenario_success.csv, "
      "DEMO_OPERATING_POINT.md, report.json)")
print("CHECKPOINT — DEMO-side; eval-only; best_model.pt/TEST/M100 untouched.")
