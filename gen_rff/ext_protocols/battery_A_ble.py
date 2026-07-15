#!/usr/bin/env python3
"""Phase 2 Stage B — LOCKED battery for Approach A (classical_b) on BLE D2.
Run ONCE. T1/T2/T3/T-RX/T5 + per-family ablation. Single-threaded BLAS enforced by caller.
Metric = ARI vs true unit labels on burst-mean embeddings. Writes results JSON + prints tables."""
import os, glob, json, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.linalg import eigh
from features_ble_classical import extract, FEATURE_NAMES, FAMILY

CACHE="/home/docker/pw26_akp_01/ext_cache/ble_classical_b"
RES="/home/docker/pw26_akp_01/ext_cache/ble_A_results"; os.makedirs(RES,exist_ok=True)
BASE="/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols"
SP=json.load(open(f"{BASE}/splits_ext_ble.json"))
TRAIN_U=SP["train_units"]; VAL_U=SP["val_units"]; EVAL_U=SP["eval_units"]
TRAIN_C=SP["train_collections"]; EVAL_C=SP["eval_collections"]
RX_FIT=SP["receiver_arm"]["fit_collections"]; RX_EVAL=SP["receiver_arm"]["eval_collections"]
SEEDS=[0,1,2,3,4]; CAP=1200; CHANCE=1/len(EVAL_U)

def key(c): return c.replace(" ","_").replace("(","").replace(")","").replace("/","__")
FEAT={}
for c in TRAIN_C+EVAL_C:
    z=np.load(f"{CACHE}/{key(c)}.npz"); FEAT[c]=(z["F"].astype(np.float64), z["y"])

# ---- standardizers ----
def global_z_params():
    pool=[FEAT[c][0][np.isin(FEAT[c][1],TRAIN_U)] for c in TRAIN_C]
    P=np.concatenate(pool); return P.mean(0), P.std(0)+1e-12
MU,SD=global_z_params()
# global-pool IQR (all collections) -> floor for per-collection IQR, guards quantized
# features (e.g. occ_bw is Welch-bin-quantized -> IQR can be 0 within one collection).
_ALL=np.concatenate([FEAT[c][0] for c in TRAIN_C+EVAL_C])
GLOB_IQR=(np.percentile(_ALL,75,0)-np.percentile(_ALL,25,0))
IQR_FLOOR=0.25*GLOB_IQR+1e-12
def _rob(F):
    med=np.median(F,0); iqr=np.percentile(F,75,0)-np.percentile(F,25,0)
    return med, np.maximum(iqr, IQR_FLOOR)
ROB={c:_rob(FEAT[c][0]) for c in TRAIN_C+EVAL_C}

def burst_embeddings(units,colls,N,seed):
    rng=np.random.default_rng(seed); E=[]; lab=[]; coll_of=[]
    for ci,c in enumerate(colls):
        F,y=FEAT[c]
        for u in units:
            idx=np.where(y==u)[0]
            if len(idx)<N: continue
            rng.shuffle(idx); ng=len(idx)//N
            for g in range(ng):
                E.append(F[idx[g*N:(g+1)*N]].mean(0)); lab.append(u); coll_of.append(c)
    return np.array(E), np.array(lab), np.array(coll_of,dtype=object)

def standardize(E,coll_of,transform,rob_params=None):
    if transform=="global_z": return (E-MU)/SD
    rp=rob_params if rob_params is not None else ROB
    out=np.empty_like(E)
    for i,c in enumerate(coll_of):
        med,iqr=rp[c] if c in rp else rp["__fixed__"]; out[i]=(E[i]-med)/iqr
    return out

def subcap(E,lab,seed):
    if len(E)<=CAP: return E,lab
    rng=np.random.default_rng(1000+seed); s=rng.choice(len(E),CAP,replace=False); return E[s],lab[s]

def estimate_K(E,kmax=15):
    A=kneighbors_graph(E,n_neighbors=10,mode="connectivity"); A=0.5*(A+A.T); A=A.toarray()
    d=A.sum(1); Dm=np.diag(1/np.sqrt(d+1e-12)); L=np.eye(len(A))-Dm@A@Dm
    w=eigh(L,eigvals_only=True); w=np.sort(w)[:kmax+1]
    gaps=np.diff(w[:kmax+1]); return int(np.argmax(gaps[1:])+2)  # skip trivial first gap

def cluster_metrics(E,lab,mask,seed,K=8,do_extra=False):
    Em=E[:,mask]
    km=KMeans(K,n_init=10,random_state=seed).fit(Em); ari_km=ARI(lab,km.labels_)
    out={"ari_kmeans":ari_km}
    try:
        sp=SpectralClustering(K,affinity="nearest_neighbors",n_neighbors=10,
              assign_labels="kmeans",random_state=seed).fit(Em); out["ari_spectral"]=ARI(lab,sp.labels_)
    except Exception as e: out["ari_spectral"]=float("nan")
    if do_extra:
        try: out["estK"]=estimate_K(Em)
        except Exception: out["estK"]=-1
        hb=HDBSCAN(min_cluster_size=15,min_samples=5).fit(Em)
        lb=hb.labels_; nz=(lb==-1).mean(); found=len(set(lb[lb!=-1]))
        out.update(estK_correct=int(out["estK"]==K), hdb_foundK=found, hdb_ari=ARI(lab,lb), hdb_noise=float(nz))
    return out

def agg(runs,keys):
    o={}
    for k in keys:
        v=np.array([r[k] for r in runs if k in r and np.isfinite(r[k])],dtype=float)
        o[k+"_mean"]=float(v.mean()) if len(v) else float("nan")
        o[k+"_std"]=float(v.std()) if len(v) else float("nan")
    return o

RESU={}
FULL=np.arange(19)

# ===== T1 transferability =====
def T1():
    print("\n===== T1 TRANSFERABILITY (linear probe + kNN-1 on eval_units) =====")
    rows={}
    for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
        for N in [1,10]:
            E,lab,co=burst_embeddings(EVAL_U,colls,N,seed=0)
            Es=standardize(E,co,"global_z")
            rng=np.random.default_rng(0); idx=rng.permutation(len(Es)); cut=len(Es)//2
            tr,te=idx[:cut],idx[cut:]
            lr=LogisticRegression(max_iter=2000).fit(Es[tr],lab[tr]); a_lr=lr.score(Es[te],lab[te])
            kn=KNeighborsClassifier(1).fit(Es[tr],lab[tr]); a_kn=kn.score(Es[te],lab[te])
            rows[f"{split}_N{N}"]={"probe_acc":a_lr,"knn1_acc":a_kn,"n_emb":len(Es)}
            print(f"  {split:10s} N={N:3d}: probe={a_lr:.3f} kNN1={a_kn:.3f} (chance {CHANCE:.3f}) n={len(Es)}")
    RESU["T1"]=rows
T1()

# ===== T3 N-sweep (oracle-K ARI, both transforms, canonical+diagnostic) =====
def T3():
    print("\n===== T3 N-SWEEP (oracle-K=8 ARI) =====")
    res={}
    for transform in ["robust","global_z"]:
        for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
            for N in [1,10,30,120]:
                runs=[]
                for sd in SEEDS:
                    E,lab,co=burst_embeddings(EVAL_U,colls,N,seed=sd)
                    if len(E)<8: continue
                    Es=standardize(E,co,transform); Es,lb=subcap(Es,lab,sd)
                    runs.append(cluster_metrics(Es,lb,FULL,sd))
                a=agg(runs,["ari_kmeans","ari_spectral"])
                res[f"{transform}/{split}/N{N}"]=a
                print(f"  {transform:8s} {split:10s} N={N:3d}: kmeans {a['ari_kmeans_mean']:.3f}±{a['ari_kmeans_std']:.3f}  "
                      f"spectral {a['ari_spectral_mean']:.3f}±{a['ari_spectral_std']:.3f}")
    RESU["T3"]=res
T3()

# ===== T2 discovery (N=120, primary=robust) full metrics =====
def T2():
    print("\n===== T2 DISCOVERY (N=120, robust transform) =====")
    res={}
    for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
        runs=[]
        for sd in SEEDS:
            E,lab,co=burst_embeddings(EVAL_U,colls,120,seed=sd)
            Es=standardize(E,co,"robust"); Es,lb=subcap(Es,lab,sd)
            runs.append(cluster_metrics(Es,lb,FULL,sd,do_extra=True))
        a=agg(runs,["ari_kmeans","ari_spectral","estK","estK_correct","hdb_foundK","hdb_ari","hdb_noise"])
        res[split]=a
        print(f"  {split:10s}: kmeans {a['ari_kmeans_mean']:.3f}±{a['ari_kmeans_std']:.3f} "
              f"spectral {a['ari_spectral_mean']:.3f} estK {a['estK_mean']:.1f} correctK {a['estK_correct_mean']:.2f} "
              f"HDBSCAN foundK {a['hdb_foundK_mean']:.1f} ari {a['hdb_ari_mean']:.3f} noise {a['hdb_noise_mean']:.2f}")
    RESU["T2"]=res
T2()

# ===== T-RX receiver-disjoint (fit stats on R1, cluster R2) + CFO ablation =====
def TRX():
    print("\n===== T-RX RECEIVER-DISJOINT (fit R1 robust stats -> cluster R2) =====")
    # robust stats from R1 pooled (both R1 collections), one shared param set
    poolR1=np.concatenate([FEAT[c][0] for c in RX_FIT])
    med,iqr=_rob(poolR1)                         # same IQR-floor safeguard
    rp={"__fixed__":(med,iqr)}
    cfo_mask=np.array([i for i in range(19) if i not in FAMILY["F-CFO"]])
    res={}
    for name,mask in [("full",FULL),("cfo_ablated",cfo_mask)]:
        runs=[]
        for sd in SEEDS:
            E,lab,co=burst_embeddings(EVAL_U,RX_EVAL,120,seed=sd)
            Es=np.empty_like(E)
            for i in range(len(E)): Es[i]=(E[i]-med)/iqr
            Es,lb=subcap(Es,lab,sd)
            runs.append(cluster_metrics(Es,lb,mask,sd))
        a=agg(runs,["ari_kmeans","ari_spectral"])
        res[name]=a
        print(f"  {name:12s}: kmeans {a['ari_kmeans_mean']:.3f}±{a['ari_kmeans_std']:.3f} spectral {a['ari_spectral_mean']:.3f}")
    # reference: same R2 eval but with R2's own robust stats (matched)
    runs=[]
    for sd in SEEDS:
        E,lab,co=burst_embeddings(EVAL_U,RX_EVAL,120,seed=sd)
        Es=standardize(E,co,"robust"); Es,lb=subcap(Es,lab,sd)
        runs.append(cluster_metrics(Es,lb,FULL,sd))
    res["r2_matched_ref"]=agg(runs,["ari_kmeans","ari_spectral"])
    print(f"  {'r2_matched':12s}: kmeans {res['r2_matched_ref']['ari_kmeans_mean']:.3f} (reference: R2's own stats)")
    RESU["TRX"]=res
TRX()

# ===== per-family ablation (canonical N=120 oracle-K) =====
def ABL():
    print("\n===== PER-FAMILY ABLATION (canonical N=120, robust, oracle-K ARI kmeans) =====")
    res={}
    base=[]
    for sd in SEEDS:
        E,lab,co=burst_embeddings(EVAL_U,EVAL_C,120,seed=sd); Es=standardize(E,co,"robust"); Es,lb=subcap(Es,lab,sd)
        base.append(cluster_metrics(Es,lb,FULL,sd)["ari_kmeans"])
    base_m=float(np.mean(base)); res["FULL"]=base_m; print(f"  FULL(19)      ari={base_m:.3f}")
    for fam,idx in FAMILY.items():
        mask=np.array([i for i in range(19) if i not in idx]); runs=[]
        for sd in SEEDS:
            E,lab,co=burst_embeddings(EVAL_U,EVAL_C,120,seed=sd); Es=standardize(E,co,"robust"); Es,lb=subcap(Es,lab,sd)
            runs.append(cluster_metrics(Es,lb,mask,sd)["ari_kmeans"])
        m=float(np.mean(runs)); res[fam]=m
        print(f"  drop {fam:7s}   ari={m:.3f}  (Δ={m-base_m:+.3f})")
    RESU["ablation"]=res
ABL()

# ===== T5 compute =====
def T5():
    import os as _os
    X=np.asarray(np.load("/home/docker/pw26_akp_01/ext_data/ble_xiao/Wired (indoors)/Ch1_R1/X_train.npy",mmap_mode="r")[:1000])
    t=time.time(); _=extract(X); ms=(time.time()-t)/1000*1000
    cache_mb=sum(_os.path.getsize(p) for p in glob.glob(f"{CACHE}/*.npz"))/1e6
    RESU["T5"]={"ms_per_segment_1core":round(ms,4),"n_features":19,"cache_mb":round(cache_mb,1)}
    print(f"\n===== T5 COMPUTE: {ms:.3f} ms/segment (1 core, incl. STFT+PSD), 19 feats, cache {cache_mb:.1f} MB =====")
T5()

json.dump(RESU, open(f"{RES}/battery_A_results.json","w"), indent=2)
print(f"\nWROTE {RES}/battery_A_results.json")
