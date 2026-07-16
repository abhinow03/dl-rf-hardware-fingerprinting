#!/usr/bin/env python3
"""Phase 3a STEP 2 — MIRRORED battery for Approach B / Arm B1 (frozen WiSig 512-D) on BLE D2.

Identical protocol to Phase 2 Stage B (battery_A_ble.py), run ONCE. Only the representation
changes: 512-D frozen-WiSig segment embeddings instead of 19-D classical_b. Two transforms,
mirroring A's two-variant structure:
  - raw   : L2-normalized burst-mean embeddings (the canonical frozen-transfer point; == F1
            frozen@N used in FINDINGS). Replaces A's global_z comparison variant.
  - robust: per-collection robust standardization (median/IQR per dim, unsupervised, IQR-floor
            safeguard identical to A). The deployable receiver-shift mitigation.
T1 / T2 / T3 / T-RX / T5. No per-family ablation (no family structure in embedding space).
Metric = ARI vs true unit labels on burst-mean embeddings. Single-threaded BLAS (caller).
"""
import os, glob, json, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.linalg import eigh

EMB="/home/docker/pw26_akp_01/ext_cache/ble_b1_emb"
RES="/home/docker/pw26_akp_01/ext_cache/ble_B1_results"; os.makedirs(RES,exist_ok=True)
BASE="/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols"
SP=json.load(open(f"{BASE}/splits_ext_ble.json"))
TRAIN_U=SP["train_units"]; VAL_U=SP["val_units"]; EVAL_U=SP["eval_units"]
TRAIN_C=SP["train_collections"]; EVAL_C=SP["eval_collections"]
RX_FIT=SP["receiver_arm"]["fit_collections"]; RX_EVAL=SP["receiver_arm"]["eval_collections"]
SEEDS=[0,1,2,3,4]; CAP=1200; CHANCE=1/len(EVAL_U)
D=512

def key(c): return c.replace(" ","_").replace("(","").replace(")","").replace("/","__")
EMBD={}
for c in TRAIN_C+EVAL_C:
    z=np.load(f"{EMB}/{key(c)}.npz"); EMBD[c]=(z["E"].astype(np.float64), z["y"])

# ---- robust per-collection stats (median/IQR) with IQR floor (== A safeguard) ----
_ALL=np.concatenate([EMBD[c][0] for c in TRAIN_C+EVAL_C])
GLOB_IQR=(np.percentile(_ALL,75,0)-np.percentile(_ALL,25,0))
IQR_FLOOR=0.25*GLOB_IQR+1e-12
def _rob(E):
    med=np.median(E,0); iqr=np.percentile(E,75,0)-np.percentile(E,25,0)
    return med, np.maximum(iqr, IQR_FLOOR)
ROB={c:_rob(EMBD[c][0]) for c in TRAIN_C+EVAL_C}

def burst_embeddings(units,colls,N,seed):
    rng=np.random.default_rng(seed); E=[]; lab=[]; coll_of=[]
    for c in colls:
        F,y=EMBD[c]
        for u in units:
            idx=np.where(y==u)[0]
            if len(idx)<N: continue
            rng.shuffle(idx); ng=len(idx)//N
            for g in range(ng):
                E.append(F[idx[g*N:(g+1)*N]].mean(0)); lab.append(u); coll_of.append(c)
    return np.array(E), np.array(lab), np.array(coll_of,dtype=object)

def _l2(E):
    return E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)

def standardize(E,coll_of,transform,rob_params=None):
    if transform=="raw": return _l2(E)                 # L2-renorm burst mean (frozen-transfer point)
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
    gaps=np.diff(w[:kmax+1]); return int(np.argmax(gaps[1:])+2)

def cluster_metrics(E,lab,seed,K=8,do_extra=False):
    km=KMeans(K,n_init=10,random_state=seed).fit(E); ari_km=ARI(lab,km.labels_)
    out={"ari_kmeans":ari_km}
    try:
        sp=SpectralClustering(K,affinity="nearest_neighbors",n_neighbors=10,
              assign_labels="kmeans",random_state=seed).fit(E); out["ari_spectral"]=ARI(lab,sp.labels_)
    except Exception: out["ari_spectral"]=float("nan")
    if do_extra:
        try: out["estK"]=estimate_K(E)
        except Exception: out["estK"]=-1
        hb=HDBSCAN(min_cluster_size=15,min_samples=5).fit(E)
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

# ===== T1 transferability (linear probe + kNN-1 on eval_units; raw L2) =====
def T1():
    print("\n===== T1 TRANSFERABILITY (linear probe + kNN-1 on eval_units; raw L2) =====")
    rows={}
    for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
        for N in [1,10]:
            E,lab,co=burst_embeddings(EVAL_U,colls,N,seed=0)
            Es=standardize(E,co,"raw")
            rng=np.random.default_rng(0); idx=rng.permutation(len(Es)); cut=len(Es)//2
            tr,te=idx[:cut],idx[cut:]
            lr=LogisticRegression(max_iter=2000).fit(Es[tr],lab[tr]); a_lr=lr.score(Es[te],lab[te])
            kn=KNeighborsClassifier(1).fit(Es[tr],lab[tr]); a_kn=kn.score(Es[te],lab[te])
            rows[f"{split}_N{N}"]={"probe_acc":a_lr,"knn1_acc":a_kn,"n_emb":len(Es)}
            print(f"  {split:10s} N={N:3d}: probe={a_lr:.3f} kNN1={a_kn:.3f} (chance {CHANCE:.3f}) n={len(Es)}")
    RESU["T1"]=rows
T1()

# ===== T3 N-sweep (oracle-K=8 ARI, both transforms, canonical+diagnostic) =====
def T3():
    print("\n===== T3 N-SWEEP (oracle-K=8 ARI) =====")
    res={}
    for transform in ["robust","raw"]:
        for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
            for N in [1,10,30,120]:
                runs=[]
                for sd in SEEDS:
                    E,lab,co=burst_embeddings(EVAL_U,colls,N,seed=sd)
                    if len(E)<8: continue
                    Es=standardize(E,co,transform); Es,lb=subcap(Es,lab,sd)
                    runs.append(cluster_metrics(Es,lb,sd))
                a=agg(runs,["ari_kmeans","ari_spectral"])
                res[f"{transform}/{split}/N{N}"]=a
                print(f"  {transform:8s} {split:10s} N={N:3d}: kmeans {a['ari_kmeans_mean']:.3f}±{a['ari_kmeans_std']:.3f}  "
                      f"spectral {a['ari_spectral_mean']:.3f}±{a['ari_spectral_std']:.3f}")
    RESU["T3"]=res
T3()

# ===== T2 discovery (N=120, both transforms) full metrics =====
def T2():
    print("\n===== T2 DISCOVERY (N=120) =====")
    res={}
    for transform in ["robust","raw"]:
        for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
            runs=[]
            for sd in SEEDS:
                E,lab,co=burst_embeddings(EVAL_U,colls,120,seed=sd)
                Es=standardize(E,co,transform); Es,lb=subcap(Es,lab,sd)
                runs.append(cluster_metrics(Es,lb,sd,do_extra=True))
            a=agg(runs,["ari_kmeans","ari_spectral","estK","estK_correct","hdb_foundK","hdb_ari","hdb_noise"])
            res[f"{transform}/{split}"]=a
            print(f"  {transform:8s} {split:10s}: kmeans {a['ari_kmeans_mean']:.3f}±{a['ari_kmeans_std']:.3f} "
                  f"spectral {a['ari_spectral_mean']:.3f} estK {a['estK_mean']:.1f} correctK {a['estK_correct_mean']:.2f} "
                  f"HDBSCAN foundK {a['hdb_foundK_mean']:.1f} ari {a['hdb_ari_mean']:.3f} noise {a['hdb_noise_mean']:.2f}")
    RESU["T2"]=res
T2()

# ===== T-RX receiver-disjoint (fit robust stats on R1 pool, cluster R2) =====
def TRX():
    print("\n===== T-RX RECEIVER-DISJOINT (fit R1 robust stats -> cluster R2) =====")
    poolR1=np.concatenate([EMBD[c][0] for c in RX_FIT])
    med,iqr=_rob(poolR1)
    res={}
    runs=[]
    for sd in SEEDS:
        E,lab,co=burst_embeddings(EVAL_U,RX_EVAL,120,seed=sd)
        Es=np.empty_like(E)
        for i in range(len(E)): Es[i]=(E[i]-med)/iqr
        Es,lb=subcap(Es,lab,sd)
        runs.append(cluster_metrics(Es,lb,sd))
    res["full"]=agg(runs,["ari_kmeans","ari_spectral"])
    print(f"  {'full(R1->R2)':14s}: kmeans {res['full']['ari_kmeans_mean']:.3f}±{res['full']['ari_kmeans_std']:.3f} spectral {res['full']['ari_spectral_mean']:.3f}")
    # reference: R2's own robust stats (matched)
    runs=[]
    for sd in SEEDS:
        E,lab,co=burst_embeddings(EVAL_U,RX_EVAL,120,seed=sd)
        Es=standardize(E,co,"robust"); Es,lb=subcap(Es,lab,sd)
        runs.append(cluster_metrics(Es,lb,sd))
    res["r2_matched_ref"]=agg(runs,["ari_kmeans","ari_spectral"])
    print(f"  {'r2_matched':14s}: kmeans {res['r2_matched_ref']['ari_kmeans_mean']:.3f} (reference: R2's own stats)")
    RESU["TRX"]=res
TRX()

# ===== T5 compute (GPU + 1-core CPU ms/segment, params, dim, cache) =====
def T5():
    import sys, torch
    _SW=os.path.abspath(os.path.join(BASE,"..","..","summer_work"))
    for p in (_SW,os.path.join(_SW,"datasets")):
        if p not in sys.path: sys.path.insert(0,p)
    from shared import RFEncoder
    from gen_rff.data import registry
    CKPT=os.path.join(_SW,"runs","wisig_supcon_fft64","retrain_best","best_model.pt")
    m=RFEncoder(); m.load_state_dict(torch.load(CKPT,map_location="cpu",weights_only=True),strict=True); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    nparams=sum(p.numel() for p in m.parameters())
    # 1-core CPU timing: one segment = 30 windows through encoder + mean-pool
    torch.set_num_threads(1)
    z=np.load(f"{EMB}/{key(EVAL_C[0])}.npz")  # for cache sizing only
    w=np.random.randn(30,2,256).astype(np.float32)
    xt=torch.from_numpy(w)
    with torch.no_grad():
        xs=registry.stft_mag(xt,64,16); _=m.get_encoder_output(xt,xs)   # warm
        t=time.time()
        for _ in range(20):
            xs=registry.stft_mag(xt,64,16); e=m.get_encoder_output(xt,xs).mean(0)
        cpu_ms=(time.time()-t)/20*1000
    # GPU ms/segment: read from extraction fixity throughput line if present
    gpu_ms=None
    fx=os.path.join(BASE,"audit_out","b1_emb_cache_fixity.md")
    if os.path.exists(fx):
        for ln in open(fx):
            if "ms/segment" in ln and "GPU" in ln:
                try: gpu_ms=float(ln.split("GPU")[1].split("ms")[0].strip())
                except Exception: pass
    cache_mb=sum(os.path.getsize(p) for p in glob.glob(f"{EMB}/*.npz"))/1e6
    RESU["T5"]={"gpu_ms_per_segment":gpu_ms,"cpu1core_ms_per_segment":round(cpu_ms,3),
                "params":int(nparams),"embedding_dim":D,"cache_mb":round(cache_mb,1)}
    print(f"\n===== T5 COMPUTE: GPU {gpu_ms} ms/seg | 1-core CPU {cpu_ms:.3f} ms/seg | params {nparams:,} | dim {D} | cache {cache_mb:.1f} MB =====")
T5()

json.dump(RESU, open(f"{RES}/battery_B1_results.json","w"), indent=2)
print(f"\nWROTE {RES}/battery_B1_results.json")
