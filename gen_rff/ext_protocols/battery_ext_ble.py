#!/usr/bin/env python3
"""Phase 3c STEP 2 — MIRRORED battery for a learned-embedding arm (B3 native, or reused for any
per-collection embedding cache). Identical protocol to Phases 2/3a. Two transforms: raw (L2-renorm
burst mean) and per-collection robust (median/IQR, IQR-floor). T1/T2/T3/T-RX/T5. 5 clustering seeds.

Usage: battery_ext_ble.py --emb <emb_dir> --out <results.json> [--t2t3only]
  --t2t3only : run just T3 (which yields N=120 canonical+diagnostic for T4) — used for data-eff arms.
Single-threaded BLAS (caller). Metric = ARI vs true unit labels on burst-mean embeddings."""
import os, glob, json, time, argparse
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
from sklearn.neighbors import KNeighborsClassifier, kneighbors_graph
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score as ARI
from scipy.linalg import eigh

BASE="/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols"
SP=json.load(open(f"{BASE}/splits_ext_ble.json"))
TRAIN_U=SP["train_units"]; VAL_U=SP["val_units"]; EVAL_U=SP["eval_units"]
TRAIN_C=SP["train_collections"]; EVAL_C=SP["eval_collections"]
RX_FIT=SP["receiver_arm"]["fit_collections"]; RX_EVAL=SP["receiver_arm"]["eval_collections"]
SEEDS=[0,1,2,3,4]; CAP=1200; CHANCE=1/len(EVAL_U)

def key(c): return c.replace(" ","_").replace("(","").replace(")","").replace("/","__")

def run(EMB_DIR,OUT,t2t3only=False):
    EMBD={}
    for c in TRAIN_C+EVAL_C:
        z=np.load(f"{EMB_DIR}/{key(c)}.npz"); EMBD[c]=(z["E"].astype(np.float64), z["y"])
    _ALL=np.concatenate([EMBD[c][0] for c in TRAIN_C+EVAL_C])
    GLOB_IQR=np.percentile(_ALL,75,0)-np.percentile(_ALL,25,0); IQR_FLOOR=0.25*GLOB_IQR+1e-12
    def _rob(E):
        med=np.median(E,0); iqr=np.percentile(E,75,0)-np.percentile(E,25,0); return med,np.maximum(iqr,IQR_FLOOR)
    ROB={c:_rob(EMBD[c][0]) for c in TRAIN_C+EVAL_C}

    def bursts(units,colls,N,seed):
        rng=np.random.default_rng(seed); E=[];lab=[];co=[]
        for c in colls:
            F,y=EMBD[c]
            for u in units:
                idx=np.where(y==u)[0]
                if len(idx)<N: continue
                rng.shuffle(idx); ng=len(idx)//N
                for g in range(ng): E.append(F[idx[g*N:(g+1)*N]].mean(0)); lab.append(u); co.append(c)
        return np.array(E),np.array(lab),np.array(co,dtype=object)
    def _l2(E): return E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
    def std(E,co,tf,rp=None):
        if tf=="raw": return _l2(E)
        rp=rp or ROB; out=np.empty_like(E)
        for i,c in enumerate(co): m,q=rp[c] if c in rp else rp["__fixed__"]; out[i]=(E[i]-m)/q
        return out
    def subcap(E,lab,seed):
        if len(E)<=CAP: return E,lab
        rng=np.random.default_rng(1000+seed); s=rng.choice(len(E),CAP,replace=False); return E[s],lab[s]
    def estK(E,kmax=15):
        A=kneighbors_graph(E,10,mode="connectivity"); A=0.5*(A+A.T); A=A.toarray()
        d=A.sum(1); Dm=np.diag(1/np.sqrt(d+1e-12)); L=np.eye(len(A))-Dm@A@Dm
        w=np.sort(eigh(L,eigvals_only=True))[:kmax+1]; return int(np.argmax(np.diff(w)[1:])+2)
    def cm(E,lab,seed,K=8,extra=False):
        o={"ari_kmeans":ARI(lab,KMeans(K,n_init=10,random_state=seed).fit(E).labels_)}
        try: o["ari_spectral"]=ARI(lab,SpectralClustering(K,affinity="nearest_neighbors",n_neighbors=10,assign_labels="kmeans",random_state=seed).fit(E).labels_)
        except Exception: o["ari_spectral"]=float("nan")
        if extra:
            try: o["estK"]=estK(E)
            except Exception: o["estK"]=-1
            hb=HDBSCAN(min_cluster_size=15,min_samples=5).fit(E); lb=hb.labels_
            o.update(estK_correct=int(o["estK"]==K),hdb_foundK=len(set(lb[lb!=-1])),hdb_ari=ARI(lab,lb),hdb_noise=float((lb==-1).mean()))
        return o
    def agg(runs,keys):
        o={}
        for k in keys:
            v=np.array([r[k] for r in runs if k in r and np.isfinite(r[k])],float)
            o[k+"_mean"]=float(v.mean()) if len(v) else float("nan"); o[k+"_std"]=float(v.std()) if len(v) else float("nan")
        return o
    R={}

    # T3 (always — gives N=120 canonical+diagnostic for T4)
    print("\n== T3 N-SWEEP =="); t3={}
    for tf in ["robust","raw"]:
        for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
            for N in [1,10,30,120]:
                runs=[]
                for sd in SEEDS:
                    E,lab,co=bursts(EVAL_U,colls,N,sd)
                    if len(E)<8: continue
                    Es,lb=subcap(std(E,co,tf),lab,sd); runs.append(cm(Es,lb,sd))
                a=agg(runs,["ari_kmeans","ari_spectral"]); t3[f"{tf}/{split}/N{N}"]=a
                print(f"  {tf:6s} {split:10s} N={N:3d}: kmeans {a['ari_kmeans_mean']:.3f}±{a['ari_kmeans_std']:.3f} spectral {a['ari_spectral_mean']:.3f}")
    R["T3"]=t3
    if t2t3only:
        json.dump(R,open(OUT,"w"),indent=2); print(f"WROTE {OUT} (T3 only)"); return R

    # T1
    print("\n== T1 PROBE/kNN =="); t1={}
    for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
        for N in [1,10]:
            E,lab,co=bursts(EVAL_U,colls,N,0); Es=std(E,co,"raw")
            rng=np.random.default_rng(0); idx=rng.permutation(len(Es)); cut=len(Es)//2; tr,te=idx[:cut],idx[cut:]
            alr=LogisticRegression(max_iter=2000).fit(Es[tr],lab[tr]).score(Es[te],lab[te])
            akn=KNeighborsClassifier(1).fit(Es[tr],lab[tr]).score(Es[te],lab[te])
            t1[f"{split}_N{N}"]={"probe_acc":alr,"knn1_acc":akn,"n":len(Es)}
            print(f"  {split:10s} N={N:3d}: probe={alr:.3f} kNN1={akn:.3f} n={len(Es)}")
    R["T1"]=t1

    # T2
    print("\n== T2 DISCOVERY =="); t2={}
    for tf in ["robust","raw"]:
        for split,colls in [("canonical",EVAL_C),("diagnostic",TRAIN_C)]:
            runs=[]
            for sd in SEEDS:
                E,lab,co=bursts(EVAL_U,colls,120,sd); Es,lb=subcap(std(E,co,tf),lab,sd); runs.append(cm(Es,lb,sd,extra=True))
            a=agg(runs,["ari_kmeans","ari_spectral","estK","estK_correct","hdb_foundK","hdb_ari","hdb_noise"]); t2[f"{tf}/{split}"]=a
            print(f"  {tf:6s} {split:10s}: kmeans {a['ari_kmeans_mean']:.3f}±{a['ari_kmeans_std']:.3f} estK {a['estK_mean']:.1f} HDB K {a['hdb_foundK_mean']:.1f} ari {a['hdb_ari_mean']:.3f} noise {a['hdb_noise_mean']:.2f}")
    R["T2"]=t2

    # T-RX
    print("\n== T-RX =="); poolR1=np.concatenate([EMBD[c][0] for c in RX_FIT]); med,iqr=_rob(poolR1); trx={}
    runs=[]
    for sd in SEEDS:
        E,lab,co=bursts(EVAL_U,RX_EVAL,120,sd); Es=np.empty_like(E)
        for i in range(len(E)): Es[i]=(E[i]-med)/iqr
        Es,lb=subcap(Es,lab,sd); runs.append(cm(Es,lb,sd))
    trx["full"]=agg(runs,["ari_kmeans","ari_spectral"])
    runs=[]
    for sd in SEEDS:
        E,lab,co=bursts(EVAL_U,RX_EVAL,120,sd); Es,lb=subcap(std(E,co,"robust"),lab,sd); runs.append(cm(Es,lb,sd))
    trx["r2_matched_ref"]=agg(runs,["ari_kmeans","ari_spectral"])
    print(f"  full R1->R2: {trx['full']['ari_kmeans_mean']:.3f}±{trx['full']['ari_kmeans_std']:.3f}  r2_matched {trx['r2_matched_ref']['ari_kmeans_mean']:.3f}")
    R["TRX"]=trx
    json.dump(R,open(OUT,"w"),indent=2); print(f"WROTE {OUT}"); return R

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--emb",required=True); ap.add_argument("--out",required=True); ap.add_argument("--t2t3only",action="store_true")
    a=ap.parse_args(); run(a.emb,a.out,a.t2t3only)
