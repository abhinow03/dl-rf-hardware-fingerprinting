#!/usr/bin/env python3
"""Phase 4 STAGE 2 — why-analysis on TRAIN units only (no eval contact).
W1 rediscovery: ridge-predict A's 19 locked features from B1(512) and B3(128) train embeddings, R².
W2 variance decomposition: fraction of (z-scored) embedding variance between-unit vs between-collection.
W3 calibration: collection-variance fraction before/after per-collection robust standardization (train side).
Caches are row-aligned (verified). Single-thread BLAS (caller)."""
import os, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

BASE=os.path.dirname(os.path.abspath(__file__))
SP=json.load(open(f"{BASE}/splits_ext_ble.json"))
TRAIN_U=SP["train_units"]; TRAIN_C=SP["train_collections"]
from rffp.discovery.ext_ble.features_ble_classical import FEATURE_NAMES, FAMILY
key=lambda c: c.replace(" ","_").replace("(","").replace(")","").replace("/","__")
Aca="/home/docker/pw26_akp_01/ext_cache/ble_classical_b"
B1c="/home/docker/pw26_akp_01/ext_cache/ble_b1_emb"
B3c="/home/docker/pw26_akp_01/ext_cache/ble_b3_emb/full_s2024"
CAP=200; RNG=np.random.default_rng(2026)

def load_train():
    Af=[];B1=[];B3=[];U=[];Cc=[]
    for ci,c in enumerate(TRAIN_C):
        A=np.load(f"{Aca}/{key(c)}.npz"); b1=np.load(f"{B1c}/{key(c)}.npz"); b3=np.load(f"{B3c}/{key(c)}.npz")
        y=A["y"]
        for u in TRAIN_U:
            idx=np.where(y==u)[0]
            if len(idx)>CAP: idx=RNG.choice(idx,CAP,replace=False)
            Af.append(A["F"][idx]); B1.append(b1["E"][idx]); B3.append(b3["E"][idx])
            U.append(np.full(len(idx),u)); Cc.append(np.full(len(idx),ci))
    return (np.concatenate(Af).astype(np.float64),np.concatenate(B1).astype(np.float64),
            np.concatenate(B3).astype(np.float64),np.concatenate(U),np.concatenate(Cc))

def zscore(X): return (X-X.mean(0))/(X.std(0)+1e-12)

def var_decomp(X,groups):
    """multivariate eta^2 = between-group SS / total SS on z-scored dims."""
    Z=zscore(X); g=Z-Z.mean(0); total=(g*g).sum()
    betw=0.0
    for gid in np.unique(groups):
        m=Z[groups==gid].mean(0)-Z.mean(0); betw+=(groups==gid).sum()*(m*m).sum()
    return betw/total

def rob_standardize(X,coll):
    out=np.empty_like(X)
    glob_iqr=np.percentile(X,75,0)-np.percentile(X,25,0); floor=0.25*glob_iqr+1e-12
    for ci in np.unique(coll):
        m=coll==ci; sub=X[m]; med=np.median(sub,0); iqr=np.maximum(np.percentile(sub,75,0)-np.percentile(sub,25,0),floor)
        out[m]=(sub-med)/iqr
    return out

def main():
    Af,B1,B3,U,C=load_train()
    print(f"[data] train segs={len(U)} units={len(set(U))} colls={len(set(C))}")
    out={}

    # ---- W1 rediscovery ----
    print("\n== W1 REDISCOVERY (ridge R^2, predict A-feature from embedding) ==")
    tgt=zscore(Af)   # standardized targets
    w1={}
    for name,E in [("B1",B1),("B3",B3)]:
        Etr,Ete,Ytr,Yte=train_test_split(E,tgt,test_size=0.3,random_state=0)
        r=Ridge(alpha=10.0).fit(Etr,Ytr); pred=r.predict(Ete)
        r2=[r2_score(Yte[:,i],pred[:,i]) for i in range(19)]
        w1[name]={FEATURE_NAMES[i]:round(float(r2[i]),3) for i in range(19)}
        fam={f:round(float(np.mean([r2[i] for i in idx])),3) for f,idx in FAMILY.items()}
        w1[name+"_family"]=fam; w1[name+"_overall"]=round(float(np.mean(r2)),3)
        print(f"  {name}: overall R2={np.mean(r2):.3f} | families "+" ".join(f"{f}={fam[f]:.2f}" for f in FAMILY))
    out["W1"]=w1

    # ---- W2 variance decomposition ----
    print("\n== W2 VARIANCE DECOMPOSITION (eta^2 between-group / total, z-scored dims) ==")
    w2={}
    for name,E in [("A(19)",Af),("B1(512)",B1),("B3(128)",B3)]:
        eu=var_decomp(E,U); ec=var_decomp(E,C)
        w2[name]={"unit_eta2":round(float(eu),3),"coll_eta2":round(float(ec),3),"ratio_u_over_c":round(float(eu/(ec+1e-9)),2)}
        print(f"  {name:9s}: unit_eta2={eu:.3f}  coll_eta2={ec:.3f}  unit/coll={eu/(ec+1e-9):.2f}")
    out["W2"]=w2

    # ---- W3 calibration (train-side collection-variance before/after robust) ----
    print("\n== W3 CALIBRATION (collection eta^2 before/after per-collection robust std, train side) ==")
    w3={}
    for name,E in [("A(19)",Af),("B1(512)",B1),("B3(128)",B3)]:
        before=var_decomp(E,C)
        Er=rob_standardize(E,C); after=var_decomp(Er,C)
        uafter=var_decomp(Er,U)
        w3[name]={"coll_eta2_raw":round(float(before),3),"coll_eta2_robust":round(float(after),3),
                  "coll_reduction":round(float(before-after),3),"unit_eta2_robust":round(float(uafter),3)}
        print(f"  {name:9s}: coll_eta2 raw={before:.3f} -> robust={after:.3f} (Δ={before-after:+.3f})  unit_eta2(robust)={uafter:.3f}")
    out["W3"]=w3

    json.dump(out,open(f"{BASE}/audit_out/why_analysis.json","w"),indent=2)
    print("\nWROTE audit_out/why_analysis.json")

if __name__=="__main__": main()
