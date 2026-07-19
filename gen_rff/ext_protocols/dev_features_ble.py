#!/usr/bin/env python3
"""Phase 2 Stage A DEV — sanity + discrimination of classical_b on TRAIN units x TRAIN
collections ONLY. No eval contact. Prints F-scores, redundancy, kNN-1; saves plots."""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from features_ble_classical import extract, FEATURE_NAMES, FAMILY

ROOT="/home/docker/pw26_akp_01/ext_data/ble_xiao"
OUT="/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols/audit_out"
SP=json.load(open("/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols/splits_ext_ble.json"))
train_units=set(SP["train_units"]); TRAIN_COLL=SP["train_collections"]
rng=np.random.default_rng(2026)
PER=60   # segs per (unit,collection)

def sample_collection(coll):
    d=os.path.join(ROOT,coll)
    X=np.load(os.path.join(d,"X_train.npy"),mmap_mode="r")
    y=np.asarray(np.load(os.path.join(d,"Y_train.npy"),mmap_mode="r")).argmax(1)
    rows=[]; labs=[]
    for u in train_units:
        idx=np.where(y==u)[0]
        pick=rng.choice(idx, size=min(PER,len(idx)), replace=False)
        rows.append(pick); labs.append(np.full(len(pick),u))
    rows=np.concatenate(rows); labs=np.concatenate(labs)
    order=np.argsort(rows); rows=rows[order]; labs=labs[order]
    F=extract(np.asarray(X[rows]))
    return F, labs

def main():
    Fs=[]; ys=[]
    for c in TRAIN_COLL:
        F,l=sample_collection(c); Fs.append(F); ys.append(l)
    F=np.concatenate(Fs); y=np.concatenate(ys)
    print(f"DEV set: {F.shape[0]} segs, {len(np.unique(y))} train units, {F.shape[1]}-D")
    print(f"finite={np.isfinite(F).all()} degenerate cols={[FEATURE_NAMES[i] for i in np.where(F.std(0)==0)[0]]}")

    # F-scores (discrimination across units)
    Fsc,_=f_classif(F,y)
    print("\n== per-feature ANOVA F-score (higher=more unit-discriminative) ==")
    for i in np.argsort(Fsc)[::-1]:
        fam=[k for k,v in FAMILY.items() if i in v][0]
        print(f"  {FEATURE_NAMES[i]:16s} [{fam:6s}] F={Fsc[i]:9.1f}")

    # redundancy
    Z=StandardScaler().fit_transform(F)
    C=np.corrcoef(Z.T)
    print("\n== redundant pairs |r|>0.9 ==")
    pairs=[]
    for i in range(19):
        for j in range(i+1,19):
            if abs(C[i,j])>0.9: pairs.append((FEATURE_NAMES[i],FEATURE_NAMES[j],C[i,j])); print(f"  {FEATURE_NAMES[i]} ~ {FEATURE_NAMES[j]}: r={C[i,j]:+.3f}")
    if not pairs: print("  none")

    # kNN-1 on train-unit classification (70/30 within DEV, global z-score)
    n=F.shape[0]; perm=rng.permutation(n); cut=int(0.7*n)
    tr,te=perm[:cut],perm[cut:]
    sc=StandardScaler().fit(F[tr])
    knn=KNeighborsClassifier(1).fit(sc.transform(F[tr]),y[tr])
    acc=knn.score(sc.transform(F[te]),y[te])
    print(f"\n== kNN-1 (train-unit ID, 70/30, global z) acc={acc:.3f}  chance={1/len(np.unique(y)):.3f} ==")
    # family-ablation kNN (drop each family)
    for fam,idx in FAMILY.items():
        keep=[i for i in range(19) if i not in idx]
        sc2=StandardScaler().fit(F[tr][:,keep]); k2=KNeighborsClassifier(1).fit(sc2.transform(F[tr][:,keep]),y[tr])
        print(f"   drop {fam:6s} -> kNN-1 {k2.score(sc2.transform(F[te][:,keep]),y[te]):.3f}")

    # distribution plots
    fig,ax=plt.subplots(4,5,figsize=(18,12))
    ax=ax.ravel()
    for i in range(19):
        for u in list(np.unique(y))[:5]:
            ax[i].hist(F[y==u,i],bins=30,alpha=0.5,density=True)
        ax[i].set_title(FEATURE_NAMES[i],fontsize=9)
    ax[19].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT,"dev_feature_dists.png"),dpi=80); plt.close(fig)
    print("WROTE audit_out/dev_feature_dists.png")

if __name__=="__main__": main()
