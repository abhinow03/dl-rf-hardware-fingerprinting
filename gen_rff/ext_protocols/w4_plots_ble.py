#!/usr/bin/env python3
"""Phase 4 W4 — audit plots to audit_out/ from ALREADY-CACHED eval embeddings (not new experiments).
1) UMAP per method (A/B1/B3), colored by unit and by collection (raw geometry).
2) Eigengap spectra (Laplacian eigenvalues) on canonical N=120 burst-means, A/B1/B3 overlaid.
3) T3 N-curves overlaid. 4) T4 data-efficiency curves.
Palette = Okabe-Ito (canonical colorblind-safe 8-set), fixed order. Single-thread BLAS (caller)."""
import os, json
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.neighbors import kneighbors_graph
import umap

BASE="/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols"; AUD=f"{BASE}/audit_out"
SP=json.load(open(f"{BASE}/splits_ext_ble.json"))
EVAL_U=SP["eval_units"]; EVAL_C=SP["eval_collections"]
OKABE=["#000000","#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7"]
key=lambda c: c.replace(" ","_").replace("(","").replace(")","").replace("/","__")
Aca="/home/docker/pw26_akp_01/ext_cache/ble_classical_b"
B1c="/home/docker/pw26_akp_01/ext_cache/ble_b1_emb"; B3c="/home/docker/pw26_akp_01/ext_cache/ble_b3_emb/full_s2024"
RNG=np.random.default_rng(7)

def load_eval(kind):
    E=[];U=[];C=[]
    for ci,c in enumerate(EVAL_C):
        if kind=="A": z=np.load(f"{Aca}/{key(c)}.npz"); X=z["F"]
        elif kind=="B1": z=np.load(f"{B1c}/{key(c)}.npz"); X=z["E"]
        else: z=np.load(f"{B3c}/{key(c)}.npz"); X=z["E"]
        y=z["y"]
        for u in EVAL_U:
            idx=np.where(y==u)[0]
            if len(idx)>60: idx=RNG.choice(idx,60,replace=False)
            E.append(X[idx].astype(np.float64)); U.append(np.full(len(idx),u)); C.append(np.full(len(idx),ci))
    return np.concatenate(E),np.concatenate(U),np.concatenate(C)

def zscore(X): return (X-X.mean(0))/(X.std(0)+1e-12)

def fig_umap():
    fig,ax=plt.subplots(3,2,figsize=(9,12))
    for r,kind in enumerate(["A","B1","B3"]):
        E,U,C=load_eval(kind)
        Ez=zscore(E) if kind=="A" else E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
        emb=umap.UMAP(n_neighbors=25,min_dist=0.1,random_state=0,metric="euclidean").fit_transform(Ez)
        for col,(lab,G) in enumerate([("unit",U),("collection",C)]):
            a=ax[r,col]
            for gi,g in enumerate(sorted(set(G))):
                m=G==g; a.scatter(emb[m,0],emb[m,1],s=6,c=OKABE[gi%8],
                                  label=(f"u{g}" if lab=="unit" else EVAL_C[g].split('/')[-1]),alpha=0.7,linewidths=0)
            a.set_title(f"{kind} — by {lab}",fontsize=10); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values(): s.set_visible(False)
            if r==0: a.legend(fontsize=6,markerscale=1.5,loc="upper right",ncol=2,framealpha=0.6)
    fig.suptitle("W4 UMAP — raw embedding geometry (canonical eval; 8 held-out units × 4 outdoor collections)",fontsize=11)
    fig.tight_layout(); fig.savefig(f"{AUD}/w4_umap.png",dpi=130); plt.close(fig); print("wrote w4_umap.png")

def burst_rob(kind,N=120):
    E=[];U=[];CO=[]
    for ci,c in enumerate(EVAL_C):
        if kind=="A": z=np.load(f"{Aca}/{key(c)}.npz"); X=z["F"].astype(np.float64)
        elif kind=="B1": z=np.load(f"{B1c}/{key(c)}.npz"); X=z["E"].astype(np.float64)
        else: z=np.load(f"{B3c}/{key(c)}.npz"); X=z["E"].astype(np.float64)
        y=z["y"]
        for u in EVAL_U:
            idx=np.where(y==u)[0]; RNG.shuffle(idx); ng=len(idx)//N
            for g in range(ng): E.append(X[idx[g*N:(g+1)*N]].mean(0)); U.append(u); CO.append(ci)
    E=np.array(E); CO=np.array(CO)
    # per-collection robust
    gi=np.percentile(E,75,0)-np.percentile(E,25,0); fl=0.25*gi+1e-12; out=np.empty_like(E)
    for ci in np.unique(CO):
        m=CO==ci; med=np.median(E[m],0); iqr=np.maximum(np.percentile(E[m],75,0)-np.percentile(E[m],25,0),fl); out[m]=(E[m]-med)/iqr
    return out

def eiggaps(E,kmax=15):
    A=kneighbors_graph(E,10,mode="connectivity"); A=0.5*(A+A.T); A=A.toarray()
    d=A.sum(1); Dm=np.diag(1/np.sqrt(d+1e-12)); L=np.eye(len(A))-Dm@A@Dm
    return np.sort(eigh(L,eigvals_only=True))[:kmax+1]

def fig_eigengap():
    fig,ax=plt.subplots(figsize=(7,4.2))
    for i,kind in enumerate(["A","B1","B3"]):
        w=eiggaps(burst_rob(kind)); ax.plot(range(1,len(w)+1),w,marker="o",ms=4,lw=2,c=OKABE[[1,5,3][i]],label=kind)
    ax.axvline(8,color="#999",lw=1,ls="--"); ax.text(8.1,ax.get_ylim()[1]*0.9,"true K=8",fontsize=8,color="#666")
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("normalized Laplacian eigenvalue")
    ax.set_title("W4 eigengap spectra — canonical N=120 robust burst-means",fontsize=10)
    ax.legend(); [ax.spines[s].set_visible(False) for s in ("top","right")]
    fig.tight_layout(); fig.savefig(f"{AUD}/w4_eigengap.png",dpi=130); plt.close(fig); print("wrote w4_eigengap.png")

def fig_t3():
    A=json.load(open("/home/docker/pw26_akp_01/ext_cache/ble_A_results/battery_A_results.json"))
    B1=json.load(open("/home/docker/pw26_akp_01/ext_cache/ble_B1_results/battery_B1_results.json"))
    B3=json.load(open("/home/docker/pw26_akp_01/ext_cache/ble_b3_results/full_s2024.json"))
    Ns=[1,10,30,120]
    def curve(d,pref): return [d["T3"][f"{pref}/N{n}"]["ari_kmeans_mean"] for n in Ns]
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for j,(reg) in enumerate(["canonical","diagnostic"]):
        ax[j].plot(Ns,curve(A,f"robust/{reg}"),marker="o",lw=2,c=OKABE[1],label="A (19-D)")
        ax[j].plot(Ns,curve(B1,f"robust/{reg}"),marker="s",lw=2,c=OKABE[5],label="B1 frozen")
        ax[j].plot(Ns,curve(B3,f"robust/{reg}"),marker="^",lw=2,c=OKABE[3],label="B3 native")
        ax[j].set_xscale("log"); ax[j].set_xticks(Ns); ax[j].set_xticklabels(Ns)
        ax[j].set_title(f"{reg} — robust k-means oracle-K ARI",fontsize=10); ax[j].set_xlabel("N (segments/burst)")
        ax[j].set_ylim(0,1); [ax[j].spines[s].set_visible(False) for s in ("top","right")]
        if j==0: ax[j].set_ylabel("ARI"); ax[j].legend()
    fig.suptitle("W4 T3 N-curves — A front-loaded / B1 integration / B3 flat",fontsize=11)
    fig.tight_layout(); fig.savefig(f"{AUD}/w4_t3_ncurves.png",dpi=130); plt.close(fig); print("wrote w4_t3_ncurves.png")

def fig_t4():
    d={u:json.load(open(f"/home/docker/pw26_akp_01/ext_cache/ble_b3_results/{t}.json")) for u,t in [(5,"de5_s2024"),(10,"de10_s2024"),(21,"full_s2024")]}
    us=[5,10,21]
    fig,ax=plt.subplots(figsize=(7,4.2))
    for tf,mk,ci in [("robust","o",3),("raw","s",6)]:
        ax.plot(us,[d[u]["T3"][f"{tf}/canonical/N120"]["ari_kmeans_mean"] for u in us],marker=mk,lw=2,c=OKABE[ci],label=f"canonical ({tf})")
        ax.plot(us,[d[u]["T3"][f"{tf}/diagnostic/N120"]["ari_kmeans_mean"] for u in us],marker=mk,lw=2,ls="--",c=OKABE[ci],label=f"diagnostic ({tf})")
    ax.axhline(0.334,color="#D55E00",lw=1,ls=":"); ax.text(5,0.345,"PRIMARY bar 0.334",fontsize=8,color="#D55E00")
    ax.set_xticks(us); ax.set_xlabel("train units"); ax.set_ylabel("N=120 oracle-K ARI"); ax.set_ylim(0,1)
    ax.set_title("W4 T4 — B3 data-efficiency",fontsize=10); ax.legend(fontsize=8); [ax.spines[s].set_visible(False) for s in ("top","right")]
    fig.tight_layout(); fig.savefig(f"{AUD}/w4_t4_dataeff.png",dpi=130); plt.close(fig); print("wrote w4_t4_dataeff.png")

if __name__=="__main__":
    fig_umap(); fig_eigengap(); fig_t3(); fig_t4(); print("W4 plots done ->",AUD)
