#!/usr/bin/env python3
"""Phase 3c STEP 1 — Approach B / Arm B3: native from-scratch SupCon encoder on BLE D2.

Fresh dual-branch RFEncoder (random init), native input: 1D (2,1850) + 2D STFT (2,128,27)
per WINDOW_SPEC_BLE. SupCon on unit labels, CONSTANT tau=0.5, NO augmentation (no CFO aug,
no time jitter — onset-aligned). Balanced P units x S segments, segments drawn uniformly
across the 8 train_collections (mechanism hypothesis). AdamW + cosine + AMP + grad-clip 1.0.
Checkpoint selection: val-unit(2) burst-mean N=10 pairwise-cosine ROC-AUC on train_collections
(tie-break val SupCon loss). Collapse guard: per-dim embedding std -> 0 aborts.

Usage: train_b3_ble.py --seed 2024 --units 21 --epochs 12 --tag full_s2024
Checkpoints off-repo: /home/docker/pw26_akp_01/ext_cache/ble_b3_ckpt/<tag>/best.pt (sha256'd).
"""
import os, sys, json, time, argparse, hashlib
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np, torch
from sklearn.metrics import roc_auc_score

_HERE=os.path.dirname(os.path.abspath(__file__))
_DLM=os.path.abspath(os.path.join(_HERE,"..",".."))
_SW=os.path.join(_DLM,"summer_work")
for p in (_DLM,_SW):
    if p not in sys.path: sys.path.insert(0,p)
from rffp.models import RFEncoder, SupervisedContrastiveLoss

ROOT="/home/docker/pw26_akp_01/ext_data/ble_xiao"
CKROOT="/home/docker/pw26_akp_01/ext_cache/ble_b3_ckpt"
import rffp.discovery.ext_ble as _extble
SP=json.load(open(os.path.join(os.path.dirname(_extble.__file__), "splits_ext_ble.json")))
TRAIN_U=SP["train_units"]; VAL_U=SP["val_units"]; TRAIN_C=SP["train_collections"]
WD=1e-4; LR=5e-4; TAU=0.5; GRAD_CLIP=1.0; S=16; WARMUP_FRAC=0.05

_HANN=None
def native_stft(x):                       # x:(B,2,1850) -> (B,2,128,27)
    global _HANN
    if _HANN is None or _HANN.device!=x.device: _HANN=torch.hann_window(128,device=x.device)
    z=torch.complex(x[:,0].float(),x[:,1].float())
    Z=torch.stft(z,n_fft=128,hop_length=64,win_length=128,window=_HANN,center=False,return_complex=True)
    return torch.stack([Z.real,Z.imag],dim=1)

def load_mmaps(colls):
    M={}
    for c in colls:
        d=os.path.join(ROOT,c)
        M[c]=dict(Xtr=np.load(f"{d}/X_train.npy",mmap_mode="r"),Xte=np.load(f"{d}/X_test.npy",mmap_mode="r"),
                  ytr=np.asarray(np.load(f"{d}/Y_train.npy",mmap_mode="r")).argmax(1),
                  yte=np.asarray(np.load(f"{d}/Y_test.npy",mmap_mode="r")).argmax(1))
    return M

def build_index(M,colls,units):
    """per unit -> list of (coll, part, row)."""
    idx={u:[] for u in units}
    for c in colls:
        for u in units:
            for r in np.where(M[c]["ytr"]==u)[0]: idx[u].append((c,0,int(r)))
            for r in np.where(M[c]["yte"]==u)[0]: idx[u].append((c,1,int(r)))
    for u in units: idx[u]=np.array(idx[u],dtype=object)
    return idx

def gather(M,triples):
    out=np.empty((len(triples),2,1850),np.float32)
    for i,(c,part,r) in enumerate(triples):
        out[i]=M[c]["Xtr"][r] if part==0 else M[c]["Xte"][r]
    return out

def make_val(M,colls,units,rng,N=10):
    """burst-mean N pooled embeddings input rows + labels (val units, train collections)."""
    rows=[]; labs=[]
    for u in units:
        tr=[(c,0,int(r)) for c in colls for r in np.where(M[c]["ytr"]==u)[0]]
        te=[(c,1,int(r)) for c in colls for r in np.where(M[c]["yte"]==u)[0]]
        allr=tr+te; rng.shuffle(allr); ng=len(allr)//N
        for g in range(ng): rows.append(allr[g*N:(g+1)*N]); labs.append(u)
    return rows,np.array(labs)

def make_val_batch(M,colls,units,rng,per=64):
    """fixed per-SEGMENT val batch (val units x per, train collections) for val SupCon loss."""
    trip=[]; ys=[]
    for u in units:
        allr=[(c,0,int(r)) for c in colls for r in np.where(M[c]["ytr"]==u)[0]]+\
             [(c,1,int(r)) for c in colls for r in np.where(M[c]["yte"]==u)[0]]
        pick=[allr[i] for i in rng.integers(0,len(allr),per)]; trip+=pick; ys+=[u]*per
    return trip,np.array(ys)

@torch.no_grad()
def val_metrics(model,M,val_groups,val_labs,vbatch,vy,crit):
    model.eval()
    E=[]
    for grp in val_groups:
        x=torch.from_numpy(gather(M,grp)).cuda()
        with torch.amp.autocast('cuda'):
            e=model(x,native_stft(x)).float().mean(0)     # burst-mean of N head-embeddings
        e=e/(e.norm()+1e-12); E.append(e.cpu().numpy())
    # val SupCon loss on a fixed per-segment val batch (tie-break)
    xb=torch.from_numpy(gather(M,vbatch)).cuda(); yb=torch.tensor(vy).cuda()
    with torch.amp.autocast('cuda'): emb=model(xb,native_stft(xb))
    vloss=float(crit(emb.float(),yb,temperature=TAU))
    model.train()
    E=np.array(E); sim=E@E.T; B=len(val_labs); iu=np.triu_indices(B,1)
    same=(val_labs[:,None]==val_labs[None,:])[iu]; s=sim[iu]
    auc=float(roc_auc_score(same.astype(int),s)) if same.sum() and (~same).sum() else float("nan")
    return auc, float(E.std(0).mean()), vloss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True); ap.add_argument("--units",type=int,required=True)
    ap.add_argument("--epochs",type=int,default=12); ap.add_argument("--tag",required=True)
    a=ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); rng=np.random.default_rng(a.seed)
    ckdir=os.path.join(CKROOT,a.tag); os.makedirs(ckdir,exist_ok=True)

    units=list(TRAIN_U)
    if a.units<len(units):
        units=sorted(np.random.default_rng(a.seed).choice(TRAIN_U,a.units,replace=False).tolist())
    P=len(units)
    print(f"[cfg] seed={a.seed} units={P}{units} P×S={P}×{S}={P*S} epochs={a.epochs} tau={TAU} lr={LR} NO-AUG")
    M=load_mmaps(TRAIN_C); idx=build_index(M,TRAIN_C,units)
    n_seg=sum(len(idx[u]) for u in units)
    steps_ep=int(np.ceil(n_seg/(P*S))); total=steps_ep*a.epochs; warm=int(total*WARMUP_FRAC)
    print(f"[data] train segs={n_seg} steps/epoch={steps_ep} total_steps={total} warmup={warm}")
    vg,vl=make_val(M,TRAIN_C,VAL_U,np.random.default_rng(7)); print(f"[val] {len(vg)} N10 bursts over units {VAL_U}")
    vbatch,vy=make_val_batch(M,TRAIN_C,VAL_U,np.random.default_rng(11))

    model=RFEncoder().cuda(); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    crit=SupervisedContrastiveLoss(); scaler=torch.amp.GradScaler('cuda')
    def lr_at(s):
        if s<warm: return LR*(s+1)/warm
        pr=(s-warm)/max(1,total-warm); return 0.5*LR*(1+np.cos(np.pi*pr))

    log=[]; best_auc=-1; best_loss=1e9; collapsed=False; t0=time.time(); step=0; ep_t0=time.time()
    model.train()
    for ep in range(a.epochs):
        for _ in range(steps_ep):
            for g in opt.param_groups: g["lr"]=lr_at(step)
            trip=[]; ys=[]
            for u in units:
                pick=idx[u][rng.integers(0,len(idx[u]),S)]
                trip+= [tuple(t) for t in pick]; ys+=[u]*S
            x=torch.from_numpy(gather(M,trip)).cuda(); y=torch.tensor(ys).cuda()
            opt.zero_grad()
            with torch.amp.autocast('cuda'): emb=model(x,native_stft(x))
            loss=crit(emb.float(),y,temperature=TAU)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),GRAD_CLIP); scaler.step(opt); scaler.update()
            step+=1
        auc,estd,vloss=val_metrics(model,M,vg,vl,vbatch,vy,crit)
        ep_dt=time.time()-ep_t0; ep_t0=time.time()
        log.append(dict(epoch=ep+1,step=step,loss=round(float(loss),4),val_auc=round(auc,4),val_loss=round(vloss,4),emb_std=round(estd,5),ep_sec=round(ep_dt,1)))
        print(f"[ep {ep+1:2d}/{a.epochs}] tr_loss={float(loss):.4f} val_auc={auc:.4f} val_loss={vloss:.4f} emb_std={estd:.5f} {ep_dt:.1f}s lr={lr_at(step):.2e}",flush=True)
        if estd<1e-4: print("!! COLLAPSE GUARD: emb_std<1e-4 -> ABORT"); collapsed=True; break
        better = auc>best_auc+1e-6 or (abs(auc-best_auc)<=1e-6 and vloss<best_loss)   # primary val-AUC, tie-break val SupCon loss
        if better:
            best_auc=auc; best_loss=vloss; torch.save(model.state_dict(),f"{ckdir}/best.pt")
            with open(f"{ckdir}/best_meta.json","w") as f: json.dump(dict(epoch=ep+1,step=step,val_auc=auc,val_loss=vloss),f)
        if ep==0:
            proj_h=(ep_dt*a.epochs)/3600
            print(f"[proj] epoch1={ep_dt:.1f}s -> projected {a.epochs} epochs ≈ {proj_h:.2f} h"+(" (>6h: HALVE)" if proj_h>6 else " (<6h ok)"),flush=True)
    torch.save(model.state_dict(),f"{ckdir}/last.pt")
    sha=hashlib.sha256(open(f"{ckdir}/best.pt","rb").read()).hexdigest() if os.path.exists(f"{ckdir}/best.pt") else None
    summ=dict(tag=a.tag,seed=a.seed,units=units,P=P,S=S,epochs=a.epochs,total_steps=total,
              best_auc=best_auc,best_meta=json.load(open(f"{ckdir}/best_meta.json")) if os.path.exists(f"{ckdir}/best_meta.json") else None,
              collapsed=collapsed,best_sha256=sha,wall_sec=round(time.time()-t0,1),log=log)
    json.dump(summ,open(f"{ckdir}/train_summary.json","w"),indent=2)
    print(f"\nDONE {a.tag}: best_auc={best_auc:.4f} best.pt sha={sha[:12] if sha else None} wall={ (time.time()-t0)/60:.1f}min")

if __name__=="__main__": main()
