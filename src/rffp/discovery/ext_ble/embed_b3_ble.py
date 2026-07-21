#!/usr/bin/env python3
"""Phase 3c — embed ALL BLE segments with a trained B3 native checkpoint -> 128-D L2-normed
head embeddings (native-arm discovery space), one npz per collection. sha256'd.

Usage: embed_b3_ble.py --ckpt <best.pt> --out <emb_dir>
Input = native (2,1850) .npy (pooled train+test per collection); 2D branch = native STFT (2,128,27).
Embedding = forward() head (128-D, already L2-normalized). GPU, read-only inference."""
import os, sys, glob, json, hashlib, time, argparse
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np, torch

_HERE=os.path.dirname(os.path.abspath(__file__))
_DLM=os.path.abspath(os.path.join(_HERE,"..","..")); _SW=os.path.join(_DLM,"summer_work")
for p in (_DLM,_SW):
    if p not in sys.path: sys.path.insert(0,p)
from rffp.models import RFEncoder

ROOT="/home/docker/pw26_akp_01/ext_data/ble_xiao"
_HANN=None
def native_stft(x):
    global _HANN
    if _HANN is None: _HANN=torch.hann_window(128,device=x.device)
    z=torch.complex(x[:,0].float(),x[:,1].float())
    Z=torch.stft(z,n_fft=128,hop_length=64,win_length=128,window=_HANN,center=False,return_complex=True)
    return torch.stack([Z.real,Z.imag],dim=1)

def collections():
    cs=[]
    for xt in sorted(glob.glob(os.path.join(ROOT,"*","*","X_train.npy"))):
        d=os.path.dirname(xt); cs.append(f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}")
    return cs

def key(c): return c.replace(" ","_").replace("(","").replace(")","").replace("/","__")

@torch.no_grad()
def embed(model,X,bs=2048):
    out=np.empty((X.shape[0],128),np.float32)
    for s in range(0,X.shape[0],bs):
        xb=torch.from_numpy(np.asarray(X[s:s+bs]).astype(np.float32)).cuda()
        with torch.amp.autocast('cuda'):
            out[s:s+bs]=model(xb,native_stft(xb)).float().cpu().numpy()   # 128-D L2-normed head
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--out",required=True)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    m=RFEncoder().cuda(); m.load_state_dict(torch.load(a.ckpt,map_location="cuda",weights_only=True),strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    stats=[]; t0=time.time(); tot=0
    for c in collections():
        d=os.path.join(ROOT,c)
        Xtr=np.load(f"{d}/X_train.npy",mmap_mode="r"); Xte=np.load(f"{d}/X_test.npy",mmap_mode="r")
        ytr=np.asarray(np.load(f"{d}/Y_train.npy",mmap_mode="r")).argmax(1)
        yte=np.asarray(np.load(f"{d}/Y_test.npy",mmap_mode="r")).argmax(1)
        n_tr=Xtr.shape[0]
        E=np.concatenate([embed(m,Xtr),embed(m,Xte)]); y=np.concatenate([ytr,yte]).astype(np.int16)
        tot+=E.shape[0]
        path=os.path.join(a.out,key(c)+".npz"); np.savez(path,E=E.astype(np.float32),y=y,n_train=np.int32(n_tr))
        h=hashlib.sha256(open(path,"rb").read()).hexdigest()
        stats.append((c,E.shape[0],round(os.path.getsize(path)/1e6,2),h))
        print(f"[emb] {c:26s} N={E.shape[0]:6d} finite={np.isfinite(E).all()} sha={h[:12]}",flush=True)
    dt=time.time()-t0
    print(f"TOTAL {tot} segs {dt:.0f}s {tot/dt:.0f} seg/s -> {a.out}")
    json.dump([dict(coll=c,N=n,mb=mb,sha256=h) for c,n,mb,h in stats], open(os.path.join(a.out,"_fixity.json"),"w"),indent=2)

if __name__=="__main__": main()
