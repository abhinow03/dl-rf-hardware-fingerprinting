#!/usr/bin/env python3
"""Phase 3a STEP 1 — B1 FROZEN WiSig TRANSFER embedding extraction (GPU, read-only encoder).

Loads the frozen WiSig encoder EXACTLY as the locked transfer harness (gen_rff/demo_router/tiers.py
_Enc / T-B / T-C): RFEncoder from summer_work runs/wisig_supcon_fft64/retrain_best/best_model.pt,
eval, requires_grad=False, embedding stage = get_encoder_output -> 512-D (LayerNorm[ cross-attn(256)
(+) spectral(256) ]). STFT front-end for L=256 windows = stft_for(256) -> nfft=64 hop=16 (fft64),
via registry.stft_mag. This is the SAME stage+dim as every prior F-series transfer eval (FINDINGS F1
frozen@N120). NO training / fine-tuning / weight update.

Input  : B1 cache /ext_cache/ble_b1_25msps/*.npz  X_win=(N,30,2,256) f32 @ 25 MS/s.
Per segment: forward all 30 windows -> (30,512), mean-pool -> 512, L2-normalize -> one embedding.
ALL segments of ALL collections embedded IN FULL (no subsampling — budget note permits <=200/unit
for stats-fit segs, but full extraction is cheap here and keeps the robust transform fit identical
to Approach A). Output per collection: E=(N,512) f32, y, n_train. sha256 each; fixity md written.
"""
import os, sys, glob, json, hashlib, time
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
import torch

_HERE=os.path.dirname(os.path.abspath(__file__))
_DLM=os.path.abspath(os.path.join(_HERE,"..",".."))
_SW=os.path.join(_DLM,"summer_work")
for p in (_DLM,_SW,os.path.join(_SW,"datasets")):
    if p not in sys.path: sys.path.insert(0,p)
from rffp.models import RFEncoder                     # frozen encoder, shared not copied
from rffp.data import registry                # stft_mag (Hann, center=False)

CKPT=os.path.join(_SW,"runs","wisig_supcon_fft64","retrain_best","best_model.pt")
SRC="/home/docker/pw26_akp_01/ext_cache/ble_b1_25msps"
OUT="/home/docker/pw26_akp_01/ext_cache/ble_b1_emb"; os.makedirs(OUT,exist_ok=True)
AUDIT=os.path.join(_HERE,"audit_out")
NFFT,HOP=64,16          # == stft_for(256) in the locked harness (L<1024)
EMB_DIM=512
SEG_CHUNK=2000          # segments loaded/forwarded per outer step
WIN_BATCH=4096          # windows per GPU sub-batch

def load_encoder():
    m=RFEncoder().cuda()
    m.load_state_dict(torch.load(CKPT,map_location="cuda",weights_only=True),strict=True)
    m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m

@torch.no_grad()
def embed_windows(m, w):                          # w: (K,2,256) np -> (K,512) np
    out=np.empty((w.shape[0],EMB_DIM),np.float32)
    xt=torch.from_numpy(np.ascontiguousarray(w.astype(np.float32)))
    for i in range(0,w.shape[0],WIN_BATCH):
        xb=xt[i:i+WIN_BATCH].cuda(non_blocking=True)
        xs=registry.stft_mag(xb,NFFT,HOP)
        with torch.amp.autocast('cuda'):
            out[i:i+WIN_BATCH]=m.get_encoder_output(xb,xs).float().cpu().numpy()
    return out

def main():
    m=load_encoder()
    nparams=sum(p.numel() for p in m.parameters())
    print(f"[enc] RFEncoder loaded read-only, params={nparams:,} stage=get_encoder_output dim={EMB_DIM} stft=({NFFT},{HOP})")
    files=sorted(glob.glob(f"{SRC}/*.npz"))
    stats=[]; t0=time.time(); tot_win=0; tot_seg=0
    for fp in files:
        name=os.path.basename(fp)[:-4]
        z=np.load(fp,mmap_mode="r")
        X=z["X_win"]; y=np.asarray(z["y_idx"]).astype(np.int16); n_tr=int(z["n_train"])
        N,W = X.shape[0], X.shape[1]              # (N,30,2,256)
        E=np.empty((N,EMB_DIM),np.float32)
        tc=time.time()
        for s in range(0,N,SEG_CHUNK):
            e=min(s+SEG_CHUNK,N)
            w=np.asarray(X[s:e]).reshape((e-s)*W,2,256)   # (chunk*30,2,256)
            emb=embed_windows(m,w).reshape(e-s,W,EMB_DIM).mean(1)   # mean-pool 30 windows
            emb/=(np.linalg.norm(emb,axis=1,keepdims=True)+1e-12)   # L2-normalize per segment
            E[s:e]=emb.astype(np.float32)
            tot_win+=(e-s)*W
        tot_seg+=N
        path=os.path.join(OUT,name+".npz")
        np.savez(path,E=E,y=y,n_train=np.int32(n_tr))
        h=hashlib.sha256(open(path,"rb").read()).hexdigest()
        dt=time.time()-tc
        stats.append(dict(coll=name,N=int(N),path=path,size_mb=round(os.path.getsize(path)/1e6,2),sha256=h))
        print(f"[emb] {name:26s} N={N:6d} {dt:5.1f}s {N/dt:7.0f} seg/s finite={np.isfinite(E).all()} {stats[-1]['size_mb']}MB sha={h[:12]}")
    dt=time.time()-t0
    wps=tot_win/dt; sps=tot_seg/dt
    tot_mb=sum(s['size_mb'] for s in stats)
    print(f"\nTOTAL {len(stats)} colls  {tot_seg:,} segs  {tot_win:,} windows  {dt:.0f}s")
    print(f"THROUGHPUT {wps:,.0f} windows/s | {sps:,.0f} segments/s | GPU {1000/sps:.4f} ms/segment | cache {tot_mb:.1f} MB")
    lines=["# B1 frozen-WiSig 512-D segment-embedding cache — fixity","",
           f"encoder: runs/wisig_supcon_fft64/retrain_best/best_model.pt (frozen, read-only)",
           f"stage: get_encoder_output  dim: {EMB_DIM}  stft: nfft={NFFT} hop={HOP} (stft_for(256))",
           f"per segment: forward 30 windows -> mean-pool -> L2-normalize",
           f"throughput: {wps:,.0f} win/s, {sps:,.0f} seg/s, GPU {1000/sps:.4f} ms/segment; cache {tot_mb:.1f} MB","",
           "| collection | N | size MB | sha256 |","|---|---|---|---|"]
    for s in stats: lines.append(f"| {s['coll']} | {s['N']} | {s['size_mb']} | {s['sha256']} |")
    open(os.path.join(AUDIT,"b1_emb_cache_fixity.md"),"w").write("\n".join(lines)+"\n")
    print("WROTE audit_out/b1_emb_cache_fixity.md")

if __name__=="__main__": main()
