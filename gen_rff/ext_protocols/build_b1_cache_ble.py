#!/usr/bin/env python3
"""
Phase 1 STEP 4 — B1 commensurability cache: resample BLE 6->25 MS/s, slice to 256-sample
WiSig windows (stride 256, 30 win/seg). One .npz per collection on the docker volume,
sha256 per file. Pooled (train+test) per collection; y_idx + seg_row kept for provenance.
NO training. Single-threaded.
"""
import os, json, hashlib, glob, time, shutil
import numpy as np
from scipy.signal import resample_poly

ROOT  = "/home/docker/pw26_akp_01/ext_data/ble_xiao"
CACHE = "/home/docker/pw26_akp_01/ext_cache/ble_b1_25msps"
AUDIT = "/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols/audit_out"
UP, DOWN, W, S = 25, 6, 256, 256
FLOOR_GB = 20.0
os.makedirs(CACHE, exist_ok=True)

def free_gb(path):
    st = shutil.disk_usage(path); return st.free/1e9

def collections():
    cs=[]
    for xt in sorted(glob.glob(os.path.join(ROOT,"*","*","X_train.npy"))):
        d=os.path.dirname(xt); cs.append(f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}")
    return cs

def build_one(coll):
    d=os.path.join(ROOT,coll)
    Xtr=np.asarray(np.load(os.path.join(d,"X_train.npy"),mmap_mode="r"))
    Xte=np.asarray(np.load(os.path.join(d,"X_test.npy"), mmap_mode="r"))
    ytr=np.asarray(np.load(os.path.join(d,"Y_train.npy"),mmap_mode="r")).argmax(1).astype(np.int16)
    yte=np.asarray(np.load(os.path.join(d,"Y_test.npy"), mmap_mode="r")).argmax(1).astype(np.int16)
    X=np.concatenate([Xtr,Xte],0); y=np.concatenate([ytr,yte],0)
    n_train=Xtr.shape[0]; N=X.shape[0]; L=X.shape[2]
    Lr=int(np.ceil(L*UP/DOWN)); nwin=(Lr-W)//S+1
    out=np.empty((N,nwin,2,W),dtype=np.float32)
    B=2000
    for s in range(0,N,B):
        chunk=X[s:s+B].astype(np.float32)                 # (b,2,L)
        r=resample_poly(chunk, UP, DOWN, axis=2)          # (b,2,Lr)
        starts=np.arange(nwin)*S
        # (b,2,nwin,W) -> (b,nwin,2,W)
        win=np.stack([r[:,:,st:st+W] for st in starts],axis=2)
        out[s:s+B]=np.transpose(win,(0,2,1,3))
    seg_row=np.arange(N,dtype=np.int32)
    path=os.path.join(CACHE, coll.replace(" ","_").replace("(","").replace(")","").replace("/","__")+".npz")
    np.savez(path, X_win=out, y_idx=y, seg_row=seg_row, n_train=np.int32(n_train),
             fs=np.float32(UP/DOWN*6e6), win=np.int32(W), stride=np.int32(S), Lr=np.int32(Lr))
    h=hashlib.sha256(open(path,"rb").read()).hexdigest()
    sz=os.path.getsize(path)/1e9
    return dict(coll=coll, path=path, N=int(N), nwin=int(nwin), Lr=int(Lr),
                total_windows=int(N*nwin), size_gb=round(sz,2), sha256=h)

def main():
    cs=collections(); print(f"{len(cs)} collections; free={free_gb(CACHE):.0f} GB (floor {FLOOR_GB})")
    stats=[]; t0=time.time()
    for c in cs:
        fg=free_gb(CACHE)
        if fg < FLOOR_GB + 6:   # ~4.5 GB per file; refuse to breach floor
            print(f"ABORT: free {fg:.1f} GB near floor — stop before {c}"); break
        t=time.time(); rec=build_one(c); dt=time.time()-t
        stats.append(rec)
        print(f"[b1] {c:26s} N={rec['N']:6d} nwin={rec['nwin']} winTot={rec['total_windows']:8d} "
              f"{rec['size_gb']:.2f}GB {dt:.0f}s sha={rec['sha256'][:12]} free={free_gb(CACHE):.0f}GB")
    tot_gb=sum(s['size_gb'] for s in stats); tot_w=sum(s['total_windows'] for s in stats)
    print(f"\nTOTAL {len(stats)} files  {tot_gb:.1f} GB  {tot_w} windows  {time.time()-t0:.0f}s  free_now={free_gb(CACHE):.0f}GB")
    # fixity md
    lines=["# B1 25 MS/s window cache — fixity + stats","",
           f"resample 6->25 (polyphase {UP}/{DOWN}); Lr={stats[0]['Lr'] if stats else '?'}; "
           f"window {W} stride {S}; {stats[0]['nwin'] if stats else '?'} win/seg; float32 (N,nwin,2,256)","",
           "| collection | segs | win/seg | total windows | size GB | sha256 |","|---|---|---|---|---|---|"]
    for s in stats:
        lines.append(f"| {s['coll']} | {s['N']} | {s['nwin']} | {s['total_windows']} | {s['size_gb']} | {s['sha256']} |")
    lines.append(f"| **TOTAL** | | | {tot_w} | **{tot_gb:.1f}** | |")
    open(os.path.join(AUDIT,"b1_cache_fixity.md"),"w").write("\n".join(lines)+"\n")
    json.dump(stats, open(os.path.join(AUDIT,"b1_cache_stats.json"),"w"), indent=2)
    print("WROTE audit_out/b1_cache_fixity.md")

if __name__=="__main__":
    main()
