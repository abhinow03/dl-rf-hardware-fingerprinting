#!/usr/bin/env python3
"""Phase 2 Stage B — extract LOCKED classical_b (19-D) for ALL segments, cache per
collection on the docker volume, sha256 each. Pooled train+test; y_idx kept. Tiny."""
import os, glob, json, hashlib, time
import numpy as np
from rffp.discovery.ext_ble.features_ble_classical import extract, FEATURE_NAMES

ROOT="/home/docker/pw26_akp_01/ext_data/ble_xiao"
CACHE="/home/docker/pw26_akp_01/ext_cache/ble_classical_b"
AUDIT="/home/pw26_akp_01/CAPSTONE/DL_model/gen_rff/ext_protocols/audit_out"
os.makedirs(CACHE,exist_ok=True)

def collections():
    cs=[]
    for xt in sorted(glob.glob(os.path.join(ROOT,"*","*","X_train.npy"))):
        d=os.path.dirname(xt); cs.append(f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}")
    return cs

def main():
    stats=[]; t0=time.time()
    for c in collections():
        d=os.path.join(ROOT,c)
        Xtr=np.load(os.path.join(d,"X_train.npy"),mmap_mode="r"); Xte=np.load(os.path.join(d,"X_test.npy"),mmap_mode="r")
        ytr=np.asarray(np.load(os.path.join(d,"Y_train.npy"),mmap_mode="r")).argmax(1)
        yte=np.asarray(np.load(os.path.join(d,"Y_test.npy"),mmap_mode="r")).argmax(1)
        n_tr=Xtr.shape[0]; N=n_tr+Xte.shape[0]
        F=np.empty((N,19),dtype=np.float32); B=4000
        for s in range(0,n_tr,B):
            e=min(s+B,n_tr); F[s:e]=extract(np.asarray(Xtr[s:e]))
        n_te=Xte.shape[0]
        for s in range(0,n_te,B):
            e=min(s+B,n_te); F[n_tr+s:n_tr+e]=extract(np.asarray(Xte[s:e]))
        y=np.concatenate([ytr,yte]).astype(np.int16)
        path=os.path.join(CACHE, c.replace(" ","_").replace("(","").replace(")","").replace("/","__")+".npz")
        np.savez(path, F=F, y=y, n_train=np.int32(n_tr))
        h=hashlib.sha256(open(path,"rb").read()).hexdigest()
        stats.append(dict(coll=c,N=int(N),path=path,size_mb=round(os.path.getsize(path)/1e6,2),sha256=h))
        print(f"[feat] {c:26s} N={N:6d} finite={np.isfinite(F).all()} {stats[-1]['size_mb']}MB sha={h[:12]}")
    tot=sum(s['size_mb'] for s in stats)
    print(f"TOTAL {len(stats)} files {tot:.1f} MB {time.time()-t0:.0f}s")
    lines=["# classical_b 19-D feature cache — fixity","",f"features: {FEATURE_NAMES}","",
           "| collection | N | size MB | sha256 |","|---|---|---|---|"]
    for s in stats: lines.append(f"| {s['coll']} | {s['N']} | {s['size_mb']} | {s['sha256']} |")
    open(os.path.join(AUDIT,"classical_b_cache_fixity.md"),"w").write("\n".join(lines)+"\n")
    print("WROTE audit_out/classical_b_cache_fixity.md")

if __name__=="__main__": main()
