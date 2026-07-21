#!/usr/bin/env python3
"""Phase 3c T5 compute for B3 native: params, ms/segment (GPU + 1-core CPU), embedding dim,
embedding cache size. Native forward = 1D (2,1850) + native STFT (2,128,27) -> 128-D head."""
import os, sys, glob, time, json, argparse
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np, torch
_HERE=os.path.dirname(os.path.abspath(__file__)); _DLM=os.path.abspath(os.path.join(_HERE,"..","..")); _SW=os.path.join(_DLM,"summer_work")
for p in (_DLM,_SW):
    if p not in sys.path: sys.path.insert(0,p)
from rffp.models import RFEncoder

def native_stft(x,w):
    z=torch.complex(x[:,0].float(),x[:,1].float())
    Z=torch.stft(z,n_fft=128,hop_length=64,win_length=128,window=w,center=False,return_complex=True)
    return torch.stack([Z.real,Z.imag],dim=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); ap.add_argument("--emb",required=True)
    a=ap.parse_args()
    m=RFEncoder(); m.load_state_dict(torch.load(a.ckpt,map_location="cpu",weights_only=True),strict=True); m.eval()
    np_=sum(p.numel() for p in m.parameters())
    # GPU ms/seg (batched 2048)
    mg=RFEncoder().cuda(); mg.load_state_dict(torch.load(a.ckpt,map_location="cuda",weights_only=True),strict=True); mg.eval()
    wg=torch.hann_window(128,device="cuda"); xb=torch.randn(2048,2,1850,device="cuda")
    with torch.no_grad():
        for _ in range(2):
            with torch.amp.autocast('cuda'): _=mg(xb,native_stft(xb,wg))
        torch.cuda.synchronize(); t=time.time()
        for _ in range(5):
            with torch.amp.autocast('cuda'): _=mg(xb,native_stft(xb,wg))
        torch.cuda.synchronize(); gpu_ms=(time.time()-t)/5/2048*1000
    # 1-core CPU ms/seg
    torch.set_num_threads(1); wc=torch.hann_window(128); xc=torch.randn(1,2,1850)
    with torch.no_grad():
        _=m(xc,native_stft(xc,wc)); t=time.time()
        for _ in range(20): _=m(xc,native_stft(xc,wc))
        cpu_ms=(time.time()-t)/20*1000
    cache=sum(os.path.getsize(p) for p in glob.glob(f"{a.emb}/*.npz"))/1e6
    r=dict(params=int(np_),embedding_dim=128,gpu_ms_per_segment=round(gpu_ms,4),cpu1core_ms_per_segment=round(cpu_ms,3),cache_mb=round(cache,1))
    print(json.dumps(r,indent=2)); json.dump(r,open(f"{a.emb}/_t5.json","w"),indent=2)

if __name__=="__main__": main()
