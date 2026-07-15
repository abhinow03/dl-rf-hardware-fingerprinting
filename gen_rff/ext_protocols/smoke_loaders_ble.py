#!/usr/bin/env python3
"""
Phase 1 STEP 5 — loader smoke. Instantiate (a) native zero-copy dataset and (b) B1 cache
dataset; pull one batch each; check 1D / 2D / B1-256 shapes; verify one-hot<->index
round-trip; print per-batch timing. NO model forward.
"""
import os, glob, time
import numpy as np
from scipy.signal import stft

ROOT  = "/home/docker/pw26_akp_01/ext_data/ble_xiao"
CACHE = "/home/docker/pw26_akp_01/ext_cache/ble_b1_25msps"

# ---------- (a) native zero-copy dataset ----------
class NativeBLE:
    """Indexes existing per-collection .npy via (collection,row) with mmap. Returns
    1D (2,1850) and on-the-fly 2D STFT (2,128,27)."""
    def __init__(self, collections):
        self.mm = {c: (np.load(os.path.join(ROOT,c,"X_train.npy"),mmap_mode="r"),
                       np.load(os.path.join(ROOT,c,"Y_train.npy"),mmap_mode="r")) for c in collections}
    def get_batch(self, coll, rows):
        X,Y = self.mm[coll]
        x = np.asarray(X[rows]).astype(np.float32)          # (B,2,1850) zero-copy view -> materialized batch
        y1h = np.asarray(Y[rows])                            # (B,31) one-hot
        y = y1h.argmax(1)
        z = x[:,0,:] + 1j*x[:,1,:]
        _,_,Zxx = stft(z, fs=25e6, nperseg=128, noverlap=64, boundary=None, padded=False)  # (B,128,27) cplx
        x2d = np.stack([Zxx.real, Zxx.imag], axis=1).astype(np.float32)                     # (B,2,128,27)
        return x, x2d, y, y1h

# ---------- (b) B1 cache dataset ----------
class B1CacheBLE:
    def __init__(self, npz_path):
        self.z = np.load(npz_path, mmap_mode="r")
        self.X = self.z["X_win"]; self.y = self.z["y_idx"]     # (N,30,2,256), (N,)
    def get_batch(self, rows):
        xb = np.asarray(self.X[rows]).astype(np.float32)       # (B,30,2,256)
        yb = np.asarray(self.y[rows]).astype(int)
        return xb, yb

def main():
    colls = ["Wired (indoors)/Ch1_R1", "Wireless (outdoors)/Loc1"]
    nd = NativeBLE(colls)
    rows = np.arange(64)

    t=time.time(); x1d,x2d,y,y1h = nd.get_batch(colls[0], rows); dt=(time.time()-t)*1000
    print(f"[native] {colls[0]}")
    print(f"  1D branch batch : {x1d.shape} {x1d.dtype}   (expect (64,2,1850))")
    print(f"  2D branch batch : {x2d.shape} {x2d.dtype}   (expect (64,2,128,27))")
    print(f"  labels idx      : {y.shape} vals[:8]={y[:8].tolist()}")
    # one-hot <-> index round-trip
    recon = np.zeros_like(y1h); recon[np.arange(len(y)), y] = 1
    rt = bool(np.array_equal(recon, y1h)) and bool(np.all(y1h.sum(1)==1))
    print(f"  one-hot<->index round-trip: {'PASS' if rt else 'FAIL'}")
    print(f"  batch time      : {dt:.1f} ms (64 segs, incl. STFT)")

    # native on a second (outdoor) collection — shape stability
    t=time.time(); x1d2,x2d2,y2,_ = nd.get_batch(colls[1], rows); dt2=(time.time()-t)*1000
    print(f"[native] {colls[1]}: 1D {x1d2.shape} 2D {x2d2.shape}  {dt2:.1f} ms")

    # B1 cache
    npzs = sorted(glob.glob(os.path.join(CACHE,"*.npz")))
    if not npzs:
        print("[b1] NO cache files found — build_b1_cache_ble.py incomplete"); return
    b1 = B1CacheBLE(npzs[0])
    t=time.time(); xb,yb = b1.get_batch(rows); dt=(time.time()-t)*1000
    print(f"[b1] {os.path.basename(npzs[0])}")
    print(f"  window batch    : {xb.shape} {xb.dtype}   (expect (64,30,2,256))")
    print(f"  per-window view : {xb.reshape(-1,2,256).shape}  (flattened for frozen encoder)")
    print(f"  labels idx      : {yb.shape} vals[:8]={yb[:8].tolist()}")
    print(f"  batch time      : {dt:.1f} ms (64 segs x 30 win)")
    print(f"[b1] cache files present: {len(npzs)}/12")

if __name__ == "__main__":
    main()
