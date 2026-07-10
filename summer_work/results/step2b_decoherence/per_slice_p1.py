"""Per-slice confirmation: does CosFace beat 128-D under P1 on EVERY DEV slice (not just pooled)?
Reuses decohere.py machinery (frozen encoder, head reproduced deterministically, DEV only)."""
import os, sys, json
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_SW, os.path.join(_SW, "datasets"), os.path.join(_SW, "discover"),
          os.path.join(_SW, "results", "step2_integrity"), _HERE):
    if p not in sys.path: sys.path.insert(0, p)
from shared import RFEncoder
import wisig_manytx as W
from sklearn.preprocessing import StandardScaler
import geometry_consolidate as GC
from burst_probe import embed_times
from integrity import score_locked, hdbscan_pred
import decohere as D

TXD, _ = W.load_manytx(eq=0, verbose=False)
D.TXD = TXD
sp = json.load(open(D.SPLIT)); part = json.load(open(D.PART))
train_tx = sp["train_tx"]; dev_tx = part["dev_tx"]; n_cls = len(train_tx)
model = RFEncoder().cuda()
model.load_state_dict(torch.load(D.BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
tt = np.concatenate([GC.consec_windows(TXD, tx, GC.TRAIN_PER_DEV) for tx in train_tx])
ty = np.concatenate([np.full(min(GC.TRAIN_PER_DEV, TXD[tx]["iq"].shape[0]), i) for i, tx in enumerate(train_tx)])
Ftr = GC.extract512(model, tt); del tt
scaler = StandardScaler().fit(Ftr)
head = GC.train_cosface(torch.from_numpy(scaler.transform(Ftr).astype(np.float32)).cuda(),
                        torch.from_numpy(ty).cuda(), n_cls, D.COS_M, D.COS_S)
cache128, cacheCF = {}, {}
for tx in dev_tx:
    t = GC.consec_windows(TXD, tx, D.WIN_CACHE)
    rx = TXD[tx]["rx"][:t.shape[0]].copy(); date = TXD[tx]["date"][:t.shape[0]].copy()
    e512 = GC.extract512(model, t)
    cache128[tx] = dict(emb=embed_times(model, t), rx=rx, date=date)
    cacheCF[tx]  = dict(emb=GC.apply_head(head, {tx: scaler.transform(e512).astype(np.float32)})[tx], rx=rx, date=date)
caches = {"128D": cache128, "CosFace": cacheCF}

print(f"{'seed':>6} {'128D_P1':>9} {'CosFace_P1':>11} {'CF>128D?':>9}")
wins = 0
for s in D.SLICE_SEEDS:
    sl = GC.scatter_slice(dev_tx, 18, s)
    row = {}
    for emb in ("128D", "CosFace"):
        # average the 3-mcs jitter per slice
        aris = []
        for mcs in D.MCS_JITTER:
            bp, bl, _ = D.build_bursts(caches[emb], sl, "P1", seed=s)
            aris.append(score_locked(hdbscan_pred(bp, mcs), bl)["ARI"])
        row[emb] = float(np.mean(aris))
    win = row["CosFace"] > row["128D"]; wins += win
    print(f"{s:>6} {row['128D']:>9.3f} {row['CosFace']:>11.3f} {'YES' if win else 'NO':>9}")
print(f"\nCosFace beats 128-D under P1 on {wins}/5 slices.")
