"""MULTI-CLASS PROBE — is the discovery gap an ENCODER problem or a WINDOW problem?
Cache + ONE small fresh classifier. NO encoder retrain, NO drones. STOP at checkpoint.

TEST 2 proved hard-pair centroids COINCIDE (distinct devices -> same point). A raw-IQ BINARY
probe already hit 96% on those pairs from [2,256]. Open question: does the signal for ~18
devices SIMULTANEOUSLY exist at 256 samples?
  - closed-set 18-way acc HIGH  -> window is FINE; the gap is the metric ENCODER's capacity/
    representation (fix via arch/width/embed-dim, NOT re-windowing).
  - closed-set 18-way acc LOW   -> 256 samples lacks multi-device info -> longer windows justified.

Fresh small dual-branch CNN (raw IQ [2,256] + STFT [2,33,13] — the SAME inputs the encoder sees),
plain cross-entropy, per-signal balanced split. Reports 18-way val acc + per-class acc + confusion,
on the random seed-123 slice AND an easy-18 control. Cross-checks confusions vs encoder coincident
centroids. Does NOT touch best_model.pt / the encoder weights (encoder loaded read-only for the
cross-check centroids only).

    python3 discover/multiclass_probe.py
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rffp.models import RFEncoder
from rffp.data import wisig_manytx as W
from sklearn.metrics import confusion_matrix

RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR   = os.path.join(RUN_DIR, "discover")
SPLIT_OLD = os.path.join(RUN_DIR, "splits", "split_manytx.json")
BASE_CKPT = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
REPORT    = os.path.join(OUT_DIR, "multiclass_probe_report.json")

RAND_N, RAND_SEED = 18, 123
PER_DEV = 2000            # signals/device (balanced); each held dev has >=2908
EPOCHS, BS, LR, WD = 60, 256, 1e-3, 1e-4
VAL_FRAC = 0.2
SEED = 42


def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)


def unitrows(M):
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)


def scatter_slice(held_tx, n, seed):
    rng = np.random.default_rng(seed)
    rows = {}
    for t in held_tx:
        rows.setdefault(int(t.split("-")[0]), []).append(t)
    for r in rows:
        rng.shuffle(rows[r])
    keys = sorted(rows); ptr = {r: 0 for r in keys}; out = []
    while len(out) < n:
        prog = False
        for r in keys:
            if len(out) >= n:
                break
            if ptr[r] < len(rows[r]):
                out.append(rows[r][ptr[r]]); ptr[r] += 1; prog = True
        if not prog:
            break
    return sorted(out)


class DualCNN(nn.Module):
    """small dual-branch classifier — mirrors the encoder's two inputs, but supervised CE."""
    def __init__(self, ncls):
        super().__init__()
        self.iq = nn.Sequential(
            nn.Conv1d(2, 64, 7, padding=3), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.GroupNorm(8, 128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.sp = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, ncls))

    def forward(self, xt, xs):
        a = self.iq(xt).flatten(1)
        b = self.sp(xs).flatten(1)
        return self.head(torch.cat([a, b], 1))


def build_dataset(tx_data, devs, per_dev, seed):
    """Balanced per-signal set: per_dev random signals/device, standardized IQ + STFT."""
    rng = np.random.default_rng(seed)
    times, labels = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        idx = rng.choice(iq.shape[0], size=min(per_dev, iq.shape[0]), replace=False)
        for k in idx:
            times.append(W.standardize(iq[k].T.copy())); labels.append(di)
    t = np.stack(times).astype(np.float32)
    s = W.compute_stft_batch(t).astype(np.float32)
    y = np.array(labels)
    return t, s, y


def train_probe(t, s, y, ncls, tag):
    """stratified 80/20 split, train DualCNN, return val acc + preds/true on val."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    tr_idx, va_idx = [], []
    for c in range(ncls):
        ci = np.where(y == c)[0]; rng.shuffle(ci)
        nv = int(len(ci) * VAL_FRAC)
        va_idx.extend(ci[:nv]); tr_idx.extend(ci[nv:])
    tr_idx = np.array(tr_idx); va_idx = np.array(va_idx)

    T = torch.from_numpy(t).cuda(); S = torch.from_numpy(s).cuda(); Y = torch.from_numpy(y).cuda()
    model = DualCNN(ncls).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    best_acc, best_pred = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = tr_idx[np.random.permutation(len(tr_idx))]
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]; bi = torch.from_numpy(b).cuda()
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(T[bi], S[bi]); loss = lossf(out, Y[bi])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        # eval
        model.eval(); preds = np.empty(len(va_idx), dtype=np.int64)
        with torch.no_grad():
            for i in range(0, len(va_idx), 1024):
                b = va_idx[i:i + 1024]; bi = torch.from_numpy(b).cuda()
                with torch.amp.autocast('cuda'):
                    preds[i:i + 1024] = model(T[bi], S[bi]).argmax(1).cpu().numpy()
        acc = float((preds == y[va_idx]).mean())
        if acc > best_acc:
            best_acc, best_pred = acc, preds.copy()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{tag}] ep {ep+1:2d} val_acc={acc:.3f} (best {best_acc:.3f})", flush=True)
    del T, S, Y, model; torch.cuda.empty_cache()
    return best_acc, best_pred, y[va_idx]


@torch.no_grad()
def encoder_centroids(tx_data, devs, per_dev=400, seed=1000):
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    m.eval()
    rng = np.random.default_rng(seed)
    embs, dids = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        idx = rng.choice(iq.shape[0], size=min(per_dev, iq.shape[0]), replace=False)
        t = np.stack([W.standardize(iq[k].T.copy()) for k in idx]).astype(np.float32)
        s = W.compute_stft_batch(t)
        with torch.amp.autocast('cuda'):
            e = m(torch.from_numpy(t).cuda(), torch.from_numpy(s).cuda()).float().cpu().numpy()
        embs.append(e); dids.append(np.full(len(idx), di))
    del m; torch.cuda.empty_cache()
    E = np.concatenate(embs); D = np.concatenate(dids)
    C = unitrows(np.stack([E[D == d].mean(0) for d in range(len(devs))]))
    M = C @ C.T; np.fill_diagonal(M, -2.0)
    return M


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    tx_data, _ = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); held_tx = sp["discover_tx"]
    rand_tx = scatter_slice(held_tx, RAND_N, RAND_SEED)
    print(f"random slice: {rand_tx}")

    # easy-18 control (reproduce burst_probe TEST 3 exactly): all-41 centroids, lowest nn-confuser
    Mall = encoder_centroids(tx_data, held_tx, per_dev=400, seed=1000)
    nn_all = Mall.max(1)
    easy_tx = [held_tx[i] for i in np.argsort(nn_all)[:RAND_N]]
    print(f"easy-18 control: {easy_tx}")

    # ═══ RANDOM SLICE 18-way ═══
    print(f"\n=== 18-way probe on RANDOM slice (raw [2,256] + STFT, {PER_DEV}/dev) ===")
    t, s, y = build_dataset(tx_data, rand_tx, PER_DEV, seed=SEED)
    r_acc, r_pred, r_true = train_probe(t, s, y, RAND_N, "rand")
    del t, s, y
    cm = confusion_matrix(r_true, r_pred, labels=list(range(RAND_N)))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    per_cls = {rand_tx[i]: round(float(cmn[i, i]), 3) for i in range(RAND_N)}

    # ═══ EASY CONTROL 18-way ═══
    print(f"\n=== 18-way probe on EASY control ===")
    te, se, ye = build_dataset(tx_data, easy_tx, PER_DEV, seed=SEED)
    e_acc, e_pred, e_true = train_probe(te, se, ye, RAND_N, "easy")
    del te, se, ye

    # ═══ CROSS-CHECK vs encoder coincident pairs (random slice) ═══
    Mr = encoder_centroids(tx_data, rand_tx, per_dev=400, seed=2000)
    enc_nn = Mr.max(1)          # encoder nearest-confuser cos per device (in slice)
    enc_nn_dev = Mr.argmax(1)

    # classifier's biggest confusion per device
    off = cmn.copy(); np.fill_diagonal(off, 0.0)
    clf_conf_dev = off.argmax(1); clf_conf_rate = off.max(1)

    print(f"\n=== PER-DEVICE (random slice): classifier acc vs encoder coincidence ===")
    print(f"{'device':>8} {'clf_acc':>8} {'clf_confused_with':>18} {'rate':>6} "
          f"{'enc_nn_cos':>10} {'enc_nn_dev':>10}")
    stuck, clean = [], []
    rows = []
    for i in range(RAND_N):
        row = {"device": rand_tx[i], "clf_acc": round(float(cmn[i, i]), 3),
               "clf_confused_with": rand_tx[clf_conf_dev[i]], "clf_conf_rate": round(float(clf_conf_rate[i]), 3),
               "enc_nn_cos": round(float(enc_nn[i]), 4), "enc_nn_dev": rand_tx[enc_nn_dev[i]]}
        rows.append(row)
        (stuck if cmn[i, i] < 0.5 else clean).append(rand_tx[i])
        print(f"{rand_tx[i]:>8} {cmn[i,i]:>8.3f} {rand_tx[clf_conf_dev[i]]:>18} "
              f"{clf_conf_rate[i]:>6.3f} {enc_nn[i]:>10.4f} {rand_tx[enc_nn_dev[i]]:>10}")

    # top confusion pairs, and how many match encoder coincident pairs
    iu = np.triu_indices(RAND_N, 1)
    pair_conf = (off + off.T)[iu]
    top = np.argsort(pair_conf)[::-1][:8]
    print(f"\n top classifier confusion pairs vs encoder centroid cos:")
    matches = 0
    for o in top:
        a, b = iu[0][o], iu[1][o]
        ec = float(Mr[a, b])
        mk = " <== also coincides in encoder (cos>0.97)" if ec > 0.97 else ""
        if ec > 0.97:
            matches += 1
        print(f"   {rand_tx[a]:>6}/{rand_tx[b]:<6} clf_conf={pair_conf[o]:.3f}  enc_cos={ec:.4f}{mk}")

    # ── interpretation ──
    print(f"\n=== INTERPRETATION ===")
    print(f"  RANDOM 18-way val acc = {r_acc:.3f}   |   EASY control = {e_acc:.3f}")
    print(f"  cleanly separable by classifier (acc>=0.5): {len(clean)}/{RAND_N}; "
          f"stuck (<0.5): {stuck}")
    if r_acc > 0.85:
        verdict = (f"WINDOW IS FINE: a fresh 18-way classifier reaches {r_acc:.3f} on [2,256] — the "
                   f"multi-device signal IS present at 256 samples. The discovery gap is the metric "
                   f"ENCODER's representation/capacity, NOT the input window. NEXT LEVER = architecture "
                   f"(embedding dim / width / arch), NOT expensive re-windowing.")
    elif r_acc < 0.55:
        verdict = (f"WINDOW PROBLEM: even a dedicated classifier caps at {r_acc:.3f} on [2,256] — 256 "
                   f"samples genuinely lacks the multi-device signal. LONGER INPUT WINDOWS justified next.")
    else:
        verdict = (f"MIXED ({r_acc:.3f}): classifier does clearly better than discovery (~0.44 ARI proxy) "
                   f"but not saturated. Partly encoder capacity, partly window. If EASY control is high "
                   f"and stuck devices coincide with encoder pairs, those specific devices are near-identical "
                   f"hardware (data ceiling); the rest is encoder-fixable.")
    if e_acc > 0.9:
        verdict += f" EASY control {e_acc:.3f}>0.9 confirms the probe is sound (low != broken probe)."
    if stuck:
        verdict += f" {len(stuck)} device(s) STUCK for even the classifier ({stuck}) -> likely genuine near-identical hardware."
    print(f"  VERDICT: {verdict}")

    with open(REPORT, "w") as f:
        json.dump({"config": {"per_dev": PER_DEV, "epochs": EPOCHS, "rand_seed": RAND_SEED},
                   "random_slice": rand_tx, "easy_slice": easy_tx,
                   "random_val_acc": r_acc, "easy_val_acc": e_acc,
                   "per_class_acc": per_cls, "per_device_rows": rows,
                   "confusion_norm": cmn.round(3).tolist(),
                   "clean_count": len(clean), "stuck_devices": stuck,
                   "enc_matches_in_top_conf": int(matches), "verdict": verdict}, f, indent=2)
    print(f"\nsaved -> {REPORT}\nCHECKPOINT — cache + one fresh probe. best_model.pt/encoder UNTOUCHED. No drones.")


if __name__ == "__main__":
    main()
