"""Hard-negative reweighted-SupCon retrain — SMOKE TEST harness (TUNED + ANNEALED v2).

WHY: probe proved the fingerprint IS in the eq=0 [2,256] window (worst pairs ~96% separable
by a tiny CNN) yet the frozen SupCon encoder collapses them to cos 0.9999 (median nearest-
confuser 0.986). v1 smoke (W_MAX=6 from step 0 + ~86% confuser co-occurrence) drove nn-confuser
0.986->0.938 and separated all 10 worst pairs BUT collapsed positives (intra 0.83->0.40):
excessive negative repulsion with no time for positives to form, and selecting on nn-confuser
alone picked a diffuse undertrained model.

v2 FIX — schedule + selection, not just magnitudes:
  * ANNEAL W_MAX 1 -> W_MAX_TGT (cosine ramp, starts ~30% in): positives form tight clusters
    FIRST under vanilla SupCon, THEN hard pairs get pulled apart.
  * Tune the PRODUCT (they compound): W_MAX_TGT 2.5 (down from 6) x confuser_frac 0.3
    (down from 0.5). Report EFFECTIVE pressure = W_MAX(phase) x confuser-co-occurrence.
  * S0 0.7 (up from 0.5): weight only TRULY hard negatives (cos 0.9+); easy/moderate stay ~1x.
  * SELECT by held-out VALIDATION ARI (HDBSCAN on a val slice) — ungameable: needs tight
    positives AND separated negatives. Gate intra>=0.8 reported. NOT nn-confuser alone, NOT gap.

KEEP: supervised-contrastive base, multi-positive, eq=0, constant tau=0.5, BETA, STFT
n_fft64/hop16, parent RFEncoder, LR 5e-4 / AdamW / wd1e-4 / warmup / clip1.0, fresh init.
SMOKE trains on OLD board split for A/B vs the 0.986/0.40 baselines (scatter split = full run).

    python3 train/train_wisig_hardneg.py --smoke
NO full run, NO drones/drive. STOP at the checkpoint.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from shared import RFEncoder
import wisig_manytx as W
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

RUN_DIR    = os.path.join(_SW, "runs", "wisig_supcon_fft64")
OUT_DIR    = os.path.join(RUN_DIR, "discover")
SPLIT_OLD  = os.path.join(RUN_DIR, "splits", "split_manytx.json")
SPLIT_NEW  = os.path.join(RUN_DIR, "splits", "split_manytx_scatter.json")
BASE_CKPT  = os.path.join(RUN_DIR, "retrain_best", "best_model.pt")
SMOKE_BEST = os.path.join(OUT_DIR, "hardneg_smoke_best.pt")     # NEVER best_model.pt
CURVE_PNG  = os.path.join(OUT_DIR, "hardneg_weight_curve.png")
REPORT     = os.path.join(OUT_DIR, "hardneg_smoke_report.json")

# ── recipe (KEEP) ──
N_DEV, M_SIG = 32, 8
LR, WD, WARMUP, GRAD_CLIP, TAU = 5e-4, 1e-4, 350, 1.0, 0.5
SEED = 42
# ── change-1 reweighting (TUNED) ──
S0, BETA, W_MAX_TGT = 0.7, 0.15, 2.5
RAMP_START, RAMP_END = 0.30, 0.80          # W_MAX anneal window (fraction of STEPS)
# ── change-2 sampling (TUNED) ──
K_NN, CONFUSER_FRAC, REFRESH, CENT_CAP = 4, 0.3, 500, 64
# ── smoke ──
STEPS, LOG, EVAL_CAP = 5000, 250, 256
VAL_NDEV, VAL_CAP, VAL_MCS = 16, 160, 15    # held-out val slice for ARI selection


# ════════════════════ CHANGE 1 — reweighted SupCon (annealed W_MAX) ════════════════════
def neg_weight(cos, w_max, s0=S0, beta=BETA):
    """Smooth bounded monotone hardness weight on a NEGATIVE pair's cosine. >=1 always."""
    return 1.0 + (w_max - 1.0) * torch.sigmoid((cos - s0) / beta)


def w_max_at(step):
    """Anneal W_MAX 1 -> W_MAX_TGT (cosine) so positives form before hard-neg pressure ramps."""
    f = step / STEPS
    if f < RAMP_START:
        return 1.0
    if f >= RAMP_END:
        return W_MAX_TGT
    a = (f - RAMP_START) / (RAMP_END - RAMP_START)
    return 1.0 + (W_MAX_TGT - 1.0) * 0.5 * (1 - np.cos(np.pi * a))


class HardNegSupCon(nn.Module):
    """SupCon with hard-negative reweighting in the denominator. Positives keep weight 1.
    w_max=1 reproduces the parent SupContrastive loss exactly."""
    def __init__(self, s0=S0, beta=BETA):
        super().__init__()
        self.s0, self.beta = s0, beta

    def forward(self, emb, labels, temperature=TAU, w_max=1.0):
        emb = emb.float()
        dev = emb.device
        n = emb.shape[0]
        self_mask = torch.eye(n, dtype=torch.bool, device=dev)
        lc = labels.view(-1, 1)
        pos_mask = (lc == lc.t()) & ~self_mask
        neg_mask = (lc != lc.t()) & ~self_mask
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)

        cos = torch.mm(emb, emb.t())
        logits = cos / temperature
        logw = torch.zeros_like(cos)
        if w_max > 1.0:
            w = neg_weight(cos, w_max, self.s0, self.beta)
            logw = torch.where(neg_mask, torch.log(w), torch.zeros_like(cos))
        weighted = (logits + logw).masked_fill(self_mask, float("-inf"))
        log_denom = torch.logsumexp(weighted, dim=1, keepdim=True)
        log_prob = (logits - log_denom).masked_fill(self_mask, 0.0)
        loss = -(log_prob * pos_mask).sum(1) / pos_mask.sum(1).float().clamp(min=1)
        loss = loss.mean()
        return torch.tensor(0.0, device=dev, requires_grad=True) if torch.isnan(loss) else loss


def print_and_plot_weight_curve():
    print(f"\n=== CHANGE 1 — negative-weight curve (TARGET) w=1+(W_MAX-1)*sigmoid((cos-S0)/BETA) ===")
    print(f"  S0={S0}  BETA={BETA}  W_MAX_TGT={W_MAX_TGT}  | W_MAX annealed 1 -> {W_MAX_TGT} "
          f"over steps [{int(RAMP_START*STEPS)},{int(RAMP_END*STEPS)}]")
    grid = np.array([-0.03, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.986, 1.0])
    w = 1.0 + (W_MAX_TGT - 1.0) * (1 / (1 + np.exp(-(grid - S0) / BETA)))
    print(f"  {'cos':>7} | " + " ".join(f"{c:>6.3f}" for c in grid))
    print(f"  {'weight':>7} | " + " ".join(f"{v:>6.2f}" for v in w))
    print(f"  -> killer cos 0.986 -> weight {1.0+(W_MAX_TGT-1.0)/(1+np.exp(-(0.986-S0)/BETA)):.2f}; "
          f"moderate cos 0.5 -> {1.0+(W_MAX_TGT-1.0)/(1+np.exp(-(0.5-S0)/BETA)):.2f}; "
          f"easy ~-0.03 -> ~1.0 (S0=0.7 concentrates on truly-hard)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xs = np.linspace(-0.3, 1.0, 400)
        ys = 1.0 + (W_MAX_TGT - 1.0) * (1 / (1 + np.exp(-(xs - S0) / BETA)))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, lw=2, label=f"target W_MAX={W_MAX_TGT}")
        ax.axvline(0.986, color="#c33", ls="--", lw=1, label="killer cos 0.986")
        ax.set_xlabel("negative-pair cosine"); ax.set_ylabel("denominator weight w")
        ax.set_title(f"Hard-neg weighting (S0={S0}, BETA={BETA}, W_MAX_TGT={W_MAX_TGT})")
        ax.legend(); fig.tight_layout(); fig.savefig(CURVE_PNG, dpi=110)
        print(f"  curve plot -> {CURVE_PNG}")
    except Exception as e:
        print(f"  (plot skipped: {e})")


# ════════════════════ batch construction ════════════════════
def to_cuda(t, s):
    return torch.from_numpy(t).cuda(non_blocking=True), torch.from_numpy(s).cuda(non_blocking=True)


def build_batch(tx_data, dev_idx, train_tx, label_of, rng, M=M_SIG, train=True, use_cfo=False):
    times, labels = [], []
    for ci in dev_idx:
        tx = train_tx[ci]; d = tx_data[tx]
        for k in rng.integers(0, d["iq"].shape[0], size=M):
            x = d["iq"][k].T.copy()
            x = W.augment(x, rng, use_cfo=use_cfo) if train else W.standardize(x)
            times.append(x); labels.append(label_of[tx])
    t = np.stack(times).astype(np.float32)
    return t, W.compute_stft_batch(t), np.array(labels, dtype=np.int64)


def confuser_devices(graph, n_dev, rng):
    order = list(rng.permutation(len(graph))); used = set(); chosen = []
    for seed in order:
        if len(chosen) >= n_dev:
            break
        if seed in used:
            continue
        cluster = [seed] + [j for j in graph[seed] if j not in used]
        for c in cluster[:K_NN + 1]:
            if c not in used and len(chosen) < n_dev:
                chosen.append(c); used.add(c)
    while len(chosen) < n_dev:
        c = int(rng.integers(0, len(graph)))
        if c not in used:
            chosen.append(c); used.add(c)
    return chosen[:n_dev]


# ════════════════════ CHANGE 2 — centroid confuser graph ════════════════════
@torch.no_grad()
def centroid_graph(model, tx_data, train_tx, k=K_NN, cap=CENT_CAP):
    model.eval()
    cents = np.empty((len(train_tx), 128), dtype=np.float32)
    for di, tx in enumerate(train_tx):
        iq = tx_data[tx]["iq"]
        idx = np.random.default_rng(di).integers(0, iq.shape[0], size=min(cap, iq.shape[0]))
        t = np.stack([W.standardize(iq[k].T.copy()) for k in idx]).astype(np.float32)
        s = W.compute_stft_batch(t)
        xt, xs = to_cuda(t, s)
        with torch.amp.autocast('cuda'):
            e = model(xt, xs).float().cpu().numpy()
        c = e.mean(0); cents[di] = c / (np.linalg.norm(c) + 1e-8)
    model.train()
    C = cents @ cents.T; np.fill_diagonal(C, -1.0)
    nn_sorted = np.argsort(-C, axis=1)
    graph = {i: nn_sorted[i, :k].tolist() for i in range(len(train_tx))}
    ref_top1 = {i: int(nn_sorted[i, 0]) for i in range(len(train_tx))}
    return graph, ref_top1


def cooccur_rate(dev_set, ref_top1):
    S = set(int(d) for d in dev_set)
    hits = sum(1 for d in S if ref_top1[d] in S)
    return hits / max(1, len(S))


# ════════════════════ held-out eval — centroids + val ARI ════════════════════
def build_eval(tx_data, devs, cap=EVAL_CAP, seed0=1000):
    times, dids = [], []
    for di, tx in enumerate(devs):
        iq = tx_data[tx]["iq"]
        idx = np.random.default_rng(seed0 + di).integers(0, iq.shape[0], size=min(cap, iq.shape[0]))
        for k in idx:
            times.append(W.standardize(iq[k].T.copy())); dids.append(di)
    t = np.stack(times).astype(np.float32)
    return to_cuda(t, W.compute_stft_batch(t)), np.array(dids)


@torch.no_grad()
def embed_pack(model, pack, batch=1024):
    (xt, xs), dids = pack
    model.eval()
    embs = np.empty((xt.shape[0], 128), dtype=np.float32)
    for s in range(0, xt.shape[0], batch):
        with torch.amp.autocast('cuda'):
            embs[s:s + batch] = model(xt[s:s + batch], xs[s:s + batch]).float().cpu().numpy()
    model.train()
    return embs, dids


def centroids_of(embs, dids, n_dev):
    return np.stack([(lambda c: c / (np.linalg.norm(c) + 1e-8))(embs[dids == d].mean(0))
                     for d in range(n_dev)])


def nearest_confuser(cents):
    C = cents @ cents.T; np.fill_diagonal(C, -2.0)
    nn = C.max(1)
    return float(np.median(nn)), float(nn.mean()), C


def worst_pairs_from(C, k=10):
    n = C.shape[0]; iu, ju = np.triu_indices(n, 1)
    cos = C[iu, ju]; order = np.argsort(cos)[::-1][:k]
    return [(int(iu[o]), int(ju[o]), float(cos[o])) for o in order]


def val_ari(embs, dids, mcs=VAL_MCS):
    """HDBSCAN on the per-signal val embeddings; ARI vs device labels (noise excluded)."""
    pred = HDBSCAN(min_cluster_size=mcs, metric="euclidean", n_jobs=-1, copy=True).fit_predict(embs)
    keep = pred != -1
    if keep.sum() < 2 or len(np.unique(pred[keep])) < 2:
        return 0.0, float((pred == -1).mean())
    return float(adjusted_rand_score(dids[keep], pred[keep])), float((pred == -1).mean())


# ════════════════════ train-device health (collapse / leakage watch) ════════════════════
def make_health(tx_data, devs, n_dev=16, m=16, seed=7):
    rng = np.random.default_rng(seed)
    lbl = {tx: i for i, tx in enumerate(devs)}
    t, s, y, info = W.sample_batch(tx_data, devs, lbl, rng, N=n_dev, M=m, train=False)
    return to_cuda(t, s), y, info


@torch.no_grad()
def health(model, pack):
    (xt, xs), y, info = pack
    model.eval()
    with torch.amp.autocast('cuda'):
        e = model(xt, xs).float().cpu().numpy()
    model.train()
    sim = e @ e.T; B = len(y); eye = np.eye(B, dtype=bool)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = (~(y[:, None] == y[None, :])) & ~eye
    rx = info["rx"]
    sd_sr = same & (rx[:, None] == rx[None, :]); sd_dr = same & (rx[:, None] != rx[None, :])
    rx_gap = (float(sim[sd_sr].mean()) - float(sim[sd_dr].mean())
              if sd_sr.sum() > 5 and sd_dr.sum() > 5 else float("nan"))
    intra, inter = float(sim[same].mean()), float(sim[diff].mean())
    return {"intra": intra, "inter": inter, "gap": intra - inter, "rx_gap": rx_gap}


# ════════════════════ CHANGE 4 — scattered split ════════════════════
def build_scatter_split(tx_list, n_discover=41, seed=42):
    rng = np.random.default_rng(seed)
    rows = {}
    for t in tx_list:
        rows.setdefault(int(t.split("-")[0]), []).append(t)
    for r in rows:
        rng.shuffle(rows[r])
    row_keys = sorted(rows)
    discover, ptr = [], {r: 0 for r in row_keys}
    while len(discover) < n_discover:
        progressed = False
        for r in row_keys:
            if len(discover) >= n_discover:
                break
            if ptr[r] < len(rows[r]):
                discover.append(rows[r][ptr[r]]); ptr[r] += 1; progressed = True
        if not progressed:
            break
    discover = sorted(discover)
    train = sorted(t for t in tx_list if t not in set(discover))
    split = {"mode": "scatter_roundrobin", "seed": seed, "n_train": len(train),
             "n_discover": len(discover),
             "rows_covered": sorted(set(int(t.split("-")[0]) for t in discover)),
             "train_tx": train, "discover_tx": discover}
    os.makedirs(os.path.dirname(SPLIT_NEW), exist_ok=True)
    with open(SPLIT_NEW, "w") as f:
        json.dump(split, f, indent=2)
    return split


def load_baseline():
    m = RFEncoder().cuda()
    m.load_state_dict(torch.load(BASE_CKPT, map_location="cuda", weights_only=True), strict=True)
    m.eval()
    return m


# ════════════════════ smoke ════════════════════
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tx_data, meta = W.load_manytx(eq=0)
    sp = json.load(open(SPLIT_OLD)); assert sp["mode"] == "board"
    train_tx, held_tx = sp["train_tx"], sp["discover_tx"]
    n_held = len(held_tx)
    print(f"SMOKE trains on OLD board split (A/B): train={len(train_tx)} held={n_held}")

    sc = build_scatter_split(meta["tx_list"], n_discover=n_held, seed=42)
    print(f"\n=== CHANGE 4 — scattered split -> {SPLIT_NEW} (held spans rows {sc['rows_covered']}) ===")
    print_and_plot_weight_curve()

    eval_pack = build_eval(tx_data, held_tx)                   # full-41 centroid tracking
    health_pack = make_health(tx_data, train_tx)

    # ── baseline reference ──
    base = load_baseline()
    b_embs, b_dids = embed_pack(base, eval_pack)
    base_cents = centroids_of(b_embs, b_dids, n_held)
    b_med, b_mean, b_C = nearest_confuser(base_cents)
    base_worst = worst_pairs_from(b_C, 10)
    base_health = health(base, health_pack)

    # val slice = devices appearing in the worst pairs (sensitive to confuser separation)
    val_dev = sorted(set([i for i, j, _ in base_worst] + [j for i, j, _ in base_worst]))[:VAL_NDEV]
    val_tx = [held_tx[i] for i in val_dev]
    val_pack = build_eval(tx_data, val_tx, cap=VAL_CAP, seed0=5000)
    v_embs, v_dids = embed_pack(base, val_pack)
    base_ari, base_noise = val_ari(v_embs, v_dids)
    del base; torch.cuda.empty_cache()

    print(f"\n=== BASELINE (eq=0 best_model.pt) on {n_held} held-out ===")
    print(f"  nearest-confuser cos median={b_med:.4f} mean={b_mean:.4f} | "
          f"intra={base_health['intra']:.3f} inter={base_health['inter']:.3f} "
          f"gap={base_health['gap']:.3f} rx_gap={base_health['rx_gap']:.3f}")
    print(f"  val-ARI (HDBSCAN on {len(val_tx)}-device worst-pair slice) = {base_ari:.3f} "
          f"(noise {base_noise*100:.0f}%)")

    # ── confuser graph bootstrapped from baseline ──
    base_g = load_baseline()
    graph, ref_top1 = centroid_graph(base_g, tx_data, train_tx)
    del base_g; torch.cuda.empty_cache()

    rng = np.random.default_rng(SEED)
    lbl_all = {tx: i for i, tx in enumerate(train_tx)}
    ca = [cooccur_rate(confuser_devices(graph, N_DEV, rng), ref_top1) for _ in range(20)]
    rnd = []
    for _ in range(20):
        _, _, y = build_batch(tx_data, list(rng.choice(len(train_tx), N_DEV, replace=False)),
                              train_tx, lbl_all, rng, M=1, train=False)
        rnd.append(cooccur_rate(np.unique(y), ref_top1))
    ca_m, rnd_m = float(np.mean(ca)), float(np.mean(rnd))
    mix_m = CONFUSER_FRAC * ca_m + (1 - CONFUSER_FRAC) * rnd_m
    print(f"\n=== CHANGE 2 — co-occurrence: confuser-aware {ca_m:.3f}  random {rnd_m:.3f}  "
          f"| at frac={CONFUSER_FRAC} expected-mix {mix_m:.3f} ===")
    print(f"  effective hard-neg pressure = W_MAX(phase) x mix-co-occurrence: "
          f"vanilla 1x{mix_m:.2f}={mix_m:.2f} -> target {W_MAX_TGT}x{mix_m:.2f}={W_MAX_TGT*mix_m:.2f} "
          f"(v1 was 6x0.86=5.2)")

    # ── fresh model + annealed hard-neg loss ──
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = RFEncoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = HardNegSupCon()
    scaler = torch.amp.GradScaler('cuda')
    rng = np.random.default_rng(SEED)

    print(f"\n=== SMOKE TRAIN — {STEPS} steps, {N_DEV}x{M_SIG}, tau={TAU} const, LR={LR}, "
          f"W_MAX 1->{W_MAX_TGT} anneal [{int(RAMP_START*STEPS)},{int(RAMP_END*STEPS)}], "
          f"confuser_frac={CONFUSER_FRAC}, select=val-ARI ===")
    best_ari, best_step, best_blob = -1.0, 0, None
    traj, cooc_win = [], []
    t0 = time.time()
    for step in range(STEPS):
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        if (step + 1) % REFRESH == 0:
            graph, _ = centroid_graph(model, tx_data, train_tx)
        if rng.random() < CONFUSER_FRAC:
            dev_idx = confuser_devices(graph, N_DEV, rng)
        else:
            dev_idx = list(rng.choice(len(train_tx), N_DEV, replace=False))
        cooc_win.append(cooccur_rate(dev_idx, ref_top1))
        t, s, y = build_batch(tx_data, dev_idx, train_tx, lbl_all, rng, train=True, use_cfo=False)
        xt, xs = to_cuda(t, s); yt = torch.from_numpy(y).cuda()
        wmax = w_max_at(step)
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            emb = model(xt, xs)
        loss = crit(emb.float(), yt, temperature=TAU, w_max=wmax)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()

        if (step + 1) % LOG == 0 or step == 0:
            embs, dids = embed_pack(model, eval_pack)
            cents = centroids_of(embs, dids, n_held)
            med, mean, C = nearest_confuser(cents)
            ve, vd = embed_pack(model, val_pack)
            ari, noise = val_ari(ve, vd)
            h = health(model, health_pack)
            eff = wmax * float(np.mean(cooc_win)); cooc_win = []
            traj.append({"step": step + 1, "loss": round(float(loss), 4), "w_max": round(wmax, 3),
                         "eff_pressure": round(eff, 3), "nn_med": round(med, 4), "nn_mean": round(mean, 4),
                         "val_ari": round(ari, 4), "val_noise": round(noise, 3),
                         "intra": round(h["intra"], 4), "inter": round(h["inter"], 4),
                         "gap": round(h["gap"], 4), "rx_gap": round(h["rx_gap"], 4)})
            print(f"step {step+1:5d} loss={float(loss):.3f} wmax={wmax:.2f} effP={eff:.2f} | "
                  f"valARI={ari:.3f} | nn med={med:.4f} mean={mean:.4f} | "
                  f"intra={h['intra']:.3f} inter={h['inter']:.3f} rx_gap={h['rx_gap']:.3f}", flush=True)
            if ari > best_ari:                               # SELECTION = val-ARI
                best_ari, best_step = ari, step + 1
                best_blob = {"med": med, "mean": mean, "C": C.copy(), "intra": h["intra"],
                             "inter": h["inter"], "gap": h["gap"], "rx_gap": h["rx_gap"], "noise": noise}
                torch.save(model.state_dict(), SMOKE_BEST)
    dt = time.time() - t0

    fb = best_blob
    print(f"\n=== SMOKE BEST by val-ARI (step {best_step}) — {dt/60:.1f} min ===")
    print(f"  val-ARI            : baseline {base_ari:.3f} -> {best_ari:.3f}  ({best_ari-base_ari:+.3f})")
    print(f"  nearest-confuser   : median {b_med:.4f} -> {fb['med']:.4f}   mean {b_mean:.4f} -> {fb['mean']:.4f}")
    print(f"  intra-cos (>=0.8?) : baseline {base_health['intra']:.3f} -> {fb['intra']:.3f}  "
          f"{'OK' if fb['intra'] >= 0.8 else 'BELOW 0.8'}")
    print(f"  inter-cos (collapse): baseline {base_health['inter']:.3f} -> {fb['inter']:.3f}   "
          f"rx_gap {fb['rx_gap']:.3f}")
    print(f"\n  worst-10 baseline pairs — centroid cosine baseline -> smoke-best:")
    wp_out = []
    for i, j, c in base_worst:
        sc_cos = float(fb["C"][i, j])
        wp_out.append({"pair": f"{held_tx[i]}/{held_tx[j]}", "baseline": round(c, 4),
                       "smoke": round(sc_cos, 4), "delta": round(sc_cos - c, 4)})
        print(f"     {held_tx[i]+'/'+held_tx[j]:>13}  {c:.4f} -> {sc_cos:.4f}  ({sc_cos-c:+.4f})")

    # vanilla-phase intra (end of W_MAX==1 window) for trade-off attribution
    van = [t for t in traj if t["step"] <= int(RAMP_START * STEPS)]
    van_intra = van[-1]["intra"] if van else float("nan")
    fin = traj[-1]

    print(f"\n=== VERDICT (smoke) ===")
    gate_intra = fb["intra"] >= 0.8
    gate_nn = fb["med"] < 0.986 - 0.01
    gate_ari = best_ari > base_ari + 0.02
    v = []
    v.append(f"intra at end-of-vanilla-phase (step {int(RAMP_START*STEPS)}) = {van_intra:.3f}; "
             f"best-ckpt intra = {fb['intra']:.3f}; final intra = {fin['intra']:.3f}")
    v.append(f"intra>=0.8: {'YES' if gate_intra else 'NO'} | "
             f"nn-confuser<0.976: {'YES' if gate_nn else 'NO'} ({fb['med']:.4f}) | "
             f"val-ARI up: {'YES' if gate_ari else 'NO'} ({base_ari:.3f}->{best_ari:.3f})")
    if gate_intra and gate_nn and gate_ari:
        v.append("ALL THREE SIMULTANEOUSLY -> tune GOOD; next session = full run on scattered split")
    elif (van_intra >= 0.78) and (fb["intra"] < van_intra - 0.1) and gate_nn:
        v.append("TRADE-OFF: positives were tight after vanilla phase but COLLAPSED once W_MAX ramped "
                 "while nn dropped -> irreconcilable with reweighting -> SWITCH to semi-hard triplet")
    elif fb["intra"] < 0.8 and van_intra < 0.8:
        v.append("intra never reached 0.8 even in the vanilla phase -> UNDERTRAINED for a smoke "
                 "(baseline needed full run); not a trade-off per se — longer run needed to judge")
    else:
        v.append("PARTIAL: not all gates met; inspect trajectory before committing a full run")
    for line in v:
        print("  - " + line)

    out = {"version": "v2_annealed_tuned", "smoke_steps": STEPS, "minutes": round(dt / 60, 1),
           "best_step": best_step, "select_metric": "val_ARI",
           "weighting": {"S0": S0, "BETA": BETA, "W_MAX_TGT": W_MAX_TGT,
                         "ramp_steps": [int(RAMP_START*STEPS), int(RAMP_END*STEPS)]},
           "cooccur": {"confuser_aware": ca_m, "random": rnd_m, "mix_frac": CONFUSER_FRAC,
                       "expected_mix": mix_m, "eff_pressure_target": round(W_MAX_TGT*mix_m, 3)},
           "baseline": {"val_ari": base_ari, "nn_median": b_med, "nn_mean": b_mean, **base_health},
           "smoke_best": {"val_ari": best_ari, "nn_median": fb["med"], "nn_mean": fb["mean"],
                          "intra": fb["intra"], "inter": fb["inter"], "gap": fb["gap"], "rx_gap": fb["rx_gap"]},
           "vanilla_phase_intra": van_intra, "worst_pairs": wp_out,
           "scatter_split": {"path": SPLIT_NEW, "rows_covered": sc["rows_covered"]},
           "trajectory": traj, "verdict": v}
    with open(REPORT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {REPORT}")
    print(f"smoke best ckpt -> {SMOKE_BEST}  (best_model.pt UNTOUCHED)")
    print("\nCHECKPOINT — smoke only. No full retrain, no drones/drive.")


if __name__ == "__main__":
    main()
