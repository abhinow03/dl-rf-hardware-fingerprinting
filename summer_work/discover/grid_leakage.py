"""PART 1 — grid-distance vs centroid-cosine on cached WiSig 41-held-out embeddings.

Decides the encoder-fix branch:
  cosine DECREASES with grid distance  -> CHANNEL/SPATIAL leakage  -> run Part 2 (eq=1)
  no grid-distance correlation         -> hardware-discrimination shortfall -> STOP

Off cached embeddings only. tx IDs parsed as ORBIT grid (row, col). Labels scoring-only.
"""
import os, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
RUN_DIR   = os.path.join(_SW, "runs", "wisig_supcon_fft64")
EMB_CACHE = os.path.join(RUN_DIR, "discover", "discover_embeddings.npz")
OUT_JSON  = os.path.join(RUN_DIR, "discover", "grid_leakage_report.json")
OUT_PNG   = os.path.join(RUN_DIR, "discover", "grid_dist_vs_cosine.png")

from scipy.stats import pearsonr, spearmanr


def unit(v, axis=-1):
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-8)


def main():
    z = np.load(EMB_CACHE)
    emb, true, tx = z["emb"].astype(np.float32), z["true"], z["tx"]
    n = len(tx)
    coords = np.array([[int(t.split("-")[0]), int(t.split("-")[1])] for t in tx], dtype=float)  # (row,col)
    print(f"{len(emb)} signals, {n} held-out tx (grid nodes)")

    cents = np.stack([unit(emb[true == i].mean(0)) for i in range(n)])     # [n,128]
    C = cents @ cents.T                                                    # cosine

    # all unordered pairs
    iu, ju = np.triu_indices(n, k=1)
    cos = C[iu, ju]
    gd = np.sqrt(((coords[iu] - coords[ju]) ** 2).sum(1))                  # grid euclidean dist
    same_row = coords[iu, 0] == coords[ju, 0]

    pr, pp = pearsonr(gd, cos)
    sr, sp = spearmanr(gd, cos)
    print(f"\nPAIRS: {len(cos)}  | grid-dist range [{gd.min():.1f}, {gd.max():.1f}]")
    print(f"Pearson  r(grid_dist, cosine) = {pr:+.3f}  (p={pp:.2e})")
    print(f"Spearman r(grid_dist, cosine) = {sr:+.3f}  (p={sp:.2e})")

    near = gd <= 2; far = gd >= 10
    print(f"\nnear pairs (dist<=2):  n={near.sum():4d}  mean cosine = {cos[near].mean():.3f}")
    print(f"far  pairs (dist>=10): n={far.sum():4d}  mean cosine = {cos[far].mean():.3f}")
    print(f"  -> near-minus-far cosine gap = {cos[near].mean() - cos[far].mean():+.3f}")

    print(f"\nsame-row pairs: n={same_row.sum():4d}  mean cosine = {cos[same_row].mean():.3f}")
    print(f"cross-row pairs:n={(~same_row).sum():4d}  mean cosine = {cos[~same_row].mean():.3f}")
    print(f"  -> same-minus-cross row cosine gap = {cos[same_row].mean() - cos[~same_row].mean():+.3f}")

    # binned mean cosine by distance
    print(f"\nmean cosine by grid-distance bin:")
    edges = [0, 1.5, 3, 5, 8, 12, 30]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gd >= lo) & (gd < hi)
        if m.any():
            print(f"  dist [{lo:>4.1f},{hi:>4.1f}): n={m.sum():4d}  mean cos={cos[m].mean():.3f}")

    # verdict
    channel = (pr < -0.2 and pp < 0.05) or (cos[near].mean() - cos[far].mean() > 0.15)
    verdict = ("CHANNEL/SPATIAL LEAKAGE (cosine falls with grid distance) -> run Part 2 (eq=1)"
               if channel else
               "NO grid-distance correlation -> hardware-discrimination shortfall -> STOP, do NOT run Part 2")
    print(f"\nVERDICT: {verdict}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(gd[~same_row], cos[~same_row], s=8, alpha=0.4, label="cross-row", color="#3b7")
        ax.scatter(gd[same_row], cos[same_row], s=12, alpha=0.6, label="same-row", color="#c33")
        ax.set_xlabel("grid Euclidean distance"); ax.set_ylabel("centroid cosine")
        ax.set_title(f"WiSig 41 held-out: grid-dist vs cosine  (Pearson {pr:+.2f}, Spearman {sr:+.2f})")
        ax.legend(); fig.tight_layout(); fig.savefig(OUT_PNG, dpi=110)
        print(f"plot -> {OUT_PNG}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    out = {"n_tx": n, "n_pairs": int(len(cos)),
           "pearson_r": float(pr), "pearson_p": float(pp),
           "spearman_r": float(sr), "spearman_p": float(sp),
           "near_le2_mean_cos": float(cos[near].mean()), "near_n": int(near.sum()),
           "far_ge10_mean_cos": float(cos[far].mean()), "far_n": int(far.sum()),
           "same_row_mean_cos": float(cos[same_row].mean()),
           "cross_row_mean_cos": float(cos[~same_row].mean()),
           "channel_leakage_verdict": bool(channel)}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
