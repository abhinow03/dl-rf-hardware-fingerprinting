"""rffp.evaluation.evaluate — Phase-1 ORACLE closed-set evaluation.

Extracts embeddings for all 16 ORACLE devices across all distances, renders t-SNE plots
(coloured by device and by distance), and reports held-out cluster purity.

    python3 -m rffp.evaluation.evaluate

Paths resolve via rffp.config (RFFP_ORACLE dataset, RFFP_ORACLE_CKPT model, RFFP_ORACLE_OUT).
"""
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

from rffp.models.encoder import RFEncoder
from rffp.config import ORACLE_DIR

# ─────────────────────────────────────────────
# CONFIG  (paths overridable via env; see rffp.config)
# ─────────────────────────────────────────────
DATASET_ROOT = ORACLE_DIR
MODEL_PATH  = os.path.expanduser(os.environ.get("RFFP_ORACLE_CKPT", "~/phase6_output_v4/best_model.pt"))
OUTPUT_DIR  = os.path.expanduser(os.environ.get("RFFP_ORACLE_OUT", "~/phase7_output_v4"))

TRAIN_DEVICES = [
    "3123D52", "3123D54", "3123D58", "3123D64",
    "3123D65", "3123D70", "3123D76", "3123D78",
    "3123D79", "3123D7B", "3123D7D", "3123D7E"
]
HELD_OUT_DEVICES = ["3123D80", "3123D89", "3123EFE", "3124E4A"]
ALL_DEVICES      = TRAIN_DEVICES + HELD_OUT_DEVICES
ALL_DISTS        = ["8ft", "14ft", "20ft", "26ft", "32ft"]

WINDOW_SIZE  = 4096
SKIP_SAMPLES = 1000
CLIP_THRESH  = 10.0
FFT_SIZE     = 256
HOP_SIZE     = 64
WINDOWS_PER_DEVICE_PER_DIST = 50  # embeddings to extract per device per distance
SEED         = 42


# ─────────────────────────────────────────────
# Helpers (same as train.py)
# ─────────────────────────────────────────────
def compute_stft(x_np):
    window = np.hanning(FFT_SIZE).astype(np.float32)
    out = []
    for ch in range(2):
        sig = x_np[ch]
        frames = []
        for start in range(0, WINDOW_SIZE - FFT_SIZE + 1, HOP_SIZE):
            frame = sig[start:start+FFT_SIZE] * window
            spec = np.fft.rfft(frame)
            frames.append(np.abs(spec).astype(np.float32))
        out.append(np.stack(frames, axis=1))
    return np.stack(out, axis=0)


def load_window(fpath, win_idx):
    sample_idx = SKIP_SAMPLES + win_idx * WINDOW_SIZE
    try:
        raw = np.memmap(fpath, dtype=np.complex64, mode='r',
                        offset=sample_idx * 8, shape=(WINDOW_SIZE,))
        x = np.stack([raw.real, raw.imag], axis=0).astype(np.float32)
        if np.any(np.abs(x) > CLIP_THRESH):
            return None
        return x
    except Exception:
        return None


def sample_window(fpath, n_windows, max_tries=20):
    for _ in range(max_tries):
        x = load_window(fpath, random.randint(0, n_windows - 1))
        if x is not None:
            return x
    return None


def preprocess(x):
    """Standardise — same as training, no augmentation."""
    mean = x.mean()
    std  = x.std() + 1e-8
    return ((x - mean) / std).astype(np.float32)


def build_index(devices, distances):
    index = defaultdict(lambda: defaultdict(list))
    for dist in distances:
        dist_path = os.path.join(DATASET_ROOT, dist)
        if not os.path.isdir(dist_path):
            continue
        for fname in os.listdir(dist_path):
            if not fname.endswith(".sigmf-data"):
                continue
            parts = fname.replace(".sigmf-data", "").split("_")
            device = parts[3]
            if device not in devices:
                continue
            fpath = os.path.join(dist_path, fname)
            fsize = os.path.getsize(fpath)
            n_windows = max(0, (fsize // 8 - SKIP_SAMPLES) // WINDOW_SIZE)
            if n_windows > 0:
                index[device][dist].append((fpath, n_windows))
    return index


# ─────────────────────────────────────────────
# Extract embeddings
# ─────────────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, index, devices, n_per_dist, device_cuda):
    """
    Returns:
        embs:         [N, 512] numpy  (encoder output, not projection head)
        device_labels:[N] int  (index into devices list)
        dist_labels:  [N] str
        device_names: [N] str
    """
    model.eval()
    all_embs, all_dev, all_dist, all_names = [], [], [], []

    for dev_idx, dev in enumerate(devices):
        for dist in ALL_DISTS:
            if dist not in index[dev]:
                continue
            count = 0
            attempts = 0
            while count < n_per_dist and attempts < n_per_dist * 5:
                attempts += 1
                fpath, n_win = random.choice(index[dev][dist])
                x = sample_window(fpath, n_win)
                if x is None:
                    continue
                x = preprocess(x)
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device_cuda)
                x_s = torch.tensor(compute_stft(x), dtype=torch.float32).unsqueeze(0).to(device_cuda)
                with torch.amp.autocast('cuda'):
                    emb = model.get_encoder_output(x_t, x_s)  # [1, 512]
                all_embs.append(emb.squeeze(0).float().cpu().numpy())
                all_dev.append(dev_idx)
                all_dist.append(dist)
                all_names.append(dev)
                count += 1

        if (dev_idx + 1) % 4 == 0:
            print(f"  Extracted {dev_idx+1}/{len(devices)} devices...")

    return (
        np.stack(all_embs),
        np.array(all_dev),
        np.array(all_dist),
        np.array(all_names)
    )


# ─────────────────────────────────────────────
# t-SNE plots
# ─────────────────────────────────────────────
def plot_tsne_by_device(tsne_embs, device_labels, device_names, devices, output_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.cm.get_cmap('tab20', len(devices))

    for i, dev in enumerate(devices):
        mask = device_labels == i
        is_held_out = dev in HELD_OUT_DEVICES
        marker = '*' if is_held_out else 'o'
        size   = 80 if is_held_out else 30
        label  = f"{dev} (held-out)" if is_held_out else dev
        ax.scatter(tsne_embs[mask, 0], tsne_embs[mask, 1],
                   c=[cmap(i)], label=label,
                   marker=marker, s=size, alpha=0.7)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax.set_title('t-SNE — coloured by DEVICE\n(stars = held-out devices, circles = train devices)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tsne_by_distance(tsne_embs, dist_labels, output_path):
    fig, ax = plt.subplots(figsize=(12, 8))
    dist_colors = {
        '8ft':  '#e41a1c',
        '14ft': '#ff7f00',
        '20ft': '#4daf4a',
        '26ft': '#377eb8',
        '32ft': '#984ea3',
    }

    for dist in ALL_DISTS:
        mask = dist_labels == dist
        if mask.sum() == 0:
            continue
        ax.scatter(tsne_embs[mask, 0], tsne_embs[mask, 1],
                   c=dist_colors.get(dist, 'gray'),
                   label=dist, s=20, alpha=0.6)

    ax.legend(title='Distance')
    ax.set_title('t-SNE — coloured by DISTANCE\n(no distance clusters = distance invariance achieved)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────
# Held-out cluster purity
# ─────────────────────────────────────────────
def compute_cluster_purity(embs, true_labels, k=5):
    """
    For each held-out embedding, find k nearest neighbours
    among ALL embeddings and check if majority are same device.
    Returns purity score 0-1.
    """
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = embs_norm @ embs_norm.T  # [N, N]

    correct = 0
    total   = 0

    for i in range(len(true_labels)):
        # Get k nearest neighbours (excluding self)
        sims = sim_matrix[i].copy()
        sims[i] = -1  # exclude self
        nn_indices = np.argsort(sims)[-k:]
        nn_labels  = true_labels[nn_indices]

        # Majority vote
        counts = np.bincount(nn_labels)
        predicted = np.argmax(counts)

        if predicted == true_labels[i]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def compute_within_between_sim(embs, device_labels):
    """Compute mean within-device and between-device cosine similarity."""
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim = embs_norm @ embs_norm.T

    n = len(device_labels)
    same_mask = (device_labels[:, None] == device_labels[None, :])
    self_mask = np.eye(n, dtype=bool)
    same_mask = same_mask & ~self_mask
    diff_mask = ~same_mask & ~self_mask

    within  = sim[same_mask].mean() if same_mask.sum() > 0 else 0.0
    between = sim[diff_mask].mean() if diff_mask.sum() > 0 else 0.0
    return within, between


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device_cuda = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading model from {MODEL_PATH}")
    model = RFEncoder().to(device_cuda)
    state = torch.load(MODEL_PATH, map_location=device_cuda, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded OK")

    # Build index for ALL 16 devices
    print("\nBuilding index for all 16 devices...")
    full_index = build_index(ALL_DEVICES, ALL_DISTS)
    print(f"Indexed {len(full_index)} devices")
    for dev in ALL_DEVICES:
        dists_found = list(full_index[dev].keys())
        tag = "(held-out)" if dev in HELD_OUT_DEVICES else "(train)"
        print(f"  {dev} {tag}: {dists_found}")

    # Extract embeddings
    print(f"\nExtracting {WINDOWS_PER_DEVICE_PER_DIST} embeddings per device per distance...")
    print("Train devices:")
    train_embs, train_dev_labels, train_dist_labels, train_names = extract_embeddings(
        model, full_index, TRAIN_DEVICES, WINDOWS_PER_DEVICE_PER_DIST, device_cuda
    )
    print(f"Train embeddings: {train_embs.shape}")

    print("Held-out devices:")
    held_embs, held_dev_labels, held_dist_labels, held_names = extract_embeddings(
        model, full_index, HELD_OUT_DEVICES, WINDOWS_PER_DEVICE_PER_DIST, device_cuda
    )
    # Offset held-out device labels so they don't overlap with train
    held_dev_labels_offset = held_dev_labels + len(TRAIN_DEVICES)
    print(f"Held-out embeddings: {held_embs.shape}")

    # Combined for t-SNE
    all_embs        = np.concatenate([train_embs, held_embs], axis=0)
    all_dev_labels  = np.concatenate([train_dev_labels, held_dev_labels_offset], axis=0)
    all_dist_labels = np.concatenate([train_dist_labels, held_dist_labels], axis=0)
    print(f"Total embeddings: {all_embs.shape}")

    # ── Within/between similarity ──────────────────────────────────────
    print("\n── Embedding Quality ─────────────────────────────────────────")
    within, between = compute_within_between_sim(train_embs, train_dev_labels)
    print(f"Train devices  — within-device sim: {within:.4f}  between-device sim: {between:.4f}  gap: {within-between:.4f}")

    within_h, between_h = compute_within_between_sim(held_embs, held_dev_labels)
    print(f"Held-out devices — within-device sim: {within_h:.4f}  between-device sim: {between_h:.4f}  gap: {within_h-between_h:.4f}")

    # ── Distance invariance check ──────────────────────────────────────
    print("\n── Distance Invariance ───────────────────────────────────────")
    for dev in TRAIN_DEVICES[:4]:  # sample 4 devices
        dev_mask = train_names == dev
        dev_embs = train_embs[dev_mask]
        dev_dists = train_dist_labels[dev_mask]

        close_mask = np.isin(dev_dists, ['8ft', '14ft'])
        far_mask   = np.isin(dev_dists, ['20ft', '26ft'])

        if close_mask.sum() < 2 or far_mask.sum() < 2:
            continue

        close_embs = dev_embs[close_mask]
        far_embs   = dev_embs[far_mask]

        # Cross-distance similarity
        cn = close_embs / (np.linalg.norm(close_embs, axis=1, keepdims=True) + 1e-8)
        fn = far_embs   / (np.linalg.norm(far_embs,   axis=1, keepdims=True) + 1e-8)
        cross_sim = (cn @ fn.T).mean()
        print(f"  {dev}: close-far cross-sim = {cross_sim:.4f}")

    # ── Held-out cluster purity ────────────────────────────────────────
    print("\n── Held-out Cluster Purity ───────────────────────────────────")
    # Use all embeddings as the reference pool for KNN
    purity_5  = compute_cluster_purity(all_embs, all_dev_labels, k=5)
    purity_10 = compute_cluster_purity(all_embs, all_dev_labels, k=10)
    print(f"KNN purity (k=5):  {purity_5*100:.1f}%")
    print(f"KNN purity (k=10): {purity_10*100:.1f}%")

    # Held-out only purity
    held_purity = compute_cluster_purity(held_embs, held_dev_labels, k=5)
    print(f"Held-out only KNN purity (k=5): {held_purity*100:.1f}%")

    # ── t-SNE ──────────────────────────────────────────────────────────
    print("\n── Running t-SNE (this takes ~2 minutes) ─────────────────────")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=SEED, verbose=1)
    tsne_embs = tsne.fit_transform(all_embs)
    print("t-SNE done")

    plot_tsne_by_device(
        tsne_embs, all_dev_labels, None,
        ALL_DEVICES,
        os.path.join(OUTPUT_DIR, "tsne_by_device.png")
    )
    plot_tsne_by_distance(
        tsne_embs, all_dist_labels,
        os.path.join(OUTPUT_DIR, "tsne_by_distance.png")
    )

    # ── Summary ────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════")
    print("PHASE 7 SUMMARY")
    print("══════════════════════════════════════════════════════")
    print(f"Train within-device sim:    {within:.4f}  (target > 0.75)")
    print(f"Train between-device sim:   {between:.4f}")
    print(f"Train sim gap:              {within-between:.4f}  (target > 0.4)")
    print(f"Held-out within-device sim: {within_h:.4f}")
    print(f"Held-out sim gap:           {within_h-between_h:.4f}")
    print(f"KNN purity (all, k=5):      {purity_5*100:.1f}%  (target > 80%)")
    print(f"KNN purity (held-out, k=5): {held_purity*100:.1f}%")
    print(f"Plots saved to: {OUTPUT_DIR}/")
    print("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
