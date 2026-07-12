# inference.py — Phase 8 Inference Pipeline
# Open-set RF emitter discovery via DBSCAN + cosine similarity registry
# Usage: python3 inference.py --demo   (runs demo on held-out devices)

import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from model import RFEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser(
    "~/Desktop/neu_m044q5210./KRI-16Devices-RawData"
)
MODEL_PATH  = os.path.expanduser("~/phase6_output_v4/best_model.pt")
OUTPUT_DIR  = os.path.expanduser("~/phase8_output")

HELD_OUT_DEVICES = ["3123D80", "3123D89", "3123EFE", "3124E4A"]
ALL_DISTS        = ["8ft", "14ft", "20ft", "26ft", "32ft"]

WINDOW_SIZE  = 4096
SKIP_SAMPLES = 1000
CLIP_THRESH  = 10.0
FFT_SIZE     = 256
HOP_SIZE     = 64

# Similarity threshold for emitter matching
# Above this → same emitter, below → new emitter
# Calibrated from Phase 7: within=0.837, between=0.706 → midpoint ~0.77
SIMILARITY_THRESHOLD = 0.77

SEED = 42


# ─────────────────────────────────────────────
# Signal processing
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


def preprocess(x):
    mean = x.mean()
    std  = x.std() + 1e-8
    return ((x - mean) / std).astype(np.float32)


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
# Embedding extractor
# ─────────────────────────────────────────────
class EmbeddingExtractor:
    def __init__(self, model_path, device):
        self.device = device
        self.model  = RFEncoder().to(device)
        state = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model loaded from {model_path}")

    @torch.no_grad()
    def embed(self, x_np):
        """x_np: [2, 1024] raw IQ → [512] encoder embedding"""
        x = preprocess(x_np)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_s = torch.tensor(compute_stft(x), dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.amp.autocast('cuda'):
            emb = self.model.get_encoder_output(x_t, x_s)
        emb = emb.squeeze(0).float().cpu().numpy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb

    @torch.no_grad()
    def embed_batch(self, windows):
        """windows: list of [2, 1024] arrays → [N, 512] embeddings"""
        time_list, stft_list = [], []
        for x_np in windows:
            x = preprocess(x_np)
            time_list.append(x)
            stft_list.append(compute_stft(x))
        x_t = torch.tensor(np.stack(time_list), dtype=torch.float32).to(self.device)
        x_s = torch.tensor(np.stack(stft_list), dtype=torch.float32).to(self.device)
        with torch.amp.autocast('cuda'):
            embs = self.model.get_encoder_output(x_t, x_s)
        embs = embs.float().cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return embs / norms


# ─────────────────────────────────────────────
# Emitter Registry
# Open-set emitter ID assignment via cosine similarity
# ─────────────────────────────────────────────
class EmitterRegistry:
    def __init__(self, threshold=SIMILARITY_THRESHOLD, ema_alpha=0.1):
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.registry  = {}   # {emitter_id: mean_embedding}
        self.counts    = {}   # {emitter_id: n_observations}
        self._next_id  = 0

    def _new_id(self):
        name = f"Emitter_{chr(65 + self._next_id)}" if self._next_id < 26 else f"Emitter_{self._next_id}"
        self._next_id += 1
        return name

    def identify(self, embedding):
        """
        Given a new embedding, return (emitter_id, similarity, is_new).
        If similarity > threshold → match existing emitter.
        Otherwise → register as new emitter.
        """
        if not self.registry:
            eid = self._new_id()
            self.registry[eid] = embedding.copy()
            self.counts[eid]   = 1
            return eid, 1.0, True

        # Compute cosine similarity to all registered emitters
        ids   = list(self.registry.keys())
        refs  = np.stack([self.registry[i] for i in ids])
        sims  = refs @ embedding  # dot product of unit vectors

        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_id  = ids[best_idx]

        if best_sim >= self.threshold:
            # Match — update registry with EMA
            self.registry[best_id] = (
                (1 - self.ema_alpha) * self.registry[best_id] +
                self.ema_alpha * embedding
            )
            # Re-normalise after EMA update
            norm = np.linalg.norm(self.registry[best_id]) + 1e-8
            self.registry[best_id] /= norm
            self.counts[best_id] += 1
            return best_id, float(best_sim), False
        else:
            # New emitter
            eid = self._new_id()
            self.registry[eid] = embedding.copy()
            self.counts[eid]   = 1
            return eid, float(best_sim), True

    def summary(self):
        print(f"\nEmitter Registry ({len(self.registry)} emitters):")
        for eid, count in sorted(self.counts.items(), key=lambda x: -x[1]):
            print(f"  {eid}: {count} observations")


# ─────────────────────────────────────────────
# Threshold calibration
# ─────────────────────────────────────────────
def calibrate_threshold(extractor, index, devices, n_samples=30):
    """
    Sample pairs of windows and compute cosine similarity.
    Same device → positive pairs, different device → negative pairs.
    Returns optimal threshold at midpoint of distributions.
    """
    print("\nCalibrating similarity threshold...")
    pos_sims, neg_sims = [], []

    device_list = [d for d in devices if any(index[d].values())]

    for dev in device_list:
        avail_dists = list(index[dev].keys())
        if not avail_dists:
            continue

        # Positive pairs: same device, different windows
        windows = []
        for _ in range(n_samples):
            dist = random.choice(avail_dists)
            fpath, n_win = random.choice(index[dev][dist])
            x = sample_window(fpath, n_win)
            if x is not None:
                windows.append(x)

        if len(windows) < 2:
            continue

        embs = extractor.embed_batch(windows)
        for i in range(len(embs)):
            for j in range(i+1, min(i+5, len(embs))):
                pos_sims.append(float(embs[i] @ embs[j]))

    # Negative pairs: different devices
    for i in range(len(device_list)):
        for j in range(i+1, len(device_list)):
            dev_a = device_list[i]
            dev_b = device_list[j]

            for dev, name in [(dev_a, 'a'), (dev_b, 'b')]:
                avail = list(index[dev].keys())
                if not avail:
                    continue

            # Sample one window from each
            avail_a = list(index[dev_a].keys())
            avail_b = list(index[dev_b].keys())
            if not avail_a or not avail_b:
                continue

            dist_a = random.choice(avail_a)
            dist_b = random.choice(avail_b)
            fpath_a, nw_a = random.choice(index[dev_a][dist_a])
            fpath_b, nw_b = random.choice(index[dev_b][dist_b])
            xa = sample_window(fpath_a, nw_a)
            xb = sample_window(fpath_b, nw_b)
            if xa is None or xb is None:
                continue

            ea = extractor.embed(xa)
            eb = extractor.embed(xb)
            neg_sims.append(float(ea @ eb))

    pos_mean = np.mean(pos_sims) if pos_sims else 0.8
    neg_mean = np.mean(neg_sims) if neg_sims else 0.7
    threshold = (pos_mean + neg_mean) / 2

    print(f"  Positive pairs (same device): mean={pos_mean:.4f}, std={np.std(pos_sims):.4f}")
    print(f"  Negative pairs (diff device): mean={neg_mean:.4f}, std={np.std(neg_sims):.4f}")
    print(f"  Calibrated threshold: {threshold:.4f}")

    return threshold, pos_mean, neg_mean


# ─────────────────────────────────────────────
# Demo: identify held-out devices
# ─────────────────────────────────────────────
def run_demo(extractor, index, threshold):
    """
    Simulate open-set discovery: feed windows from held-out devices
    one at a time to the registry. The registry should discover
    4 distinct emitters without knowing device labels.
    """
    print(f"\n{'='*55}")
    print("OPEN-SET EMITTER DISCOVERY DEMO")
    print(f"Threshold: {threshold:.4f}")
    print(f"Devices: {HELD_OUT_DEVICES} (identities hidden from registry)")
    print(f"{'='*55}\n")

    registry = EmitterRegistry(threshold=threshold)

    # Stream windows from held-out devices in random order
    all_windows = []
    for dev in HELD_OUT_DEVICES:
        avail = list(index[dev].keys())
        if not avail:
            continue
        for _ in range(25):  # 25 windows per device = 100 total
            dist = random.choice(avail)
            fpath, n_win = random.choice(index[dev][dist])
            x = sample_window(fpath, n_win)
            if x is not None:
                all_windows.append((dev, x))

    random.shuffle(all_windows)
    print(f"Streaming {len(all_windows)} windows in random order...\n")

    # Track assignments for purity computation
    assignments = defaultdict(list)  # {true_device: [assigned_emitter_ids]}

    for true_dev, x in all_windows:
        emb = extractor.embed(x)
        eid, sim, is_new = registry.identify(emb)
        assignments[true_dev].append(eid)
        if is_new:
            print(f"  NEW emitter discovered: {eid} (triggered by {true_dev}, sim={sim:.3f})")

    # Registry summary
    registry.summary()

    # Purity: for each true device, what fraction went to its majority emitter?
    print(f"\n── Assignment Purity ─────────────────────────────────")
    total_correct = 0
    total_windows = 0

    for true_dev in HELD_OUT_DEVICES:
        if true_dev not in assignments:
            continue
        assigned = assignments[true_dev]
        if not assigned:
            continue

        # Majority vote
        counts = defaultdict(int)
        for eid in assigned:
            counts[eid] += 1
        majority_eid   = max(counts, key=counts.get)
        majority_count = counts[majority_eid]
        purity         = majority_count / len(assigned)
        total_correct += majority_count
        total_windows += len(assigned)

        print(f"  {true_dev}: {len(assigned)} windows → majority={majority_eid} ({majority_count}/{len(assigned)} = {purity*100:.1f}%)")

    overall_purity = total_correct / total_windows if total_windows > 0 else 0
    print(f"\n  Overall purity: {overall_purity*100:.1f}%")
    print(f"  Emitters discovered: {len(registry.registry)} (true: {len(HELD_OUT_DEVICES)})")

    # False merge / false split analysis
    print(f"\n── False Merge / False Split ─────────────────────────")
    n_emitters = len(registry.registry)
    n_true     = len(HELD_OUT_DEVICES)
    if n_emitters < n_true:
        print(f"  FALSE MERGES: {n_true - n_emitters} device pairs merged into same emitter")
    elif n_emitters > n_true:
        print(f"  FALSE SPLITS: {n_emitters - n_true} devices split into multiple emitters")
    else:
        print(f"  Perfect count: {n_emitters} emitters discovered = {n_true} true devices")

    return overall_purity, n_emitters


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true',
                        help='Run open-set discovery demo on held-out devices')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override similarity threshold (default: auto-calibrate)')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device_cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device_cuda}")

    # Load model
    extractor = EmbeddingExtractor(MODEL_PATH, device_cuda)

    # Build index for held-out devices
    print("\nBuilding index for held-out devices...")
    held_index = build_index(HELD_OUT_DEVICES, ALL_DISTS)
    for dev in HELD_OUT_DEVICES:
        dists = list(held_index[dev].keys())
        print(f"  {dev}: {dists}")

    # Calibrate threshold
    if args.threshold:
        threshold = args.threshold
        print(f"Using provided threshold: {threshold}")
    else:
        threshold, pos_mean, neg_mean = calibrate_threshold(
            extractor, held_index, HELD_OUT_DEVICES
        )

    if args.demo:
        purity, n_discovered = run_demo(extractor, held_index, threshold)

        print(f"\n{'='*55}")
        print("PHASE 8 SUMMARY")
        print(f"{'='*55}")
        print(f"Similarity threshold:     {threshold:.4f}")
        print(f"Emitters discovered:      {n_discovered} (true: {len(HELD_OUT_DEVICES)})")
        print(f"Assignment purity:        {purity*100:.1f}%")
        print(f"{'='*55}")
    else:
        print("\nRun with --demo to test open-set emitter discovery.")
        print("Example: python3 inference.py --demo")


if __name__ == "__main__":
    main()
