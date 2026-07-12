# train.py — Phase 6 v4 — Window size 4096
# Train only on 8ft + 14ft (high SNR, no distance leakage)
# 4096 samples = 819.2 us at 5 MS/s — captures more IQ imbalance cycles

import os
import json
import random
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime

from model import RFEncoder
from losses import SupervisedContrastiveLoss, get_temperature

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_ROOT = os.path.expanduser(
    "~/Desktop/neu_m044q5210./KRI-16Devices-RawData"
)
OUTPUT_DIR = os.path.expanduser("~/phase6_output_v4")

TRAIN_DEVICES = [
    "3123D52", "3123D54", "3123D58", "3123D64",
    "3123D65", "3123D70", "3123D76", "3123D78",
    "3123D79", "3123D7B", "3123D7D", "3123D7E"
]
HELD_OUT_DEVICES = ["3123D80", "3123D89", "3123EFE", "3124E4A"]

TRAIN_DISTS = ["8ft", "14ft"]
EVAL_DISTS  = ["20ft", "26ft"]

WINDOW_SIZE        = 4096   # 4x larger than v3 — more IQ imbalance signal
SKIP_SAMPLES       = 1000
CLIP_THRESH        = 10.0
WINDOWS_PER_DEVICE = 8      # smaller per batch — windows are 4x larger
STEPS_PER_EPOCH    = 150
N_EPOCHS           = 80
LR                 = 1e-3
WEIGHT_DECAY       = 1e-4
GRAD_CLIP          = 1.0
WARMUP_EPOCHS      = 5
CKPT_EVERY         = 10
SEED               = 42

# STFT params for 4096-sample windows
# FFT=256, hop=64 -> [129, 61] per channel
FFT_SIZE = 256
HOP_SIZE = 64


def compute_stft(x_np):
    """x_np: [2, 4096] -> [2, 129, 61]"""
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
    return np.stack(out, axis=0)  # [2, 129, 61]


def augment(x):
    # 1. Phase rotation p=1.0
    phi = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(phi).astype(np.float32), np.sin(phi).astype(np.float32)
    i, q = x[0].copy(), x[1].copy()
    x = np.stack([c*i - s*q, s*i + c*q], axis=0)

    # 2. CFO injection p=0.8
    if np.random.rand() < 0.8:
        cfo = np.random.uniform(-0.005, 0.005)
        t = np.arange(WINDOW_SIZE, dtype=np.float32)
        phase = (2 * np.pi * cfo * t).astype(np.float32)
        c2, s2 = np.cos(phase), np.sin(phase)
        i2, q2 = x[0], x[1]
        x = np.stack([c2*i2 - s2*q2, s2*i2 + c2*q2], axis=0)

    # 3. AWGN p=0.5 — wide SNR range to teach distance invariance
    if np.random.rand() < 0.5:
        snr_db = np.random.uniform(15, 50)
        sig_power = np.mean(x**2) + 1e-8
        noise_power = sig_power / (10 ** (snr_db / 10))
        x = x + np.random.randn(*x.shape).astype(np.float32) * np.sqrt(noise_power)

    # 4. Amplitude scale p=0.5
    if np.random.rand() < 0.5:
        x = x * np.random.uniform(0.5, 2.0)

    # 5. Standardise
    mean = x.mean()
    std  = x.std() + 1e-8
    x = (x - mean) / std
    return x.astype(np.float32)


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


def sample_batch(index, devices, dists, windows_per_device):
    time_list, stft_list, label_list = [], [], []
    for label_idx, device in enumerate(devices):
        avail = [d for d in dists if d in index[device]]
        if not avail:
            continue
        count = attempts = 0
        while count < windows_per_device and attempts < windows_per_device * 5:
            attempts += 1
            dist = random.choice(avail)
            fpath, n_win = random.choice(index[device][dist])
            x = sample_window(fpath, n_win)
            if x is None:
                continue
            x = augment(x)
            time_list.append(x)
            stft_list.append(compute_stft(x))
            label_list.append(label_idx)
            count += 1
    return (
        torch.tensor(np.stack(time_list), dtype=torch.float32),
        torch.tensor(np.stack(stft_list), dtype=torch.float32),
        torch.tensor(label_list, dtype=torch.long)
    )


@torch.no_grad()
def compute_distance_sim(model, index, device_list,
                         close_dists, far_dists, n_samples=5):
    model.eval()
    close_sims, far_sims = [], []
    for device in device_list:
        for dist_group, sim_list in [(close_dists, close_sims),
                                     (far_dists, far_sims)]:
            avail = [d for d in dist_group if d in index[device]]
            if not avail:
                continue
            embs = []
            for _ in range(n_samples):
                dist = random.choice(avail)
                fpath, n_win = random.choice(index[device][dist])
                x = sample_window(fpath, n_win)
                if x is None:
                    continue
                mean = x.mean(); std = x.std() + 1e-8
                x = (x - mean) / std
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).cuda()
                x_s = torch.tensor(compute_stft(x), dtype=torch.float32).unsqueeze(0).cuda()
                with torch.amp.autocast('cuda'):
                    emb = model(x_t, x_s)
                embs.append(emb.squeeze(0).float())
            if len(embs) < 2:
                continue
            embs = torch.stack(embs)
            sim_mat = torch.mm(embs, embs.t())
            mask = ~torch.eye(len(embs), dtype=torch.bool, device=embs.device)
            sim_list.append(sim_mat[mask].mean().item())
    model.train()
    return (
        float(np.mean(close_sims)) if close_sims else 0.0,
        float(np.mean(far_sims))   if far_sims   else 0.0
    )


def run_epoch(model, criterion, optimizer, scaler, index, epoch, device):
    model.train()
    total_loss = total_gap = 0.0
    tau = get_temperature(epoch)

    for step in range(STEPS_PER_EPOCH):
        time_t, stft_t, labels = sample_batch(
            index, TRAIN_DEVICES, TRAIN_DISTS, WINDOWS_PER_DEVICE
        )
        time_t = time_t.to(device)
        stft_t = stft_t.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            emb = model(time_t, stft_t)

        emb_f = emb.float()
        loss  = criterion(emb_f, labels, temperature=tau)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        with torch.no_grad():
            sim = torch.mm(emb_f, emb_f.t())
            same      = (labels.unsqueeze(0) == labels.unsqueeze(1))
            self_mask = torch.eye(len(labels), dtype=torch.bool, device=device)
            same = same & ~self_mask
            diff = ~same & ~self_mask
            within  = sim[same].mean().item() if same.sum() > 0 else 0.0
            between = sim[diff].mean().item() if diff.sum() > 0 else 0.0
            total_gap += within - between

    n = STEPS_PER_EPOCH
    return total_loss / n, total_gap / n


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "train_log.json")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Verify STFT shape
    dummy = np.zeros((2, WINDOW_SIZE), dtype=np.float32)
    stft_shape = compute_stft(dummy).shape
    print(f"STFT output shape: {stft_shape}  (expected: (2, 129, 61))")
    assert stft_shape == (2, 129, 61), f"STFT shape mismatch: {stft_shape}"

    print("Building dataset index...")
    all_dists   = TRAIN_DISTS + EVAL_DISTS
    train_index = build_index(TRAIN_DEVICES, all_dists)

    for d in HELD_OUT_DEVICES:
        assert d not in train_index, f"HELD-OUT DEVICE {d} IN TRAINING INDEX!"
    print("Held-out device check: PASSED")
    print(f"Training on: {TRAIN_DISTS}  |  Eval on: {EVAL_DISTS}")
    for d in TRAIN_DEVICES:
        print(f"  {d}: {list(train_index[d].keys())}")

    model = RFEncoder().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params/1e6:.2f}M")

    criterion = SupervisedContrastiveLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(N_EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda')

    log      = []
    best_gap = -999.0

    print(f"\n{'='*55}")
    print(f"BATCHED SUPCON v4 — window={WINDOW_SIZE} samples")
    print(f"Batch: {len(TRAIN_DEVICES)}x{WINDOWS_PER_DEVICE}={len(TRAIN_DEVICES)*WINDOWS_PER_DEVICE}")
    print(f"Steps/epoch: {STEPS_PER_EPOCH}  Epochs: {N_EPOCHS}")
    print(f"{'='*55}\n")

    for epoch in range(N_EPOCHS):
        t0 = datetime.now()

        loss, gap = run_epoch(
            model, criterion, optimizer, scaler,
            train_index, epoch, device
        )

        scheduler.step()
        lr      = optimizer.param_groups[0]['lr']
        elapsed = (datetime.now() - t0).total_seconds()
        tau     = get_temperature(epoch)

        close_sim = far_sim = 0.0
        if (epoch + 1) % 10 == 0:
            close_sim, far_sim = compute_distance_sim(
                model, train_index, TRAIN_DEVICES,
                close_dists=TRAIN_DISTS,
                far_dists=EVAL_DISTS
            )

        log.append({
            "epoch":     epoch + 1,
            "loss":      round(loss, 4),
            "sim_gap":   round(gap, 4),
            "close_sim": round(close_sim, 4),
            "far_sim":   round(far_sim, 4),
            "lr":        round(lr, 6),
            "tau":       tau,
            "elapsed_s": round(elapsed, 1)
        })

        print(
            f"Epoch {epoch+1:02d}/{N_EPOCHS} | "
            f"loss={loss:.4f} sim_gap={gap:.4f} | "
            f"train_sim={close_sim:.3f} eval_sim={far_sim:.3f} | "
            f"lr={lr:.5f} tau={tau} | {elapsed:.1f}s"
        )

        if close_sim > 0 and far_sim > 0:
            gap_dist = close_sim - far_sim
            if gap_dist < 0.15:
                print(f"  Distance invariance OK: train={close_sim:.3f} eval={far_sim:.3f}")
            else:
                print(f"  WARNING DISTANCE LEAKAGE: train={close_sim:.3f} eval={far_sim:.3f}")

        if not (epoch + 1) % CKPT_EVERY:
            ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_epoch{epoch+1}.pt")
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss":            loss,
                "sim_gap":         gap,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")
            if gap > best_gap:
                best_gap = gap
                torch.save(model.state_dict(),
                           os.path.join(OUTPUT_DIR, "best_model.pt"))
                print(f"  New best model (gap={gap:.4f})")

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved: {log_path}")
    print(f"TRAINING COMPLETE — best sim_gap: {best_gap:.4f}")
    print(f"Best model: {OUTPUT_DIR}/best_model.pt")


if __name__ == "__main__":
    main()
