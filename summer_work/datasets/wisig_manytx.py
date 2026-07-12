"""WiSig ManyTx dataloader — metric-learning pool for open-world emitter discovery.

Source: ~/CAPSTONE/phase3_dataset/ManyTx.pkl
Layout (verified): dict with
    tx_list   (150)  : transmitter IDs, 'board-antenna' e.g. '8-13'   -> device label
    rx_list   (18)   : receiver IDs
    capture_date_list(4): '2021_03_01' ... '2021_03_23'
    equalized_list[0,1] : 0 = RAW (channel present), 1 = EQUALIZED (channel divided out)
    max_sig   (50)   : cap per (tx,rx,date,eq) cell
    data[tx][rx][date][eq] -> ndarray (n, 256, 2) float64   (I/Q), n in [0,50]

DESIGN DECISIONS (see summer_work/CONTEXT.md / session Part 3):
  * EQ = 0 (RAW). Keep the channel present so the metric must learn channel/receiver-robust
    hardware features, not lean on the dataset's equalizer (which would be an easier task
    that doesn't reflect deployment and undercuts the cross-domain transfer claim).
  * Signals are pooled across ALL rx/date per Tx, labelled ONLY by Tx identity, so the
    encoder is forced to be rx/date-invariant. rx/date indices are retained per signal so
    the Step-3 health check can verify embeddings separate by Tx, not by receiver/day.
  * STFT n_fft=64, hop=16 on the 256-sample packet -> spectral input [2, 33, 13]. Same
    config stays valid for 4096-sample drone windows later (encoder GAP is length-agnostic).
  * Split is DEVICE-DISJOINT. Default split_by='board' (hold out whole boards) to guarantee
    no shared-board hardware leaks across train/discover; split_by='tx' available for the
    standard WiSig per-Tx protocol. Exact held-out IDs are written to the run folder.
"""
import os
import json
import numpy as np

PKL_PATH = os.path.expanduser("~/CAPSTONE/phase3_dataset/ManyTx.pkl")

EQ        = 0       # 0 = raw (chosen), 1 = equalized
SIG_LEN   = 256
N_FFT     = 64
HOP       = 16
# frames = len(range(0, SIG_LEN - N_FFT + 1, HOP)) = 13 ; bins = N_FFT//2 + 1 = 33
STFT_SHAPE = (2, N_FFT // 2 + 1, len(range(0, SIG_LEN - N_FFT + 1, HOP)))  # (2, 33, 13)

_HANN = np.hanning(N_FFT).astype(np.float32)
_STARTS = list(range(0, SIG_LEN - N_FFT + 1, HOP))


# ─────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────
def compute_stft(x):
    """x: [2, 256] float32 -> [2, 33, 13] STFT magnitude (matches V4 SpectralBranch)."""
    out = np.empty(STFT_SHAPE, dtype=np.float32)
    for ch in range(2):
        sig = x[ch]
        for j, s in enumerate(_STARTS):
            spec = np.fft.rfft(sig[s:s + N_FFT] * _HANN)
            out[ch, :, j] = np.abs(spec).astype(np.float32)
    return out


def compute_stft_batch(xb):
    """xb: [B, 2, 256] float32 -> [B, 2, 33, 13]  (vectorised)."""
    B = xb.shape[0]
    frames = np.stack([xb[:, :, s:s + N_FFT] for s in _STARTS], axis=2)  # [B,2,13,64]
    frames = frames * _HANN
    spec = np.fft.rfft(frames, axis=-1)                                  # [B,2,13,33]
    return np.abs(spec).astype(np.float32).transpose(0, 1, 3, 2)         # [B,2,33,13]


def standardize(x):
    """Per-window zero-mean unit-std (removes absolute amplitude — not a stable fingerprint)."""
    m = x.mean()
    s = x.std() + 1e-8
    return ((x - m) / s).astype(np.float32)


# Augmentation — V4 order, ranges retuned for 256-sample WiFi packets.
# FLAGGED for confirmation at Checkpoint 2 (see report):
#   CFO_MAX 0.005->0.003  and  AWGN 15-50dB -> 10-40dB  (256-sample packets are short;
#   ±0.005 cyc/sample is ~1.3 phase cycles across the window, possibly too aggressive).
CFO_MAX   = 0.003
AWGN_SNR  = (10.0, 40.0)
AMP_RANGE = (0.5, 2.0)

def augment(x, rng, use_cfo=True):
    """x: [2, 256] float32 -> augmented + standardised [2, 256].

    use_cfo: gate the CFO-injection step. CFO is itself a transmitter cue, so augmenting it
    away may erase fingerprint signal — toggled for the Checkpoint-3 ablation.
    """
    # 1. Phase rotation p=1.0  (absolute phase is not a stable fingerprint — always destroy)
    phi = rng.uniform(0, 2 * np.pi)
    c, s = np.float32(np.cos(phi)), np.float32(np.sin(phi))
    i, q = x[0], x[1]
    x = np.stack([c * i - s * q, s * i + c * q], axis=0)

    # 2. CFO injection p=0.8  (oscillator offset — nuisance OR cue; see ablation)
    if use_cfo and rng.random() < 0.8:
        cfo = rng.uniform(-CFO_MAX, CFO_MAX)
        ph = (2 * np.pi * cfo * np.arange(SIG_LEN, dtype=np.float32)).astype(np.float32)
        c2, s2 = np.cos(ph), np.sin(ph)
        x = np.stack([c2 * x[0] - s2 * x[1], s2 * x[0] + c2 * x[1]], axis=0)

    # 3. AWGN p=0.5
    if rng.random() < 0.5:
        snr = rng.uniform(*AWGN_SNR)
        p_sig = np.mean(x ** 2) + 1e-8
        p_noise = p_sig / (10 ** (snr / 10))
        x = x + rng.standard_normal(x.shape).astype(np.float32) * np.float32(np.sqrt(p_noise))

    # 4. Amplitude scale p=0.5
    if rng.random() < 0.5:
        x = x * np.float32(rng.uniform(*AMP_RANGE))

    # 5. Standardise
    return standardize(x.astype(np.float32))


# ─────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────
def load_manytx(eq=EQ, verbose=True):
    """Load ManyTx.pkl into a per-Tx dict pooled across rx/date for the chosen eq.

    Returns:
        tx_data: { tx_id: {'iq':[N,256,2]f32, 'rx':[N]i16, 'date':[N]i8} }
        meta:    {'tx_list','rx_list','date_list','eq'}
    """
    import pickle
    if verbose:
        print(f"Loading {PKL_PATH} (eq={eq}) ...")
    with open(PKL_PATH, "rb") as f:
        D = pickle.load(f)
    tx_list, rx_list, dates = D["tx_list"], D["rx_list"], D["capture_date_list"]
    data = D["data"]

    tx_data = {}
    for ti, tx in enumerate(tx_list):
        iqs, rxs, dts = [], [], []
        for ri in range(len(rx_list)):
            for di in range(len(dates)):
                a = data[ti][ri][di][eq]
                if a.shape[0] > 0:
                    iqs.append(a.astype(np.float32))
                    rxs.append(np.full(a.shape[0], ri, dtype=np.int16))
                    dts.append(np.full(a.shape[0], di, dtype=np.int8))
        tx_data[tx] = {
            "iq":   np.concatenate(iqs, axis=0),    # [N,256,2]
            "rx":   np.concatenate(rxs, axis=0),
            "date": np.concatenate(dts, axis=0),
        }
    meta = {"tx_list": tx_list, "rx_list": rx_list, "date_list": dates, "eq": eq}
    if verbose:
        tot = sum(v["iq"].shape[0] for v in tx_data.values())
        print(f"  loaded {len(tx_data)} Tx, {tot} signals, sig dtype={tx_data[tx_list[0]]['iq'].dtype}")
    return tx_data, meta


# ─────────────────────────────────────────────────────────────
# Device-disjoint split
# ─────────────────────────────────────────────────────────────
def make_split(tx_list, split_by="board", n_discover=30, seed=42, out_path=None):
    """Device-disjoint train/discover split.

    split_by='board' : hold out WHOLE boards (board = tx_id.split('-')[0]) until ~n_discover
                       Tx accumulate. Guarantees no shared-board hardware across the split.
    split_by='tx'    : standard WiSig per-Tx random split (n_discover Tx held out).

    Writes the exact split (with seed + mode) to out_path if given. Returns (train, discover).
    """
    rng = np.random.default_rng(seed)
    if split_by == "tx":
        order = list(tx_list)
        rng.shuffle(order)
        discover = sorted(order[:n_discover])
        train = sorted(order[n_discover:])
        held_boards = None
    elif split_by == "board":
        boards = {}
        for t in tx_list:
            boards.setdefault(t.split("-")[0], []).append(t)
        border = list(boards.keys())
        rng.shuffle(border)
        discover, held_boards = [], []
        for b in border:
            if len(discover) >= n_discover:
                break
            discover.extend(boards[b]); held_boards.append(b)
        discover = sorted(discover)
        train = sorted([t for t in tx_list if t not in set(discover)])
    else:
        raise ValueError(f"split_by must be 'board' or 'tx', got {split_by!r}")

    assert set(train).isdisjoint(set(discover)), "SPLIT LEAK: train/discover overlap!"
    split = {
        "mode": split_by, "seed": seed,
        "n_train": len(train), "n_discover": len(discover),
        "held_out_boards": held_boards,
        "train_tx": train, "discover_tx": discover,
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"  split written -> {out_path}")
    return train, discover, split


# ─────────────────────────────────────────────────────────────
# Class-balanced batch sampler  (N devices x M signals)
# ─────────────────────────────────────────────────────────────
DEFAULT_N = 32   # devices per batch  (negatives diversity for SupCon)
DEFAULT_M = 8    # signals per device (positives for SupCon) -> batch = 256

def sample_batch(tx_data, tx_subset, label_of, rng, N=DEFAULT_N, M=DEFAULT_M,
                 train=True, use_cfo=True):
    """Return (time[B,2,256] f32, stft[B,2,33,13] f32, labels[B] int64, info dict)."""
    chosen = rng.choice(len(tx_subset), size=min(N, len(tx_subset)), replace=False)
    times, labels, rxs, dts = [], [], [], []
    for ci in chosen:
        tx = tx_subset[ci]
        d = tx_data[tx]
        idx = rng.integers(0, d["iq"].shape[0], size=M)
        for k in idx:
            x = d["iq"][k].T.copy()                  # [256,2] -> [2,256]
            x = augment(x, rng, use_cfo=use_cfo) if train else standardize(x)
            times.append(x)
            labels.append(label_of[tx])
            rxs.append(int(d["rx"][k])); dts.append(int(d["date"][k]))
    time_arr = np.stack(times).astype(np.float32)    # [B,2,256]
    stft_arr = compute_stft_batch(time_arr)          # [B,2,33,13]
    info = {"rx": np.array(rxs), "date": np.array(dts)}
    return time_arr, stft_arr, np.array(labels, dtype=np.int64), info


# ─────────────────────────────────────────────────────────────
# Checkpoint-2 diagnostics
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RUN_DIR = os.path.expanduser("~/CAPSTONE/DL_model/summer_work/runs/wisig_supcon_fft64")
    SPLIT_PATH = os.path.join(RUN_DIR, "splits", "split_manytx.json")

    tx_data, meta = load_manytx()
    train_tx, discover_tx, split = make_split(
        meta["tx_list"], split_by="board", n_discover=30, seed=42, out_path=SPLIT_PATH
    )
    label_of = {tx: i for i, tx in enumerate(train_tx)}

    print(f"\nSplit mode={split['mode']} seed={split['seed']}")
    print(f"  train Tx   = {len(train_tx)}")
    print(f"  discover Tx= {len(discover_tx)}  (held-out boards: {split['held_out_boards']})")
    print(f"  discover IDs: {discover_tx}")

    rng = np.random.default_rng(0)
    t, s, y, info = sample_batch(tx_data, train_tx, label_of, rng, train=True)
    print(f"\nOne training batch (N={DEFAULT_N} x M={DEFAULT_M}):")
    print(f"  time  {t.shape} {t.dtype}   stft {s.shape} {s.dtype}   labels {y.shape}")
    print(f"  unique devices in batch = {len(np.unique(y))}  | per-device counts = {np.bincount(np.searchsorted(np.unique(y), y)).tolist()[:8]}...")
    print(f"  NaN/Inf check: time={np.isnan(t).any() or np.isinf(t).any()}  stft={np.isnan(s).any() or np.isinf(s).any()}")
    print(f"  time per-window mean~{t.mean():.3e} std~{t.std():.3f} (standardised)")

    # Leakage sanity: confirm device-disjoint, note rx/date overlap
    leak = set(train_tx) & set(discover_tx)
    print(f"\nLeakage check: train∩discover = {len(leak)} (must be 0)  -> {'OK' if not leak else 'LEAK!'}")
    rx_tr = set(int(r) for tx in train_tx for r in np.unique(tx_data[tx]['rx']))
    rx_di = set(int(r) for tx in discover_tx for r in np.unique(tx_data[tx]['rx']))
    print(f"  Rx overlap train/discover = {len(rx_tr & rx_di)} receivers shared "
          f"(EXPECTED: within-domain split shares receivers/days; cross-Rx/day transfer is a separate ManySig experiment)")

    # Plot a couple of example signals + their STFTs
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    for col, tx in enumerate(train_tx[:2] + discover_tx[:1]):
        x = standardize(tx_data[tx]["iq"][0].T.copy())
        st = compute_stft(x)
        ax[0, col].plot(x[0], lw=.6, label="I"); ax[0, col].plot(x[1], lw=.6, label="Q")
        ax[0, col].set_title(f"Tx {tx}  IQ [2,256]"); ax[0, col].legend(fontsize=7)
        im = ax[1, col].imshow(st[0], aspect="auto", origin="lower")
        ax[1, col].set_title(f"Tx {tx}  STFT[0] [33,13]")
    fig.tight_layout()
    figp = os.path.join(RUN_DIR, "checkpoint2_examples.png")
    fig.savefig(figp, dpi=110)
    print(f"\nExample plot saved -> {figp}")
