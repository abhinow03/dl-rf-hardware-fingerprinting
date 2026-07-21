"""Temperature sweep + clean retrain for the WiSig ManyTx metric encoder.

WHY: the locked full run (train_wisig.py --full) peaked at gap 0.834 (step 3250, tau~0.41)
then COLLAPSED to ~0.36 in the tau=0.07 tail — global contraction (inter-cos 0.0->0.55),
NOT receiver leakage (rx_gap stayed ~0.02). Loss kept dropping the whole time, so LOSS IS
NOT A VALID SELECTION SIGNAL here. This script finds a temperature schedule whose separation
is STABLE (doesn't collapse), then retrains once with best-by-GAP checkpoint selection.

Everything matches the full run except temperature: CFO OFF, board-disjoint 109-Tx train,
32x8 batch, LR=5e-4, wd=1e-4, cosine+warmup, grad-clip 1.0, AMP. Metric trains on WiSig
ManyTx ONLY — ManySig stays held out as the transfer benchmark.

    python3 train/sweep_temp.py --sweep     # PART A: 3 temp configs x 5k steps -> CSVs + table
    python3 train/sweep_temp.py --retrain --tau <name>   # PART B: clean retrain, best-by-gap
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SW = os.path.dirname(_HERE)
for p in (_SW, os.path.join(_SW, "datasets")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rffp.models import RFEncoder, SupervisedContrastiveLoss
from rffp.data import wisig_manytx as W
from train_wisig import make_fixed_eval, eval_fixed, to_cuda, RUN_DIR, SPLIT_PATH

# ── locked-from-full-run knobs (only temperature varies) ───────
N_DEV, M_SIG = W.DEFAULT_N, W.DEFAULT_M     # 32 x 8 = 256
LR0       = 5e-4
WD        = 1e-4
GRAD_CLIP = 1.0
SEED      = 42

SWEEP_STEPS  = 5000
SWEEP_WARMUP = 500                          # ~10%
LOG_STEP     = 250
SWEEP_DIR    = os.path.join(RUN_DIR, "temp_sweep")


# ── schedules ──────────────────────────────────────────────────
def lr_at(step, total, warmup, lr0=LR0):
    if step < warmup:
        return lr0 * (step + 1) / warmup
    prog = (step - warmup) / (total - warmup)
    return 0.5 * lr0 * (1 + np.cos(np.pi * prog))


def make_tau(name):
    """Return tau(step, total). Sweep configs + the retrain winner all live here."""
    if name == "const0.3":
        return lambda s, t: 0.3
    if name == "const0.5":
        return lambda s, t: 0.5
    if name == "anneal0.5to0.2":
        # linear 0.5 -> 0.2 over the first 60% of the run, then hold 0.2 (never lower)
        def f(s, t):
            frac, hi, lo, knee = s / t, 0.5, 0.2, 0.6
            if frac >= knee:
                return lo
            return hi + (frac / knee) * (lo - hi)
        return f
    raise ValueError(f"unknown tau config {name!r}")


# ── one training run ───────────────────────────────────────────
def train_one(tau_name, total_steps, warmup, tx_data, train_tx, out_dir,
              save_ckpts=False, ckpt_step=2000):
    """Train fresh; log health every LOG_STEP; track + (optionally) save best-by-GAP.

    Returns dict with the full trajectory and peak/final summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    label_of = {tx: i for i, tx in enumerate(train_tx)}
    tau_fn = make_tau(tau_name)

    model = RFEncoder().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR0, weight_decay=WD)
    crit = SupervisedContrastiveLoss()
    scaler = torch.amp.GradScaler('cuda')
    fixed = make_fixed_eval(tx_data, train_tx)          # seed=7, no-aug, shared across configs

    csv_path = os.path.join(out_dir, f"sweep_{tau_name}.csv")
    cf = open(csv_path, "w", newline="")
    wr = csv.writer(cf)
    wr.writerow(["step", "loss", "intra", "inter", "gap", "rx_gap", "lr", "tau"])

    traj = []
    best_gap, best_step = -1.0, 0
    best_path = os.path.join(out_dir, "best_model.pt") if save_ckpts else None

    model.train()
    for step in range(total_steps):
        lr = lr_at(step, total_steps, warmup); tau = float(tau_fn(step, total_steps))
        for g in opt.param_groups:
            g["lr"] = lr
        t, s, y, _ = W.sample_batch(tx_data, train_tx, label_of, rng,
                                    N=N_DEV, M=M_SIG, train=True, use_cfo=False)
        xt, xs = to_cuda(t, s)
        yt = torch.from_numpy(y).cuda()
        opt.zero_grad()
        with torch.amp.autocast('cuda'):
            emb = model(xt, xs)
        loss = crit(emb.float(), yt, temperature=tau)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt); scaler.update()

        if (step + 1) % LOG_STEP == 0 or step == 0:
            intra, inter, gap, rx_gap = eval_fixed(model, fixed)
            wr.writerow([step + 1, round(float(loss), 4), round(intra, 4), round(inter, 4),
                         round(gap, 4), round(rx_gap, 4), round(lr, 7), round(tau, 4)])
            cf.flush()
            traj.append({"step": step + 1, "loss": float(loss), "intra": intra,
                         "inter": inter, "gap": gap, "rx_gap": rx_gap, "tau": tau})
            print(f"[{tau_name}] step {step+1:5d}/{total_steps} loss={float(loss):.3f} "
                  f"intra={intra:.3f} inter={inter:.3f} gap={gap:.3f} rx_gap={rx_gap:.3f} "
                  f"tau={tau:.3f}", flush=True)
            if gap > best_gap:
                best_gap, best_step = gap, step + 1
                if best_path:
                    torch.save(model.state_dict(), best_path)

        if save_ckpts and (step + 1) % ckpt_step == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"ckpt_step{step+1}.pt"))

    if save_ckpts:
        torch.save(model.state_dict(), os.path.join(out_dir, "last_model.pt"))
    cf.close()

    final = traj[-1]
    # "collapsed" = lost a meaningful chunk of peak separation by the end of the run
    collapsed = (best_gap - final["gap"]) > 0.10
    summary = {
        "tau_config": tau_name, "total_steps": total_steps,
        "peak_gap": best_gap, "peak_step": best_step,
        "final_gap": final["gap"], "final_inter": final["inter"],
        "drop_from_peak": best_gap - final["gap"], "collapsed": collapsed,
        "rx_gap_min": min(p["rx_gap"] for p in traj),
        "rx_gap_max": max(p["rx_gap"] for p in traj),
        "best_model": best_path, "csv": csv_path, "traj": traj,
    }
    print(f"[{tau_name}] DONE  peak={best_gap:.4f}@{best_step}  final={final['gap']:.4f}  "
          f"drop={best_gap-final['gap']:.4f}  {'COLLAPSED' if collapsed else 'STABLE'}", flush=True)
    return summary


# ── PART A ─────────────────────────────────────────────────────
def run_sweep(tx_data, train_tx):
    os.makedirs(SWEEP_DIR, exist_ok=True)
    configs = ["const0.3", "const0.5", "anneal0.5to0.2"]
    results = {}
    for name in configs:
        # clean any stale bytecode before each fresh config
        for d in (_HERE, os.path.join(_SW, "datasets")):
            pc = os.path.join(d, "__pycache__")
            if os.path.isdir(pc):
                for fn in os.listdir(pc):
                    os.remove(os.path.join(pc, fn))
        print(f"\n{'='*64}\nSWEEP CONFIG: {name}  ({SWEEP_STEPS} steps)\n{'='*64}", flush=True)
        results[name] = train_one(name, SWEEP_STEPS, SWEEP_WARMUP,
                                   tx_data, train_tx, SWEEP_DIR, save_ckpts=False)

    # comparison table
    print(f"\n{'='*72}\nSWEEP COMPARISON  (winner = highest STABLE gap, not highest peak)\n{'='*72}")
    print(f"{'config':16s} {'peak_gap':>9s} {'@step':>7s} {'final':>8s} {'drop':>7s} "
          f"{'verdict':>10s} {'rx_gap[min,max]':>18s}")
    for name, r in results.items():
        verdict = "COLLAPSED" if r["collapsed"] else "STABLE"
        print(f"{name:16s} {r['peak_gap']:>9.4f} {r['peak_step']:>7d} {r['final_gap']:>8.4f} "
              f"{r['drop_from_peak']:>7.4f} {verdict:>10s} "
              f"[{r['rx_gap_min']:.3f},{r['rx_gap_max']:.3f}]")

    stable = {k: v for k, v in results.items() if not v["collapsed"]}
    pool = stable if stable else results
    winner = max(pool, key=lambda k: pool[k]["peak_gap"])
    print(f"\nWINNER = {winner}  (peak {results[winner]['peak_gap']:.4f}, "
          f"{'stable' if not results[winner]['collapsed'] else 'least-bad'})")

    out = os.path.join(SWEEP_DIR, "sweep_summary.json")
    with open(out, "w") as f:
        json.dump({"winner": winner,
                   "results": {k: {kk: vv for kk, vv in v.items() if kk != "traj"}
                               for k, v in results.items()}}, f, indent=2)
    print(f"saved -> {out}")
    return winner, results


# ── PART B ─────────────────────────────────────────────────────
RETRAIN_DIR = os.path.join(RUN_DIR, "retrain_best")

def run_retrain(tau_name, total_steps, tx_data, train_tx):
    warmup = int(0.10 * total_steps)
    print(f"\n{'='*64}\nCLEAN RETRAIN — tau={tau_name}, {total_steps} steps, "
          f"best-by-GAP checkpoint\n{'='*64}", flush=True)
    summary = train_one(tau_name, total_steps, warmup, tx_data, train_tx,
                        RETRAIN_DIR, save_ckpts=True, ckpt_step=2000)
    out = os.path.join(RETRAIN_DIR, "retrain_summary.json")
    with open(out, "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "traj"}, f, indent=2)
    print(f"\nRETRAIN DONE. best-by-gap encoder -> {summary['best_model']}")
    print(f"  peak gap {summary['peak_gap']:.4f} @ step {summary['peak_step']} | "
          f"final {summary['final_gap']:.4f} | {'STABLE' if not summary['collapsed'] else 'COLLAPSED'}")
    print(f"  rx_gap range [{summary['rx_gap_min']:.3f}, {summary['rx_gap_max']:.3f}] | summary -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--tau", type=str, default=None, help="tau config for --retrain")
    ap.add_argument("--steps", type=int, default=8000, help="retrain step budget")
    args = ap.parse_args()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tx_data, _ = W.load_manytx()
    sp = json.load(open(SPLIT_PATH))
    assert sp["mode"] == "board", "split must be board-disjoint"
    train_tx = sp["train_tx"]
    print(f"Split: board-disjoint  train={sp['n_train']}  discover={sp['n_discover']}")

    if args.sweep:
        run_sweep(tx_data, train_tx)
    if args.retrain:
        assert args.tau, "--retrain needs --tau <config name>"
        run_retrain(args.tau, args.steps, tx_data, train_tx)
    if not (args.sweep or args.retrain):
        print("Pass --sweep and/or --retrain --tau <name>.")


if __name__ == "__main__":
    main()
