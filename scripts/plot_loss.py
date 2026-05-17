#!/usr/bin/env python3
"""Parse SSTrack training logs and plot loss / IoU curves.

Usage (requires conda env `must` for Python 3.9+ and matplotlib):

    conda activate must
    python scripts/plot_loss.py logs/baseline_must_trans_enc_cope_cvtp_3.log
    python scripts/plot_loss.py path/to/other.log -o logs/my_plots --smooth 100
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LINE_RE = re.compile(
    r"^\[(train|val):\s*(\d+),\s*(\d+)\s*/\s*(\d+)\]"
)
METRIC_RE = re.compile(
    r"(\d+)frame_(Loss/(\w+)|IoU):\s*([\d.]+)"
)


def parse_log(log_path: Path):
    """Return list of records: {phase, epoch, iter, total_iters, metrics}."""
    records = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            phase, epoch, it, total = m.groups()
            metrics = {}
            for fm in METRIC_RE.finditer(line):
                frame = int(fm.group(1))
                if fm.group(2).startswith("Loss"):
                    key = f"{frame}frame_Loss/{fm.group(3)}"
                else:
                    key = f"{frame}frame_IoU"
                metrics[key] = float(fm.group(4))
            records.append(
                {
                    "phase": phase,
                    "epoch": int(epoch),
                    "iter": int(it),
                    "total_iters": int(total),
                    "metrics": metrics,
                }
            )
    return records


def avg_metrics(records, metric_key):
    vals = [r["metrics"][metric_key] for r in records if metric_key in r["metrics"]]
    return float(np.mean(vals)) if vals else np.nan


def epoch_summary(records, phase, frame=0, loss_type="total"):
    """Per-epoch mean loss."""
    key = f"{frame}frame_Loss/{loss_type}"
    by_epoch = defaultdict(list)
    for r in records:
        if r["phase"] != phase:
            continue
        if key in r["metrics"]:
            by_epoch[r["epoch"]].append(r["metrics"][key])
    epochs = sorted(by_epoch)
    means = [float(np.mean(by_epoch[e])) for e in epochs]
    return epochs, means


def epoch_iou_summary(records, phase, frame=0):
    """Per-epoch mean IoU for a given frame."""
    key = f"{frame}frame_IoU"
    by_epoch = defaultdict(list)
    for r in records:
        if r["phase"] != phase:
            continue
        if key in r["metrics"]:
            by_epoch[r["epoch"]].append(r["metrics"][key])
    epochs = sorted(by_epoch)
    means = [float(np.mean(by_epoch[e])) for e in epochs]
    return epochs, means


def global_step(records, phase):
    """Assign monotonic step index for plotting iteration-level curves."""
    steps, values = [], []
    key = "0frame_Loss/total"
    step = 0
    for r in records:
        if r["phase"] != phase or key not in r["metrics"]:
            continue
        steps.append(step)
        values.append(r["metrics"][key])
        step += 1
    return np.array(steps), np.array(values)


def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_all(records, out_dir: Path, log_name: str, smooth_win: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(
        {int(k.split("frame")[0]) for r in records for k in r["metrics"] if "frame" in k}
    )
    loss_types = ["total", "giou", "l1", "location"]

    # --- 1. Train / val total loss per epoch (frame 0) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for phase, color, label in [("train", "C0", "Train"), ("val", "C1", "Val")]:
        ep, vals = epoch_summary(records, phase, frame=0, loss_type="total")
        if ep:
            ax.plot(ep, vals, "o-", color=color, label=label, linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / total (frame 0)")
    ax.set_title(f"{log_name} — Epoch-averaged total loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_epoch_total.png", dpi=150)
    plt.close(fig)

    # --- 2. Iteration-level train loss (smoothed) ---
    steps, train_loss = global_step(records, "train")
    if len(train_loss) > 0:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(steps, train_loss, alpha=0.25, color="C0", linewidth=0.8, label="raw")
        if len(train_loss) >= smooth_win:
            sm = smooth(train_loss, smooth_win)
            ax.plot(
                np.arange(smooth_win - 1, len(train_loss)),
                sm,
                color="C0",
                linewidth=2,
                label=f"MA-{smooth_win}",
            )
        ax.set_xlabel("Log step (train prints)")
        ax.set_ylabel("Loss / total (frame 0)")
        ax.set_title(f"{log_name} — Training loss curve")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_train_iter.png", dpi=150)
        plt.close(fig)

    # --- 3. Loss components per epoch ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    for ax, lt in zip(axes, loss_types):
        ep, vals = epoch_summary(records, "train", frame=0, loss_type=lt)
        if ep:
            ax.plot(ep, vals, "o-", color="C0", linewidth=1.5, markersize=3)
        ep_v, vals_v = epoch_summary(records, "val", frame=0, loss_type=lt)
        if ep_v:
            ax.plot(ep_v, vals_v, "s--", color="C1", linewidth=1.5, markersize=3)
        ax.set_ylabel(lt)
        ax.grid(True, alpha=0.3)
    for ax in axes:
        ax.set_xlabel("Epoch")
    axes[0].legend(["train", "val"], loc="upper right")
    fig.suptitle(f"{log_name} — Loss components (frame 0)")
    fig.tight_layout()
    fig.savefig(out_dir / "loss_components.png", dpi=150)
    plt.close(fig)

    # --- 4. Per-frame total loss (train, epoch avg) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        ep, vals = epoch_summary(records, "train", frame=frame, loss_type="total")
        if ep:
            ax.plot(ep, vals, "o-", label=f"frame {frame}", linewidth=1.5, markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss / total")
    ax.set_title(f"{log_name} — Per-frame train loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_per_frame.png", dpi=150)
    plt.close(fig)

    # --- 5. IoU per epoch (train & val, frame 0) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for phase, style, color in [("train", "-", "C2"), ("val", "--", "C3")]:
        ep, vals = epoch_iou_summary(records, phase, frame=0)
        if ep:
            ax.plot(ep, vals, style + "o", color=color, label=phase, linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU (frame 0)")
    ax.set_title(f"{log_name} — IoU (frame 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iou_epoch.png", dpi=150)
    plt.close(fig)

    # --- 6. Val IoU per frame (overlay) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for frame in frames:
        ep, vals = epoch_iou_summary(records, "val", frame=frame)
        if ep:
            ax.plot(ep, vals, "o-", label=f"frame {frame}", linewidth=1.5, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title(f"{log_name} — Val IoU per frame")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iou_val_per_frame.png", dpi=150)
    plt.close(fig)

    # --- 7. Val IoU per frame (subplots) ---
    n = len(frames)
    if n > 0:
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, frame in zip(axes, frames):
            ep, vals = epoch_iou_summary(records, "val", frame=frame)
            if ep:
                ax.plot(ep, vals, "o-", color="C1", linewidth=1.5, markersize=4)
            ax.set_ylabel("IoU")
            ax.set_title(f"frame {frame}")
            ax.grid(True, alpha=0.3)
        for ax in axes[n:]:
            ax.set_visible(False)
        for ax in axes[max(0, n - ncols) : n]:
            ax.set_xlabel("Epoch")
        fig.suptitle(f"{log_name} — Val IoU per frame")
        fig.tight_layout()
        fig.savefig(out_dir / "iou_val_per_frame_grid.png", dpi=150)
        plt.close(fig)

    print(f"Saved plots to {out_dir}/")
    print(f"  - loss_epoch_total.png")
    print(f"  - loss_train_iter.png")
    print(f"  - loss_components.png")
    print(f"  - loss_per_frame.png")
    print(f"  - iou_epoch.png")
    print(f"  - iou_val_per_frame.png")
    print(f"  - iou_val_per_frame_grid.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize SSTrack training log losses.")
    parser.add_argument(
        "log_file",
        type=Path,
        nargs="?",
        default=Path("logs/baseline_must_trans_enc_cope_cvtp_3.log"),
        help="Path to training log",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: logs/<log_stem>_plots)",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=50,
        help="Moving-average window for iteration plot (default: 50)",
    )
    args = parser.parse_args()

    log_path = args.log_file.resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    out_dir = args.output or (log_path.parent / f"{log_path.stem}_plots")
    records = parse_log(log_path)
    n_train = sum(1 for r in records if r["phase"] == "train")
    n_val = sum(1 for r in records if r["phase"] == "val")
    print(f"Parsed {log_path.name}: {n_train} train steps, {n_val} val steps")
    if not records:
        raise SystemExit("No train/val metric lines found in log.")

    plot_all(records, out_dir, log_path.stem, args.smooth)


if __name__ == "__main__":
    main()
