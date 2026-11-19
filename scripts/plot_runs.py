"""Plot training curves and side-by-side samples for runs in runs/*.

    uv run python scripts/plot_runs.py                      # all runs -> assets/
    uv run python scripts/plot_runs.py --runs flow selfflow --out assets

Writes:
    curves.png            gen loss / rep loss / grad norm vs step, and gen loss vs wall-clock
    samples_<step>.png    latest common sample grid of each run, side by side
    eval.png              FID / vanilla val loss / linear probe vs step, and probe accuracy per block
                          (from runs/<name>/eval.jsonl, written by `python -m minselfflow.evaluate`)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from minselfflow.utils.log import logger


def load_metrics(run: Path) -> dict[str, np.ndarray]:
    """metrics.jsonl -> {key: array}, keeping only the last entry per step (resumed runs repeat steps)."""
    rows = {r["step"]: r for r in map(json.loads, (run / "metrics.jsonl").read_text().splitlines())}
    rows = [rows[s] for s in sorted(rows)]
    keys = rows[0].keys()
    return {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in keys}


def smooth(x: np.ndarray, k: int = 20) -> np.ndarray:
    """Trailing moving average; same length as x."""
    c = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - k, 0)
    return (c[i] - c[lo]) / (i - lo)


def plot_curves(runs: dict[str, dict[str, np.ndarray]], path: Path) -> None:
    """2x2 grid: gen loss, rep loss, grad norm vs step; gen loss vs GPU-hours."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for name, m in runs.items():
        step = m["step"]
        hours = np.cumsum(100 * 512 / m["img_per_s"]) / 3600  # wall-clock from logged throughput
        ax[0, 0].plot(step, smooth(m["gen"]), label=name)
        if "rep" in m:
            ax[0, 1].plot(step, smooth(m["rep"]), label=name)
        ax[1, 0].plot(step, smooth(m["grad_norm"]), label=name)
        ax[1, 1].plot(hours, smooth(m["gen"]), label=name)
    titles = [
        "flow loss (gen) vs step",
        "representation loss (-cos) vs step",
        "grad norm vs step",
        "flow loss vs hours",
    ]
    for a, title in zip(ax.flat, titles):
        a.set_title(title)
        a.legend()
        a.grid(alpha=0.3)
    ax[0, 0].set_ylim(top=0.2)
    ax[1, 1].set_ylim(top=0.2)
    ax[1, 1].set_xlabel("hours (2x A100)")
    for a in ax[:, 0]:
        a.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    logger.info("saved %s", path)


def plot_eval(runs: dict[str, list[dict]], path: Path) -> None:
    """2x2 grid: FID, vanilla val loss, best probe accuracy vs step; probe accuracy per block at the last step."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for name, rows in runs.items():
        rows = sorted(rows, key=lambda r: r["step"])
        step = [r["step"] for r in rows]
        ax[0, 0].plot(step, [r["fid"] for r in rows], "o-", label=name)
        ax[0, 1].plot(step, [r["val_loss"] for r in rows], "o-", label=name)
        ax[1, 0].plot(step, [r["probe_best"] for r in rows], "o-", label=name)
        last = rows[-1]
        blocks = np.arange(1, len(last["probe_acc"]) + 1)
        ax[1, 1].plot(blocks, last["probe_acc"], "o-", label="%s @ %dk" % (name, last["step"] // 1000))
    titles = [
        "FID-10k (lower is better)",
        "val flow loss, same t for all tokens",
        "linear probe, best block",
        "linear probe per block",
    ]
    for a, title in zip(ax.flat, titles):
        a.set_title(title)
        a.legend()
        a.grid(alpha=0.3)
    for a in [ax[0, 0], ax[0, 1], ax[1, 0]]:
        a.set_xlabel("step")
    ax[1, 1].set_xlabel("block")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    logger.info("saved %s", path)


def plot_samples(run_dirs: list[Path], out: Path) -> None:
    """Latest sample grid that every run has, side by side."""
    steps = [{p.stem.split("_")[1] for p in d.glob("samples_*.png")} for d in run_dirs]
    common = sorted(set.intersection(*steps))
    if not common:
        logger.info("no common sample step yet")
        return
    step = common[-1]
    fig, ax = plt.subplots(1, len(run_dirs), figsize=(6 * len(run_dirs), 6.4))
    for a, d in zip(np.atleast_1d(ax), run_dirs):
        a.imshow(Image.open(d / ("samples_%s.png" % step)))
        a.set_title("%s @ step %d" % (d.name, int(step)))
        a.axis("off")
    fig.tight_layout()
    path = out / ("samples_%s.png" % step)
    fig.savefig(path, dpi=120)
    logger.info("saved %s", path)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="runs")
    p.add_argument("--runs", nargs="*", default=None)
    p.add_argument("--out", default="assets")
    args = p.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = args.runs or sorted(d.name for d in root.iterdir() if (d / "metrics.jsonl").exists())
    dirs = [root / n for n in names]
    plot_curves({d.name: load_metrics(d) for d in dirs}, out / "curves.png")
    plot_samples(dirs, out)
    evals = {
        d.name: [json.loads(x) for x in (d / "eval.jsonl").read_text().splitlines()]
        for d in dirs
        if (d / "eval.jsonl").exists()
    }
    if evals:
        plot_eval(evals, out / "eval.png")


if __name__ == "__main__":
    main()
