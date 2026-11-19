"""Evaluate EMA checkpoints (runs/<name>/ema_*.pt) on three axes, appending to runs/<name>/eval.jsonl.

    # one GPU per run; re-running skips steps already evaluated
    CUDA_VISIBLE_DEVICES=0 python -m minselfflow.evaluate --run runs/flow
    CUDA_VISIBLE_DEVICES=1 python -m minselfflow.evaluate --run runs/selfflow

1. FID      - sample quality: n_fid samples (50 Euler steps, no CFG) vs CIFAR-10 train Inception stats.
2. val loss - vanilla flow loss (same t for every token) on the test set at fixed t, fixed noise.
              Fair across methods: Self-Flow's *training* loss is measured on dual-timestep inputs.
3. probe    - linear probe accuracy on mean-pooled block features of the test set, for every block.
              Images are lightly noised (probe_t) and the label is the "no class" token, so the model
              cannot read the answer off its conditioning.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch import Tensor, nn

from minselfflow.data import load_cifar10, to_model_range
from minselfflow.flow import add_noise, euler_sample
from minselfflow.model import DiT, DiTConfig, patchify, unpatchify
from minselfflow.utils.log import logger


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation settings."""

    n_fid: int = 10_000
    sample_steps: int = 50
    batch: int = 1000
    val_ts: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    probe_t: float = 0.1
    probe_epochs: int = 30
    seed: int = 0


def load_ema(path: Path, device: torch.device) -> tuple[DiT, int, int]:
    """Load an EMA snapshot written by train.py.

    Returns:
        tuple[DiT, int, int]: model in eval mode, patch size, training step.
    """
    ckpt = torch.load(path, map_location=device)
    c = ckpt["cfg"]
    num_tokens = (32 // c["patch"]) ** 2
    mcfg = DiTConfig(
        in_dim=c["patch"] ** 2 * 3, num_tokens=num_tokens, dim=c["dim"], depth=c["depth"], heads=c["heads"]
    )
    model = DiT(mcfg).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval().requires_grad_(False), c["patch"], ckpt["step"]


def to_uint8(x: Tensor) -> Tensor:
    """float [-1, 1] (B, 3, H, W) -> uint8 [0, 255]."""
    return ((x.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def generate(model: DiT, n: int, patch: int, cfg: EvalConfig, device: torch.device) -> Tensor:
    """n class-balanced samples from fixed noise.

    Returns:
        Tensor: uint8 images, shape (n, 3, 32, 32).
    """
    gen = torch.Generator(device).manual_seed(cfg.seed)
    out = []
    for i in range(0, n, cfg.batch):
        b = min(cfg.batch, n - i)
        y = torch.arange(i, i + b, device=device) % model.cfg.num_classes
        x1 = torch.randn(b, model.cfg.num_tokens, model.cfg.in_dim, device=device, generator=gen)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            x0 = euler_sample(model, x1, y, steps=cfg.sample_steps)
        out.append(to_uint8(unpatchify(x0.float(), patch)))
    return torch.cat(out)


@torch.no_grad()
def inception_stats(images: Tensor, net: nn.Module, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of Inception pool3 features.

    Args:
        images (Tensor): uint8, shape (B, 3, H, W).
        net (nn.Module): pytorch_fid InceptionV3 (2048-d block), resizes to 299 internally.

    Returns:
        tuple[np.ndarray, np.ndarray]: mu (2048,), sigma (2048, 2048), float64.
    """
    device = next(net.parameters()).device
    feats = [net(images[i : i + batch].to(device).float() / 255.0)[0].flatten(1) for i in range(0, len(images), batch)]
    f = torch.cat(feats).double().cpu().numpy()
    return f.mean(0), np.cov(f, rowvar=False)


def reference_stats(net: nn.Module, data_dir: str, batch: int) -> tuple[np.ndarray, np.ndarray]:
    """Inception stats of the CIFAR-10 train set, cached in data_dir."""
    cache = Path(data_dir) / "fid_stats_cifar10_train.npz"
    if cache.exists():
        s = np.load(cache)
        return s["mu"], s["sigma"]
    images, _ = load_cifar10(data_dir, "train")
    mu, sigma = inception_stats(images, net, batch)
    np.savez(cache, mu=mu, sigma=sigma)
    logger.info("cached FID reference stats to %s", cache)
    return mu, sigma


@torch.no_grad()
def val_loss(model: DiT, images: Tensor, labels: Tensor, patch: int, cfg: EvalConfig) -> dict[str, float]:
    """Vanilla flow-matching loss at fixed t values with fixed noise.

    Args:
        images (Tensor): uint8 test images on device, shape (M, 3, 32, 32).
        labels (Tensor): int64 labels on device, shape (M,).

    Returns:
        dict[str, float]: {"t=0.1": mse, ..., "mean": mse averaged over t}.
    """
    gen = torch.Generator(images.device).manual_seed(cfg.seed)
    totals = dict.fromkeys(cfg.val_ts, 0.0)
    for i in range(0, len(images), cfg.batch):
        x0 = patchify(to_model_range(images[i : i + cfg.batch]), patch)
        y = labels[i : i + cfg.batch]
        x1 = torch.randn(x0.shape, device=x0.device, generator=gen)
        B, N, _ = x0.shape
        for t in cfg.val_ts:
            tau = torch.full((B, N), t, device=x0.device)
            with torch.autocast(x0.device.type, dtype=torch.bfloat16, enabled=x0.device.type == "cuda"):
                v = model(add_noise(x0, x1, tau), tau, y)
            totals[t] += F.mse_loss(v.float(), x1 - x0, reduction="sum").item()
    per_t = {"t=%.1f" % t: s / images.numel() for t, s in totals.items()}
    return {**per_t, "mean": float(np.mean(list(per_t.values())))}


@torch.no_grad()
def block_features(model: DiT, images: Tensor, patch: int, cfg: EvalConfig) -> Tensor:
    """Mean-pooled output of every block, for lightly noised images with the "no class" label.

    Returns:
        Tensor: float32, shape (depth, M, dim).
    """
    gen = torch.Generator(images.device).manual_seed(cfg.seed)
    out = []
    for i in range(0, len(images), cfg.batch):
        x0 = patchify(to_model_range(images[i : i + cfg.batch]), patch)
        B, N, _ = x0.shape
        tau = torch.full((B, N), cfg.probe_t, device=x0.device)
        x = add_noise(x0, torch.randn(x0.shape, device=x0.device, generator=gen), tau)
        y = torch.full((B,), model.cfg.num_classes, device=x0.device)
        with torch.autocast(x0.device.type, dtype=torch.bfloat16, enabled=x0.device.type == "cuda"):
            _, feats = model.forward_features(x, tau, y)
        out.append(torch.stack([f.float().mean(1) for f in feats]))
    return torch.cat(out, dim=1)


def linear_probe(
    train_x: Tensor, train_y: Tensor, test_x: Tensor, test_y: Tensor, epochs: int, seed: int = 0
) -> list[float]:
    """Train one linear classifier per layer (all layers at once) and return test accuracy per layer.

    Args:
        train_x (Tensor): features, shape (L, M, D).
        train_y (Tensor): labels, shape (M,).
        test_x (Tensor): features, shape (L, M_test, D).
        test_y (Tensor): labels, shape (M_test,).
        epochs (int): passes over the training features.

    Returns:
        list[float]: test accuracy for each of the L layers.
    """
    torch.manual_seed(seed)
    L, M, D = train_x.shape
    mean, std = train_x.mean(1, keepdim=True), train_x.std(1, keepdim=True) + 1e-6
    train_x, test_x = (train_x - mean) / std, (test_x - mean) / std
    num_classes = int(train_y.max()) + 1
    W = torch.zeros(L, D, num_classes, device=train_x.device, requires_grad=True)
    b = torch.zeros(L, 1, num_classes, device=train_x.device, requires_grad=True)
    opt = torch.optim.AdamW([W, b], lr=1e-2, weight_decay=1e-4)
    steps_per_epoch = (M + 1023) // 1024
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch)
    for _ in range(epochs):
        perm = torch.randperm(M, device=train_x.device)
        for idx in perm.split(1024):
            logits = torch.bmm(train_x[:, idx], W) + b  # (L, B, K)
            loss = F.cross_entropy(logits.flatten(0, 1), train_y[idx].repeat(L))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        pred = (torch.bmm(test_x, W) + b).argmax(-1)  # (L, M_test)
        return (pred == test_y).float().mean(1).tolist()


def evaluate_checkpoint(
    path: Path, cfg: EvalConfig, data: dict[str, Tensor], inception: nn.Module, ref: tuple[np.ndarray, np.ndarray]
) -> dict:
    """All metrics for one EMA checkpoint."""
    device = data["train_x"].device
    model, patch, step = load_ema(path, device)
    samples = generate(model, cfg.n_fid, patch, cfg, device)
    mu, sigma = inception_stats(samples, inception, cfg.batch)
    fid = float(calculate_frechet_distance(mu, sigma, *ref))
    vl = val_loss(model, data["test_x"], data["test_y"], patch, cfg)
    tr = block_features(model, data["train_x"], patch, cfg)
    te = block_features(model, data["test_x"], patch, cfg)
    acc = linear_probe(tr, data["train_y"], te, data["test_y"], cfg.probe_epochs, cfg.seed)
    best = int(np.argmax(acc))
    return {
        "step": step,
        "fid": fid,
        "val_loss": vl["mean"],
        "val_loss_t": {k: v for k, v in vl.items() if k != "mean"},
        "probe_acc": acc,
        "probe_best": acc[best],
        "probe_best_block": best + 1,
    }


def main(argv: list[str]) -> None:
    """CLI entry point: evaluate every not-yet-evaluated ema_*.pt in --run."""
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--n_fid", type=int, default=EvalConfig.n_fid)
    p.add_argument("--probe_t", type=float, default=EvalConfig.probe_t)
    args = p.parse_args(argv)
    cfg = EvalConfig(n_fid=args.n_fid, probe_t=args.probe_t)
    run, device = Path(args.run), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True

    out = run / "eval.jsonl"
    done = {json.loads(line)["step"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [c for c in sorted(run.glob("ema_*.pt")) if int(c.stem.split("_")[1]) not in done]
    logger.info("%s: %d checkpoints to evaluate (%d done)", run, len(todo), len(done))
    if not todo:
        return

    inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    ref = reference_stats(inception, args.data_dir, cfg.batch)
    (train_x, train_y), (test_x, test_y) = load_cifar10(args.data_dir, "train"), load_cifar10(args.data_dir, "test")
    data = {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}
    data = {k: v.to(device) for k, v in data.items()}

    for ckpt in todo:
        row = {**evaluate_checkpoint(ckpt, cfg, data, inception, ref), "eval_cfg": asdict(cfg)}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        logger.info(
            "%s step %d | FID %.2f | val loss %.4f | probe %.3f (block %d)",
            run.name,
            row["step"],
            row["fid"],
            row["val_loss"],
            row["probe_best"],
            row["probe_best_block"],
        )


if __name__ == "__main__":
    main(sys.argv[1:])
