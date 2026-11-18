"""Training loop (single GPU or multi-GPU DDP).

    # 4 GPUs
    torchrun --nproc_per_node=4 -m minselfflow.train --out_dir runs/flow --steps 100000
    # 1 GPU / CPU
    python -m minselfflow.train --steps 1000

Every rank keeps the whole CIFAR-10 train set on its GPU (150MB) and samples random
batches from it, so there is no DataLoader. The global batch is split across ranks.
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import save_image

from minselfflow.data import load_cifar10, to_image_range, to_model_range
from minselfflow.flow import euler_sample, flow_loss
from minselfflow.model import DiT, DiTConfig, patchify, unpatchify
from minselfflow.utils.log import logger


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters; every field is also a --flag on the command line."""

    method: Literal["flow"] = "flow"
    out_dir: str = "runs/flow"
    data_dir: str = "data"
    steps: int = 100_000
    batch_size: int = 512  # global, split across GPUs
    lr: float = 2e-4
    warmup: int = 1000  # linear LR warmup steps
    grad_clip: float = 1.0
    ema_decay: float = 0.9999  # max decay; ramps up from 0 early in training (see ema_decay_at)
    class_dropout: float = 0.1  # fraction of labels replaced by "no class", enables CFG later
    dim: int = 384
    depth: int = 12
    heads: int = 6
    patch: int = 2
    log_every: int = 100
    sample_every: int = 5000
    ckpt_every: int = 10_000
    seed: int = 0
    compile: bool = True


def parse_config(argv: list[str]) -> TrainConfig:
    """Build a TrainConfig from --field value flags; types come from the defaults."""
    parser = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        kind = type(f.default)
        parse = (lambda s: s.lower() in ("1", "true", "yes")) if kind is bool else kind
        parser.add_argument("--" + f.name, type=parse, default=f.default)
    return TrainConfig(**vars(parser.parse_args(argv)))


@dataclass(frozen=True)
class Dist:
    """Where this process sits in the distributed job."""

    rank: int
    world: int
    device: torch.device

    @property
    def main(self) -> bool:
        return self.rank == 0


def setup_dist() -> Dist:
    """Init NCCL when launched by torchrun, otherwise run as a single process."""
    if "RANK" not in os.environ:
        return Dist(0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    return Dist(dist.get_rank(), dist.get_world_size(), device)


def ema_decay_at(step: int, max_decay: float) -> float:
    """EMA warmup: decay (1+step)/(10+step) capped at max_decay.

    Early weights change fast and averaging them gives a broken model, so the EMA tracks the
    model closely at first (decay 0.1 at step 0, 0.9 at step 80, 0.999 at step ~9000).
    """
    return min(max_decay, (1 + step) / (10 + step))


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    """ema <- decay * ema + (1 - decay) * model, parameter-wise."""
    for e, p in zip(ema.parameters(), model.parameters()):
        e.lerp_(p, 1.0 - decay)


def sample_batch(images: Tensor, labels: Tensor, n: int, cfg: TrainConfig, num_classes: int) -> tuple[Tensor, Tensor]:
    """Random batch with horizontal flips and label dropout.

    Args:
        images (Tensor): uint8 dataset on device, shape (M, 3, H, W).
        labels (Tensor): int64 labels on device, shape (M,).
        n (int): per-rank batch size.

    Returns:
        tuple[Tensor, Tensor]: tokens (n, N, C) in [-1, 1] and labels (n,), num_classes = dropped.
    """
    idx = torch.randint(len(images), (n,), device=images.device)
    x = to_model_range(images[idx])
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    y = torch.where(torch.rand(n, device=x.device) < cfg.class_dropout, num_classes, labels[idx])
    return patchify(x, cfg.patch), y


@torch.no_grad()
def save_samples(model: nn.Module, cfg: TrainConfig, step: int, device: torch.device) -> None:
    """Save a 10x10 grid (one row per class) from fixed noise, 50 Euler steps."""
    gen = torch.Generator(device).manual_seed(1234)
    y = torch.arange(10, device=device).repeat_interleave(10)
    x1 = torch.randn(100, (32 // cfg.patch) ** 2, cfg.patch**2 * 3, device=device, generator=gen)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = euler_sample(model, x1, y, steps=50)
    path = Path(cfg.out_dir) / ("samples_%06d.png" % step)
    save_image(to_image_range(unpatchify(out.float(), cfg.patch)), path, nrow=10)
    logger.info("saved %s", path)


def train(cfg: TrainConfig, images: Tensor, labels: Tensor, d: Dist) -> None:
    """Run training; rank 0 writes metrics.jsonl, sample grids and ckpt.pt into cfg.out_dir.

    Args:
        cfg (TrainConfig): hyperparameters.
        images (Tensor): uint8 training images, shape (M, 3, 32, 32).
        labels (Tensor): int64 labels, shape (M,).
        d (Dist): distributed context.
    """
    assert cfg.batch_size % d.world == 0, (cfg.batch_size, d.world)
    torch.manual_seed(cfg.seed + d.rank)  # different data/noise per rank
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    num_tokens = (images.shape[-1] // cfg.patch) ** 2
    mcfg = DiTConfig(in_dim=cfg.patch**2 * 3, num_tokens=num_tokens, dim=cfg.dim, depth=cfg.depth, heads=cfg.heads)
    model = DiT(mcfg).to(d.device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))

    start = 0
    ckpt_path = out / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=d.device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start = ckpt["step"]
        logger.info("resumed from %s at step %d", ckpt_path, start)

    net: nn.Module = torch.compile(model) if cfg.compile else model  # ty: ignore[invalid-assignment]
    if d.world > 1:
        net = DistributedDataParallel(net, device_ids=[d.device.index])
    images, labels = images.to(d.device), labels.to(d.device)
    per_rank = cfg.batch_size // d.world

    if d.main:
        logger.info(
            "params %.1fM | %d GPUs x %d per GPU", sum(p.numel() for p in model.parameters()) / 1e6, d.world, per_rank
        )
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    t0, metrics = time.time(), (out / "metrics.jsonl").open("a")
    for step in range(start + 1, cfg.steps + 1):
        x0, y = sample_batch(images, labels, per_rank, cfg, mcfg.num_classes)
        with torch.autocast(d.device.type, dtype=torch.bfloat16, enabled=d.device.type == "cuda"):
            match cfg.method:
                case "flow":
                    loss = flow_loss(net, x0, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        update_ema(ema, model, ema_decay_at(step, cfg.ema_decay))

        if step % cfg.log_every == 0:
            stats = torch.stack([loss.detach().float(), grad_norm.float()])
            if d.world > 1:
                dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            elapsed, t0 = time.time() - t0, time.time()
            img_per_s = cfg.log_every * cfg.batch_size / elapsed
            if d.main:
                row = {"step": step, "loss": stats[0].item(), "grad_norm": stats[1].item(), "img_per_s": img_per_s}
                metrics.write(json.dumps(row) + "\n")
                metrics.flush()
                logger.info(
                    "step %d | loss %.4f | grad %.3f | %.0f img/s", step, row["loss"], row["grad_norm"], img_per_s
                )
        if d.main and (step % cfg.sample_every == 0 or step == cfg.steps):
            save_samples(ema, cfg, step, d.device)
        if d.main and (step % cfg.ckpt_every == 0 or step == cfg.steps):
            state = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
            }
            torch.save({**state, "step": step, "cfg": asdict(cfg)}, ckpt_path)
            logger.info("saved %s", ckpt_path)
    metrics.close()


def main(argv: list[str]) -> None:
    """CLI entry point."""
    cfg = parse_config(argv)
    d = setup_dist()
    if d.main:
        load_cifar10(cfg.data_dir, "train")  # download once before other ranks read the cache
    if d.world > 1:
        dist.barrier()
    images, labels = load_cifar10(cfg.data_dir, "train")
    train(cfg, images, labels, d)
    if d.world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
