# minSelfFlow

A minimal, hackable, from-scratch implementation of **Self-Flow** — *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis* ([arXiv:2603.06507](https://arxiv.org/abs/2603.06507), Chefer, Esser, Lorenz, Podell, Raja, Tong, Torralba, Rombach).

In the spirit of [cloneofsimo](https://github.com/cloneofsimo)'s `min*` repos: small enough to read in one sitting, real enough to train and see the difference.

> 🚧 **Work in progress.** This repo is being built step by step as a learning log. Each step lands as its own commit so you can follow along.

## What is Self-Flow? (TL;DR)

Standard flow matching learns to generate, but its internal representations are weak. Methods like **REPA** fix this by aligning features with an *external* frozen encoder (e.g. DINOv2). Self-Flow removes that dependency:

- **Dual-Timestep Scheduling** — different subsets of tokens get *different* noise levels, creating an information asymmetry: the model must infer the heavily-corrupted tokens from the cleaner ones.
- **EMA teacher** — a self-distillation target (teacher sees a cleaner view) gives a semantic feature-prediction loss.
- **One unified objective** — standard flow loss on the heterogeneously-noised input + self-supervised representation loss. No external models.

## Goal of this repo

1. Implement a **baseline flow-matching DiT** (small, ~tens of M params).
2. Add **Self-Flow** on top (dual-timestep scheduling + EMA teacher + representation loss).
3. Train both on a small dataset with the **same compute** on 4× A100 and **compare**: sample quality (FID), convergence speed, and representation quality (linear probe).
4. Write up what we learned.

## Roadmap

- [ ] Step 0 — Project setup (this commit)
- [ ] Step 1 — Rectified flow / flow matching baseline (loss, sampler)
- [ ] Step 2 — Minimal DiT backbone
- [ ] Step 3 — Data pipeline + multi-GPU (DDP) training loop
- [ ] Step 4 — Dual-Timestep Scheduling
- [ ] Step 5 — EMA teacher + self-supervised representation loss
- [ ] Step 6 — Evaluation: FID, linear probe
- [ ] Step 7 — Baseline vs Self-Flow results & learnings

## Repo layout

```
src/minselfflow/   # the implementation
scripts/           # training / sampling / eval entry points
tests/             # small sanity tests
```

## Setup

```bash
make install   # uv venv + deps
make test      # sanity tests
make lint      # ruff + ty
```

## Learnings

_Coming as we go._

## References

- Self-Flow: [arXiv:2603.06507](https://arxiv.org/abs/2603.06507)
- Flow Matching: Lipman et al., 2022 · Rectified Flow: Liu et al., 2022
- DiT: Peebles & Xie, 2023 · SiT: Ma et al., 2024
- REPA: Yu et al., 2024
- Inspiration: [cloneofsimo/minRF](https://github.com/cloneofsimo/minRF) and friends

## License

MIT
