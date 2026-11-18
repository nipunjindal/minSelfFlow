# minSelfFlow

Minimal implementation of **Self-Flow** — *Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis* ([arXiv:2603.06507](https://arxiv.org/abs/2603.06507)).

> 🚧 Work in progress, built step by step.

## Idea

Self-Flow learns strong representations inside flow matching without external encoders (unlike REPA):

- **Dual-Timestep Scheduling** — tokens get different noise levels, so the model must infer heavily-noised tokens from cleaner ones.
- **EMA teacher** — self-distillation gives a representation loss alongside the standard flow loss.

## Plan

Train a small baseline flow-matching DiT and a Self-Flow DiT with the same compute, and compare.

## Setup

```bash
make install
make test
```

## License

MIT
