# minSelfFlow

Minimal implementation of the Self-Flow paper (arXiv 2603.06507), built on a uv/ruff/ty/pre-commit template.

## Quick Reference

```bash
make install        # Create venv + install deps (including dev)
make test           # Run pytest
make lint           # Format + lint + type-check (ruff, shfmt, toml-sort, ty)
make type-check     # ty check only
make pre-commit-run # Run all pre-commit hooks
make clean          # Remove .venv and caches
make rename NEW=x   # (template) rename package
```

## Project Layout

- `src/minselfflow/` — Main package
  - `flow.py` — flow matching: per-token noising, dual-timestep sampling, loss, Euler sampler
  - `model.py` — DiT with per-token timestep conditioning, patchify/unpatchify
  - `selfflow.py` — Self-Flow: student w/ projection head, EMA-teacher representation loss
  - `train.py` — DDP training (`--method flow|dual|selfflow`), outputs in `runs/<name>/`
  - `data.py` — CIFAR-10 from Hugging Face, cached as uint8 tensors in `data/`
  - `evaluate.py` — per-checkpoint FID-10k, vanilla val loss, per-block linear probe → `runs/<name>/eval.jsonl`
- `tests/` — Pytest test files
- `scripts/` — Standalone scripts (`toy_2d.py`: 2D flow matching demo, `plot_runs.py`: curves/eval/sample plots → `assets/`)
- `assets/` — plots referenced from the README
- `notebooks/minselfflow.ipynb` — step-by-step learning notebook (run from `notebooks/`)

Conventions: tokens are `(B, N, C)`, per-token timesteps `tau` are `(B, N)`; t=0 is data, t=1 is noise.

## Tooling

- **Package manager**: uv (not pip/poetry)
- **Linter/formatter**: ruff (line-length 120, target py310)
- **Type checker**: ty
- **Tests**: pytest (testpaths=tests, pythonpath=src)
- **Pre-commit**: ruff-format, ruff lint, shfmt, toml-sort

## Agent Team

Agent definitions live in `.claude/agents/` (orchestrator, implementer, reviewer). When asked to "pick up issue(s)" or "start polling for issues", follow the orchestrator workflow directly in the main conversation — do not spawn it as a sub-agent.

## Conventions

- Run `make lint` before committing — pre-commit hooks enforce this
- All source goes under `src/minselfflow/`; tests mirror the source structure in `tests/`
- Python >=3.10 required
- Keep this file updated as the project evolves — new deps, conventions, or structural changes should be reflected here

## Coding Standards

- Functional core, imperative shell — FP patterns internally, straightforward public APIs
- Code duplication is often better than the wrong abstraction
- Errors-as-types / errors-as-return-values over try-except; nested try-except is prohibited
- Composition over inheritance
- Early returns and sensible defaults over if-else
- Prefer match-case over if-elif-else
- Liberal use of asserts to affirm expected state
- Global import statements only
- Modern Python 3.10+ type annotations (`list[int]` not `List[int]`)
- Prefer Literals over string enumerations
- Products are `@dataclass` (preferably frozen), sums are `|`, eliminate with `match`
- Protocols and ABCs for decoupling and type lambdas
- Docstrings with argument types, return types, and exceptions; include tensor shapes if applicable
- Liberal logging for observability; never use f-strings in log calls — use structured logging or logger formatting
- Be judicious with comments — skip the obvious, preserve the complex
