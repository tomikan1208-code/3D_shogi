# GPU Env Rebuild (Separated Project)

This is a fully separated experimental project for GPU-first environment stepping.
It does not modify or import the existing ai_training env implementation.

## Scope

- Batched 3D board state on torch tensors
- Separated full-rule move system (all major pieces, promotions, drops, hands)
- King-safety filtering (self-check illegal move exclusion)
- End reasons: `ou_capture`, `mate`, `no_legal`, `max_plies`
- Sampling modes for training phases:
	- `fast`: early-exit legal sampling (high throughput, approximate)
	- `exact`: full legal-list sampling (slower, stricter)
- Random self-play benchmark with steps/s and matches/s

Note: current version is correctness-first. State is tensorized on device, but legal
move generation is still Python-loop heavy and not yet fully vectorized.

## Files

- gpu_env.py: GPU batched environment core
- run_benchmark.py: benchmark runner
- START_GPU_ENV_BENCH.bat: double-click benchmark launcher

## Run

```bash
python gpu_env_rebuild/run_benchmark.py --device cuda --batch-size 512 --max-plies 512 --seconds 20
```

Two-phase style (start fast, then switch to exact at a chosen step):

```bash
python gpu_env_rebuild/run_benchmark.py --device cuda --force-cuda --batch-size 256 --seconds 20 --sampling-mode fast --switch-after-steps 12
```

Exact-only mode:

```bash
python gpu_env_rebuild/run_benchmark.py --device cuda --force-cuda --batch-size 64 --seconds 10 --sampling-mode exact
```

If CUDA is not available, it will fail fast when `--force-cuda` is set.
