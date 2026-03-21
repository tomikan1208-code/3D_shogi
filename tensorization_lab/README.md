# Tensorization Lab

Separate prototype project for tensorizing 3D board operations.

## What this includes

- `tensorized_ops.py`
  - Batched tensor state generation
  - Vectorized FU legal-move mask (SENTE/GOTE)
  - Vectorized king-adjacent attack map via `conv3d`
- `benchmark.py`
  - Compares tensorized ops vs NumPy loop baseline
- `gpu_match_runner.py`
  - Runs batched matches on GPU/CPU (prototype rule set: FU + OU)
- `gpu_full_match_runner.py`
  - Runs batched matches from standard initial position with all on-board piece moves

## Run

```bash
python tensorization_lab/benchmark.py
```

Run batched matches on GPU:

```bash
python tensorization_lab/gpu_match_runner.py --device cuda --batch 512 --max-plies 512
```

Fail fast when GPU is not available:

```bash
python tensorization_lab/gpu_match_runner.py --device cuda --batch 512 --max-plies 512
```

Run full-piece batched matches on GPU:

```bash
python tensorization_lab/gpu_full_match_runner.py --device cuda --batch 128 --max-plies 512
```

If CUDA is unavailable:

```bash
python tensorization_lab/gpu_match_runner.py --device cpu --batch 256 --max-plies 256
```

## Notes

- This is intentionally isolated from the main training pipeline.
- It is a minimal proof-of-concept to validate speed and shape conventions before integrating deeper move generation tensorization.
- Match runner currently uses a simplified rule set (FU + OU movement/capture) for GPU pipeline validation.
- Full-piece runner supports board-piece movement and promotion candidates, but does not yet implement drops/hands or self-check filtering.
