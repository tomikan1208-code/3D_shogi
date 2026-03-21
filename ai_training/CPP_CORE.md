# C++ Core Integration (Python Fallback)

This adds an optional C++ backend for king-in-check ray evaluation and legal move generation.
If C++ module is unavailable, Python implementation is used automatically.

## What is included

- cpp_core/shogi3d_core.cpp
  - is_king_in_check_ray_cpp(board, owner, player)
  - generate_legal_moves_cpp(board, owner, hand_counts, player)
  - Includes KEI (knight) check handling.
- setup_cpp_core.py
  - Builds the pybind11 extension module named shogi3d_core.
- cpp_kingcheck.py
  - Runtime wrapper with safe fallback.

## Build

From ai_training folder:

python -m pip install pybind11 setuptools wheel
python setup_cpp_core.py build_ext --inplace

## Enable in env

Set environment variable:

SHOGI_USE_CPP_KINGCHECK=1
SHOGI_USE_CPP_LEGAL_MOVES=1

If module load fails, env falls back to Python logic.

## 60s speed comparison

From ai_training folder:

python benchmark_legal_moves_cpp.py --seconds 60 --opening-random-plies 2

## Legal move equivalence validation

From ai_training folder:

python validate_legal_moves_equivalence.py --positions 2000 --opening-random-plies 2

The script compares Python and C++ legal-move outputs per position and prints
the first mismatch with move-level diffs.

## Strategy note

- Caching only initial position is not enough for speed in middle/end game.
- Better approach is incremental update from move source/target and affected rays.
- KEI handling must stay separate because it is not a slider.
- Python-side ray-watch cache is a good place to start because update set is small and easy to maintain.
