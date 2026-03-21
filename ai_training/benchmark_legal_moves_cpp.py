import argparse
import os
import time
from collections import Counter

import numpy as np

from shogi_env import Shogi3DEnv


def _capture_state(env: Shogi3DEnv):
    return {
        "board": env.board.copy(),
        "owner": env.owner.copy(),
        "hands": {
            int(k): dict(v)
            for k, v in env.hands.items()
        },
        "current_player": int(env.current_player),
    }


def _restore_state(env: Shogi3DEnv, st):
    env.board[:, :, :] = st["board"]
    env.owner[:, :, :] = st["owner"]
    env.hands = {
        env.current_player.__class__.SENTE: Counter(st["hands"].get(1, {})),
        env.current_player.__class__.GOTE: Counter(st["hands"].get(2, {})),
    }
    env.current_player = env.current_player.__class__(st["current_player"])
    env._rebuild_all_caches()
    env.position_version += 1
    env.legal_moves_cache.clear()
    env.legal_action_cache.clear()


def collect_fixed_positions(count: int, opening_random_plies: int, seed: int):
    os.environ["SHOGI_USE_CPP_LEGAL_MOVES"] = "0"
    os.environ["SHOGI_USE_CPP_KINGCHECK"] = "0"

    env = Shogi3DEnv(opening_random_plies=opening_random_plies)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    states = []
    while len(states) < count:
        states.append(_capture_state(env))
        legal = env.get_legal_moves(env.current_player)
        if len(legal) == 0:
            env.reset(seed=seed + len(states))
            continue
        legal_actions = list(env._get_legal_action_map(env.current_player).keys())
        idx = int(legal_actions[int(rng.integers(0, len(legal_actions)))])
        _, _, term, trunc, _ = env.step(idx)
        if term or trunc:
            env.reset(seed=seed + len(states))
    return states


def run_once(seconds: float, use_cpp: bool, opening_random_plies: int, seed: int):
    os.environ["SHOGI_USE_CPP_LEGAL_MOVES"] = "1" if use_cpp else "0"
    os.environ["SHOGI_USE_CPP_KINGCHECK"] = "1" if use_cpp else "0"

    env = Shogi3DEnv(opening_random_plies=opening_random_plies)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    deadline = t0 + seconds

    calls = 0
    total_moves = 0
    steps = 0

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break

        legal = env.get_legal_moves(env.current_player)
        calls += 1
        total_moves += len(legal)

        if len(legal) == 0:
            env.reset()
            continue

        # Keep state evolving so benchmark includes realistic variety.
        legal_actions = list(env._get_legal_action_map(env.current_player).keys())
        idx = int(legal_actions[int(rng.integers(0, len(legal_actions)))])
        _, _, term, trunc, _ = env.step(idx)
        steps += 1
        if term or trunc:
            env.reset()

    elapsed = time.perf_counter() - t0
    cache_stats = env.get_legal_cache_stats()
    return {
        "mode": "cpp" if use_cpp else "python",
        "bench": "rollout",
        "seconds": round(elapsed, 3),
        "calls": int(calls),
        "calls_per_sec": round(calls / max(elapsed, 1e-9), 2),
        "steps": int(steps),
        "steps_per_sec": round(steps / max(elapsed, 1e-9), 2),
        "avg_legal_moves": round(total_moves / max(calls, 1), 2),
        "cache_hit_rate": round(cache_stats["hit_rate"], 4),
    }


def run_fixed_positions(seconds: float, use_cpp: bool, positions):
    os.environ["SHOGI_USE_CPP_LEGAL_MOVES"] = "1" if use_cpp else "0"
    os.environ["SHOGI_USE_CPP_KINGCHECK"] = "1" if use_cpp else "0"

    env = Shogi3DEnv(opening_random_plies=0)
    env.reset(seed=0)

    t0 = time.perf_counter()
    calls = 0
    total_moves = 0
    i = 0
    gen_elapsed = 0.0

    while True:
        if gen_elapsed >= seconds:
            break
        st = positions[i % len(positions)]
        i += 1
        _restore_state(env, st)
        g0 = time.perf_counter()
        legal = env.get_legal_moves(env.current_player)
        gen_elapsed += (time.perf_counter() - g0)
        calls += 1
        total_moves += len(legal)

    elapsed = time.perf_counter() - t0
    cache_stats = env.get_legal_cache_stats()
    return {
        "mode": "cpp" if use_cpp else "python",
        "bench": "fixed_positions",
        "seconds": round(elapsed, 3),
        "gen_seconds": round(gen_elapsed, 3),
        "calls": int(calls),
        "calls_per_sec": round(calls / max(gen_elapsed, 1e-9), 2),
        "avg_legal_moves": round(total_moves / max(calls, 1), 2),
        "cache_hit_rate": round(cache_stats["hit_rate"], 4),
    }


def main():
    p = argparse.ArgumentParser(description="60s legal move generation benchmark: Python vs C++")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--opening-random-plies", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-positions", type=int, default=0, help="Use shared fixed positions count for fair comparison")
    args = p.parse_args()

    if args.fixed_positions > 0:
        positions = collect_fixed_positions(args.fixed_positions, args.opening_random_plies, args.seed)
        py_res = run_fixed_positions(args.seconds, use_cpp=False, positions=positions)
        cpp_res = run_fixed_positions(args.seconds, use_cpp=True, positions=positions)
    else:
        py_res = run_once(args.seconds, use_cpp=False, opening_random_plies=args.opening_random_plies, seed=args.seed)
        cpp_res = run_once(args.seconds, use_cpp=True, opening_random_plies=args.opening_random_plies, seed=args.seed)

    speedup = cpp_res["calls_per_sec"] / max(py_res["calls_per_sec"], 1e-9)

    print("=== Legal Move Benchmark ===")
    print(py_res)
    print(cpp_res)
    print({"calls_per_sec_speedup_cpp_vs_py": round(speedup, 3)})


if __name__ == "__main__":
    main()
