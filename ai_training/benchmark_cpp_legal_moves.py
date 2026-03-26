import argparse
import copy
import os
import time
from statistics import mean, median

import numpy as np

from shogi_env import Shogi3DEnv


def _sample_snapshots(count: int, seed: int):
    rng = np.random.default_rng(seed)
    env = Shogi3DEnv(opening_random_plies=0)
    env.use_legal_moves_cache = False
    env.cpp_strict_filter = False

    snapshots = []
    env.reset(seed=seed)
    safety = 0
    while len(snapshots) < count and safety < count * 200:
        safety += 1
        legal_map = env._get_legal_action_map(env.current_player)
        if not legal_map:
            env.reset()
            continue
        action_keys = list(legal_map.keys())
        action = int(action_keys[int(rng.integers(0, len(action_keys)))])
        _, _, terminated, truncated, _ = env.step(action)
        snapshots.append(copy.deepcopy(env))
        if terminated or truncated:
            env.reset()
    return snapshots


def _bench_one_config(
    snapshots,
    use_cpp_legal_moves: bool,
    use_cpp_pinned: bool,
    strict_filter: bool,
    cpp_filter_level: int,
):
    lat_ms = []
    move_counts = []

    for snap in snapshots:
        env = copy.deepcopy(snap)
        env.use_cpp_legal_moves = bool(use_cpp_legal_moves)
        env.use_cpp_pinned = bool(use_cpp_pinned)
        env.cpp_strict_filter = bool(strict_filter)
        env.cpp_filter_level = int(cpp_filter_level)
        env.use_legal_moves_cache = False

        t0 = time.perf_counter()
        moves = env.get_legal_moves(env.current_player)
        t1 = time.perf_counter()

        lat_ms.append((t1 - t0) * 1000.0)
        move_counts.append(len(moves))

    return {
        "avg_ms": float(mean(lat_ms)) if lat_ms else 0.0,
        "p50_ms": float(median(lat_ms)) if lat_ms else 0.0,
        "max_ms": float(max(lat_ms)) if lat_ms else 0.0,
        "moves_avg": float(mean(move_counts)) if move_counts else 0.0,
        "moves_total": int(sum(move_counts)),
        "calls": int(len(lat_ms)),
        "moves_each": move_counts,
    }


def _fmt(label: str, r: dict):
    return (
        f"{label:<24} calls={r['calls']:>4} avg={r['avg_ms']:>7.3f}ms "
        f"p50={r['p50_ms']:>7.3f}ms max={r['max_ms']:>7.3f}ms "
        f"moves_avg={r['moves_avg']:>8.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark legal move generation (C++ path / pinned helper on-off)")
    parser.add_argument("--samples", type=int, default=80, help="number of sampled positions")
    parser.add_argument("--seed", type=int, default=20260324, help="rng seed")
    parser.add_argument("--strict", action="store_true", help="enable SHOGI_CPP_STRICT_FILTER during cpp runs")
    args = parser.parse_args()

    os.environ.setdefault("SHOGI_FIXED_ACTION_SPACE", "1")
    os.environ.setdefault("SHOGI_USE_CPP_LEGAL_MOVES", "1")
    os.environ.setdefault("SHOGI_USE_CPP_KINGCHECK", "1")
    os.environ.setdefault("SHOGI_CPP_STRICT_FILTER", "0")
    os.environ.setdefault("SHOGI_CPP_FILTER_LEVEL", "0")
    os.environ.setdefault("SHOGI_USE_LEGAL_MOVES_CACHE", "0")

    snapshots = _sample_snapshots(args.samples, args.seed)
    if not snapshots:
        print("[bench] no snapshots sampled")
        return

    print(f"[bench] sampled positions: {len(snapshots)}")

    py_ref = _bench_one_config(
        snapshots,
        use_cpp_legal_moves=False,
        use_cpp_pinned=False,
        strict_filter=False,
        cpp_filter_level=0,
    )
    cpp_l0_no_pin = _bench_one_config(
        snapshots,
        use_cpp_legal_moves=True,
        use_cpp_pinned=False,
        strict_filter=args.strict,
        cpp_filter_level=0,
    )
    cpp_l0_with_pin = _bench_one_config(
        snapshots,
        use_cpp_legal_moves=True,
        use_cpp_pinned=True,
        strict_filter=args.strict,
        cpp_filter_level=0,
    )
    cpp_l1 = _bench_one_config(
        snapshots,
        use_cpp_legal_moves=True,
        use_cpp_pinned=True,
        strict_filter=args.strict,
        cpp_filter_level=1,
    )
    cpp_l2 = _bench_one_config(
        snapshots,
        use_cpp_legal_moves=True,
        use_cpp_pinned=True,
        strict_filter=args.strict,
        cpp_filter_level=2,
    )

    print(_fmt("python_legal_moves", py_ref))
    print(_fmt("cpp_l0_no_cpp_pin", cpp_l0_no_pin))
    print(_fmt("cpp_l0_with_cpp_pin", cpp_l0_with_pin))
    print(_fmt("cpp_l1_structural_only", cpp_l1))
    print(_fmt("cpp_l2_trust_cpp", cpp_l2))

    if cpp_l0_no_pin["avg_ms"] > 0:
        speedup = cpp_l0_no_pin["avg_ms"] / max(cpp_l0_with_pin["avg_ms"], 1e-9)
        print(f"[bench] cpp pin helper speedup (l0 with-pin vs no-pin): x{speedup:.3f}")
    if cpp_l0_with_pin["avg_ms"] > 0:
        speedup_l1 = cpp_l0_with_pin["avg_ms"] / max(cpp_l1["avg_ms"], 1e-9)
        speedup_l2 = cpp_l0_with_pin["avg_ms"] / max(cpp_l2["avg_ms"], 1e-9)
        print(f"[bench] staged speedup (l0 -> l1): x{speedup_l1:.3f}")
        print(f"[bench] staged speedup (l0 -> l2): x{speedup_l2:.3f}")

    if cpp_l2["avg_ms"] > 0:
        speedup_py = py_ref["avg_ms"] / max(cpp_l2["avg_ms"], 1e-9)
        print(f"[bench] end-to-end speedup (python -> cpp l2): x{speedup_py:.3f}")

    # Consistency check (move count only): l0_with_pin を基準に差分件数を表示。
    def _count_mismatch(ref, target):
        mism = 0
        for a, b in zip(ref["moves_each"], target["moves_each"]):
            if a != b:
                mism += 1
        return mism

    mm_no_pin = _count_mismatch(cpp_l0_with_pin, cpp_l0_no_pin)
    mm_l1 = _count_mismatch(cpp_l0_with_pin, cpp_l1)
    mm_l2 = _count_mismatch(cpp_l0_with_pin, cpp_l2)
    total = len(cpp_l0_with_pin["moves_each"])
    print(f"[bench] move-count mismatch (l0_with_pin vs l0_no_pin): {mm_no_pin}/{total}")
    print(f"[bench] move-count mismatch (l0_with_pin vs l1): {mm_l1}/{total}")
    print(f"[bench] move-count mismatch (l0_with_pin vs l2): {mm_l2}/{total}")


if __name__ == "__main__":
    main()
