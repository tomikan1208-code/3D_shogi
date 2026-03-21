import argparse
import random
from collections import Counter

import numpy as np

from shogi_env import Shogi3DEnv


def _to_counter(moves):
    # Keep multiplicity to detect duplicate-related mismatches too.
    return Counter(tuple(int(v) for v in m) for m in moves)


def _compute_moves(env, player, use_cpp):
    env.use_cpp_legal_moves = bool(use_cpp)
    env.use_cpp_kingcheck = bool(use_cpp)
    return env.get_legal_moves(player)


def validate(positions, seed, opening_random_plies, max_steps_per_game, max_diff_print, rebuild_caches_before_compare):
    random.seed(seed)
    np.random.seed(seed)

    env = Shogi3DEnv(opening_random_plies=opening_random_plies)
    env.reset(seed=seed)

    checked = 0
    mismatch = 0
    games = 0

    while checked < positions:
        player = env.current_player

        if rebuild_caches_before_compare and hasattr(env, "_rebuild_all_caches"):
            env._rebuild_all_caches()

        py_moves = _compute_moves(env, player, use_cpp=False)
        cpp_moves = _compute_moves(env, player, use_cpp=True)

        py_counter = _to_counter(py_moves)
        cpp_counter = _to_counter(cpp_moves)

        if py_counter != cpp_counter:
            mismatch += 1
            print("[MISMATCH] position=", checked)
            print("  player=", int(player))
            print("  py_count=", len(py_moves), "cpp_count=", len(cpp_moves))
            print("  python_in_check=", bool(env._is_king_in_check(player)))
            print("  python_in_check_ray=", bool(env._is_king_in_check_ray(player)))

            only_py = list((py_counter - cpp_counter).elements())
            only_cpp = list((cpp_counter - py_counter).elements())

            if only_py:
                print("  only_python(sample)=")
                for m in only_py[:max_diff_print]:
                    print("   ", m)
            if only_cpp:
                print("  only_cpp(sample)=")
                for m in only_cpp[:max_diff_print]:
                    print("   ", m)

            # Print source-square piece info for first few only_cpp moves.
            if only_cpp:
                print("  only_cpp_sources(sample)=")
                seen = set()
                for m in only_cpp:
                    fx, fy, fz = m[0], m[1], m[2]
                    key = (fx, fy, fz)
                    if key in seen:
                        continue
                    seen.add(key)
                    piece = int(env.board[fx, fy, fz])
                    owner = int(env.owner[fx, fy, fz])
                    print("   ", key, "piece=", piece, "owner=", owner)
                    if len(seen) >= max_diff_print:
                        break

            # Fail fast to make debugging quicker.
            return {
                "ok": False,
                "checked_positions": checked + 1,
                "mismatch_count": mismatch,
                "games": games,
            }

        checked += 1

        if len(py_moves) == 0:
            games += 1
            env.reset()
            continue

        legal_actions = list(env._get_legal_action_map(env.current_player).keys())
        idx = random.choice(legal_actions)
        _, _, term, trunc, _ = env.step(idx)

        if term or trunc or env.turn_count >= max_steps_per_game:
            games += 1
            env.reset()

    return {
        "ok": True,
        "checked_positions": checked,
        "mismatch_count": mismatch,
        "games": games,
    }


def main():
    p = argparse.ArgumentParser(description="Validate legal move equivalence between Python and C++ backends")
    p.add_argument("--positions", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opening-random-plies", type=int, default=2)
    p.add_argument("--max-steps-per-game", type=int, default=1200)
    p.add_argument("--max-diff-print", type=int, default=15)
    p.add_argument("--rebuild-caches-before-compare", action="store_true")
    args = p.parse_args()

    result = validate(
        positions=args.positions,
        seed=args.seed,
        opening_random_plies=args.opening_random_plies,
        max_steps_per_game=args.max_steps_per_game,
        max_diff_print=args.max_diff_print,
        rebuild_caches_before_compare=args.rebuild_caches_before_compare,
    )

    print(result)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
