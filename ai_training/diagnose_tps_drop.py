import argparse
import json
import time
import types

import numpy as np

from shogi_env import Shogi3DEnv, Player

try:
    from sb3_contrib import MaskablePPO
except Exception:
    MaskablePPO = None


def pearson(x, y):
    if len(x) < 2:
        return 0.0
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xstd = xa.std()
    ystd = ya.std()
    if xstd <= 1e-12 or ystd <= 1e-12:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def choose_action(env, obs, model, deterministic):
    legal_action_map = env._get_legal_action_map(env.current_player)
    legal_count = len(legal_action_map)
    if legal_count == 0:
        return 0, 0

    if model is None:
        legal_actions = list(legal_action_map.keys())
        idx = int(legal_actions[int(np.random.randint(0, len(legal_actions)))])
        return idx, legal_count

    mask = env.action_masks()
    action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
    return int(action), legal_count


def run_diagnosis(args):
    env = Shogi3DEnv(opening_random_plies=args.opening_random_plies)
    obs, _ = env.reset(seed=args.seed)

    model = None
    if args.model:
        if MaskablePPO is None:
            raise RuntimeError("sb3_contrib is not available but --model was specified")
        model = MaskablePPO.load(args.model, device=args.device)

    timing = {
        "get_legal_moves": [0.0, 0],
        "_is_move_safe": [0.0, 0],
        "_is_drop_safe": [0.0, 0],
        "_is_king_in_check_ray": [0.0, 0],
        "_compute_piece_moves_uncached": [0.0, 0],
    }

    def wrap_method(name):
        fn = getattr(env, name)

        def wrapped(self, *a, **kw):
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                timing[name][0] += dt
                timing[name][1] += 1

        setattr(env, name, types.MethodType(wrapped, env))

    for m in timing.keys():
        if hasattr(env, m):
            wrap_method(m)

    step_times = []
    legal_counts = []
    turn_counts = []
    rep_sizes = []
    hand_counts = []
    empty_counts = []
    cache_sizes = []

    episodes = 0
    t0 = time.perf_counter()

    local_steps = 0
    for _ in range(args.steps):
        action_idx, legal_count = choose_action(env, obs, model, args.deterministic)

        pre_turn = int(env.turn_count)
        pre_rep = len(env.repetition_map)
        pre_hands = len(env.hands[Player.SENTE]) + len(env.hands[Player.GOTE])
        pre_empty = int(np.count_nonzero(env.board == 0))
        pre_cache = len(env.piece_move_cache)

        s0 = time.perf_counter()
        obs, _, term, trunc, _ = env.step(action_idx)
        local_steps += 1
        dt = time.perf_counter() - s0

        step_times.append(dt)
        legal_counts.append(int(legal_count))
        turn_counts.append(pre_turn)
        rep_sizes.append(pre_rep)
        hand_counts.append(int(pre_hands))
        empty_counts.append(pre_empty)
        cache_sizes.append(int(pre_cache))

        force_reset = args.force_reset_interval > 0 and local_steps >= args.force_reset_interval
        if term or trunc or force_reset:
            episodes += 1
            obs, _ = env.reset(seed=args.seed + episodes)
            local_steps = 0

    elapsed = time.perf_counter() - t0

    # 前半・後半の比較
    n = len(step_times)
    mid = max(1, n // 2)
    first_mean = float(np.mean(step_times[:mid]))
    second_mean = float(np.mean(step_times[mid:]))

    # 手数帯別平均 (0-99, 100-199, ...)
    bucket = args.turn_bucket
    bucket_map = {}
    for t, st in zip(turn_counts, step_times):
        b = (t // bucket) * bucket
        bucket_map.setdefault(b, []).append(st)
    bucket_stats = {f"{b}-{b+bucket-1}": float(np.mean(v)) for b, v in sorted(bucket_map.items())}

    result = {
        "steps": n,
        "episodes": episodes,
        "elapsed_seconds": round(elapsed, 3),
        "steps_per_sec": round(n / max(elapsed, 1e-9), 3),
        "mean_step_ms": round(float(np.mean(step_times)) * 1000.0, 3),
        "first_half_step_ms": round(first_mean * 1000.0, 3),
        "second_half_step_ms": round(second_mean * 1000.0, 3),
        "slowdown_ratio_second_over_first": round(second_mean / max(first_mean, 1e-12), 4),
        "corr_step_ms_vs_legal_count": round(pearson(step_times, legal_counts), 4),
        "corr_step_ms_vs_turn_count": round(pearson(step_times, turn_counts), 4),
        "corr_step_ms_vs_repetition_size": round(pearson(step_times, rep_sizes), 4),
        "corr_step_ms_vs_hand_count": round(pearson(step_times, hand_counts), 4),
        "corr_step_ms_vs_empty_count": round(pearson(step_times, empty_counts), 4),
        "corr_step_ms_vs_piece_cache_size": round(pearson(step_times, cache_sizes), 4),
        "turn_bucket_step_ms": {k: round(v * 1000.0, 3) for k, v in bucket_stats.items()},
        "timing_breakdown": {
            k: {
                "calls": int(v[1]),
                "total_ms": round(v[0] * 1000.0, 3),
                "avg_us": round((v[0] / max(v[1], 1)) * 1e6, 3),
            }
            for k, v in timing.items()
        },
    }
    return result


def main():
    p = argparse.ArgumentParser(description="Diagnose why steps/sec drops during training")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--opening-random-plies", type=int, default=2)
    p.add_argument("--turn-bucket", type=int, default=100)
    p.add_argument("--model", type=str, default="")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--force-reset-interval", type=int, default=0, help="Force reset every N steps (0 disables)")
    args = p.parse_args()

    out = run_diagnosis(args)
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
