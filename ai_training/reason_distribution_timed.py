import argparse
import json
import time
from collections import Counter

import torch
from sb3_contrib import MaskablePPO

from shogi_env import Shogi3DEnv


def run_timed_distribution(
    model_path: str,
    budget_seconds: float,
    opening_random_plies: int,
    deterministic: bool,
    seed: int,
    per_episode_step_cap: int,
    device: str,
):
    setup_start = time.perf_counter()

    env = Shogi3DEnv(opening_random_plies=opening_random_plies)
    model = MaskablePPO.load(model_path, device=device)

    start = time.perf_counter()
    deadline = start + budget_seconds
    setup_seconds = start - setup_start

    reasons = Counter()
    turns = []
    episodes_completed = 0
    episodes_started = 0
    steps_total = 0

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break

        obs, _ = env.reset(seed=seed + episodes_started)
        episodes_started += 1
        step_count = 0

        while True:
            now = time.perf_counter()
            if now >= deadline:
                reasons["time_budget_hit_mid_game"] += 1
                elapsed = now - start
                return {
                    "budget_seconds": round(budget_seconds, 3),
                    "setup_seconds": round(setup_seconds, 3),
                    "elapsed_seconds": round(elapsed, 3),
                    "episodes_started": episodes_started,
                    "episodes_completed": episodes_completed,
                    "steps_total": steps_total,
                    "avg_turns_completed": round(sum(turns) / len(turns), 3) if turns else 0.0,
                    "reason_counts": dict(reasons),
                    "reason_ratio_completed_only": {
                        k: round(v / episodes_completed, 6) if episodes_completed > 0 else 0.0
                        for k, v in sorted(reasons.items())
                        if k != "time_budget_hit_mid_game"
                    },
                }

            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(int(action))
            step_count += 1
            steps_total += 1

            if step_count >= per_episode_step_cap and not (terminated or truncated):
                terminated = True
                truncated = False
                info = {"reason": "episode_step_cap"}

            if terminated or truncated:
                reason = info.get("reason", "unknown")
                reasons[reason] += 1
                turns.append(step_count)
                episodes_completed += 1
                break

    elapsed = time.perf_counter() - start
    return {
        "budget_seconds": round(budget_seconds, 3),
        "setup_seconds": round(setup_seconds, 3),
        "elapsed_seconds": round(elapsed, 3),
        "episodes_started": episodes_started,
        "episodes_completed": episodes_completed,
        "steps_total": steps_total,
        "avg_turns_completed": round(sum(turns) / len(turns), 3) if turns else 0.0,
        "reason_counts": dict(reasons),
        "reason_ratio_completed_only": {
            k: round(v / episodes_completed, 6) if episodes_completed > 0 else 0.0
            for k, v in sorted(reasons.items())
            if k != "time_budget_hit_mid_game"
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Timed reason distribution for self-play")
    parser.add_argument("--model", default="./models/shogi3d_ppo_latest.zip", help="Model path")
    parser.add_argument("--seconds", type=float, default=600.0, help="Time budget in seconds")
    parser.add_argument("--opening-random-plies", type=int, default=2, help="Opening random plies")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed")
    parser.add_argument("--per-episode-step-cap", type=int, default=800, help="Force-finish an episode after this many steps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Model inference device")
    parser.add_argument("--force-cuda", action="store_true", help="Fail if CUDA is unavailable")
    args = parser.parse_args()

    if args.force_cuda and not torch.cuda.is_available():
        raise RuntimeError("--force-cuda 指定ですが CUDA が利用できません。")

    summary = run_timed_distribution(
        model_path=args.model,
        budget_seconds=args.seconds,
        opening_random_plies=args.opening_random_plies,
        deterministic=args.deterministic,
        seed=args.seed,
        per_episode_step_cap=args.per_episode_step_cap,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
