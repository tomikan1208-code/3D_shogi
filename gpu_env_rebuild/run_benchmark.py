import argparse
import json
import time

import torch

from gpu_env import GPUShogiEnv


def run(seconds, batch_size, max_plies, device, force_cuda, seed, sampling_mode, switch_after_steps):
    if force_cuda and not torch.cuda.is_available():
        raise RuntimeError("--force-cuda specified but CUDA is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda specified but CUDA is unavailable")

    env = GPUShogiEnv(
        batch_size=batch_size,
        device=device,
        max_plies=max_plies,
        seed=seed,
        sampling_mode=sampling_mode,
    )

    t0 = time.perf_counter()
    deadline = t0 + seconds
    steps = 0

    while True:
        now = time.perf_counter()
        if now >= deadline:
            break

        actions = env.sample_actions()
        env.step(actions)
        steps += 1

        if switch_after_steps >= 0 and steps == switch_after_steps:
            env.set_sampling_mode(GPUShogiEnv.SAMPLING_EXACT)

        if torch.all(env.done):
            break

    elapsed = time.perf_counter() - t0
    finished = int(torch.sum(env.done).item())
    reason_hist = env.reason_hist()
    avg_plies = float(env.plies.float().mean().item())

    out = {
        "device": str(device),
        "seconds": round(elapsed, 3),
        "batch_size": int(batch_size),
        "sampling_mode_start": str(sampling_mode),
        "sampling_mode_end": str(env.sampling_mode),
        "switch_after_steps": int(switch_after_steps),
        "env_steps": int(steps),
        "env_steps_per_sec": round(steps / max(elapsed, 1e-9), 2),
        "finished_matches": finished,
        "matches_per_sec": round(finished / max(elapsed, 1e-9), 2),
        "avg_plies": round(avg_plies, 3),
        "reasons": reason_hist,
    }
    print(json.dumps(out, ensure_ascii=True, indent=2))


def main():
    parser = argparse.ArgumentParser(description="GPU-first environment benchmark (separated)")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-mode", choices=["fast", "exact"], default="fast")
    parser.add_argument("--switch-after-steps", type=int, default=-1)
    args = parser.parse_args()

    run(
        seconds=args.seconds,
        batch_size=args.batch_size,
        max_plies=args.max_plies,
        device=args.device,
        force_cuda=args.force_cuda,
        seed=args.seed,
        sampling_mode=args.sampling_mode,
        switch_after_steps=args.switch_after_steps,
    )


if __name__ == "__main__":
    main()
