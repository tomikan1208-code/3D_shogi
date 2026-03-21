import argparse
import concurrent.futures
import json
import os
import sys
import time

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from shogi_env import Shogi3DEnv, Player


_WORKER_MODEL = None


REASON_LABEL_JP = {
    "mate": "詰み",
    "illegal": "不正手",
    "ou_capture": "王捕獲",
    "sennichite": "千日手",
    "no_legal_moves": "合法手なし",
    "timeout": "時間切れ",
    "unknown": "不明",
}


def choose_random_action(env):
    legal_action_map = env._get_legal_action_map(env.current_player)
    if not legal_action_map:
        return 0
    legal_actions = list(legal_action_map.keys())
    return int(legal_actions[int(np.random.randint(0, len(legal_actions)))])


def choose_model_action(env, model, obs, deterministic):
    masks = env.action_masks()
    model_action_n = int(getattr(getattr(model, "action_space", None), "n", -1))
    if model_action_n > 0 and int(masks.shape[0]) != model_action_n:
        raise RuntimeError(
            f"Action-space mismatch: model expects {model_action_n}, env mask has {int(masks.shape[0])}. "
            "Set SHOGI_FIXED_ACTION_SPACE to match the model."
        )
    action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
    return int(action)


def winner_from_terminal(last_actor, reward, terminated, truncated):
    if truncated:
        return "draw"
    if not terminated:
        return "unknown"
    if reward > 0:
        return last_actor
    if reward < 0:
        return Player.GOTE if last_actor == Player.SENTE else Player.SENTE
    return "unknown"


def _build_env_for_model(model, opening_random_plies):
    model_action_n = int(getattr(getattr(model, "action_space", None), "n", 0))
    if model_action_n <= 0:
        env = Shogi3DEnv(opening_random_plies=opening_random_plies)
        return env, int(getattr(env.action_space, "n", 0))

    # fixed-action model has large discrete action-space (e.g. 238383),
    # legacy legal-index model uses max_legal_moves (1500).
    desired_fixed = 1 if model_action_n > 1500 else 0
    os.environ["SHOGI_FIXED_ACTION_SPACE"] = str(desired_fixed)
    env = Shogi3DEnv(opening_random_plies=opening_random_plies)
    env_action_n = int(getattr(env.action_space, "n", 0))
    if env_action_n != model_action_n:
        raise RuntimeError(
            f"Failed to align env action space with model: model={model_action_n}, env={env_action_n}. "
            "Check SHOGI_FIXED_ACTION_SPACE and model compatibility."
        )
    return env, model_action_n


def run_game(model, model_side, seed, opening_random_plies, deterministic, live_context=None):
    env, _ = _build_env_for_model(model, opening_random_plies)
    obs, _ = env.reset(seed=seed)

    last_actor = None
    last_reward = 0.0
    last_info = {}
    terminated = False
    truncated = False

    while not (terminated or truncated):
        last_actor = env.current_player
        if env.current_player == model_side:
            action = choose_model_action(env, model, obs, deterministic)
        else:
            action = choose_random_action(env)

        obs, reward, terminated, truncated, info = env.step(action)
        last_reward = float(reward)
        last_info = info if isinstance(info, dict) else {}

        if live_context is not None:
            now = time.time()
            if now - live_context["last_live_print_ts"] >= 0.2:
                current_turn = int(env.turn_count)
                elapsed = now - live_context["start_time"]
                steps_done = int(live_context["total_turns_before"] + current_turn)
                tps_total = steps_done / max(elapsed, 1e-6)
                sys.stdout.write(
                    f"\r  [進行中] 現在対局手数: {current_turn:4d} | 手数/s(合計): {tps_total:6.2f} | 1環境換算: {tps_total:6.2f}"
                )
                sys.stdout.flush()
                live_context["last_live_print_ts"] = now
            periodic_sec = float(live_context.get("periodic_log_interval_sec", 3.0))
            if periodic_sec > 0 and (now - live_context["last_periodic_log_ts"]) >= periodic_sec:
                current_turn = int(env.turn_count)
                elapsed = now - live_context["start_time"]
                steps_done = int(live_context["total_turns_before"] + current_turn)
                tps_total = steps_done / max(elapsed, 1e-6)
                print(
                    f"\n  [進行ログ] 現在対局手数: {current_turn:4d} | 手数/s(合計): {tps_total:6.2f} | 1環境換算: {tps_total:6.2f}"
                )
                live_context["last_periodic_log_ts"] = now

    winner = winner_from_terminal(last_actor, last_reward, terminated, truncated)
    reason = last_info.get("reason", "unknown")
    if truncated and reason == "unknown":
        reason = "timeout"

    if winner == model_side:
        model_result = "win"
        winner_role = "ai"
    elif winner == "draw":
        model_result = "draw"
        winner_role = "draw"
    elif winner in (Player.SENTE, Player.GOTE):
        model_result = "loss"
        winner_role = "random"
    else:
        model_result = "unknown"
        winner_role = "unknown"

    return {
        "model_side": int(model_side),
        "result": model_result,
        "reason": reason,
        "turn_count": int(env.turn_count),
        "winner": winner,
        "winner_role": winner_role,
    }


def _resolve_model_path(model_path):
    if os.path.exists(model_path):
        return model_path
    probes = [
        model_path,
        "./ai_training/models/shogi3d_ppo_latest.zip",
        "./models/shogi3d_ppo_latest.zip",
    ]
    resolved = next((p for p in probes if p and os.path.exists(p)), None)
    if resolved is None:
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Resolved model path: {resolved}")
    return resolved


def _load_model_for_inference(model_path, device):
    # Load-time override: evaluation does not need large rollout buffers.
    # Without this, a training checkpoint may allocate huge (n_steps, n_envs, action_n) masks.
    custom_objects = {
        "n_steps": 1,
        "n_envs": 1,
        "batch_size": 1,
    }
    return MaskablePPO.load(model_path, device=device, custom_objects=custom_objects)


def _init_eval_worker(model_path, device):
    global _WORKER_MODEL
    _WORKER_MODEL = _load_model_for_inference(model_path, device)


def _evaluate_one_game(game_index, opening_random_plies, deterministic, seed):
    global _WORKER_MODEL
    model = _WORKER_MODEL
    if model is None:
        raise RuntimeError("Worker model is not initialized")
    model_side = Player.SENTE if (game_index % 2 == 0) else Player.GOTE
    game = run_game(
        model=model,
        model_side=model_side,
        seed=seed + game_index,
        opening_random_plies=opening_random_plies,
        deterministic=deterministic,
        live_context=None,
    )
    game["game_index"] = int(game_index)
    return game


def evaluate(model_path, games, opening_random_plies, deterministic, seed, device, workers):
    model_path = _resolve_model_path(model_path)
    print(f"[EVAL] model path resolved: {model_path}", flush=True)
    print(f"[EVAL] loading model on device={device} ...", flush=True)

    model = _load_model_for_inference(model_path, device)
    print("[EVAL] model loaded", flush=True)
    model_action_n = int(getattr(getattr(model, "action_space", None), "n", 0))
    if model_action_n > 0:
        fixed_flag = 1 if model_action_n > 1500 else 0
        print(f"Detected model action space: {model_action_n} (SHOGI_FIXED_ACTION_SPACE={fixed_flag})", flush=True)
    del model

    summary = {
        "games": games,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "unknown": 0,
        "avg_turns": 0.0,
        "reasons": {},
        "by_side": {
            "sente": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
            "gote": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        },
    }

    total_turns = 0
    workers = max(1, int(workers))
    workers = min(workers, max(games, 1))
    print(f"Parallel workers: {workers}", flush=True)

    game_results = []
    if workers == 1:
        model = _load_model_for_inference(model_path, device)
        print("[EVAL] single-worker loop start", flush=True)
        start_time = time.time()
        total_turns_before = 0
        last_live_print_ts = 0.0
        last_periodic_log_ts = 0.0
        periodic_log_interval_sec = float(os.getenv("SHOGI_EVAL_LOG_INTERVAL_SEC", "3"))
        for i in range(games):
            model_side = Player.SENTE if (i % 2 == 0) else Player.GOTE
            game = run_game(
                model=model,
                model_side=model_side,
                seed=seed + i,
                opening_random_plies=opening_random_plies,
                deterministic=deterministic,
                live_context={
                    "start_time": start_time,
                    "total_turns_before": total_turns_before,
                    "last_live_print_ts": last_live_print_ts,
                    "last_periodic_log_ts": last_periodic_log_ts,
                    "periodic_log_interval_sec": periodic_log_interval_sec,
                },
            )
            last_live_print_ts = time.time()
            last_periodic_log_ts = last_live_print_ts
            game["game_index"] = int(i)
            game_results.append(game)
            total_turns_before += int(game.get("turn_count", 0))
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()
        print("")
    else:
        print("[INFO] workers>1 のため、現在対局手数/TPS の逐次表示は省略し、完了対局数のみ表示します。")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_eval_worker,
            initargs=(model_path, device),
        ) as ex:
            pending = {
                ex.submit(_evaluate_one_game, i, opening_random_plies, deterministic, seed): i
                for i in range(games)
            }
            done_games = 0
            while pending:
                done, not_done = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    game = fut.result()
                    game_results.append(game)
                    done_games += 1
                pending = not_done
                sys.stdout.write(f"\r[進行中] 完了対局数: {done_games}/{games}")
                sys.stdout.flush()
            print("")

    game_results.sort(key=lambda g: int(g.get("game_index", 0)))

    for i, game in enumerate(game_results):
        model_side = int(game["model_side"])

        total_turns += game["turn_count"]
        reason = game["reason"]
        summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1

        side_key = "sente" if model_side == Player.SENTE else "gote"
        summary["by_side"][side_key]["games"] += 1

        if game["result"] == "win":
            summary["wins"] += 1
            summary["by_side"][side_key]["wins"] += 1
        elif game["result"] == "loss":
            summary["losses"] += 1
            summary["by_side"][side_key]["losses"] += 1
        elif game["result"] == "draw":
            summary["draws"] += 1
            summary["by_side"][side_key]["draws"] += 1
        else:
            summary["unknown"] += 1

        avg_turns = total_turns / max(i + 1, 1)
        reason_jp = REASON_LABEL_JP.get(reason, reason)
        ai_side_jp = "先手" if model_side == Player.SENTE else "後手"
        random_side_jp = "後手" if model_side == Player.SENTE else "先手"
        winner_role = game.get("winner_role", "unknown")
        if winner_role == "ai":
            winner_jp = f"AI({ai_side_jp})"
        elif winner_role == "random":
            winner_jp = f"Random({random_side_jp})"
        elif winner_role == "draw":
            winner_jp = "引き分け"
        else:
            winner_jp = "不明"

        print(
            f"\n  [対局 {i + 1:5d}] 終了理由: {reason_jp:8s} | 手数: {game['turn_count']:4d} | "
            f"平均: {avg_turns:6.1f} | 勝者: {winner_jp}"
        )

    summary["avg_turns"] = round(total_turns / max(games, 1), 3)
    summary["win_rate"] = round(summary["wins"] / max(games, 1), 4)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate latest model vs random player")
    parser.add_argument("--model", default="./ai_training/models/shogi3d_ppo_latest.zip")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--workers", type=int, default=int(os.getenv("SHOGI_EVAL_WORKERS", "1")))
    args = parser.parse_args()

    result = evaluate(
        model_path=args.model,
        games=args.games,
        opening_random_plies=args.opening_random_plies,
        deterministic=args.deterministic,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
    )

    print("=== Eval: Model vs Random ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
