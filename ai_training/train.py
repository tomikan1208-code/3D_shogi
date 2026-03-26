import os
import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from shogi_env import Shogi3DEnv, PieceType
from model import Shogi3DFeaturesExtractor

import sys
import time
import math
import json
import random


REASON_LABEL_JP = {
    "mate": "詰み",
    "illegal": "不正手",
    "ou_capture": "王捕獲",
    "sennichite": "千日手",
    "no_legal_moves": "合法手なし",
    "timeout": "時間切れ",
    "unknown": "不明",
}


class ProgressMaskablePPO(MaskablePPO):
    """MaskablePPO with in-place update-phase progress output.
    This avoids monkey-patching instance methods, keeping save/pickle stable.
    """

    def train(self):
        t_train_start = time.time()
        rb = getattr(self, "rollout_buffer", None)
        if rb is None or not hasattr(rb, "get"):
            ret = super().train()
            self.last_train_duration_sec = time.time() - t_train_start
            return ret

        sample_count = int(getattr(rb, "buffer_size", 0)) * int(getattr(rb, "n_envs", 1))
        batch_size = int(getattr(self, "batch_size", 1))
        n_epochs = int(getattr(self, "n_epochs", 1))
        batches_per_epoch = max(1, math.ceil(sample_count / max(batch_size, 1)))
        total_updates = max(1, n_epochs * batches_per_epoch)

        sys.stdout.write(
            f"\r[更新フェーズ] 進捗:   0.0% (0/{total_updates}) | epochs={n_epochs}, batches/epoch={batches_per_epoch}"
        )
        sys.stdout.flush()

        orig_get = rb.get
        state = {"updates": 0}

        def wrapped_get(batch_size_arg=None):
            for data in orig_get(batch_size_arg):
                state["updates"] += 1
                pct = min(100.0, (state["updates"] / total_updates) * 100.0)
                sys.stdout.write(f"\r[更新フェーズ] 進捗: {pct:5.1f}% ({state['updates']}/{total_updates})")
                sys.stdout.flush()
                yield data

        rb.get = wrapped_get
        try:
            ret = super().train()
        finally:
            rb.get = orig_get
            self.last_train_duration_sec = time.time() - t_train_start
            sys.stdout.write(
                f"\r[更新フェーズ] 完了: 100.0% ({state['updates']}/{total_updates})" + " " * 20 + "\n"
            )
            sys.stdout.flush()
        return ret

PIECE_LABEL_JP = {
    int(PieceType.FU): "歩",
    int(PieceType.KYO): "香",
    int(PieceType.KEI): "桂",
    int(PieceType.GIN): "銀",
    int(PieceType.KIN): "金",
    int(PieceType.KAKU): "角",
    int(PieceType.HI): "飛",
    int(PieceType.OU): "王",
    int(PieceType.NARI_FU): "と",
    int(PieceType.NARI_KYO): "成香",
    int(PieceType.NARI_KEI): "成桂",
    int(PieceType.NARI_GIN): "成銀",
    int(PieceType.UMA): "馬",
    int(PieceType.RYU): "龍",
}

class MatchCountCallback(BaseCallback):
    """
    累積試合数、平均手数、および終了理由の統計を表示するコールバック
    """
    def __init__(self, verbose=0):
        super(MatchCountCallback, self).__init__(verbose)
        self.total_matches = 0
        self.total_steps_in_finished_matches = 0
        self.env_steps = None
        # 終了理由の統計
        self.stats = {
            "mate": 0,
            "illegal": 0,
            "ou_capture": 0,
            "sennichite": 0,
            "no_legal_moves": 0,
            "timeout": 0,
            "unknown": 0,
        }
        self.start_time = time.time()
        self.base_timesteps = 0
        self.gen_matches = 0
        self.gen_steps_in_finished_matches = 0
        self.gen_stats = self.stats.copy()
        self.ou_capture_by_total = {}
        self.ou_capture_by_gen = {}
        self.last_live_print_ts = 0.0
        self.save_reason_replays = os.getenv("SHOGI_SAVE_REASON_REPLAYS", "1") == "1"
        default_replay_dir = os.path.join(os.path.dirname(__file__), "replay_logs")
        self.replay_dir = os.getenv("SHOGI_REPLAY_LOG_DIR", default_replay_dir)
        self.replay_target_reasons = {"mate", "ou_capture", "timeout"}
        self.replay_seen_counts = {k: 0 for k in self.replay_target_reasons}
        self.replay_samples = {}

    def _on_training_start(self) -> None:
        # 既存チェックポイントを読み込んだ場合でも、FPSは「今回learnで進んだ分」だけで計算する
        self.start_time = time.time()
        self.base_timesteps = int(self.model.num_timesteps)

    def reset_generation_stats(self) -> None:
        self.gen_matches = 0
        self.gen_steps_in_finished_matches = 0
        self.gen_stats = {k: 0 for k in self.gen_stats.keys()}
        self.ou_capture_by_gen = {}
        self.last_live_print_ts = 0.0
        self.replay_seen_counts = {k: 0 for k in self.replay_target_reasons}
        self.replay_samples = {}

    def _try_sample_replay(self, reason: str, info: dict):
        if not self.save_reason_replays:
            return
        if reason not in self.replay_target_reasons:
            return
        if not isinstance(info, dict):
            return
        def _as_dict(v):
            if isinstance(v, dict):
                return v
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
                return v[0]
            return None

        game_record = info.get("game_record")
        if not game_record:
            terminal_info = _as_dict(info.get("terminal_info"))
            if terminal_info is not None:
                game_record = terminal_info.get("game_record")
        if not game_record:
            final_info = _as_dict(info.get("final_info"))
            if final_info is not None:
                game_record = final_info.get("game_record")
        if not game_record:
            return

        self.replay_seen_counts[reason] += 1
        seen = self.replay_seen_counts[reason]
        # Reservoir sampling: 各理由カテゴリからランダムに1局だけ保持
        if seen == 1 or random.random() < (1.0 / float(seen)):
            winner = info.get("winner", 0)
            if winner == 0:
                terminal_info = _as_dict(info.get("terminal_info"))
                if terminal_info is not None:
                    winner = terminal_info.get("winner", 0)
            if winner == 0:
                final_info = _as_dict(info.get("final_info"))
                if final_info is not None:
                    winner = final_info.get("winner", 0)
            self.replay_samples[reason] = {
                "reason": reason,
                "winner": int(winner) if winner is not None else 0,
                "turn_count": int(len(game_record)),
                "moves": game_record,
            }

    def save_generation_replays(self, generation: int):
        if not self.save_reason_replays:
            return []

        os.makedirs(self.replay_dir, exist_ok=True)
        saved = []
        for reason in sorted(self.replay_target_reasons):
            sample = self.replay_samples.get(reason)
            if not sample:
                continue
            payload = {
                "format": "3d_shogi_replay_v1",
                "source": "train_generation",
                "generation": int(generation),
                "reason": str(sample["reason"]),
                "winner": int(sample["winner"]),
                "turn_count": int(sample["turn_count"]),
                "moves": sample["moves"],
            }
            out_path = os.path.join(self.replay_dir, f"gen_{int(generation):05d}_{reason}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            size_bytes = int(os.path.getsize(out_path))
            saved.append({
                "reason": reason,
                "path": out_path,
                "size_bytes": size_bytes,
            })
        return saved

    def replay_debug_summary(self):
        seen = ", ".join(
            f"{k}:{int(self.replay_seen_counts.get(k, 0))}" for k in sorted(self.replay_target_reasons)
        )
        sampled = ", ".join(sorted(self.replay_samples.keys())) if self.replay_samples else "none"
        return f"seen=({seen}) sampled=({sampled})"

    def _record_ou_capture_by(self, piece_type_int):
        label = PIECE_LABEL_JP.get(int(piece_type_int), f"駒{int(piece_type_int)}")
        self.ou_capture_by_total[label] = self.ou_capture_by_total.get(label, 0) + 1
        self.ou_capture_by_gen[label] = self.ou_capture_by_gen.get(label, 0) + 1

    def get_generation_avg_turns(self) -> float:
        return self.gen_steps_in_finished_matches / max(self.gen_matches, 1)

    def _on_step(self) -> bool:
        if self.env_steps is None:
            self.env_steps = np.zeros(self.training_env.num_envs, dtype=np.int32)

        self.env_steps += 1
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if infos is not None:
            for i, info in enumerate(infos):
                done_by_dones = bool(dones[i]) if (dones is not None and i < len(dones)) else False
                done_by_episode = isinstance(info, dict) and ("episode" in info)
                if not (done_by_dones or done_by_episode):
                    continue

                self.total_matches += 1
                self.total_steps_in_finished_matches += self.env_steps[i]
                self.gen_matches += 1
                self.gen_steps_in_finished_matches += self.env_steps[i]

                reason = "unknown"
                if isinstance(info, dict):
                    def _as_dict(v):
                        if isinstance(v, dict):
                            return v
                        if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], dict):
                            return v[0]
                        return None

                    reason = info.get("reason", reason)
                    terminal_info = _as_dict(info.get("terminal_info"))
                    final_info = _as_dict(info.get("final_info"))
                    if reason == "unknown" and terminal_info is not None:
                        reason = terminal_info.get("reason", reason)
                    if reason == "unknown" and final_info is not None:
                        reason = final_info.get("reason", reason)
                    if reason == "unknown" and info.get("TimeLimit.truncated"):
                        reason = "timeout"

                self.stats[reason] = self.stats.get(reason, 0) + 1
                self.gen_stats[reason] = self.gen_stats.get(reason, 0) + 1
                self._try_sample_replay(reason, info if isinstance(info, dict) else {})

                if reason == "ou_capture":
                    capture_piece = None
                    if isinstance(info, dict):
                        capture_piece = info.get("ou_capture_by", None)
                        if capture_piece is None and isinstance(info.get("terminal_info"), dict):
                            capture_piece = info["terminal_info"].get("ou_capture_by", None)
                    if capture_piece is not None:
                        self._record_ou_capture_by(capture_piece)

                # 試合終了時に1行ログを表示
                avg_turns = self.get_generation_avg_turns()
                reason_jp = REASON_LABEL_JP.get(reason, reason)
                # 進捗の上書き行を消してから、確定ログを改行で出す
                sys.stdout.write("\r" + " " * 120 + "\r")
                sys.stdout.flush()
                print(f"\n  [対局 {self.total_matches:5d}] 終了理由: {reason_jp:8s} | 手数: {self.env_steps[i]:3d} | 平均: {avg_turns:5.1f}")

                self.env_steps[i] = 0 # リセット
            
        # 現在対局の手数は1行上書きで表示（新しい行は増やさない）
        now = time.time()
        if now - self.last_live_print_ts >= 0.2:
            current_turn = int(np.max(self.env_steps)) if self.env_steps is not None and len(self.env_steps) > 0 else 0
            elapsed = now - self.start_time
            steps_this_session = max(int(self.num_timesteps) - self.base_timesteps, 0)
            tps_total = steps_this_session / max(elapsed, 1e-6)
            n_envs = max(int(self.training_env.num_envs), 1)
            tps_per_env = tps_total / n_envs
            sys.stdout.write(
                f"\r  [進行中] 現在対局手数: {current_turn:4d} | 手数/s(合計): {tps_total:6.2f} | 1環境換算: {tps_per_env:6.2f}"
            )
            sys.stdout.flush()
            self.last_live_print_ts = now

        return True


def safe_model_save(model, target_path, retries=2):
    """Save to a temporary file first, then atomically replace target.
    This keeps the previous model valid even if save gets interrupted.
    """
    tmp_path = target_path + ".tmp"
    last_err = None

    for attempt in range(retries + 1):
        try:
            model.save(tmp_path)
            os.replace(tmp_path, target_path)
            return True
        except PermissionError as e:
            last_err = repr(e)
            # Windows では他プロセスが掴んでいると置換に失敗することがある。
            fallback = target_path.replace(".zip", f".{int(time.time())}.zip")
            print(
                f"[WARN] Atomic replace failed due to file lock/permission. "
                f"Saving fallback file: {fallback}"
            )
            try:
                model.save(fallback)
                return True
            except Exception as e2:
                last_err = repr(e2)
                print(f"[WARN] Fallback save failed: {e2}")
        except KeyboardInterrupt:
            last_err = "KeyboardInterrupt"
            print("\n[WARN] Save interrupted by KeyboardInterrupt. Keeping previous model file.")
            break
        except Exception as e:
            last_err = repr(e)
            print(f"[WARN] Save attempt {attempt + 1}/{retries + 1} failed: {e}")
            time.sleep(0.2)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    print(f"[ERROR] Failed to save model to {target_path}. Last error: {last_err}")
    return False


def train():
    print("[TRAIN] script version: replay-save-v3", flush=True)
    # 1. 環境の並列生成
    cpu_count = os.cpu_count() or 2
    default_envs = max(1, min(8, cpu_count // 2))
    n_envs = int(os.getenv("SHOGI_N_ENVS", default_envs))
    n_steps = int(os.getenv("SHOGI_N_STEPS", 1024))
    n_epochs = int(os.getenv("SHOGI_N_EPOCHS", 10))
    max_mask_gb = float(os.getenv("SHOGI_MAX_MASK_GB", "2.0"))
    total_target_steps = int(os.getenv("SHOGI_TOTAL_STEPS", 20000000))
    opening_random_plies = int(os.getenv("SHOGI_OPENING_RANDOM_PLIES", 2))

    print(f"Creating {n_envs} parallel environments...")
    print(f"Requested rollout settings: n_steps={n_steps}, n_envs={n_envs}, n_epochs={n_epochs}")
    env_kwargs = {"opening_random_plies": opening_random_plies}
    force_dummyvec = os.getenv("SHOGI_FORCE_DUMMYVEC", "0") == "1"
    if n_envs > 1 and not force_dummyvec:
        env = make_vec_env(
            Shogi3DEnv,
            n_envs=n_envs,
            env_kwargs=env_kwargs,
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "spawn"}
        )
    else:
        if n_envs > 1 and force_dummyvec:
            print("[WARN] SHOGI_FORCE_DUMMYVEC=1: using DummyVecEnv to avoid Subproc spawn.")
        env = make_vec_env(Shogi3DEnv, n_envs=n_envs, env_kwargs=env_kwargs, vec_env_cls=DummyVecEnv)

    # MaskablePPO の action mask バッファ想定サイズを使って n_steps を安全側へ調整する。
    # 既定は 1 byte/要素想定（過去運用値）。必要なら SHOGI_MASK_ITEM_BYTES で上書き可能。
    action_n = int(getattr(env.action_space, "n", 0))
    mask_item_bytes = int(os.getenv("SHOGI_MASK_ITEM_BYTES", "1"))
    if mask_item_bytes <= 0:
        mask_item_bytes = 1
    bytes_per_step = 0
    if action_n > 0:
        bytes_per_step = int(n_envs) * int(action_n) * int(mask_item_bytes)
        requested_mask_gb = (bytes_per_step * int(n_steps)) / (1024 ** 3)
        budget_bytes = int(max(max_mask_gb, 0.25) * (1024 ** 3))
        if bytes_per_step > 0 and requested_mask_gb > max_mask_gb:
            safe_n_steps = max(1, budget_bytes // bytes_per_step)
            print(
                "[WARN] Requested n_steps is too large for action-mask buffer: "
                f"requested_mask_mem={requested_mask_gb:.2f} GiB > budget={max_mask_gb:.2f} GiB. "
                f"Auto-adjust n_steps: {n_steps} -> {safe_n_steps}."
            )
            n_steps = int(safe_n_steps)
    obs_bytes_per_step = 0
    obs_space = getattr(env, "observation_space", None)
    if hasattr(obs_space, "spaces"):
        for space in obs_space.spaces.values():
            shape = getattr(space, "shape", None)
            if shape is None:
                continue
            dtype = np.dtype(getattr(space, "dtype", np.float32))
            obs_bytes_per_step += int(np.prod(shape)) * int(dtype.itemsize) * int(n_envs)
    obs_rollout_gb = (obs_bytes_per_step * int(n_steps)) / (1024 ** 3) if obs_bytes_per_step > 0 else 0.0
    mask_rollout_gb = (bytes_per_step * int(n_steps)) / (1024 ** 3) if bytes_per_step > 0 else 0.0
    print(
        f"Applied rollout settings: n_steps={n_steps}, n_envs={n_envs}, action_n={action_n}, "
        f"mask_item_bytes={mask_item_bytes}, mask_budget_gb={max_mask_gb}"
    )
    print(f"Estimated rollout memory: obs={obs_rollout_gb:.2f} GiB, mask={mask_rollout_gb:.2f} GiB")
    
    # 2. パス設定
    model_dir = os.getenv("SHOGI_MODEL_DIR", "./models/")
    log_dir = os.getenv("SHOGI_LOG_DIR", "./logs/")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    latest_model_path = os.path.join(model_dir, "shogi3d_ppo_latest.zip")

    # 3. PPOモデルの初期化またはロード
    requested_device = os.getenv("SHOGI_DEVICE", "auto").lower()
    force_cuda = os.getenv("SHOGI_FORCE_CUDA", "0") == "1"
    if requested_device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    if force_cuda and (not torch.cuda.is_available() or device != "cuda"):
        raise RuntimeError("SHOGI_FORCE_CUDA=1 が指定されていますが、CUDA が利用できません。")

    print(f"Training on device: {device}")
    print("Note: self-play environment stepping runs on CPU; GPU is used for neural network forward/backward.")
    print(f"Opening random plies per reset: {opening_random_plies}")

    rollout_size = n_steps * n_envs
    # 既定は割り切れるバッチを選ぶ。必要なら SHOGI_BATCH_SIZE で明示上書き可能。
    candidate_batch = [512, 256, 128, 64]
    batch_size = 64
    for b in candidate_batch:
        if rollout_size % b == 0:
            batch_size = b
            break
    env_batch = int(os.getenv("SHOGI_BATCH_SIZE", "0"))
    if env_batch > 0:
        batch_size = env_batch
    if batch_size > rollout_size:
        batch_size = rollout_size

    policy_kwargs = dict(
        features_extractor_class=Shogi3DFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 128], vf=[256, 128])
    )

    # ロードするモデルの探索
    load_path = None
    if os.path.exists(model_dir):
        files = [f for f in os.listdir(model_dir) if f.endswith(".zip")]
        if files:
            import re
            def get_steps(fn):
                # Prefer checkpoint step suffix, e.g. shogi3d_checkpoint_123456_steps.zip
                m = re.search(r'checkpoint_(\d+)_steps', fn)
                if m:
                    return int(m.group(1))
                nums = re.findall(r'\d+', fn)
                return int(nums[-1]) if nums else 0
            # 1. チェックポイントの中から最新（最大ステップ）を探す
            checkpoints = [f for f in files if "checkpoint" in f]
            target_file = None
            if checkpoints:
                target_file = max(checkpoints, key=get_steps)
            # 2. チェックポイントがなければ shogi3d_ppo_latest.zip を探す
            elif "shogi3d_ppo_latest.zip" in files:
                target_file = "shogi3d_ppo_latest.zip"
            
            if target_file:
                load_path = os.path.join(model_dir, target_file)

    if load_path:
        try:
            print(f"Loading existing model from {load_path}...")
            loaded_model = ProgressMaskablePPO.load(load_path, env=env, device=device)
            loaded_n_steps = int(getattr(loaded_model, "n_steps", n_steps))
            loaded_batch_size = int(getattr(loaded_model, "batch_size", batch_size))
            loaded_n_envs = int(getattr(loaded_model, "n_envs", env.num_envs))
            loaded_timesteps = int(getattr(loaded_model, "num_timesteps", 0))

            # 既存モデルの rollout 設定が異なる場合は、重みだけ移して再構築する
            if (
                loaded_n_steps != n_steps
                or loaded_batch_size != batch_size
                or loaded_n_envs != int(env.num_envs)
            ):
                print(
                    "Loaded model hyperparameters differ from requested settings: "
                    f"loaded(n_steps={loaded_n_steps}, batch_size={loaded_batch_size}, n_envs={loaded_n_envs}) "
                    f"-> requested(n_steps={n_steps}, batch_size={batch_size}, n_envs={int(env.num_envs)})."
                )
                print("Rebuilding PPO with requested rollout settings and transferring network weights...")
                model = ProgressMaskablePPO(
                    "MultiInputPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=log_dir,
                    device=device,
                    learning_rate=1e-4,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    ent_coef=0.01,
                )
                model.set_parameters(loaded_model.get_parameters(), exact_match=False)
                model.num_timesteps = loaded_timesteps
                del loaded_model
            else:
                model = loaded_model
        except Exception as e:
            print(f"Could not load old model: {e}")
            print("Creating a new model instead.")
            model = ProgressMaskablePPO(
                "MultiInputPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir,
                device=device,
                learning_rate=1e-4,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                ent_coef=0.01,
            )
    else:
        print("Creating new model...")
        model = ProgressMaskablePPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
            learning_rate=1e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            ent_coef=0.01,
        )

    effective_n_steps = int(getattr(model, "n_steps", n_steps))
    effective_batch_size = int(getattr(model, "batch_size", batch_size))
    effective_n_envs = int(env.num_envs)
    effective_rollout_steps = max(effective_n_steps * effective_n_envs, 1)
    print(
        "Effective PPO rollout settings: "
        f"n_steps={effective_n_steps}, batch_size={effective_batch_size}, "
        f"n_envs={effective_n_envs}, learn_timesteps_per_generation={effective_rollout_steps}"
    )
    updates_per_generation = (effective_rollout_steps // max(effective_batch_size, 1)) * max(int(getattr(model, "n_epochs", n_epochs)), 1)
    print(f"Estimated gradient updates per generation: {updates_per_generation}")

    # 4. コールバック設定
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="shogi3d_checkpoint"
    )
    match_count_callback = MatchCountCallback()
    print(
        f"Replay save enabled: {match_count_callback.save_reason_replays} | "
        f"dir: {match_count_callback.replay_dir} | "
        f"target reasons: {','.join(sorted(match_count_callback.replay_target_reasons))}"
    )

    # 5. 学習開始 (世代ループ)
    print("\nBeginning PPO Self-Play Training (Ctrl+C to stop and save)...")
    
    current_total_steps = model.num_timesteps
    generation = current_total_steps // effective_rollout_steps

    try:
        while current_total_steps < total_target_steps:
            generation += 1
            print(f"\n--- Generation {generation} ---")
            match_count_callback.reset_generation_stats()
            print(f"[収集フェーズ] 開始: 目標 timesteps = {effective_rollout_steps}")
            learn_t0 = time.time()
            model.last_train_duration_sec = 0.0
            model.learn(
                total_timesteps=effective_rollout_steps,
                callback=[checkpoint_callback, match_count_callback],
                progress_bar=False,
                reset_num_timesteps=False
            )
            learn_elapsed = time.time() - learn_t0
            update_elapsed = float(getattr(model, "last_train_duration_sec", 0.0))
            collect_elapsed = max(0.0, learn_elapsed - update_elapsed)
            
            current_total_steps = model.num_timesteps
            # 世代終了時のサマリ
            avg_turns = match_count_callback.get_generation_avg_turns()
            elapsed = time.time() - match_count_callback.start_time
            steps_this_session = max(int(model.num_timesteps) - match_count_callback.base_timesteps, 0)
            fps = steps_this_session / max(elapsed, 1e-6)
            n_envs_live = max(int(env.num_envs), 1)
            tps_per_env = fps / n_envs_live
            gen = match_count_callback.gen_stats
            stats_str = (
                f"詰み:{gen.get('mate', 0)}, 不正:{gen.get('illegal', 0)}, 王捕獲:{gen.get('ou_capture', 0)}, "
                f"千日手:{gen.get('sennichite', 0)}, 合法手なし:{gen.get('no_legal_moves', 0)}, "
                f"時間切れ:{gen.get('timeout', 0)}, 不明:{gen.get('unknown', 0)}"
            )
            ou_by = match_count_callback.ou_capture_by_gen
            if ou_by:
                ou_by_str = ", ".join(f"{k}:{v}" for k, v in sorted(ou_by.items(), key=lambda kv: (-kv[1], kv[0])))
                stats_str += f" | 王捕獲内訳: {ou_by_str}"
            print(
                f"\n[世代 {generation}] 終了 | 世代対局数: {int(match_count_callback.gen_matches)} | "
                f"世代平均手数: {avg_turns:.1f} | 世代理由: {stats_str} | 総Step: {current_total_steps} | "
                f"手数/s(合計): {fps:.2f} | 1環境換算: {tps_per_env:.2f} | "
                f"収集時間: {collect_elapsed:.2f}s | 学習時間: {update_elapsed:.2f}s"
            )

            replay_saved = match_count_callback.save_generation_replays(generation)
            if replay_saved:
                size_summary = ", ".join(
                    f"{r['reason']}={r['size_bytes'] / 1024.0:.1f}KB" for r in replay_saved
                )
                total_kb = sum(int(r["size_bytes"]) for r in replay_saved) / 1024.0
                print(
                    f"[世代 {generation}] リプレイ保存: {size_summary} | 合計={total_kb:.1f}KB | "
                    f"{match_count_callback.replay_debug_summary()}"
                )
            else:
                print(
                    f"[世代 {generation}] リプレイ保存: 対象理由の終局なし | "
                    f"{match_count_callback.replay_debug_summary()}"
                )
            
            safe_model_save(model, latest_model_path)

    except (KeyboardInterrupt, SystemExit):
        print("\n\n[INFO] Training interrupted. Saving current progress...")
    finally:
        if safe_model_save(model, latest_model_path):
            print(f"Model saved to {latest_model_path}")
        else:
            print(f"[WARN] Model save skipped/failed. Previous valid file should remain: {latest_model_path}")

if __name__ == "__main__":
    train()
