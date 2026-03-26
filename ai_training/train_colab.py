import argparse
import os
from pathlib import Path
import warnings


def maybe_mount_drive() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception:
        # Not running on Colab, ignore.
        pass


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def set_default_env(key: str, value: str) -> None:
    if os.getenv(key) is None:
        os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D Shogi training on Colab with Google Drive persistence")
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/3d_shogi_train")
    parser.add_argument("--model-subdir", default="models")
    parser.add_argument("--log-subdir", default="logs")
    parser.add_argument("--n-envs", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-mask-gb", type=float, default=8.0)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--total-steps", type=int, default=20000000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-cuda", action="store_true")
    parser.add_argument("--use-cpp", action="store_true", help="Enable C++ legal move/kingcheck if extension is available")
    parser.add_argument("--fixed-action-space", action="store_true", default=True)
    args = parser.parse_args()

    maybe_mount_drive()

    # Colab logs can be noisy due to transitive TensorBoard/TensorFlow imports.
    # These are not fatal for SB3+PyTorch training, so we mute known noisy warnings.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("PYTHONWARNINGS", "ignore:Gym has been unmaintained")
    warnings.filterwarnings("ignore", message="Gym has been unmaintained.*")

    drive_root = Path(args.drive_root)
    model_dir = drive_root / args.model_subdir
    log_dir = drive_root / args.log_subdir
    ensure_dirs(model_dir, log_dir)

    os.environ["SHOGI_MODEL_DIR"] = str(model_dir)
    os.environ["SHOGI_LOG_DIR"] = str(log_dir)
    os.environ["SHOGI_N_ENVS"] = str(args.n_envs)
    os.environ["SHOGI_N_STEPS"] = str(args.n_steps)
    os.environ["SHOGI_N_EPOCHS"] = str(args.n_epochs)
    os.environ["SHOGI_BATCH_SIZE"] = str(args.batch_size)
    os.environ["SHOGI_MAX_MASK_GB"] = str(args.max_mask_gb)
    os.environ["SHOGI_OPENING_RANDOM_PLIES"] = str(args.opening_random_plies)
    os.environ["SHOGI_TOTAL_STEPS"] = str(args.total_steps)
    os.environ["SHOGI_DEVICE"] = args.device
    os.environ["SHOGI_FORCE_CUDA"] = "1" if args.force_cuda else "0"
    os.environ["SHOGI_FIXED_ACTION_SPACE"] = "1" if args.fixed_action_space else "0"

    if args.use_cpp:
        set_default_env("SHOGI_USE_CPP_KINGCHECK", "1")
        set_default_env("SHOGI_USE_CPP_LEGAL_MOVES", "1")
        set_default_env("SHOGI_USE_CPP_PINNED", "1")
        set_default_env("SHOGI_CPP_FILTER_LEVEL", "1")
    else:
        set_default_env("SHOGI_USE_CPP_KINGCHECK", "0")
        set_default_env("SHOGI_USE_CPP_LEGAL_MOVES", "0")

    # Safe defaults for training quality and speed.
    set_default_env("SHOGI_CPP_STRICT_FILTER", "0")
    set_default_env("SHOGI_DETERMINISTIC_MOVE_ORDER", "1")
    set_default_env("SHOGI_FAST_DROP_GENERATION", "1")
    set_default_env("SHOGI_USE_LEGAL_MOVES_CACHE", "1")
    set_default_env("SHOGI_USE_UINT8_OBS", "1")
    set_default_env("SHOGI_TACTICAL_REWARD", "1")
    set_default_env("SHOGI_COLAB_LOG_INTERVAL_SEC", "20")

    print("=== Colab Training Config ===")
    print(f"drive_root: {drive_root}")
    print(f"model_dir : {model_dir}")
    print(f"log_dir   : {log_dir}")
    print(f"n_envs={args.n_envs} n_steps={args.n_steps} n_epochs={args.n_epochs} batch_size={args.batch_size}")
    print("If model files already exist in model_dir, train.py will auto-load the latest one.")

    from train import train

    train()


if __name__ == "__main__":
    main()
