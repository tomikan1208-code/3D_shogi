import sys

import torch


def main() -> int:
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
        return 0
    print("CUDA is not available in this Python environment.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
