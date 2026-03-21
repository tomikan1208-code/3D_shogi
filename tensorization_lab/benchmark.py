import time
import numpy as np
import torch

from tensorized_ops import TensorizedBoardOps


def fu_moves_numpy(piece_np, owner_np, player):
    bsz = piece_np.shape[0]
    out = np.zeros_like(piece_np, dtype=bool)
    fu = 1

    for b in range(bsz):
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    if piece_np[b, x, y, z] != fu or owner_np[b, x, y, z] != player:
                        continue
                    if player == 1 and y > 0 and piece_np[b, x, y - 1, z] == 0:
                        out[b, x, y, z] = True
                    if player == 2 and y < 8 and piece_np[b, x, y + 1, z] == 0:
                        out[b, x, y, z] = True
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ops = TensorizedBoardOps(device=device)

    batch = 256
    piece_t, owner_t = ops.random_state(batch)

    # Warm-up
    _ = ops.legal_fu_moves_mask(piece_t, owner_t, player=1)
    _ = ops.king_adjacent_attack_map(piece_t, owner_t, player=1)
    if device == "cuda":
        torch.cuda.synchronize()

    # Tensorized timing
    t0 = time.perf_counter()
    for _ in range(50):
        _ = ops.legal_fu_moves_mask(piece_t, owner_t, player=1)
        _ = ops.king_adjacent_attack_map(piece_t, owner_t, player=1)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # NumPy loop baseline (FU only)
    piece_np = piece_t.detach().cpu().numpy()
    owner_np = owner_t.detach().cpu().numpy()
    t2 = time.perf_counter()
    for _ in range(5):
        _ = fu_moves_numpy(piece_np, owner_np, player=1)
    t3 = time.perf_counter()

    tensor_ms = (t1 - t0) * 1000 / 50
    numpy_ms = (t3 - t2) * 1000 / 5

    print({
        "device": device,
        "batch": batch,
        "tensorized_ms_per_iter": round(tensor_ms, 3),
        "numpy_fu_loop_ms_per_iter": round(numpy_ms, 3),
        "speedup_vs_numpy_fu_only": round(numpy_ms / max(tensor_ms, 1e-9), 2),
    })


if __name__ == "__main__":
    main()
