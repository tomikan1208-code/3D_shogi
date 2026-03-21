import argparse
import json
import time
from collections import Counter

import torch

from tensorized_ops import TensorizedBoardOps


OFFSETS_26 = torch.tensor(
    [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == 0 and dy == 0 and dz == 0)],
    dtype=torch.int64,
)


def legal_moves_one_board(piece_b: torch.Tensor, owner_b: torch.Tensor, player: int):
    fu = TensorizedBoardOps.FU
    ou = TensorizedBoardOps.OU

    src_list = []
    dst_list = []

    fu_src = torch.nonzero((piece_b == fu) & (owner_b == player), as_tuple=False)
    if fu_src.numel() > 0:
        dy = -1 if player == TensorizedBoardOps.SENTE else 1
        fu_dst = fu_src.clone()
        fu_dst[:, 1] += dy
        in_board = (
            (fu_dst[:, 0] >= 0)
            & (fu_dst[:, 0] < 9)
            & (fu_dst[:, 1] >= 0)
            & (fu_dst[:, 1] < 9)
            & (fu_dst[:, 2] >= 0)
            & (fu_dst[:, 2] < 9)
        )
        if torch.any(in_board):
            fu_src = fu_src[in_board]
            fu_dst = fu_dst[in_board]
            dst_owner = owner_b[fu_dst[:, 0], fu_dst[:, 1], fu_dst[:, 2]]
            legal = dst_owner != player
            if torch.any(legal):
                src_list.append(fu_src[legal])
                dst_list.append(fu_dst[legal])

    king_src = torch.nonzero((piece_b == ou) & (owner_b == player), as_tuple=False)
    if king_src.numel() > 0:
        k = king_src[0]
        cand = k.unsqueeze(0) + OFFSETS_26.to(piece_b.device)
        in_board = (
            (cand[:, 0] >= 0)
            & (cand[:, 0] < 9)
            & (cand[:, 1] >= 0)
            & (cand[:, 1] < 9)
            & (cand[:, 2] >= 0)
            & (cand[:, 2] < 9)
        )
        cand = cand[in_board]
        if cand.numel() > 0:
            dst_owner = owner_b[cand[:, 0], cand[:, 1], cand[:, 2]]
            legal = dst_owner != player
            cand = cand[legal]
            if cand.numel() > 0:
                src = k.unsqueeze(0).repeat(cand.shape[0], 1)
                src_list.append(src)
                dst_list.append(cand)

    if not src_list:
        return None, None
    return torch.cat(src_list, dim=0), torch.cat(dst_list, dim=0)


def run_matches(batch_size: int, max_plies: int, seed: int, device: str):
    ops = TensorizedBoardOps(device=device)
    piece, owner = ops.random_state(batch_size=batch_size, seed=seed)

    current_player = torch.full((batch_size,), TensorizedBoardOps.SENTE, dtype=torch.int64, device=piece.device)
    active = torch.ones((batch_size,), dtype=torch.bool, device=piece.device)
    reason = [None] * batch_size
    plies = torch.zeros((batch_size,), dtype=torch.int64, device=piece.device)

    for _ in range(max_plies):
        active_idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        if active_idx.numel() == 0:
            break

        for b in active_idx.tolist():
            p = int(current_player[b].item())
            src, dst = legal_moves_one_board(piece[b], owner[b], p)

            if src is None:
                active[b] = False
                reason[b] = "legal_none"
                continue

            n = src.shape[0]
            r = int(torch.randint(0, n, (1,), device=piece.device).item())
            s = src[r]
            d = dst[r]

            captured_piece = int(piece[b, d[0], d[1], d[2]].item())

            piece[b, d[0], d[1], d[2]] = piece[b, s[0], s[1], s[2]]
            owner[b, d[0], d[1], d[2]] = owner[b, s[0], s[1], s[2]]
            piece[b, s[0], s[1], s[2]] = 0
            owner[b, s[0], s[1], s[2]] = 0

            plies[b] += 1

            if captured_piece == TensorizedBoardOps.OU:
                active[b] = False
                reason[b] = "ou_capture"
                continue

            current_player[b] = TensorizedBoardOps.GOTE if p == TensorizedBoardOps.SENTE else TensorizedBoardOps.SENTE

    still_active = torch.nonzero(active, as_tuple=False).squeeze(1).tolist()
    for b in still_active:
        reason[b] = "max_plies"

    rc = Counter(reason)
    avg_ply = float(plies.float().mean().item()) if batch_size > 0 else 0.0
    return {
        "batch_size": batch_size,
        "max_plies": max_plies,
        "avg_plies": round(avg_ply, 3),
        "reasons": dict(rc),
    }


def main():
    parser = argparse.ArgumentParser(description="GPU match runner (tensorization prototype)")
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--max-plies", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-cuda", action="store_true")
    args = parser.parse_args()

    if args.force_cuda and not torch.cuda.is_available():
        raise RuntimeError("--force-cuda 指定ですが CUDA が利用できません。")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda 指定ですが CUDA が利用できません。")

    start = time.perf_counter()
    out = run_matches(args.batch, args.max_plies, args.seed, args.device)
    elapsed = time.perf_counter() - start
    out["device"] = args.device
    out["elapsed_sec"] = round(elapsed, 3)
    out["matches_per_sec"] = round(args.batch / max(elapsed, 1e-9), 2)

    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
