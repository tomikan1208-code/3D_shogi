import argparse
import json
import time
from collections import Counter

import torch


class P:
    NONE = 0
    SENTE = 1
    GOTE = 2


class T:
    NONE = 0
    FU = 1
    KYO = 2
    KEI = 3
    GIN = 4
    KIN = 5
    KAKU = 6
    HI = 7
    OU = 8
    NARI_FU = 9
    NARI_KYO = 10
    NARI_KEI = 11
    NARI_GIN = 12
    UMA = 13
    RYU = 14


PROMOTION_MAP = {
    T.FU: T.NARI_FU,
    T.KYO: T.NARI_KYO,
    T.KEI: T.NARI_KEI,
    T.GIN: T.NARI_GIN,
    T.KAKU: T.UMA,
    T.HI: T.RYU,
}

HI_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

KAKU_DIRS = set()
for sx in (-1, 1):
    for sy in (-1, 1):
        for sz in (-1, 1):
            KAKU_DIRS.add((sx, 0, sz))
            KAKU_DIRS.add((sx, sy, 0))
            KAKU_DIRS.add((0, sy, sz))
KAKU_DIRS = list(KAKU_DIRS)

ALL_ADJ = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def kin_moves(dy_sign):
    out = []
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            out.append((dx, dy_sign, dz))
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx != 0 or dz != 0:
                out.append((dx, 0, dz))
    out.append((0, -dy_sign, 0))
    return out


def gin_moves(dy_sign):
    out = []
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            out.append((dx, dy_sign, dz))
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx != 0 or dz != 0:
                out.append((dx, -dy_sign, dz))
    return out


def in_board(x, y, z):
    return 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9


def in_promo_zone(y, player):
    if player == P.SENTE:
        return y <= 2
    return y >= 6


def standard_initial_state(batch_size, device):
    piece = torch.zeros((batch_size, 9, 9, 9), dtype=torch.int64, device=device)
    owner = torch.zeros((batch_size, 9, 9, 9), dtype=torch.int64, device=device)

    plane_1_7 = [
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
        [2, 3, 3, 3, 3, 3, 3, 3, 2],
        [2, 3, 4, 4, 5, 4, 4, 3, 2],
        [2, 3, 4, 5, 5, 5, 4, 3, 2],
        [2, 3, 5, 5, 8, 5, 5, 3, 2],
        [2, 3, 4, 5, 5, 5, 4, 3, 2],
        [2, 3, 4, 4, 5, 4, 4, 3, 2],
        [2, 3, 3, 3, 3, 3, 3, 3, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2],
    ]
    plane_2_8 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 7, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 0, 0, 0, 7, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 6, 0, 0, 7, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]

    def apply_plane(y, data, pl):
        for z in range(9):
            for x in range(9):
                t = data[z][x]
                if t != 0:
                    piece[:, x, y, z] = t
                    owner[:, x, y, z] = pl

    apply_plane(0, plane_1_7, P.GOTE)
    apply_plane(1, plane_2_8, P.GOTE)
    piece[:, :, 2, :] = T.FU
    owner[:, :, 2, :] = P.GOTE

    apply_plane(8, plane_1_7, P.SENTE)
    apply_plane(7, plane_2_8, P.SENTE)
    piece[:, :, 6, :] = T.FU
    owner[:, :, 6, :] = P.SENTE

    return piece, owner


def step_moves(piece_b, owner_b, x, y, z, dirs, player):
    out = []
    for dx, dy, dz in dirs:
        tx, ty, tz = x + dx, y + dy, z + dz
        if not in_board(tx, ty, tz):
            continue
        o = int(owner_b[tx, ty, tz].item())
        if o != player:
            out.append((tx, ty, tz))
    return out


def slide_moves(piece_b, owner_b, x, y, z, dirs, player):
    out = []
    for dx, dy, dz in dirs:
        tx, ty, tz = x + dx, y + dy, z + dz
        while in_board(tx, ty, tz):
            o = int(owner_b[tx, ty, tz].item())
            if o != P.NONE:
                if o != player:
                    out.append((tx, ty, tz))
                break
            out.append((tx, ty, tz))
            tx += dx
            ty += dy
            tz += dz
    return out


def piece_moves(piece_b, owner_b, x, y, z, ptype, player):
    dy_sign = -1 if player == P.SENTE else 1

    if ptype == T.OU:
        return step_moves(piece_b, owner_b, x, y, z, ALL_ADJ, player)
    if ptype in (T.KIN, T.NARI_FU, T.NARI_KYO, T.NARI_KEI, T.NARI_GIN):
        return step_moves(piece_b, owner_b, x, y, z, kin_moves(dy_sign), player)
    if ptype == T.GIN:
        return step_moves(piece_b, owner_b, x, y, z, gin_moves(dy_sign), player)
    if ptype == T.FU:
        return step_moves(piece_b, owner_b, x, y, z, [(0, dy_sign, 0)], player)
    if ptype == T.KEI:
        return step_moves(
            piece_b,
            owner_b,
            x,
            y,
            z,
            [(1, 2 * dy_sign, 0), (-1, 2 * dy_sign, 0), (0, 2 * dy_sign, 1), (0, 2 * dy_sign, -1)],
            player,
        )
    if ptype == T.KYO:
        return slide_moves(piece_b, owner_b, x, y, z, [(0, dy_sign, 0)], player)
    if ptype == T.KAKU:
        return slide_moves(piece_b, owner_b, x, y, z, KAKU_DIRS, player)
    if ptype == T.HI:
        return slide_moves(piece_b, owner_b, x, y, z, HI_DIRS, player)
    if ptype == T.RYU:
        s = slide_moves(piece_b, owner_b, x, y, z, HI_DIRS, player)
        a = step_moves(piece_b, owner_b, x, y, z, ALL_ADJ, player)
        return list(dict.fromkeys(s + a))
    if ptype == T.UMA:
        s = slide_moves(piece_b, owner_b, x, y, z, KAKU_DIRS, player)
        a = step_moves(piece_b, owner_b, x, y, z, ALL_ADJ, player)
        return list(dict.fromkeys(s + a))
    return []


def legal_moves_one_board(piece_b, owner_b, player):
    srcs = []
    dsts = []
    promotes = []

    coords = torch.nonzero(owner_b == player, as_tuple=False)
    for c in coords.tolist():
        x, y, z = c
        p = int(piece_b[x, y, z].item())
        dst_list = piece_moves(piece_b, owner_b, x, y, z, p, player)
        for tx, ty, tz in dst_list:
            srcs.append((x, y, z))
            dsts.append((tx, ty, tz))
            can_promote = p in PROMOTION_MAP and (in_promo_zone(y, player) or in_promo_zone(ty, player))
            promotes.append(bool(can_promote))

    if not srcs:
        return None
    return srcs, dsts, promotes


def apply_move(piece_b, owner_b, src, dst, promote, player):
    sx, sy, sz = src
    tx, ty, tz = dst
    p = int(piece_b[sx, sy, sz].item())
    captured = int(piece_b[tx, ty, tz].item())

    if promote and p in PROMOTION_MAP:
        p = PROMOTION_MAP[p]

    piece_b[tx, ty, tz] = p
    owner_b[tx, ty, tz] = player
    piece_b[sx, sy, sz] = T.NONE
    owner_b[sx, sy, sz] = P.NONE

    return captured


def run_matches(batch_size, max_plies, seed, device):
    piece, owner = standard_initial_state(batch_size, device)
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    current = torch.full((batch_size,), P.SENTE, dtype=torch.int64, device=device)
    active = torch.ones((batch_size,), dtype=torch.bool, device=device)
    plies = torch.zeros((batch_size,), dtype=torch.int64, device=device)
    reasons = [None] * batch_size

    for _ in range(max_plies):
        idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            break

        for b in idx.tolist():
            player = int(current[b].item())
            lm = legal_moves_one_board(piece[b], owner[b], player)
            if lm is None:
                active[b] = False
                reasons[b] = "no_legal_moves"
                continue

            srcs, dsts, promotes = lm
            n = len(srcs)
            r = int(torch.randint(0, n, (1,), generator=g, device=device).item())

            promote = False
            if promotes[r]:
                promote = bool(torch.randint(0, 2, (1,), generator=g, device=device).item())

            captured = apply_move(piece[b], owner[b], srcs[r], dsts[r], promote, player)
            plies[b] += 1

            if captured == T.OU:
                active[b] = False
                reasons[b] = "ou_capture"
                continue

            current[b] = P.GOTE if player == P.SENTE else P.SENTE

    for b in torch.nonzero(active, as_tuple=False).squeeze(1).tolist():
        reasons[b] = "max_plies"

    rc = Counter(reasons)
    avg_plies = float(plies.float().mean().item()) if batch_size > 0 else 0.0
    return {
        "batch_size": batch_size,
        "max_plies": max_plies,
        "avg_plies": round(avg_plies, 3),
        "reasons": dict(rc),
    }


def main():
    parser = argparse.ArgumentParser(description="Full-piece GPU match runner (no drops/check filtering)")
    parser.add_argument("--batch", type=int, default=128)
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
    out["matches_per_sec"] = round(args.batch / max(elapsed, 1e-9), 3)
    print(json.dumps(out, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
