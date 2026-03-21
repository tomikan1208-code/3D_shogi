import argparse
import random
from collections import Counter

from shogi_env import (
    HAND_INDEX,
    PROMOTION_MAP,
    PieceType,
    Player,
    Shogi3DEnv,
)


def _to_counter(moves):
    return Counter(tuple(int(v) for v in m) for m in moves)


def _hand_unique_sorted(env, player):
    return sorted(set(env.hands[player]), key=lambda p: int(p))


def _find_king(board, owner, player):
    for x in range(9):
        for y in range(9):
            for z in range(9):
                if owner[x, y, z] == player and board[x, y, z] == PieceType.OU:
                    return (x, y, z)
    return None


def _is_king_in_check_ray_from_state(env, board, owner, player):
    king = _find_king(board, owner, player)
    if king is None:
        return True

    saved_board = env.board
    saved_owner = env.owner
    saved_kings = env.king_positions
    try:
        env.board = board
        env.owner = owner
        env.king_positions = {
            Player.SENTE: _find_king(board, owner, Player.SENTE),
            Player.GOTE: _find_king(board, owner, Player.GOTE),
        }
        return bool(env._is_king_in_check_ray(player))
    finally:
        env.board = saved_board
        env.owner = saved_owner
        env.king_positions = saved_kings


def _is_move_safe_oracle(env, player, fx, fy, fz, tx, ty, tz):
    board = env.board.copy()
    owner = env.owner.copy()
    board[tx, ty, tz] = board[fx, fy, fz]
    owner[tx, ty, tz] = owner[fx, fy, fz]
    board[fx, fy, fz] = PieceType.NONE
    owner[fx, fy, fz] = Player.NONE
    return not _is_king_in_check_ray_from_state(env, board, owner, player)


def _is_drop_safe_oracle(env, player, ptype, tx, ty, tz):
    board = env.board.copy()
    owner = env.owner.copy()
    board[tx, ty, tz] = ptype
    owner[tx, ty, tz] = player
    return not _is_king_in_check_ray_from_state(env, board, owner, player)


def get_legal_moves_oracle(env, player):
    moves = []

    currently_in_check = _is_king_in_check_ray_from_state(env, env.board, env.owner, player)

    # Board moves: full scan (no piece_positions / no cache reliance)
    for x in range(9):
        for y in range(9):
            for z in range(9):
                if env.owner[x, y, z] != player:
                    continue
                p = env.board[x, y, z]
                pseudo = env._compute_piece_moves_uncached(x, y, z)
                for tx, ty, tz in pseudo:
                    # Oracle safety: always verify by make/unmake
                    if not _is_move_safe_oracle(env, player, x, y, z, tx, ty, tz):
                        continue

                    moves.append([x, y, z, tx, ty, tz, False, False, 0])
                    if p in PROMOTION_MAP and (
                        env._is_promotion_zone(y, player) or env._is_promotion_zone(ty, player)
                    ):
                        moves.append([x, y, z, tx, ty, tz, True, False, 0])

    # Drops
    unique_hand = _hand_unique_sorted(env, player)
    piece_map_inv = {
        PieceType.FU: 0,
        PieceType.KYO: 1,
        PieceType.KEI: 2,
        PieceType.GIN: 3,
        PieceType.KIN: 4,
        PieceType.KAKU: 5,
        PieceType.HI: 6,
    }

    nifu_xz = set()
    for x in range(9):
        for y in range(9):
            for z in range(9):
                if env.owner[x, y, z] == player and env.board[x, y, z] == PieceType.FU:
                    nifu_xz.add((x, z))

    for ptype in unique_hand:
        dt = piece_map_inv.get(ptype, 0)
        is_fu = ptype == PieceType.FU
        for x in range(9):
            for z in range(9):
                if is_fu and (x, z) in nifu_xz:
                    continue
                for y in range(9):
                    if env.board[x, y, z] != PieceType.NONE:
                        continue

                    forward_limit = (player == Player.SENTE and y == 0) or (player == Player.GOTE and y == 8)
                    kei_limit = (player == Player.SENTE and y <= 1) or (player == Player.GOTE and y >= 7)
                    if is_fu and forward_limit:
                        continue
                    if ptype == PieceType.KYO and forward_limit:
                        continue
                    if ptype == PieceType.KEI and kei_limit:
                        continue

                    if currently_in_check and not _is_drop_safe_oracle(env, player, ptype, x, y, z):
                        continue

                    moves.append([0, 0, 0, x, y, z, False, True, dt])

    return moves


def run_validation(positions, seed, opening_random_plies, max_steps_per_game, max_diff_print):
    random.seed(seed)

    env = Shogi3DEnv(opening_random_plies=opening_random_plies)
    env.reset(seed=seed)

    checked = 0
    mismatches = 0

    while checked < positions:
        player = env.current_player

        oracle = get_legal_moves_oracle(env, player)

        env.use_cpp_legal_moves = False
        env.use_cpp_kingcheck = False
        py_moves = env.get_legal_moves(player)

        env.use_cpp_legal_moves = True
        env.use_cpp_kingcheck = True
        cpp_moves = env.get_legal_moves(player)

        c_oracle = _to_counter(oracle)
        c_py = _to_counter(py_moves)
        c_cpp = _to_counter(cpp_moves)

        py_ok = c_oracle == c_py
        cpp_ok = c_oracle == c_cpp

        if (not py_ok) or (not cpp_ok):
            mismatches += 1
            print("[MISMATCH] position", checked, "player", int(player))
            print("  oracle", len(oracle), "python", len(py_moves), "cpp", len(cpp_moves))

            if not py_ok:
                only_py = list((c_py - c_oracle).elements())
                miss_py = list((c_oracle - c_py).elements())
                print("  python extra(sample)", only_py[:max_diff_print])
                print("  python miss(sample)", miss_py[:max_diff_print])

            if not cpp_ok:
                only_cpp = list((c_cpp - c_oracle).elements())
                miss_cpp = list((c_oracle - c_cpp).elements())
                print("  cpp extra(sample)", only_cpp[:max_diff_print])
                print("  cpp miss(sample)", miss_cpp[:max_diff_print])
                seen_src = set()
                for m in only_cpp[:max_diff_print]:
                    fx, fy, fz = m[0], m[1], m[2]
                    src = (fx, fy, fz)
                    if src in seen_src:
                        continue
                    seen_src.add(src)
                    print("  cpp extra src", src, "piece", int(env.board[fx, fy, fz]), "owner", int(env.owner[fx, fy, fz]))
                    if len(m) >= 6:
                        tx, ty, tz = m[3], m[4], m[5]
                        print("   src_in_piece_positions", src in env.piece_positions[player])
                        pseudo = env._compute_piece_moves_uncached(fx, fy, fz)
                        print("   in_python_pseudo", [tx, ty, tz] in pseudo)
                        try:
                            print("   env_is_move_safe", bool(env._is_move_safe(fx, fy, fz, tx, ty, tz, player)))
                        except Exception as exc:
                            print("   env_is_move_safe_error", str(exc))
                        try:
                            print("   oracle_is_move_safe", bool(_is_move_safe_oracle(env, player, fx, fy, fz, tx, ty, tz)))
                        except Exception as exc:
                            print("   oracle_is_move_safe_error", str(exc))

            return {
                "ok": False,
                "checked_positions": checked + 1,
                "mismatches": mismatches,
                "python_ok": py_ok,
                "cpp_ok": cpp_ok,
            }

        checked += 1

        if len(oracle) == 0:
            env.reset()
            continue

        legal_actions = list(env._get_legal_action_map(env.current_player).keys())
        idx = random.choice(legal_actions)
        _, _, term, trunc, _ = env.step(idx)
        if term or trunc or env.turn_count >= max_steps_per_game:
            env.reset()

    return {
        "ok": True,
        "checked_positions": checked,
        "mismatches": mismatches,
        "python_ok": True,
        "cpp_ok": True,
    }


def main():
    p = argparse.ArgumentParser(description="Oracle-based legal move validation for Python and C++")
    p.add_argument("--positions", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opening-random-plies", type=int, default=2)
    p.add_argument("--max-steps-per-game", type=int, default=1200)
    p.add_argument("--max-diff-print", type=int, default=10)
    args = p.parse_args()

    result = run_validation(
        positions=args.positions,
        seed=args.seed,
        opening_random_plies=args.opening_random_plies,
        max_steps_per_game=args.max_steps_per_game,
        max_diff_print=args.max_diff_print,
    )

    print(result)
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
