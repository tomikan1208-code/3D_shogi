import os
import sys


def _load_cpp_module():
    here = os.path.dirname(__file__)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import shogi3d_core
        return shogi3d_core
    except Exception:
        return None


_cpp = _load_cpp_module()


def cpp_available():
    return _cpp is not None


def is_king_in_check_ray(board, owner, player):
    if _cpp is None:
        return None
    return bool(_cpp.is_king_in_check_ray_cpp(board, owner, int(player)))


def generate_legal_moves(board, owner, hand_counts, player, king_pos=None):
    if _cpp is None:
        return None
    if king_pos is not None:
        kx, ky, kz = (int(king_pos[0]), int(king_pos[1]), int(king_pos[2]))
        arr = _cpp.generate_legal_moves_cpp(board, owner, hand_counts, int(player), kx, ky, kz)
    else:
        arr = _cpp.generate_legal_moves_cpp(board, owner, hand_counts, int(player))
    return arr.tolist()


def compute_pinned_pieces(board, owner, player, currently_in_check):
    if _cpp is None:
        return None
    arr = _cpp.compute_pinned_pieces_cpp(board, owner, int(player), bool(currently_in_check))
    return arr.tolist()
