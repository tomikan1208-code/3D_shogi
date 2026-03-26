import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from enum import IntEnum
from collections import defaultdict, Counter, deque

try:
    import cpp_kingcheck
except Exception:
    cpp_kingcheck = None

# --- 定数定義 (最適化) ---
_ALL_ADJACENT = [[dx, dy, dz] for dx in [-1,0,1] for dy in [-1,0,1] for dz in [-1,0,1] if not (dx==0 and dy==0 and dz==0)]
_HI_DIRS = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]

_kaku_set = set()
for sx in [-1, 1]:
    for sy in [-1, 1]:
        for sz in [-1, 1]:
            _kaku_set.add((sx, 0, sz))
            _kaku_set.add((sx, sy, 0))
            _kaku_set.add((0, sy, sz))
_KAKU_DIRS = [list(d) for d in _kaku_set]

_KIN_MOVES = {
    1: [[dx, 1, dz] for dx in [-1,0,1] for dz in [-1,0,1]] + \
       [[dx, 0, dz] for dx in [-1,0,1] for dz in [-1,0,1] if dx!=0 or dz!=0] + [[0, -1, 0]],
    -1: [[dx, -1, dz] for dx in [-1,0,1] for dz in [-1,0,1]] + \
        [[dx, 0, dz] for dx in [-1,0,1] for dz in [-1,0,1] if dx!=0 or dz!=0] + [[0, 1, 0]],
}

_GIN_MOVES = {
    1: [[dx, 1, dz] for dx in [-1,0,1] for dz in [-1,0,1]] + \
       [[dx, -1, dz] for dx in [-1,0,1] for dz in [-1,0,1] if dx!=0 or dz!=0],
    -1: [[dx, -1, dz] for dx in [-1,0,1] for dz in [-1,0,1]] + \
        [[dx, 1, dz] for dx in [-1,0,1] for dz in [-1,0,1] if dx!=0 or dz!=0],
}

# 飛車・角の動きに含まれない隣接方向 (王手判定の高速化用)
_ALL_ADJACENT_FILTERED = [d for d in _ALL_ADJACENT if d not in _HI_DIRS and d not in _KAKU_DIRS]
_HI_DIRS_SET = set(tuple(d) for d in _HI_DIRS)
_KAKU_DIRS_SET = set(tuple(d) for d in _KAKU_DIRS)
_KIN_MOVES_SET = {
    1: set(tuple(d) for d in _KIN_MOVES[1]),
    -1: set(tuple(d) for d in _KIN_MOVES[-1]),
}
_GIN_MOVES_SET = {
    1: set(tuple(d) for d in _GIN_MOVES[1]),
    -1: set(tuple(d) for d in _GIN_MOVES[-1]),
}

# プレイヤーの定義
class Player(IntEnum):
    NONE = 0
    SENTE = 1 # 先手 (y=8からy=0へ進む)
    GOTE = 2  # 後手 (y=0からy=8へ進む)

# 駒の定義
class PieceType(IntEnum):
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

# 成り変換マップ
PROMOTION_MAP = {
    PieceType.FU: PieceType.NARI_FU,
    PieceType.KYO: PieceType.NARI_KYO,
    PieceType.KEI: PieceType.NARI_KEI,
    PieceType.GIN: PieceType.NARI_GIN,
    PieceType.KAKU: PieceType.UMA,
    PieceType.HI: PieceType.RYU,
}

# 駒の価値 (簡易版)
PIECE_VALUES = {
    PieceType.NONE: 0,
    PieceType.FU: 10,
    PieceType.KYO: 30,
    PieceType.KEI: 40,
    PieceType.GIN: 50,
    PieceType.KIN: 60,
    PieceType.KAKU: 80,
    PieceType.HI: 100,
    PieceType.OU: 10000,
    PieceType.NARI_FU: 70,
    PieceType.NARI_KYO: 70,
    PieceType.NARI_KEI: 70,
    PieceType.NARI_GIN: 70,
    PieceType.UMA: 120,
    PieceType.RYU: 150,
}

HAND_PIECE_ORDER = [
    PieceType.FU, PieceType.KYO, PieceType.KEI,
    PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI
]
HAND_INDEX = {p: i for i, p in enumerate(HAND_PIECE_ORDER)}
DROP_TYPES = [PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI]
DROP_INDEX = {p: i for i, p in enumerate(DROP_TYPES)}


def _build_fixed_move_deltas():
    deltas = set()

    # Slider deltas (rook-like and bishop-like rays)
    for dx, dy, dz in _HI_DIRS + _KAKU_DIRS:
        for dist in range(1, 9):
            deltas.add((dx * dist, dy * dist, dz * dist))

    # King-only space diagonals (all three axes non-zero)
    for dx in (-1, 1):
        for dy in (-1, 1):
            for dz in (-1, 1):
                deltas.add((dx, dy, dz))

    # Knight deltas for both players
    knight = [
        (1, -2, 0), (-1, -2, 0), (0, -2, 1), (0, -2, -1),
        (1, 2, 0), (-1, 2, 0), (0, 2, 1), (0, 2, -1),
    ]
    deltas.update(knight)

    out = sorted(deltas)
    return out, {d: i for i, d in enumerate(out)}


_MOVE_DELTAS, _DELTA_TO_INDEX = _build_fixed_move_deltas()

_ZOBRIST_RNG = np.random.default_rng(20260321)
_ZOBRIST_TABLE = _ZOBRIST_RNG.integers(
    low=0,
    high=np.iinfo(np.uint64).max,
    size=(15, 3, 9, 9, 9),
    dtype=np.uint64,
)


def _zobrist_value(ptype, owner, x, y, z):
    if owner == Player.NONE or ptype == PieceType.NONE:
        return np.uint64(0)
    return _ZOBRIST_TABLE[int(ptype), int(owner), int(x), int(y), int(z)]

class Shogi3DEnv(gym.Env):
    def __init__(self, opening_random_plies=0):
        super(Shogi3DEnv, self).__init__()
        self.use_cpp_kingcheck = bool(int(os.getenv("SHOGI_USE_CPP_KINGCHECK", "0")))
        self.use_cpp_legal_moves = bool(int(os.getenv("SHOGI_USE_CPP_LEGAL_MOVES", "0")))
        self.use_cpp_pinned = bool(int(os.getenv("SHOGI_USE_CPP_PINNED", "1")))
        self.cpp_filter_level = int(os.getenv("SHOGI_CPP_FILTER_LEVEL", "0"))
        self.use_legal_moves_cache = bool(int(os.getenv("SHOGI_USE_LEGAL_MOVES_CACHE", "1")))
        self.cpp_strict_filter = bool(int(os.getenv("SHOGI_CPP_STRICT_FILTER", "0")))
        self.deterministic_move_order = bool(int(os.getenv("SHOGI_DETERMINISTIC_MOVE_ORDER", "0")))
        self.fast_drop_generation = bool(int(os.getenv("SHOGI_FAST_DROP_GENERATION", "1")))
        self.use_fixed_action_space = bool(int(os.getenv("SHOGI_FIXED_ACTION_SPACE", "0")))
        self.use_uint8_obs = bool(int(os.getenv("SHOGI_USE_UINT8_OBS", "1")))
        self.enable_tactical_reward = bool(int(os.getenv("SHOGI_TACTICAL_REWARD", "1")))
        self.hanging_penalty_weight = float(os.getenv("SHOGI_HANGING_PENALTY_WEIGHT", "0.35"))
        self.free_capture_bonus_weight = float(os.getenv("SHOGI_FREE_CAPTURE_BONUS_WEIGHT", "0.20"))
        self.free_capture_bonus_cap = float(os.getenv("SHOGI_FREE_CAPTURE_BONUS_CAP", "0.50"))
        self.hanging_penalty_cap = float(os.getenv("SHOGI_HANGING_PENALTY_CAP", "0.60"))
        obs_dtype = np.uint8 if self.use_uint8_obs else np.float32
        board_high = 1 if self.use_uint8_obs else 1.0
        hand_high = 81 if self.use_uint8_obs else 81.0
        player_high = 1 if self.use_uint8_obs else 1.0
        # 観測空間: 盤面(3D-CNN) + 持ち駒ベクトル(MLP用)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=board_high, shape=(30, 9, 9, 9), dtype=obs_dtype),
            "hand_sente": spaces.Box(low=0, high=hand_high, shape=(7,), dtype=obs_dtype), # FU, KYO, KEI, GIN, KIN, KAKU, HI
            "hand_gote": spaces.Box(low=0, high=hand_high, shape=(7,), dtype=obs_dtype),
            "current_player": spaces.Box(low=0, high=player_high, shape=(1,), dtype=obs_dtype)
        })
        
        # 行動空間のフラット化 (Discrete)
        # 移動: 9^3 (from) * 9^3 (to) * 2 (promote) は巨大すぎるため、
        # 相対的な動きに基づいてインデックス化するのが一般的ですが、
        # ここでは「(from_idx, to_idx, promote)」と「(drop_idx, piece_type)」を
        # 現実的な範囲で表現できるよう、少し工夫した Discrete 空間にします。
        
        # 簡易的な実装として、以下の合計をアクション数とします：
        # 1. 盤上の駒の移動: 729 (from) * 1 (to_offset) * 2 (promote) ... これも多い
        # 結局、MaskablePPO を使うのであれば、「その局面の合法手リスト」への
        # インデックスを直接選ばせるのが最も確実です。
        # ただし、モデル作成時に空間サイズが決まっている必要があるため、
        # 合法手の最大数（想定 1500 程度）を固定サイズとして定義します。
        self.max_legal_moves = 1500
        self.num_move_deltas = len(_MOVE_DELTAS)
        self.fixed_move_action_count = 729 * self.num_move_deltas * 2
        self.fixed_drop_action_base = self.fixed_move_action_count
        self.fixed_drop_action_count = 729 * len(DROP_TYPES)
        self.fixed_action_space_n = self.fixed_move_action_count + self.fixed_drop_action_count
        if self.use_fixed_action_space:
            self.action_space = spaces.Discrete(self.fixed_action_space_n)
        else:
            self.action_space = spaces.Discrete(self.max_legal_moves)
        self.opening_random_plies = int(opening_random_plies)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((9, 9, 9), dtype=np.int32) # [x, y, z]
        self.owner = np.zeros((9, 9, 9), dtype=np.int8)  # Player
        self.hands = {Player.SENTE: Counter(), Player.GOTE: Counter()}
        # 駒の位置を管理する辞書 (Player -> set of (x,y,z))
        self.piece_positions = {Player.SENTE: set(), Player.GOTE: set()}
        self.king_positions = {Player.SENTE: None, Player.GOTE: None}
        self.attack_map = np.zeros((3, 9, 9, 9), dtype=np.int16)
        self.piece_move_cache = {}
        self.piece_cache_owner = {}
        self.piece_watch_squares = {}
        self.ray_watchers = defaultdict(set)
        self.turn_count = 0
        self.current_player = Player.SENTE
        self.zobrist_hash = np.uint64(0)
        self.repetition_history_cap = int(os.getenv("SHOGI_REPETITION_HISTORY_CAP", "6000"))
        self.repetition_truncate_count = int(os.getenv("SHOGI_REPETITION_TRUNCATE_COUNT", "4"))
        self.position_version = 0
        self.legal_moves_cache = {}
        self.legal_action_cache = {}
        self.legal_cache_hits = 0
        self.legal_cache_misses = 0
        self.game_record_moves = []
        
        self._init_board()
        self._rebuild_all_caches()
        self.repetition_map = {}
        self.repetition_history = deque()

        # 学習多様性のための序盤ランダム化（推論環境では 0 推奨）
        if self.opening_random_plies > 0:
            for _ in range(self.opening_random_plies):
                legal_map = self._get_legal_action_map(self.current_player)
                if not legal_map:
                    break
                legal_actions = list(legal_map.keys())
                idx = int(legal_actions[int(self.np_random.integers(0, len(legal_actions)))])
                _, _, term, trunc, _ = self.step(idx)
                if term or trunc:
                    break
        return self._get_obs(), {}

    def _get_pos_hash(self):
        hs = tuple(int(v) for v in self._hand_counts_vec(Player.SENTE))
        hg = tuple(int(v) for v in self._hand_counts_vec(Player.GOTE))
        return (int(self.zobrist_hash), hs, hg, int(self.current_player))

    def _recompute_zobrist_hash(self):
        zh = np.uint64(0)
        xs, ys, zs = np.where((self.owner != Player.NONE) & (self.board > PieceType.NONE))
        for x, y, z in zip(xs, ys, zs):
            p = self.board[x, y, z]
            o = self.owner[x, y, z]
            zh ^= _zobrist_value(p, o, x, y, z)
        self.zobrist_hash = zh

    def _init_board(self):
        # board.js の PLANE_1_7 等に基づき配置
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

        # 後手 (y=0,1,2)
        self._apply_plane(0, plane_1_7, Player.GOTE)
        self._apply_plane(1, plane_2_8, Player.GOTE)
        for x in range(9):
            for z in range(9):
                self.board[x, 2, z] = PieceType.FU
                self.owner[x, 2, z] = Player.GOTE
                self.piece_positions[Player.GOTE].add((x, 2, z))

        # 先手 (y=8,7,6)
        self._apply_plane(8, plane_1_7, Player.SENTE)
        self._apply_plane(7, plane_2_8, Player.SENTE)
        for x in range(9):
            for z in range(9):
                self.board[x, 6, z] = PieceType.FU
                self.owner[x, 6, z] = Player.SENTE
                self.piece_positions[Player.SENTE].add((x, 6, z))

    def _apply_plane(self, y, data, player):
        for z in range(9):
            for x in range(9):
                t = data[z][x]
                if t != 0:
                    self.board[x, y, z] = t
                    self.owner[x, y, z] = player
                    self.piece_positions[player].add((x, y, z))
                    if t == PieceType.OU:
                        self.king_positions[player] = (x, y, z)

    def _coord_to_index(self, x, y, z):
        return int(x) * 81 + int(y) * 9 + int(z)

    def _move_to_action_index(self, move):
        fx, fy, fz, tx, ty, tz, promote, is_drop, drop_type = move
        if is_drop:
            to_idx = self._coord_to_index(tx, ty, tz)
            return int(self.fixed_drop_action_base + int(drop_type) * 729 + to_idx)

        delta = (int(tx - fx), int(ty - fy), int(tz - fz))
        d_idx = _DELTA_TO_INDEX.get(delta)
        if d_idx is None:
            return None
        from_idx = self._coord_to_index(fx, fy, fz)
        return int(((from_idx * self.num_move_deltas + d_idx) * 2) + (1 if promote else 0))

    def _get_legal_action_map(self, player):
        cache_key = (int(player), self.position_version)
        if self.use_legal_moves_cache:
            cached = self.legal_action_cache.get(cache_key)
            if cached is not None:
                return cached

        moves = self.get_legal_moves(player)
        action_map = {}
        if self.use_fixed_action_space:
            for m in moves:
                a = self._move_to_action_index(m)
                if a is None:
                    continue
                if 0 <= a < self.action_space.n and a not in action_map:
                    action_map[a] = m
        else:
            for i, m in enumerate(moves[:self.max_legal_moves]):
                action_map[int(i)] = m

        if self.use_legal_moves_cache:
            self.legal_action_cache[cache_key] = action_map
        return action_map

    def _has_any_legal_move(self, player):
        return len(self.get_legal_moves(player, limit=1)) > 0

    def step(self, action_idx):
        # action_idx は固定ID（推奨）または旧来の合法手インデックス
        acting_player = self.current_player
        chosen_move = None
        if self.use_fixed_action_space:
            legal_action_map = self._get_legal_action_map(self.current_player)
            if not legal_action_map:
                terminated = True
                truncated = False
                in_check = self._is_king_in_check(self.current_player)
                winner = Player.GOTE if self.current_player == Player.SENTE else Player.SENTE
                info = {
                    "reason": "mate" if in_check else "no_legal_moves",
                    "winner": int(winner),
                    "game_record": list(self.game_record_moves),
                }
                reward = -20.0 if in_check else 0.0
                return self._get_obs(), reward, terminated, truncated, info
            chosen_move = legal_action_map.get(int(action_idx))
            if chosen_move is None:
                # マスク不整合時は最初の合法手へ丸めて継続
                chosen_move = next(iter(legal_action_map.values()))
        else:
            moves = self.get_legal_moves(self.current_player)
            if len(moves) == 0:
                # 手番側に合法手がない場合（詰み等）
                terminated = True
                truncated = False
                in_check = self._is_king_in_check(self.current_player)
                winner = Player.GOTE if self.current_player == Player.SENTE else Player.SENTE
                info = {
                    "reason": "mate" if in_check else "no_legal_moves",
                    "winner": int(winner),
                    "game_record": list(self.game_record_moves),
                }
                reward = -20.0 if in_check else 0.0
                return self._get_obs(), reward, terminated, truncated, info
            if action_idx >= len(moves):
                action_idx = 0
            chosen_move = moves[action_idx]

        # 合法手の取得
        fx, fy, fz, tx, ty, tz, promote, is_drop, drop_type = chosen_move
        move_record = {
            "fx": int(fx),
            "fy": int(fy),
            "fz": int(fz),
            "tx": int(tx),
            "ty": int(ty),
            "tz": int(tz),
            "promote": bool(promote),
            "is_drop": bool(is_drop),
            "drop_type": int(drop_type),
            "player": int(acting_player),
            "moving_piece": None,
            "captured_piece": None,
            "position_hash_after": None,
            "repetition_count_after": 1,
        }
        self.game_record_moves.append(move_record)
        
        # 報酬の基本計算
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if is_drop:
            # 打ちのロジック
            ptype = DROP_TYPES[drop_type]
            move_record["moving_piece"] = int(ptype)

            self.zobrist_hash ^= _zobrist_value(ptype, self.current_player, tx, ty, tz)
                
            self.board[tx, ty, tz] = ptype
            self.owner[tx, ty, tz] = self.current_player
            self.piece_positions[self.current_player].add((tx, ty, tz))
            self.hands[self.current_player][ptype] -= 1
            if self.hands[self.current_player][ptype] <= 0:
                self.hands[self.current_player].pop(ptype, None)
            self._invalidate_and_refresh({(tx, ty, tz)}, set())
        else:
            # 移動の実行
            target_ptype = self.board[tx, ty, tz]
            target_owner = self.owner[tx, ty, tz]
            moving_ptype_before_move = self.board[fx, fy, fz]
            moving_ptype_after_move = moving_ptype_before_move
            move_record["moving_piece"] = int(moving_ptype_before_move)
            if target_owner != Player.NONE:
                move_record["captured_piece"] = int(target_ptype)
            
            if target_owner != Player.NONE:
                self.zobrist_hash ^= _zobrist_value(target_ptype, target_owner, tx, ty, tz)
                val = PIECE_VALUES[target_ptype] / 100.0
                reward += val

                if target_ptype == PieceType.OU:
                    reward += 10.0
                    terminated = True
                    info["reason"] = "ou_capture"
                    info["ou_capture_by"] = int(moving_ptype_before_move)
                    info["winner"] = int(acting_player)
                
                # 相手のリストから削除
                self.piece_positions[target_owner].discard((tx, ty, tz))
                if target_ptype == PieceType.OU:
                    self.king_positions[target_owner] = None
                
                raw_type = target_ptype
                for promo, raw in [(PieceType.NARI_FU, PieceType.FU), (PieceType.NARI_KYO, PieceType.KYO), 
                                   (PieceType.NARI_KEI, PieceType.KEI), (PieceType.NARI_GIN, PieceType.GIN),
                                   (PieceType.UMA, PieceType.KAKU), (PieceType.RYU, PieceType.HI)]:
                    if target_ptype == promo: raw_type = raw; break
                self.hands[self.current_player][raw_type] += 1

            moving_ptype = moving_ptype_before_move
            is_in_zone = self._is_promotion_zone(fy, self.current_player) or self._is_promotion_zone(ty, self.current_player)
            
            # 強制成りチェック
            forward_limit = True if (self.current_player == Player.SENTE and ty == 0) or (self.current_player == Player.GOTE and ty == 8) else False
            kei_limit = True if (self.current_player == Player.SENTE and ty <= 1) or (self.current_player == Player.GOTE and ty >= 7) else False
            
            forced_promote = False
            if moving_ptype == PieceType.FU and forward_limit: forced_promote = True
            if moving_ptype == PieceType.KYO and forward_limit: forced_promote = True
            if moving_ptype == PieceType.KEI and kei_limit: forced_promote = True
            
            if (promote or forced_promote) and is_in_zone and moving_ptype in PROMOTION_MAP:
                moving_ptype = PROMOTION_MAP[moving_ptype]
            moving_ptype_after_move = moving_ptype

            self.zobrist_hash ^= _zobrist_value(moving_ptype_before_move, self.current_player, fx, fy, fz)
            self.zobrist_hash ^= _zobrist_value(moving_ptype_after_move, self.current_player, tx, ty, tz)
            
            self.board[tx, ty, tz] = moving_ptype
            self.owner[tx, ty, tz] = self.current_player
            self.piece_positions[self.current_player].add((tx, ty, tz))
            
            self.board[fx, fy, fz] = PieceType.NONE
            self.owner[fx, fy, fz] = Player.NONE
            self.piece_positions[self.current_player].discard((fx, fy, fz))
            if moving_ptype_before_move == PieceType.OU:
                self.king_positions[self.current_player] = (tx, ty, tz)

            self._invalidate_and_refresh({(fx, fy, fz), (tx, ty, tz)}, {(fx, fy, fz), (tx, ty, tz)})

        # 実局面が更新されたので合法手キャッシュを無効化
        self.position_version += 1
        self.legal_moves_cache.clear()
        self.legal_action_cache.clear()
        reward += self._tactical_delta_after_move(acting_player, tx, ty, tz)

        # 相手の状態チェック (王手・詰み報酬)
        opp = Player.GOTE if self.current_player == Player.SENTE else Player.SENTE
        if (not terminated) and self._is_king_in_check(opp):
            # 王手報酬は Reward Farming 対策で削除（または極小化）
            # 詰み判定 (合法手が一つもなければ詰み)
            if not self._has_any_legal_move(opp):
                reward += 20.0 # 詰み勝利報酬
                terminated = True
                info["reason"] = "mate"
                info["winner"] = int(acting_player)

        # ターン交代
        self.current_player = opp
        self.turn_count += 1
        
        # 1000手以降のペナルティ
        if self.turn_count >= 1000:
            reward -= 0.001

        # 制限は事実上撤廃 (安全策として10000手までは許可)
        # 手数制限の修正 (300 -> 10000)
        if self.turn_count >= 10000:
            truncated = True
            info.setdefault("reason", "timeout")
            info.setdefault("winner", 0)

        # --- 千日手（繰り返し）チェック ---
        pos_hash = self._get_pos_hash()
        self.repetition_map[pos_hash] = self.repetition_map.get(pos_hash, 0) + 1
        self.repetition_history.append(pos_hash)
        if self.repetition_history_cap > 0 and len(self.repetition_history) > self.repetition_history_cap:
            old_hash = self.repetition_history.popleft()
            old_cnt = self.repetition_map.get(old_hash, 0)
            if old_cnt <= 1:
                self.repetition_map.pop(old_hash, None)
            else:
                self.repetition_map[old_hash] = old_cnt - 1
        rep_count = self.repetition_map[pos_hash]
        move_record["position_hash_after"] = int(pos_hash[0]) if isinstance(pos_hash, tuple) else int(pos_hash)
        move_record["repetition_count_after"] = int(rep_count)
        
        if rep_count >= 2:
            # 繰り返しが発生したら累積的にペナルティ
            reward -= 0.01 * (rep_count - 1)
        
        if rep_count >= self.repetition_truncate_count:
            # 規定回数以上の繰り返しで打ち切り
            truncated = True
            reward -= 2.0 # 強めのペナルティを与えてループを確実に避けさせる
            info["reason"] = "sennichite"
            info["winner"] = 0

        if terminated or truncated:
            info.setdefault("winner", 0)
            info["game_record"] = list(self.game_record_moves)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        obs_dtype = np.uint8 if self.use_uint8_obs else np.float32
        one_val = 1 if self.use_uint8_obs else 1.0
        obs_board = np.zeros((30, 9, 9, 9), dtype=obs_dtype)
        sente_mask = (self.owner == Player.SENTE) & (self.board > PieceType.NONE)
        if np.any(sente_mask):
            xs, ys, zs = np.where(sente_mask)
            pcs = self.board[sente_mask] - 1
            obs_board[pcs, xs, ys, zs] = one_val

        gote_mask = (self.owner == Player.GOTE) & (self.board > PieceType.NONE)
        if np.any(gote_mask):
            xg, yg, zg = np.where(gote_mask)
            pcg = self.board[gote_mask] - 1 + 15
            obs_board[pcg, xg, yg, zg] = one_val
        
        def get_hand_vec(player):
            vec = np.zeros(7, dtype=obs_dtype)
            for p, cnt in self.hands[player].items():
                idx = HAND_INDEX.get(p)
                if idx is not None:
                    vec[idx] += int(cnt) if self.use_uint8_obs else float(cnt)
            return vec

        return {
            "board": obs_board,
            "hand_sente": get_hand_vec(Player.SENTE),
            "hand_gote": get_hand_vec(Player.GOTE),
            "current_player": np.array([1 if self.current_player == Player.SENTE else 0], dtype=obs_dtype)
        }

    def _piece_reward_unit(self, ptype):
        return PIECE_VALUES.get(ptype, 0) / 100.0

    def _tactical_delta_after_move(self, acting_player, tx, ty, tz):
        if not self.enable_tactical_reward:
            return 0.0
        if not (0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
            return 0.0

        moved_piece = self.board[tx, ty, tz]
        if moved_piece == PieceType.NONE:
            return 0.0

        opp = Player.GOTE if acting_player == Player.SENTE else Player.SENTE
        delta = 0.0

        # 移動先が敵に利かれていて味方の利きがないなら「ただ取られ」ペナルティ。
        if moved_piece != PieceType.OU:
            attacked = self.attack_map[opp, tx, ty, tz] > 0
            defended = self.attack_map[acting_player, tx, ty, tz] > 0
            if attacked and not defended:
                hanging = self._piece_reward_unit(moved_piece) * self.hanging_penalty_weight
                delta -= min(hanging, self.hanging_penalty_cap)

        # 移動した駒が狙える敵駒のうち「無防備」な最大価値をボーナスにする。
        best_free_capture = 0.0
        moved_attacks = self.piece_move_cache.get((tx, ty, tz), ())
        for ax, ay, az in moved_attacks:
            if self.owner[ax, ay, az] != opp:
                continue
            target_ptype = self.board[ax, ay, az]
            if target_ptype in (PieceType.NONE, PieceType.OU):
                continue
            if self.attack_map[opp, ax, ay, az] <= 0:
                best_free_capture = max(best_free_capture, self._piece_reward_unit(target_ptype))

        if best_free_capture > 0.0:
            gain = best_free_capture * self.free_capture_bonus_weight
            delta += min(gain, self.free_capture_bonus_cap)

        return float(delta)

    # --- アクションマスク (MaskablePPO 用) ---
    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        legal_action_map = self._get_legal_action_map(self.current_player)
        if legal_action_map:
            idx = np.fromiter(legal_action_map.keys(), dtype=np.int64)
            mask[idx] = True
        return mask

    # --- 移動ルールロジック (Internal) ---
    def _hand_counts_vec(self, player):
        vec = np.zeros(7, dtype=np.int16)
        for p, cnt in self.hands[player].items():
            idx = HAND_INDEX.get(p)
            if idx is not None:
                vec[idx] += int(cnt)
        return vec

    def _compute_pinned_pieces(self, player, currently_in_check):
        # {(px,py,pz): (dx,dy,dz,attacker_t)}
        # dx,dy,dz は王から見たピン方向、attacker_t はその方向での攻撃駒までの距離。
        pinned_info = {}
        if currently_in_check:
            return pinned_info

        if self.use_cpp_pinned and cpp_kingcheck is not None and cpp_kingcheck.cpp_available():
            cpp_rows = cpp_kingcheck.compute_pinned_pieces(self.board, self.owner, int(player), bool(currently_in_check))
            if cpp_rows is not None:
                for row in cpp_rows:
                    px, py, pz, dx, dy, dz, attacker_t = row
                    pinned_info[(int(px), int(py), int(pz))] = (int(dx), int(dy), int(dz), int(attacker_t))
                return pinned_info

        king = self.king_positions.get(player)
        if king is None:
            return pinned_info
        kx, ky, kz = king

        opp = Player.GOTE if player == Player.SENTE else Player.SENTE
        dy_opp = -1 if opp == Player.SENTE else 1
        for dx, dy, dz in _HI_DIRS + _KAKU_DIRS:
            tx, ty, tz = kx + dx, ky + dy, kz + dz
            first_piece = None
            dist = 1
            while 0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9:
                o = self.owner[tx, ty, tz]
                if o != Player.NONE:
                    if o == player:
                        if first_piece is None:
                            first_piece = (tx, ty, tz)
                        else:
                            break
                    else:
                        if first_piece:
                            p = self.board[tx, ty, tz]
                            is_pinned = False
                            ray = (dx, dy, dz)
                            if p in [PieceType.HI, PieceType.RYU] and ray in _HI_DIRS_SET:
                                is_pinned = True
                            elif p in [PieceType.KAKU, PieceType.UMA] and ray in _KAKU_DIRS_SET:
                                is_pinned = True
                            elif p == PieceType.KYO and dx == 0 and dy == -dy_opp and dz == 0:
                                is_pinned = True
                            if is_pinned:
                                pinned_info[first_piece] = (dx, dy, dz, dist)
                        break
                tx += dx
                ty += dy
                tz += dz
                dist += 1
        return pinned_info

    def _ray_param_from_king(self, king, pos, ray_dir):
        kx, ky, kz = king
        px, py, pz = pos
        dx, dy, dz = ray_dir

        t = None
        for delta, d in ((px - kx, dx), (py - ky, dy), (pz - kz, dz)):
            if d == 0:
                if delta != 0:
                    return None
                continue
            if delta % d != 0:
                return None
            v = delta // d
            if v <= 0:
                return None
            if t is None:
                t = v
            elif t != v:
                return None
        return t

    def get_legal_moves(self, player, limit=None):
        cache_key = (int(player), self.position_version)
        if self.use_legal_moves_cache and limit is None:
            cached_moves = self.legal_moves_cache.get(cache_key)
            if cached_moves is not None:
                self.legal_cache_hits += 1
                return cached_moves
        self.legal_cache_misses += 1

        if self.use_cpp_legal_moves and cpp_kingcheck is not None and cpp_kingcheck.cpp_available():
            hm = self._hand_counts_vec(player)
            cpp_moves = cpp_kingcheck.generate_legal_moves(self.board, self.owner, hm, int(player))
            if cpp_moves is not None:
                # C++ legal-move outputをどこまでPython側で再検証するかを段階化する。
                # level 0: 既存挙動(安全性重視)
                # level 1: 盤面整合の軽量チェックのみ(王手/ピン再計算は省略)
                # level 2: C++出力を全面信頼(最速)
                cpp_filter_level = int(self.cpp_filter_level)
                if cpp_filter_level < 0:
                    cpp_filter_level = 0
                if cpp_filter_level > 2:
                    cpp_filter_level = 2
                if self.cpp_strict_filter:
                    cpp_filter_level = 0

                use_python_pin_filter = (cpp_filter_level == 0)
                use_python_safety = (cpp_filter_level == 0)
                use_structural_validation = (cpp_filter_level <= 1)

                king = None
                currently_in_check = False
                pinned_info = {}
                if use_python_pin_filter or use_python_safety:
                    currently_in_check = self._is_king_in_check(player)
                if use_python_pin_filter:
                    king = self.king_positions.get(player)
                    pinned_info = self._compute_pinned_pieces(player, currently_in_check)

                filtered = []
                src_cache = {} if self.cpp_strict_filter else None
                for m in cpp_moves:
                    fx, fy, fz, tx, ty, tz, promote, is_drop, dt = m
                    if is_drop:
                        if use_structural_validation:
                            if not (0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
                                continue
                            if not (0 <= dt < len(DROP_TYPES)):
                                continue
                            if self.board[tx, ty, tz] != PieceType.NONE:
                                continue
                        if use_python_safety and currently_in_check:
                            # dt mapping: FU,KYO,KEI,GIN,KIN,KAKU,HI
                            ptype = [PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI][dt]
                            if not self._is_drop_safe(ptype, tx, ty, tz, player):
                                continue
                        filtered.append(m)
                        continue

                    skey = (fx, fy, fz)
                    if self.cpp_strict_filter:
                        # Defensive validation for C++ output consistency.
                        if not (0 <= fx < 9 and 0 <= fy < 9 and 0 <= fz < 9 and 0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
                            continue
                        if self.owner[fx, fy, fz] != player:
                            continue
                        if self.owner[tx, ty, tz] == player:
                            continue

                        meta = src_cache.get(skey)
                        if meta is None:
                            pseudo = self._compute_piece_moves_uncached(fx, fy, fz)
                            pseudo_set = set((int(px), int(py), int(pz)) for px, py, pz in pseudo)
                            is_ou = self.board[fx, fy, fz] == PieceType.OU
                            pin_meta = pinned_info.get(skey)
                            meta = (pseudo_set, is_ou, pin_meta)
                            src_cache[skey] = meta

                        pseudo_set, is_ou, pin_meta = meta
                        if (tx, ty, tz) not in pseudo_set:
                            continue
                    else:
                        if use_structural_validation:
                            if not (0 <= fx < 9 and 0 <= fy < 9 and 0 <= fz < 9 and 0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9):
                                continue
                            if self.owner[fx, fy, fz] != player:
                                continue
                            if self.owner[tx, ty, tz] == player:
                                continue
                            is_ou = self.board[fx, fy, fz] == PieceType.OU
                            pin_meta = pinned_info.get(skey) if use_python_pin_filter else None
                        else:
                            is_ou = False
                            pin_meta = None

                    if use_python_pin_filter:
                        is_pinned = pin_meta is not None
                        if is_pinned and (not currently_in_check) and (not is_ou):
                            pdx, pdy, pdz, attacker_t = pin_meta
                            t = self._ray_param_from_king(king, (tx, ty, tz), (pdx, pdy, pdz))
                            if t is None or t > attacker_t:
                                continue
                            need_safety = False
                        else:
                            need_safety = currently_in_check or is_ou or is_pinned
                    else:
                        need_safety = False

                    if use_python_safety and need_safety:
                        if not self._is_move_safe(fx, fy, fz, tx, ty, tz, player):
                            continue
                    filtered.append(m)
                    if limit is not None and len(filtered) >= limit:
                        return filtered
                if self.use_legal_moves_cache and limit is None:
                    self.legal_moves_cache[cache_key] = filtered
                if limit is not None:
                    return filtered[:limit]
                return filtered

        moves = []
        # 王の位置と「ピンされている可能性のある駒」の特定
        king = self.king_positions.get(player)
        if king is None:
            return moves
        kx, ky, kz = king
        
        currently_in_check = self._is_king_in_check(player)
        pinned_info = self._compute_pinned_pieces(player, currently_in_check)

        piece_iter = sorted(self.piece_positions[player]) if self.deterministic_move_order else self.piece_positions[player]
        # 盤面の駒による移動
        for x, y, z in piece_iter:
            is_ou = (self.board[x, y, z] == PieceType.OU)
            pin_meta = pinned_info.get((x, y, z))
            is_pinned = pin_meta is not None
            
            # Use uncached move generation for legality to avoid stale-cache artifacts.
            for tx, ty, tz in self._compute_piece_moves_uncached(x, y, z):
                # 安全性チェックのスキップ条件:
                # 1. 王自身ではない 2. 王手されていない 3. ピンされていない駒
                if is_pinned and (not currently_in_check) and (not is_ou):
                    pdx, pdy, pdz, attacker_t = pin_meta
                    t = self._ray_param_from_king(king, (tx, ty, tz), (pdx, pdy, pdz))
                    if t is None or t > attacker_t:
                        continue
                elif is_ou or currently_in_check:
                    if not self._is_move_safe(x, y, z, tx, ty, tz, player):
                        continue
                
                moves.append([x, y, z, tx, ty, tz, False, False, 0])
                if limit is not None and len(moves) >= limit:
                    return moves
                if self._is_promotion_zone(y, player) or self._is_promotion_zone(ty, player):
                    if self.board[x, y, z] in PROMOTION_MAP:
                        moves.append([x, y, z, tx, ty, tz, True, False, 0])
                        if limit is not None and len(moves) >= limit:
                            return moves

        # 持ち駒による打ち
        unique_hand = set(self.hands[player].keys())
        piece_map_inv = {PieceType.FU: 0, PieceType.KYO: 1, PieceType.KEI: 2, PieceType.GIN: 3, PieceType.KIN: 4, PieceType.KAKU: 5, PieceType.HI: 6}
        
        # 自分が既に歩を置いている (x, z) のセットを事前に作る (二歩チェック高速化)
        nifu_xz = set()
        for px, py, pz in piece_iter:
            if self.board[px, py, pz] == PieceType.FU:
                nifu_xz.add((px, pz))

        if self.fast_drop_generation:
            empty_indices = np.where(self.board == PieceType.NONE)
            empty_squares = list(zip(empty_indices[0], empty_indices[1], empty_indices[2]))
        else:
            empty_squares = None
        
        for ptype in HAND_PIECE_ORDER:
            if ptype not in unique_hand:
                continue
            dt = piece_map_inv.get(ptype, 0)
            is_fu = (ptype == PieceType.FU)
            if empty_squares is not None:
                for x, y, z in empty_squares:
                    if is_fu and (x, z) in nifu_xz:
                        continue

                    # 行き場のない駒のチェック
                    forward_limit = True if (player == Player.SENTE and y == 0) or (player == Player.GOTE and y == 8) else False
                    kei_limit = True if (player == Player.SENTE and y <= 1) or (player == Player.GOTE and y >= 7) else False
                    if is_fu and forward_limit:
                        continue
                    if ptype == PieceType.KYO and forward_limit:
                        continue
                    if ptype == PieceType.KEI and kei_limit:
                        continue

                    if not currently_in_check:
                        moves.append([0, 0, 0, x, y, z, False, True, dt])
                        if limit is not None and len(moves) >= limit:
                            return moves
                    else:
                        if self._is_drop_safe(ptype, x, y, z, player):
                            moves.append([0, 0, 0, x, y, z, False, True, dt])
                            if limit is not None and len(moves) >= limit:
                                return moves
            else:
                for x in range(9):
                    for z in range(9):
                        if is_fu and (x, z) in nifu_xz:
                            continue
                        for y in range(9):
                            if self.board[x, y, z] == PieceType.NONE:
                                # 行き場のない駒のチェック
                                forward_limit = True if (player == Player.SENTE and y == 0) or (player == Player.GOTE and y == 8) else False
                                kei_limit = True if (player == Player.SENTE and y <= 1) or (player == Player.GOTE and y >= 7) else False
                                if is_fu and forward_limit: continue
                                if ptype == PieceType.KYO and forward_limit: continue
                                if ptype == PieceType.KEI and kei_limit: continue

                                if not currently_in_check:
                                    moves.append([0, 0, 0, x, y, z, False, True, dt])
                                    if limit is not None and len(moves) >= limit:
                                        return moves
                                else:
                                    if self._is_drop_safe(ptype, x, y, z, player):
                                        moves.append([0, 0, 0, x, y, z, False, True, dt])
                                        if limit is not None and len(moves) >= limit:
                                            return moves
        if self.use_legal_moves_cache and limit is None:
            self.legal_moves_cache[cache_key] = moves
        return moves

    def get_legal_cache_stats(self):
        total = self.legal_cache_hits + self.legal_cache_misses
        return {
            "hits": int(self.legal_cache_hits),
            "misses": int(self.legal_cache_misses),
            "hit_rate": float(self.legal_cache_hits / total) if total > 0 else 0.0,
        }

    def _get_piece_moves(self, x, y, z):
        if not (0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9):
            return []
        if self.owner[x, y, z] == Player.NONE or self.board[x, y, z] == PieceType.NONE:
            return []

        key = (x, y, z)
        cached = self.piece_move_cache.get(key)
        if cached is not None:
            return [list(m) for m in cached]

        self._register_piece_cache(key)
        cached = self.piece_move_cache.get(key, [])
        return [list(m) for m in cached]

    def _compute_piece_moves_uncached(self, x, y, z):
        ptype = self.board[x, y, z]
        player = self.owner[x, y, z]
        dy = -1 if player == Player.SENTE else 1
        
        if ptype == PieceType.OU:
            return self._step_moves(x, y, z, _ALL_ADJACENT)
        elif ptype in [PieceType.KIN, PieceType.NARI_FU, PieceType.NARI_KYO, PieceType.NARI_KEI, PieceType.NARI_GIN]:
            return self._step_moves(x, y, z, _KIN_MOVES[dy])
        elif ptype == PieceType.GIN:
            return self._step_moves(x, y, z, _GIN_MOVES[dy])
        elif ptype == PieceType.FU:
            return self._step_moves(x, y, z, [[0, dy, 0]])
        elif ptype == PieceType.KEI:
            return self._step_moves(x, y, z, [[1, 2*dy, 0], [-1, 2*dy, 0], [0, 2*dy, 1], [0, 2*dy, -1]])
        elif ptype == PieceType.KYO:
            return self._slide_moves(x, y, z, [[0, dy, 0]])
        elif ptype == PieceType.KAKU:
            return self._slide_moves(x, y, z, _KAKU_DIRS)
        elif ptype == PieceType.HI:
            return self._slide_moves(x, y, z, _HI_DIRS)
        elif ptype == PieceType.RYU:
            return self._dedup_moves(self._slide_moves(x, y, z, _HI_DIRS) + self._step_moves(x, y, z, _ALL_ADJACENT))
        elif ptype == PieceType.UMA:
            return self._dedup_moves(self._slide_moves(x, y, z, _KAKU_DIRS) + self._step_moves(x, y, z, _ALL_ADJACENT))
        return []

    def _dedup_moves(self, moves):
        seen = set()
        out = []
        for tx, ty, tz in moves:
            k = (tx, ty, tz)
            if k in seen:
                continue
            seen.add(k)
            out.append([tx, ty, tz])
        return out

    def _all_adjacent(self): return _ALL_ADJACENT
    def _kin_moves(self, dir): return _KIN_MOVES[dir]
    def _gin_moves(self, dir): return _GIN_MOVES[dir]
    def _kaku_dirs(self): return _KAKU_DIRS
    def _hi_dirs(self): return _HI_DIRS

    def _step_moves(self, fx, fy, fz, rels):
        moves = []
        player = self.owner[fx, fy, fz]
        for dx, dy, dz in rels:
            tx, ty, tz = fx+dx, fy+dy, fz+dz
            if 0<=tx<9 and 0<=ty<9 and 0<=tz<9:
                if self.owner[tx, ty, tz] != player:
                    moves.append([tx, ty, tz])
        return moves

    def _slide_moves(self, fx, fy, fz, dirs):
        moves = []
        player = self.owner[fx, fy, fz]
        for dx, dy, dz in dirs:
            tx, ty, tz = fx+dx, fy+dy, fz+dz
            while 0<=tx<9 and 0<=ty<9 and 0<=tz<9:
                if self.owner[tx, ty, tz] != Player.NONE:
                    if self.owner[tx, ty, tz] != player:
                        moves.append([tx, ty, tz])
                    break
                moves.append([tx, ty, tz])
                tx += dx
                ty += dy
                tz += dz
        return moves

    def _is_promotion_zone(self, y, player):
        if player == Player.SENTE:
            return y <= 2 # 0,1,2
        else:
            return y >= 6 # 6,7,8

    def _is_king_in_check(self, player):
        king = self.king_positions.get(player)
        if king is None:
            return True
        opp = Player.GOTE if player == Player.SENTE else Player.SENTE
        kx, ky, kz = king
        return self.attack_map[opp, kx, ky, kz] > 0

    def _is_king_in_check_ray(self, player):
        if self.use_cpp_kingcheck and cpp_kingcheck is not None and cpp_kingcheck.cpp_available():
            cpp_res = cpp_kingcheck.is_king_in_check_ray(self.board, self.owner, int(player))
            if cpp_res is not None:
                return bool(cpp_res)

        king = self.king_positions.get(player)
        if king is None:
            return True
        kx, ky, kz = king

        opp = Player.GOTE if player == Player.SENTE else Player.SENTE
        dy_opp = -1 if opp == Player.SENTE else 1

        for dx, dy, dz in _HI_DIRS:
            tx, ty, tz = kx+dx, ky+dy, kz+dz
            dist = 1
            while 0<=tx<9 and 0<=ty<9 and 0<=tz<9:
                o = self.owner[tx, ty, tz]
                if o != Player.NONE:
                    if o == opp:
                        p = self.board[tx, ty, tz]
                        if p in [PieceType.HI, PieceType.RYU]: return True
                        if p == PieceType.KYO and dx == 0 and dy == -dy_opp and dz == 0: return True
                        if dist == 1:
                            if p == PieceType.OU: return True
                            if p in [PieceType.KIN, PieceType.NARI_FU, PieceType.NARI_KYO, PieceType.NARI_KEI, PieceType.NARI_GIN]:
                                if (dx, dy, dz) in _KIN_MOVES_SET[dy_opp]: return True
                            if p == PieceType.FU and dx == 0 and dy == -dy_opp and dz == 0: return True
                            if p == PieceType.GIN and (dx, dy, dz) in _GIN_MOVES_SET[dy_opp]: return True
                    break
                tx += dx; ty += dy; tz += dz; dist += 1

        for dx, dy, dz in _KAKU_DIRS:
            tx, ty, tz = kx+dx, ky+dy, kz+dz
            dist = 1
            while 0<=tx<9 and 0<=ty<9 and 0<=tz<9:
                o = self.owner[tx, ty, tz]
                if o != Player.NONE:
                    if o == opp:
                        p = self.board[tx, ty, tz]
                        if p in [PieceType.KAKU, PieceType.UMA]: return True
                        if dist == 1:
                            if p == PieceType.OU: return True
                            if p in [PieceType.KIN, PieceType.NARI_FU, PieceType.NARI_KYO, PieceType.NARI_KEI, PieceType.NARI_GIN]:
                                if (dx, dy, dz) in _KIN_MOVES_SET[dy_opp]: return True
                            if p == PieceType.GIN and (dx, dy, dz) in _GIN_MOVES_SET[dy_opp]: return True
                    break
                tx += dx; ty += dy; tz += dz; dist += 1

        for dx, dy, dz in _ALL_ADJACENT_FILTERED:
            tx, ty, tz = kx+dx, ky+dy, kz+dz
            if 0<=tx<9 and 0<=ty<9 and 0<=tz<9:
                o = self.owner[tx, ty, tz]
                if o == opp:
                    p = self.board[tx, ty, tz]
                    if p == PieceType.OU: return True
                    if p in [PieceType.KIN, PieceType.NARI_FU, PieceType.NARI_KYO, PieceType.NARI_KEI, PieceType.NARI_GIN] and (dx, dy, dz) in _KIN_MOVES_SET[dy_opp]: return True
                    if p == PieceType.GIN and (dx, dy, dz) in _GIN_MOVES_SET[dy_opp]: return True
                    if p == PieceType.RYU or p == PieceType.UMA: return True

        for dx, dy, dz in [[1, -2*dy_opp, 0], [-1, -2*dy_opp, 0], [0, -2*dy_opp, 1], [0, -2*dy_opp, -1]]:
            tx, ty, tz = kx+dx, ky+dy, kz+dz
            if 0<=tx<9 and 0<=ty<9 and 0<=tz<9:
                if self.board[tx, ty, tz] == PieceType.KEI and self.owner[tx, ty, tz] == opp:
                    return True

        return False

    def _is_move_safe(self, fx, fy, fz, tx, ty, tz, player):
        # 実際に動かしてみて王手されていないか確認
        # 高速化のため piece_positions は触らず、board/owner と king_positions のみを一時更新する。
        saved_captured = self.board[tx, ty, tz]
        saved_captured_owner = self.owner[tx, ty, tz]
        saved_piece = self.board[fx, fy, fz]
        saved_owner = self.owner[fx, fy, fz]

        old_king = self.king_positions[player]
        if saved_piece == PieceType.OU:
            self.king_positions[player] = (tx, ty, tz)
        
        # 仮の移動
        self.board[tx, ty, tz] = saved_piece
        self.owner[tx, ty, tz] = saved_owner
        self.board[fx, fy, fz] = PieceType.NONE
        self.owner[fx, fy, fz] = Player.NONE
        
        safe = not self._is_king_in_check_ray(player)
        
        # 復元
        self.board[fx, fy, fz] = saved_piece
        self.owner[fx, fy, fz] = saved_owner
        self.board[tx, ty, tz] = saved_captured
        self.owner[tx, ty, tz] = saved_captured_owner

        if saved_piece == PieceType.OU:
            self.king_positions[player] = old_king
        
        return safe

    def _is_drop_safe(self, ptype, tx, ty, tz, player):
        # 仮に打ってみる
        saved_captured = self.board[tx, ty, tz]
        saved_captured_owner = self.owner[tx, ty, tz]
        
        self.board[tx, ty, tz] = ptype
        self.owner[tx, ty, tz] = player
        
        safe = not self._is_king_in_check_ray(player)
        
        # 復元
        self.board[tx, ty, tz] = saved_captured
        self.owner[tx, ty, tz] = saved_captured_owner
        
        return safe

    def _is_slider_piece(self, ptype):
        return ptype in [PieceType.KYO, PieceType.KAKU, PieceType.HI, PieceType.UMA, PieceType.RYU]

    def _slider_dirs_for_piece(self, ptype, player):
        dy = -1 if player == Player.SENTE else 1
        if ptype == PieceType.KYO:
            return [[0, dy, 0]]
        if ptype in [PieceType.KAKU, PieceType.UMA]:
            return _KAKU_DIRS
        if ptype in [PieceType.HI, PieceType.RYU]:
            return _HI_DIRS
        return []

    def _calc_slider_watch_squares(self, x, y, z, ptype, player):
        watch = set()
        dirs = self._slider_dirs_for_piece(ptype, player)
        for dx, dy, dz in dirs:
            tx, ty, tz = x + dx, y + dy, z + dz
            while 0 <= tx < 9 and 0 <= ty < 9 and 0 <= tz < 9:
                sq = (tx, ty, tz)
                watch.add(sq)
                if self.owner[tx, ty, tz] != Player.NONE:
                    break
                tx += dx
                ty += dy
                tz += dz
        return watch

    def _register_piece_cache(self, pos):
        x, y, z = pos
        player = self.owner[x, y, z]
        ptype = self.board[x, y, z]
        if player == Player.NONE or ptype == PieceType.NONE:
            return

        moves = self._compute_piece_moves_uncached(x, y, z)
        norm_moves = [tuple(m) for m in moves]
        self.piece_move_cache[pos] = norm_moves
        self.piece_cache_owner[pos] = player

        for tx, ty, tz in norm_moves:
            self.attack_map[player, tx, ty, tz] += 1

        if self._is_slider_piece(ptype):
            watch_squares = self._calc_slider_watch_squares(x, y, z, ptype, player)
            self.piece_watch_squares[pos] = watch_squares
            for sq in watch_squares:
                self.ray_watchers[sq].add(pos)

    def _unregister_piece_cache(self, pos):
        moves = self.piece_move_cache.pop(pos, None)
        player = self.piece_cache_owner.pop(pos, None)
        if moves is not None and player is not None:
            for tx, ty, tz in moves:
                self.attack_map[player, tx, ty, tz] -= 1

        watch = self.piece_watch_squares.pop(pos, None)
        if watch:
            for sq in watch:
                bucket = self.ray_watchers.get(sq)
                if bucket is None:
                    continue
                bucket.discard(pos)
                if not bucket:
                    self.ray_watchers.pop(sq, None)

    def _collect_affected_pieces(self, dirty_squares, old_piece_positions):
        affected = set(old_piece_positions)
        for sq in dirty_squares:
            watchers = self.ray_watchers.get(sq)
            if watchers:
                affected.update(watchers)
            x, y, z = sq
            if self.owner[x, y, z] != Player.NONE:
                affected.add((x, y, z))
        return affected

    def _invalidate_and_refresh(self, dirty_squares, old_piece_positions):
        affected = self._collect_affected_pieces(dirty_squares, old_piece_positions)
        if not affected:
            return

        for pos in list(affected):
            if pos in self.piece_move_cache:
                self._unregister_piece_cache(pos)

        for pos in affected:
            x, y, z = pos
            if self.owner[x, y, z] != Player.NONE and self.board[x, y, z] != PieceType.NONE:
                self._register_piece_cache(pos)

    def _rebuild_all_caches(self):
        # Re-sync piece/king positions from authoritative board tensors.
        self.piece_positions = {Player.SENTE: set(), Player.GOTE: set()}
        self.king_positions = {Player.SENTE: None, Player.GOTE: None}
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    o = self.owner[x, y, z]
                    p = self.board[x, y, z]
                    if o in (Player.SENTE, Player.GOTE) and p != PieceType.NONE:
                        self.piece_positions[o].add((x, y, z))
                        if p == PieceType.OU:
                            self.king_positions[o] = (x, y, z)

        self.attack_map.fill(0)
        self.piece_move_cache.clear()
        self.piece_cache_owner.clear()
        self.piece_watch_squares.clear()
        self.ray_watchers.clear()
        self._recompute_zobrist_hash()
        for player in [Player.SENTE, Player.GOTE]:
            for pos in list(self.piece_positions[player]):
                self._register_piece_cache(pos)
