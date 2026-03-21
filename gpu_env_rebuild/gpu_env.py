import torch


class GPUShogiEnv:
    """Separated full-rule environment (board moves + drops + promotions + king safety).

    This implementation is fully separated from existing ai_training env files.
    It uses torch tensors for state storage on a chosen device, while move generation
    still uses Python loops for correctness-first behavior.
    """

    # Players
    NONE = 0
    SENTE = 1
    GOTE = 2

    # Piece types
    PT_NONE = 0
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
        FU: NARI_FU,
        KYO: NARI_KYO,
        KEI: NARI_KEI,
        GIN: NARI_GIN,
        KAKU: UMA,
        HI: RYU,
    }

    DEMOTION_MAP = {
        NARI_FU: FU,
        NARI_KYO: KYO,
        NARI_KEI: KEI,
        NARI_GIN: GIN,
        UMA: KAKU,
        RYU: HI,
    }

    HAND_PIECES = [FU, KYO, KEI, GIN, KIN, KAKU, HI]
    HAND_TO_IDX = {p: i for i, p in enumerate(HAND_PIECES)}

    REASON_NONE = 0
    REASON_OU_CAPTURE = 1
    REASON_MATE = 2
    REASON_NO_LEGAL = 3
    REASON_MAX_PLIES = 4
    SAMPLING_FAST = "fast"
    SAMPLING_EXACT = "exact"

    DROP_RANDOM_TRIES = 96

    ALL_ADJ = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if not (dx == 0 and dy == 0 and dz == 0)]
    HI_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    _kset = set()
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                _kset.add((sx, 0, sz))
                _kset.add((sx, sy, 0))
                _kset.add((0, sy, sz))
    KAKU_DIRS = list(_kset)

    KIN_MOVES_SENTE = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
        (0, 0, -1), (0, 0, 1),
        (1, 0, -1), (1, 0, 0), (1, 0, 1),
        (0, 1, 0),
    ]
    KIN_MOVES_GOTE = [(dx, -dy, dz) for (dx, dy, dz) in KIN_MOVES_SENTE]
    GIN_MOVES_SENTE = [
        (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
        (0, -1, -1), (0, -1, 0), (0, -1, 1),
        (1, -1, -1), (1, -1, 0), (1, -1, 1),
        (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
        (0, 1, -1), (0, 1, 1),
        (1, 1, -1), (1, 1, 0), (1, 1, 1),
    ]
    GIN_MOVES_GOTE = [(dx, -dy, dz) for (dx, dy, dz) in GIN_MOVES_SENTE]
    KIN_SET_SENTE = set(KIN_MOVES_SENTE)
    KIN_SET_GOTE = set(KIN_MOVES_GOTE)
    GIN_SET_SENTE = set(GIN_MOVES_SENTE)
    GIN_SET_GOTE = set(GIN_MOVES_GOTE)

    def __init__(self, batch_size=128, device="cpu", max_plies=512, seed=0, sampling_mode="fast"):
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_plies = int(max_plies)
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))
        self.sampling_mode = self.SAMPLING_FAST

        self.board = torch.zeros((self.batch_size, 9, 9, 9), dtype=torch.int16, device=self.device)
        self.owner = torch.zeros((self.batch_size, 9, 9, 9), dtype=torch.int8, device=self.device)
        self.hands = torch.zeros((self.batch_size, 2, 7), dtype=torch.int16, device=self.device)
        self.current_player = torch.full((self.batch_size,), self.SENTE, dtype=torch.int8, device=self.device)
        self.done = torch.zeros((self.batch_size,), dtype=torch.bool, device=self.device)
        self.reason = torch.zeros((self.batch_size,), dtype=torch.int8, device=self.device)
        self.plies = torch.zeros((self.batch_size,), dtype=torch.int16, device=self.device)

        self.set_sampling_mode(sampling_mode)

        self.reset()

    def set_sampling_mode(self, mode):
        m = str(mode).lower()
        if m not in (self.SAMPLING_FAST, self.SAMPLING_EXACT):
            raise ValueError(f"Unsupported sampling mode: {mode}")
        self.sampling_mode = m

    def reset(self):
        self.board.zero_()
        self.owner.zero_()
        self.hands.zero_()
        self.current_player.fill_(self.SENTE)
        self.done.zero_()
        self.reason.zero_()
        self.plies.zero_()

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

        for b in range(self.batch_size):
            self._apply_plane(b, 0, plane_1_7, self.GOTE)
            self._apply_plane(b, 1, plane_2_8, self.GOTE)
            self.board[b, :, 2, :] = self.FU
            self.owner[b, :, 2, :] = self.GOTE

            self._apply_plane(b, 8, plane_1_7, self.SENTE)
            self._apply_plane(b, 7, plane_2_8, self.SENTE)
            self.board[b, :, 6, :] = self.FU
            self.owner[b, :, 6, :] = self.SENTE

    def _apply_plane(self, b, y, data, player):
        for z in range(9):
            for x in range(9):
                t = data[z][x]
                if t != 0:
                    self.board[b, x, y, z] = t
                    self.owner[b, x, y, z] = player

    def _in_board(self, x, y, z):
        return 0 <= x < 9 and 0 <= y < 9 and 0 <= z < 9

    def _opp(self, p):
        return self.GOTE if p == self.SENTE else self.SENTE

    def _is_promo_zone(self, y, player):
        if player == self.SENTE:
            return y <= 2
        return y >= 6

    def _hand_idx(self, ptype):
        return self.HAND_TO_IDX.get(ptype, None)

    def _step_moves(self, b, fx, fy, fz, rels, player):
        out = []
        for dx, dy, dz in rels:
            tx, ty, tz = fx + dx, fy + dy, fz + dz
            if not self._in_board(tx, ty, tz):
                continue
            if int(self.owner[b, tx, ty, tz].item()) != player:
                out.append((tx, ty, tz))
        return out

    def _slide_moves(self, b, fx, fy, fz, dirs, player):
        out = []
        for dx, dy, dz in dirs:
            tx, ty, tz = fx + dx, fy + dy, fz + dz
            while self._in_board(tx, ty, tz):
                o = int(self.owner[b, tx, ty, tz].item())
                if o != self.NONE:
                    if o != player:
                        out.append((tx, ty, tz))
                    break
                out.append((tx, ty, tz))
                tx += dx
                ty += dy
                tz += dz
        return out

    def _kin_moves(self, dy):
        return self.KIN_MOVES_SENTE if dy == -1 else self.KIN_MOVES_GOTE

    def _gin_moves(self, dy):
        return self.GIN_MOVES_SENTE if dy == -1 else self.GIN_MOVES_GOTE

    def _get_piece_moves(self, b, x, y, z):
        p = int(self.board[b, x, y, z].item())
        player = int(self.owner[b, x, y, z].item())
        if p == self.PT_NONE or player == self.NONE:
            return []
        dy = -1 if player == self.SENTE else 1

        if p == self.OU:
            return self._step_moves(b, x, y, z, self.ALL_ADJ, player)
        if p in (self.KIN, self.NARI_FU, self.NARI_KYO, self.NARI_KEI, self.NARI_GIN):
            return self._step_moves(b, x, y, z, self._kin_moves(dy), player)
        if p == self.GIN:
            return self._step_moves(b, x, y, z, self._gin_moves(dy), player)
        if p == self.FU:
            return self._step_moves(b, x, y, z, [(0, dy, 0)], player)
        if p == self.KEI:
            return self._step_moves(b, x, y, z, [(1, 2 * dy, 0), (-1, 2 * dy, 0), (0, 2 * dy, 1), (0, 2 * dy, -1)], player)
        if p == self.KYO:
            return self._slide_moves(b, x, y, z, [(0, dy, 0)], player)
        if p == self.KAKU:
            return self._slide_moves(b, x, y, z, self.KAKU_DIRS, player)
        if p == self.HI:
            return self._slide_moves(b, x, y, z, self.HI_DIRS, player)
        if p == self.RYU:
            s = self._slide_moves(b, x, y, z, self.HI_DIRS, player)
            a = self._step_moves(b, x, y, z, self.ALL_ADJ, player)
            return list(dict.fromkeys(s + a))
        if p == self.UMA:
            s = self._slide_moves(b, x, y, z, self.KAKU_DIRS, player)
            a = self._step_moves(b, x, y, z, self.ALL_ADJ, player)
            return list(dict.fromkeys(s + a))
        return []

    def _find_king(self, b, player):
        k = torch.nonzero((self.board[b] == self.OU) & (self.owner[b] == player), as_tuple=False)
        if k.numel() == 0:
            return None
        k = k[0]
        return int(k[0].item()), int(k[1].item()), int(k[2].item())

    def _is_king_in_check_ray(self, b, player):
        king = self._find_king(b, player)
        if king is None:
            return True
        kx, ky, kz = king
        opp = self._opp(player)
        dy_opp = -1 if opp == self.SENTE else 1
        kin_set_opp = self.KIN_SET_SENTE if opp == self.SENTE else self.KIN_SET_GOTE
        gin_set_opp = self.GIN_SET_SENTE if opp == self.SENTE else self.GIN_SET_GOTE

        # Rook-like rays
        for dx, dy, dz in self.HI_DIRS:
            tx, ty, tz = kx + dx, ky + dy, kz + dz
            dist = 1
            while self._in_board(tx, ty, tz):
                o = int(self.owner[b, tx, ty, tz].item())
                if o != self.NONE:
                    if o == opp:
                        p = int(self.board[b, tx, ty, tz].item())
                        if p in (self.HI, self.RYU):
                            return True
                        if p == self.KYO and dx == 0 and dy == -dy_opp and dz == 0:
                            return True
                        if dist == 1:
                            if p == self.OU:
                                return True
                            if p in (self.KIN, self.NARI_FU, self.NARI_KYO, self.NARI_KEI, self.NARI_GIN) and (dx, dy, dz) in kin_set_opp:
                                return True
                            if p == self.FU and dx == 0 and dy == -dy_opp and dz == 0:
                                return True
                            if p == self.GIN and (dx, dy, dz) in gin_set_opp:
                                return True
                    break
                tx += dx
                ty += dy
                tz += dz
                dist += 1

        # Bishop-like rays
        for dx, dy, dz in self.KAKU_DIRS:
            tx, ty, tz = kx + dx, ky + dy, kz + dz
            dist = 1
            while self._in_board(tx, ty, tz):
                o = int(self.owner[b, tx, ty, tz].item())
                if o != self.NONE:
                    if o == opp:
                        p = int(self.board[b, tx, ty, tz].item())
                        if p in (self.KAKU, self.UMA):
                            return True
                        if dist == 1:
                            if p == self.OU:
                                return True
                            if p in (self.KIN, self.NARI_FU, self.NARI_KYO, self.NARI_KEI, self.NARI_GIN) and (dx, dy, dz) in kin_set_opp:
                                return True
                            if p == self.GIN and (dx, dy, dz) in gin_set_opp:
                                return True
                    break
                tx += dx
                ty += dy
                tz += dz
                dist += 1

        # Knight checks
        for dx, dy, dz in [(1, -2 * dy_opp, 0), (-1, -2 * dy_opp, 0), (0, -2 * dy_opp, 1), (0, -2 * dy_opp, -1)]:
            tx, ty, tz = kx + dx, ky + dy, kz + dz
            if self._in_board(tx, ty, tz):
                if int(self.board[b, tx, ty, tz].item()) == self.KEI and int(self.owner[b, tx, ty, tz].item()) == opp:
                    return True

        # Adjacent specials (king, horse, dragon)
        for dx, dy, dz in self.ALL_ADJ:
            tx, ty, tz = kx + dx, ky + dy, kz + dz
            if not self._in_board(tx, ty, tz):
                continue
            if int(self.owner[b, tx, ty, tz].item()) != opp:
                continue
            p = int(self.board[b, tx, ty, tz].item())
            if p == self.OU:
                return True
            if p in (self.RYU, self.UMA):
                return True

        return False

    def _is_move_safe(self, b, fx, fy, fz, tx, ty, tz, player):
        sp = int(self.board[b, fx, fy, fz].item())
        so = int(self.owner[b, fx, fy, fz].item())
        cp = int(self.board[b, tx, ty, tz].item())
        co = int(self.owner[b, tx, ty, tz].item())

        self.board[b, tx, ty, tz] = sp
        self.owner[b, tx, ty, tz] = so
        self.board[b, fx, fy, fz] = self.PT_NONE
        self.owner[b, fx, fy, fz] = self.NONE

        safe = not self._is_king_in_check_ray(b, player)

        self.board[b, fx, fy, fz] = sp
        self.owner[b, fx, fy, fz] = so
        self.board[b, tx, ty, tz] = cp
        self.owner[b, tx, ty, tz] = co
        return safe

    def _is_drop_safe(self, b, ptype, tx, ty, tz, player):
        cp = int(self.board[b, tx, ty, tz].item())
        co = int(self.owner[b, tx, ty, tz].item())
        self.board[b, tx, ty, tz] = ptype
        self.owner[b, tx, ty, tz] = player
        safe = not self._is_king_in_check_ray(b, player)
        self.board[b, tx, ty, tz] = cp
        self.owner[b, tx, ty, tz] = co
        return safe

    def _has_nifu_file(self, b, x, z, player):
        for y in range(9):
            if int(self.owner[b, x, y, z].item()) == player and int(self.board[b, x, y, z].item()) == self.FU:
                return True
        return False

    def _is_aligned_with_king_ray(self, king_pos, fx, fy, fz):
        """Return True if source square is on any rook/bishop ray from own king.

        Conservative optimization: if not aligned, moving this piece cannot expose a slider line to king.
        """
        if king_pos is None:
            return True
        kx, ky, kz = king_pos
        dx, dy, dz = fx - kx, fy - ky, fz - kz
        if dx == 0 and dy == 0 and dz == 0:
            return True

        # Rook rays: exactly one non-zero axis.
        non_zero = (1 if dx != 0 else 0) + (1 if dy != 0 else 0) + (1 if dz != 0 else 0)
        if non_zero == 1:
            return True

        # Bishop rays in this variant: exactly one zero axis and equal absolute of the other two.
        if dx == 0 and abs(dy) == abs(dz):
            return True
        if dy == 0 and abs(dx) == abs(dz):
            return True
        if dz == 0 and abs(dx) == abs(dy):
            return True

        return False

    def _legal_moves_for_env(self, b, player):
        moves = []
        in_check = self._is_king_in_check_ray(b, player)
        king_pos = self._find_king(b, player)

        coords = torch.nonzero(self.owner[b] == player, as_tuple=False)
        for c in coords.tolist():
            x, y, z = c
            p = int(self.board[b, x, y, z].item())
            need_full_safety = in_check or p == self.OU or self._is_aligned_with_king_ray(king_pos, x, y, z)
            for tx, ty, tz in self._get_piece_moves(b, x, y, z):
                if need_full_safety:
                    if not self._is_move_safe(b, x, y, z, tx, ty, tz, player):
                        continue

                forced = False
                if p in (self.FU, self.KYO):
                    if (player == self.SENTE and ty == 0) or (player == self.GOTE and ty == 8):
                        forced = True
                if p == self.KEI:
                    if (player == self.SENTE and ty <= 1) or (player == self.GOTE and ty >= 7):
                        forced = True

                can_promote = p in self.PROMOTION_MAP and (self._is_promo_zone(y, player) or self._is_promo_zone(ty, player))
                if forced and can_promote:
                    moves.append(("move", x, y, z, tx, ty, tz, True))
                else:
                    moves.append(("move", x, y, z, tx, ty, tz, False))
                    if can_promote:
                        moves.append(("move", x, y, z, tx, ty, tz, True))

        # Drops
        pidx = 0 if player == self.SENTE else 1
        for ptype in self.HAND_PIECES:
            hidx = self.HAND_TO_IDX[ptype]
            count = int(self.hands[b, pidx, hidx].item())
            if count <= 0:
                continue
            for x in range(9):
                for y in range(9):
                    for z in range(9):
                        if int(self.board[b, x, y, z].item()) != self.PT_NONE:
                            continue
                        # dead-square restrictions
                        if ptype in (self.FU, self.KYO):
                            if (player == self.SENTE and y == 0) or (player == self.GOTE and y == 8):
                                continue
                        if ptype == self.KEI:
                            if (player == self.SENTE and y <= 1) or (player == self.GOTE and y >= 7):
                                continue
                        if ptype == self.FU and self._has_nifu_file(b, x, z, player):
                            continue
                        if in_check and (not self._is_drop_safe(b, ptype, x, y, z, player)):
                            continue
                        moves.append(("drop", ptype, x, y, z))

        return moves

    def _sample_one_legal_action(self, b, player):
        """Sample one legal action with early exit to avoid full legal-list materialization."""
        in_check = self._is_king_in_check_ray(b, player)
        king_pos = self._find_king(b, player)

        coords = torch.nonzero(self.owner[b] == player, as_tuple=False)
        ncoords = int(coords.shape[0])
        if ncoords > 0:
            order = torch.randperm(ncoords, generator=self.rng, device=self.device)
            for oi in order.tolist():
                x, y, z = [int(v) for v in coords[oi].tolist()]
                p = int(self.board[b, x, y, z].item())
                need_full_safety = in_check or p == self.OU or self._is_aligned_with_king_ray(king_pos, x, y, z)
                cand = self._get_piece_moves(b, x, y, z)
                if not cand:
                    continue
                move_order = torch.randperm(len(cand), generator=self.rng, device=self.device)
                for mi in move_order.tolist():
                    tx, ty, tz = cand[mi]
                    if need_full_safety:
                        if not self._is_move_safe(b, x, y, z, tx, ty, tz, player):
                            continue

                    forced = False
                    if p in (self.FU, self.KYO):
                        if (player == self.SENTE and ty == 0) or (player == self.GOTE and ty == 8):
                            forced = True
                    if p == self.KEI:
                        if (player == self.SENTE and ty <= 1) or (player == self.GOTE and ty >= 7):
                            forced = True

                    can_promote = p in self.PROMOTION_MAP and (self._is_promo_zone(y, player) or self._is_promo_zone(ty, player))
                    if forced and can_promote:
                        return ("move", x, y, z, tx, ty, tz, True)
                    if can_promote:
                        # Randomize promote/non-promote without creating two entries.
                        promote = bool(torch.randint(0, 2, (1,), generator=self.rng, device=self.device).item())
                        return ("move", x, y, z, tx, ty, tz, promote)
                    return ("move", x, y, z, tx, ty, tz, False)

        # Try drops only if no board move found.
        pidx = 0 if player == self.SENTE else 1
        has_hand = False
        for ptype in self.HAND_PIECES:
            hidx = self.HAND_TO_IDX[ptype]
            if int(self.hands[b, pidx, hidx].item()) > 0:
                has_hand = True
                break
        if not has_hand:
            return None

        for ptype in self.HAND_PIECES:
            hidx = self.HAND_TO_IDX[ptype]
            if int(self.hands[b, pidx, hidx].item()) <= 0:
                continue

            # Fast random probing before fallback exhaustive scan.
            for _ in range(self.DROP_RANDOM_TRIES):
                x = int(torch.randint(0, 9, (1,), generator=self.rng, device=self.device).item())
                y = int(torch.randint(0, 9, (1,), generator=self.rng, device=self.device).item())
                z = int(torch.randint(0, 9, (1,), generator=self.rng, device=self.device).item())
                if int(self.board[b, x, y, z].item()) != self.PT_NONE:
                    continue
                if ptype in (self.FU, self.KYO):
                    if (player == self.SENTE and y == 0) or (player == self.GOTE and y == 8):
                        continue
                if ptype == self.KEI:
                    if (player == self.SENTE and y <= 1) or (player == self.GOTE and y >= 7):
                        continue
                if ptype == self.FU and self._has_nifu_file(b, x, z, player):
                    continue
                if in_check and (not self._is_drop_safe(b, ptype, x, y, z, player)):
                    continue
                return ("drop", ptype, x, y, z)

            # Exhaustive fallback keeps correctness complete.
            for x in range(9):
                for y in range(9):
                    for z in range(9):
                        if int(self.board[b, x, y, z].item()) != self.PT_NONE:
                            continue
                        if ptype in (self.FU, self.KYO):
                            if (player == self.SENTE and y == 0) or (player == self.GOTE and y == 8):
                                continue
                        if ptype == self.KEI:
                            if (player == self.SENTE and y <= 1) or (player == self.GOTE and y >= 7):
                                continue
                        if ptype == self.FU and self._has_nifu_file(b, x, z, player):
                            continue
                        if in_check and (not self._is_drop_safe(b, ptype, x, y, z, player)):
                            continue
                        return ("drop", ptype, x, y, z)

        return None

    def _sample_one_legal_action_exact(self, b, player):
        """Uniform sample from full legal list (correctness-first, slower)."""
        legal = self._legal_moves_for_env(b, player)
        if len(legal) == 0:
            return None
        ridx = int(torch.randint(0, len(legal), (1,), generator=self.rng, device=self.device).item())
        return legal[ridx]

    def _has_any_legal_action(self, b, player):
        """Early-exit legality check for mate/no-legal judgment."""
        in_check = self._is_king_in_check_ray(b, player)
        king_pos = self._find_king(b, player)
        coords = torch.nonzero(self.owner[b] == player, as_tuple=False)

        for c in coords.tolist():
            x, y, z = c
            p = int(self.board[b, x, y, z].item())
            need_full_safety = in_check or p == self.OU or self._is_aligned_with_king_ray(king_pos, x, y, z)
            for tx, ty, tz in self._get_piece_moves(b, x, y, z):
                if not need_full_safety:
                    return True
                if self._is_move_safe(b, x, y, z, tx, ty, tz, player):
                    return True

        pidx = 0 if player == self.SENTE else 1
        for ptype in self.HAND_PIECES:
            hidx = self.HAND_TO_IDX[ptype]
            if int(self.hands[b, pidx, hidx].item()) <= 0:
                continue
            for x in range(9):
                for y in range(9):
                    for z in range(9):
                        if int(self.board[b, x, y, z].item()) != self.PT_NONE:
                            continue
                        if ptype in (self.FU, self.KYO):
                            if (player == self.SENTE and y == 0) or (player == self.GOTE and y == 8):
                                continue
                        if ptype == self.KEI:
                            if (player == self.SENTE and y <= 1) or (player == self.GOTE and y >= 7):
                                continue
                        if ptype == self.FU and self._has_nifu_file(b, x, z, player):
                            continue
                        if in_check and (not self._is_drop_safe(b, ptype, x, y, z, player)):
                            continue
                        return True
        return False

    def sample_actions(self):
        actions = [None] * self.batch_size
        exact = self.sampling_mode == self.SAMPLING_EXACT
        for b in range(self.batch_size):
            if bool(self.done[b].item()):
                continue
            player = int(self.current_player[b].item())
            if exact:
                actions[b] = self._sample_one_legal_action_exact(b, player)
            else:
                actions[b] = self._sample_one_legal_action(b, player)
        return actions

    def step(self, actions):
        for b in range(self.batch_size):
            if bool(self.done[b].item()):
                continue

            player = int(self.current_player[b].item())
            action = actions[b]

            if action is None:
                self.done[b] = True
                self.reason[b] = self.REASON_MATE if self._is_king_in_check_ray(b, player) else self.REASON_NO_LEGAL
                continue

            if action[0] == "drop":
                _, ptype, x, y, z = action
                pidx = 0 if player == self.SENTE else 1
                hidx = self.HAND_TO_IDX[ptype]
                self.board[b, x, y, z] = int(ptype)
                self.owner[b, x, y, z] = player
                self.hands[b, pidx, hidx] -= 1
            else:
                _, fx, fy, fz, tx, ty, tz, promote = action
                moving = int(self.board[b, fx, fy, fz].item())
                captured = int(self.board[b, tx, ty, tz].item())
                captured_owner = int(self.owner[b, tx, ty, tz].item())

                if promote and moving in self.PROMOTION_MAP:
                    moving = self.PROMOTION_MAP[moving]

                self.board[b, tx, ty, tz] = moving
                self.owner[b, tx, ty, tz] = player
                self.board[b, fx, fy, fz] = self.PT_NONE
                self.owner[b, fx, fy, fz] = self.NONE

                if captured_owner != self.NONE and captured != self.PT_NONE:
                    if captured == self.OU:
                        self.done[b] = True
                        self.reason[b] = self.REASON_OU_CAPTURE
                    else:
                        raw = self.DEMOTION_MAP.get(captured, captured)
                        hidx = self._hand_idx(raw)
                        if hidx is not None:
                            pidx = 0 if player == self.SENTE else 1
                            self.hands[b, pidx, hidx] += 1

            self.plies[b] += 1
            if bool(self.done[b].item()):
                continue

            if int(self.plies[b].item()) >= self.max_plies:
                self.done[b] = True
                self.reason[b] = self.REASON_MAX_PLIES
                continue

            opp = self._opp(player)
            if self._is_king_in_check_ray(b, opp):
                if not self._has_any_legal_action(b, opp):
                    self.done[b] = True
                    self.reason[b] = self.REASON_MATE
                    continue

            self.current_player[b] = opp

    def reason_hist(self):
        vals, counts = torch.unique(self.reason[self.done], return_counts=True)
        mapping = {
            self.REASON_OU_CAPTURE: "ou_capture",
            self.REASON_MATE: "mate",
            self.REASON_NO_LEGAL: "no_legal",
            self.REASON_MAX_PLIES: "max_plies",
        }
        out = {}
        for v, c in zip(vals.tolist(), counts.tolist()):
            if v == self.REASON_NONE:
                continue
            out[mapping.get(v, f"reason_{v}")] = int(c)
        return out
