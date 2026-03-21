import torch


class TensorizedBoardOps:
    """Small tensorization prototype for 3D-shogi-like board operations.

    Board layout: [B, 9, 9, 9] as [batch, x, y, z]
    owner: 0 none, 1 sente, 2 gote
    piece: 0 none, 1 FU, 8 OU
    """

    EMPTY = 0
    SENTE = 1
    GOTE = 2

    FU = 1
    OU = 8

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def random_state(self, batch_size: int, seed: int = 42):
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        # Sparse-ish random board: mostly empty, some FU and OU
        piece = torch.zeros((batch_size, 9, 9, 9), dtype=torch.int64, device=self.device)
        owner = torch.zeros((batch_size, 9, 9, 9), dtype=torch.int64, device=self.device)

        rnd = torch.rand((batch_size, 9, 9, 9), generator=g, device=self.device)
        fu_mask = rnd < 0.10
        piece[fu_mask] = self.FU
        owner[fu_mask] = torch.where(
            torch.rand_like(rnd[fu_mask]) < 0.5,
            torch.tensor(self.SENTE, device=self.device),
            torch.tensor(self.GOTE, device=self.device),
        )

        # Place one king for each side per board
        for b in range(batch_size):
            k1 = torch.randint(0, 9, (3,), generator=g, device=self.device)
            k2 = torch.randint(0, 9, (3,), generator=g, device=self.device)
            piece[b, k1[0], k1[1], k1[2]] = self.OU
            owner[b, k1[0], k1[1], k1[2]] = self.SENTE
            piece[b, k2[0], k2[1], k2[2]] = self.OU
            owner[b, k2[0], k2[1], k2[2]] = self.GOTE

        return piece, owner

    def legal_fu_moves_mask(self, piece: torch.Tensor, owner: torch.Tensor, player: int):
        """Vectorized FU forward move mask.

        Returns a bool mask [B, 9, 9, 9] where True means
        a FU at (x, y, z) can move one step forward for the given player.
        """
        assert player in (self.SENTE, self.GOTE)

        is_fu = (piece == self.FU) & (owner == player)

        if player == self.SENTE:
            # SENTE moves to y-1
            dest_empty = torch.zeros_like(is_fu)
            dest_empty[:, :, 1:, :] = (piece[:, :, :-1, :] == self.EMPTY)
            movable = torch.zeros_like(is_fu)
            movable[:, :, 1:, :] = is_fu[:, :, 1:, :] & dest_empty[:, :, 1:, :]
            return movable

        # GOTE moves to y+1
        dest_empty = torch.zeros_like(is_fu)
        dest_empty[:, :, :-1, :] = (piece[:, :, 1:, :] == self.EMPTY)
        movable = torch.zeros_like(is_fu)
        movable[:, :, :-1, :] = is_fu[:, :, :-1, :] & dest_empty[:, :, :-1, :]
        return movable

    def king_adjacent_attack_map(self, piece: torch.Tensor, owner: torch.Tensor, player: int):
        """Vectorized king-adjacent attack map for a single player.

        Returns bool map [B, 9, 9, 9] of squares attacked by king adjacency.
        """
        kings = (piece == self.OU) & (owner == player)
        if kings.dtype != torch.float32:
            kings_f = kings.float()
        else:
            kings_f = kings

        # 3x3x3 kernel with center excluded
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32, device=piece.device)
        kernel[:, :, 1, 1, 1] = 0.0

        # conv3d expects [B, C, D, H, W]. We map [x,y,z] as [D,H,W].
        x = kings_f.unsqueeze(1)
        attacked = torch.nn.functional.conv3d(x, kernel, padding=1)
        return attacked.squeeze(1) > 0
