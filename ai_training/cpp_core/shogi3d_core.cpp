#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

static inline bool in_board(int x, int y, int z) {
    return x >= 0 && x < 9 && y >= 0 && y < 9 && z >= 0 && z < 9;
}

// board: int32[9,9,9], owner: int8[9,9,9], player: int (1|2)
bool is_king_in_check_ray_cpp(py::array_t<int, py::array::c_style | py::array::forcecast> board,
                              py::array_t<signed char, py::array::c_style | py::array::forcecast> owner,
                              int player) {
    auto b = board.unchecked<3>();
    auto o = owner.unchecked<3>();

    const int NONE = 0;
    const int SENTE = 1;
    const int GOTE = 2;

    const int FU = 1;
    const int KYO = 2;
    const int KEI = 3;
    const int GIN = 4;
    const int KIN = 5;
    const int KAKU = 6;
    const int HI = 7;
    const int OU = 8;
    const int NARI_FU = 9;
    const int NARI_KYO = 10;
    const int NARI_KEI = 11;
    const int NARI_GIN = 12;
    const int UMA = 13;
    const int RYU = 14;

    int kx = -1, ky = -1, kz = -1;
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 9; ++y) {
            for (int z = 0; z < 9; ++z) {
                if (o(x, y, z) == player && b(x, y, z) == OU) {
                    kx = x;
                    ky = y;
                    kz = z;
                    goto king_found;
                }
            }
        }
    }
king_found:
    if (kx < 0) return true;

    const int opp = (player == SENTE) ? GOTE : SENTE;
    const int dy_opp = (opp == SENTE) ? 1 : -1;

    static const int HI_DIRS[6][3] = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
    };

    // Kaku directions in this rule set: exactly one axis zero, other two +/-1.
    static const int KAKU_DIRS[12][3] = {
        {1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0},
        {1, 0, 1}, {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1},
        {0, 1, 1}, {0, 1, -1}, {0, -1, 1}, {0, -1, -1}
    };

    auto is_kin_dir = [&](int dx, int dy, int dz, int d) -> bool {
        // d == 1 (forward +y), d == -1 (forward -y)
        if (dy == d && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        if (dy == 0 && !(dx == 0 && dz == 0) && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        if (dx == 0 && dz == 0 && dy == -d) return true;
        return false;
    };

    auto is_gin_dir = [&](int dx, int dy, int dz, int d) -> bool {
        if (dy == d && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        if (dy == -d && !(dx == 0 && dz == 0) && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        return false;
    };

    // Rook rays
    for (int i = 0; i < 6; ++i) {
        int dx = HI_DIRS[i][0], dy = HI_DIRS[i][1], dz = HI_DIRS[i][2];
        int tx = kx + dx, ty = ky + dy, tz = kz + dz;
        int dist = 1;
        while (in_board(tx, ty, tz)) {
            int own = o(tx, ty, tz);
            if (own != NONE) {
                if (own == opp) {
                    int p = b(tx, ty, tz);
                    if (p == HI || p == RYU) return true;
                    if (p == KYO && dx == 0 && dy == -dy_opp && dz == 0) return true;
                    if (dist == 1) {
                        if (p == OU) return true;
                        if ((p == KIN || p == NARI_FU || p == NARI_KYO || p == NARI_KEI || p == NARI_GIN) && is_kin_dir(dx, dy, dz, dy_opp)) return true;
                        if (p == FU && dx == 0 && dy == -dy_opp && dz == 0) return true;
                        if (p == GIN && is_gin_dir(dx, dy, dz, dy_opp)) return true;
                    }
                }
                break;
            }
            tx += dx;
            ty += dy;
            tz += dz;
            dist += 1;
        }
    }

    // Bishop rays
    for (int i = 0; i < 12; ++i) {
        int dx = KAKU_DIRS[i][0], dy = KAKU_DIRS[i][1], dz = KAKU_DIRS[i][2];
        int tx = kx + dx, ty = ky + dy, tz = kz + dz;
        int dist = 1;
        while (in_board(tx, ty, tz)) {
            int own = o(tx, ty, tz);
            if (own != NONE) {
                if (own == opp) {
                    int p = b(tx, ty, tz);
                    if (p == KAKU || p == UMA) return true;
                    if (dist == 1) {
                        if (p == OU) return true;
                        if ((p == KIN || p == NARI_FU || p == NARI_KYO || p == NARI_KEI || p == NARI_GIN) && is_kin_dir(dx, dy, dz, dy_opp)) return true;
                        if (p == GIN && is_gin_dir(dx, dy, dz, dy_opp)) return true;
                    }
                }
                break;
            }
            tx += dx;
            ty += dy;
            tz += dz;
            dist += 1;
        }
    }

    // Knight checks
    static const int KEI_REL[4][3] = {
        {1, 2, 0}, {-1, 2, 0}, {0, 2, 1}, {0, 2, -1}
    };
    for (int i = 0; i < 4; ++i) {
        int tx = kx + KEI_REL[i][0];
        int ty = ky - 2 * dy_opp;
        int tz = kz + KEI_REL[i][2];
        if (!in_board(tx, ty, tz)) continue;
        if (o(tx, ty, tz) == opp && b(tx, ty, tz) == KEI) return true;
    }

    // Adjacent checks for OU/UMA/RYU and near Kin/Gin patterns
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int tx = kx + dx, ty = ky + dy, tz = kz + dz;
                if (!in_board(tx, ty, tz)) continue;
                if (o(tx, ty, tz) != opp) continue;
                int p = b(tx, ty, tz);
                if (p == OU || p == UMA || p == RYU) return true;
                if ((p == KIN || p == NARI_FU || p == NARI_KYO || p == NARI_KEI || p == NARI_GIN) && is_kin_dir(dx, dy, dz, dy_opp)) return true;
                if (p == GIN && is_gin_dir(dx, dy, dz, dy_opp)) return true;
            }
        }
    }

    return false;
}

static inline int idx3(int x, int y, int z) {
    return (x * 9 + y) * 9 + z;
}

static inline bool is_promo_zone(int y, int player) {
    return (player == 1) ? (y <= 2) : (y >= 6);
}

template <typename BO, typename OO>
bool is_king_in_check_ray_view(BO &b, OO &o, int player) {
    const int NONE = 0;
    const int SENTE = 1;
    const int GOTE = 2;

    const int FU = 1;
    const int KYO = 2;
    const int KEI = 3;
    const int GIN = 4;
    const int KIN = 5;
    const int KAKU = 6;
    const int HI = 7;
    const int OU = 8;
    const int NARI_FU = 9;
    const int NARI_KYO = 10;
    const int NARI_KEI = 11;
    const int NARI_GIN = 12;
    const int UMA = 13;
    const int RYU = 14;

    int kx = -1, ky = -1, kz = -1;
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 9; ++y) {
            for (int z = 0; z < 9; ++z) {
                if (o(x, y, z) == player && b(x, y, z) == OU) {
                    kx = x;
                    ky = y;
                    kz = z;
                    goto king_found2;
                }
            }
        }
    }
king_found2:
    if (kx < 0) return true;

    const int opp = (player == SENTE) ? GOTE : SENTE;
    const int dy_opp = (opp == SENTE) ? 1 : -1;

    static const int HI_DIRS[6][3] = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
    };
    static const int KAKU_DIRS[12][3] = {
        {1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0},
        {1, 0, 1}, {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1},
        {0, 1, 1}, {0, 1, -1}, {0, -1, 1}, {0, -1, -1}
    };

    auto is_kin_dir = [&](int dx, int dy, int dz, int d) -> bool {
        if (dy == d && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        if (dy == 0 && !(dx == 0 && dz == 0) && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        if (dx == 0 && dz == 0 && dy == -d) return true;
        return false;
    };
    auto is_gin_dir = [&](int dx, int dy, int dz, int d) -> bool {
        if (dy == d && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        if (dy == -d && !(dx == 0 && dz == 0) && dx >= -1 && dx <= 1 && dz >= -1 && dz <= 1) return true;
        return false;
    };

    for (int i = 0; i < 6; ++i) {
        int dx = HI_DIRS[i][0], dy = HI_DIRS[i][1], dz = HI_DIRS[i][2];
        int tx = kx + dx, ty = ky + dy, tz = kz + dz;
        int dist = 1;
        while (in_board(tx, ty, tz)) {
            int own = o(tx, ty, tz);
            if (own != NONE) {
                if (own == opp) {
                    int p = b(tx, ty, tz);
                    if (p == HI || p == RYU) return true;
                    if (p == KYO && dx == 0 && dy == -dy_opp && dz == 0) return true;
                    if (dist == 1) {
                        if (p == OU) return true;
                        if ((p == KIN || p == NARI_FU || p == NARI_KYO || p == NARI_KEI || p == NARI_GIN) && is_kin_dir(dx, dy, dz, dy_opp)) return true;
                        if (p == FU && dx == 0 && dy == -dy_opp && dz == 0) return true;
                        if (p == GIN && is_gin_dir(dx, dy, dz, dy_opp)) return true;
                    }
                }
                break;
            }
            tx += dx;
            ty += dy;
            tz += dz;
            dist += 1;
        }
    }

    for (int i = 0; i < 12; ++i) {
        int dx = KAKU_DIRS[i][0], dy = KAKU_DIRS[i][1], dz = KAKU_DIRS[i][2];
        int tx = kx + dx, ty = ky + dy, tz = kz + dz;
        int dist = 1;
        while (in_board(tx, ty, tz)) {
            int own = o(tx, ty, tz);
            if (own != NONE) {
                if (own == opp) {
                    int p = b(tx, ty, tz);
                    if (p == KAKU || p == UMA) return true;
                    if (dist == 1) {
                        if (p == OU) return true;
                        if ((p == KIN || p == NARI_FU || p == NARI_KYO || p == NARI_KEI || p == NARI_GIN) && is_kin_dir(dx, dy, dz, dy_opp)) return true;
                        if (p == GIN && is_gin_dir(dx, dy, dz, dy_opp)) return true;
                    }
                }
                break;
            }
            tx += dx;
            ty += dy;
            tz += dz;
            dist += 1;
        }
    }

    static const int KEI_REL[4][3] = {
        {1, 2, 0}, {-1, 2, 0}, {0, 2, 1}, {0, 2, -1}
    };
    for (int i = 0; i < 4; ++i) {
        int tx = kx + KEI_REL[i][0];
        int ty = ky - 2 * dy_opp;
        int tz = kz + KEI_REL[i][2];
        if (!in_board(tx, ty, tz)) continue;
        if (o(tx, ty, tz) == opp && b(tx, ty, tz) == KEI) return true;
    }

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int tx = kx + dx, ty = ky + dy, tz = kz + dz;
                if (!in_board(tx, ty, tz)) continue;
                if (o(tx, ty, tz) != opp) continue;
                int p = b(tx, ty, tz);
                if (p == OU || p == UMA || p == RYU) return true;
                if ((p == KIN || p == NARI_FU || p == NARI_KYO || p == NARI_KEI || p == NARI_GIN) && is_kin_dir(dx, dy, dz, dy_opp)) return true;
                if (p == GIN && is_gin_dir(dx, dy, dz, dy_opp)) return true;
            }
        }
    }

    return false;
}

py::array_t<int> generate_legal_moves_cpp(
    py::array_t<int, py::array::c_style | py::array::forcecast> board,
    py::array_t<signed char, py::array::c_style | py::array::forcecast> owner,
    py::array_t<short, py::array::c_style | py::array::forcecast> hand_counts,
    int player) {

    auto b = board.mutable_unchecked<3>();
    auto o = owner.mutable_unchecked<3>();
    auto hc = hand_counts.unchecked<1>();

    const int NONE = 0;
    const int SENTE = 1;
    const int GOTE = 2;

    const int FU = 1;
    const int KYO = 2;
    const int KEI = 3;
    const int GIN = 4;
    const int KIN = 5;
    const int KAKU = 6;
    const int HI = 7;
    const int OU = 8;
    const int NARI_FU = 9;
    const int NARI_KYO = 10;
    const int NARI_KEI = 11;
    const int NARI_GIN = 12;
    const int UMA = 13;
    const int RYU = 14;

    static const int HI_DIRS[6][3] = {
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}
    };
    static const int KAKU_DIRS[12][3] = {
        {1, 1, 0}, {1, -1, 0}, {-1, 1, 0}, {-1, -1, 0},
        {1, 0, 1}, {1, 0, -1}, {-1, 0, 1}, {-1, 0, -1},
        {0, 1, 1}, {0, 1, -1}, {0, -1, 1}, {0, -1, -1}
    };

    auto add_step_moves = [&](int fx, int fy, int fz, const std::vector<std::array<int, 3>> &rels, std::vector<std::array<int, 3>> &out) {
        for (auto r : rels) {
            int tx = fx + r[0], ty = fy + r[1], tz = fz + r[2];
            if (!in_board(tx, ty, tz)) continue;
            if (o(tx, ty, tz) != player) out.push_back({tx, ty, tz});
        }
    };

    auto add_slide_moves = [&](int fx, int fy, int fz, const int dirs[][3], int nd, std::vector<std::array<int, 3>> &out) {
        for (int i = 0; i < nd; ++i) {
            int dx = dirs[i][0], dy = dirs[i][1], dz = dirs[i][2];
            int tx = fx + dx, ty = fy + dy, tz = fz + dz;
            while (in_board(tx, ty, tz)) {
                int own = o(tx, ty, tz);
                if (own != NONE) {
                    if (own != player) out.push_back({tx, ty, tz});
                    break;
                }
                out.push_back({tx, ty, tz});
                tx += dx;
                ty += dy;
                tz += dz;
            }
        }
    };

    auto make_kin = [&](int dy) {
        std::vector<std::array<int, 3>> rel;
        for (int dx = -1; dx <= 1; ++dx) for (int dz = -1; dz <= 1; ++dz) rel.push_back({dx, dy, dz});
        for (int dx = -1; dx <= 1; ++dx) for (int dz = -1; dz <= 1; ++dz) if (!(dx == 0 && dz == 0)) rel.push_back({dx, 0, dz});
        rel.push_back({0, -dy, 0});
        return rel;
    };

    auto make_gin = [&](int dy) {
        std::vector<std::array<int, 3>> rel;
        for (int dx = -1; dx <= 1; ++dx) for (int dz = -1; dz <= 1; ++dz) rel.push_back({dx, dy, dz});
        for (int dx = -1; dx <= 1; ++dx) for (int dz = -1; dz <= 1; ++dz) if (!(dx == 0 && dz == 0)) rel.push_back({dx, -dy, dz});
        return rel;
    };

    auto piece_moves = [&](int fx, int fy, int fz, std::vector<std::array<int, 3>> &out) {
        int p = b(fx, fy, fz);
        int dy = (player == SENTE) ? -1 : 1;
        if (p == OU) {
            for (int dx = -1; dx <= 1; ++dx) for (int dy2 = -1; dy2 <= 1; ++dy2) for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy2 == 0 && dz == 0) continue;
                int tx = fx + dx, ty = fy + dy2, tz = fz + dz;
                if (!in_board(tx, ty, tz)) continue;
                if (o(tx, ty, tz) != player) out.push_back({tx, ty, tz});
            }
            return;
        }
        if (p == KIN || p == NARI_FU || p == NARI_KYO || p == NARI_KEI || p == NARI_GIN) {
            auto rel = make_kin(dy);
            add_step_moves(fx, fy, fz, rel, out);
            return;
        }
        if (p == GIN) {
            auto rel = make_gin(dy);
            add_step_moves(fx, fy, fz, rel, out);
            return;
        }
        if (p == FU) {
            add_step_moves(fx, fy, fz, {{0, dy, 0}}, out);
            return;
        }
        if (p == KEI) {
            add_step_moves(fx, fy, fz, {{1, 2 * dy, 0}, {-1, 2 * dy, 0}, {0, 2 * dy, 1}, {0, 2 * dy, -1}}, out);
            return;
        }
        if (p == KYO) {
            const int d[1][3] = {{0, dy, 0}};
            add_slide_moves(fx, fy, fz, d, 1, out);
            return;
        }
        if (p == KAKU) {
            add_slide_moves(fx, fy, fz, KAKU_DIRS, 12, out);
            return;
        }
        if (p == HI) {
            add_slide_moves(fx, fy, fz, HI_DIRS, 6, out);
            return;
        }
        if (p == UMA) {
            add_slide_moves(fx, fy, fz, KAKU_DIRS, 12, out);
            for (int dx = -1; dx <= 1; ++dx) for (int dy2 = -1; dy2 <= 1; ++dy2) for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy2 == 0 && dz == 0) continue;
                int tx = fx + dx, ty = fy + dy2, tz = fz + dz;
                if (!in_board(tx, ty, tz)) continue;
                if (o(tx, ty, tz) != player) out.push_back({tx, ty, tz});
            }
            return;
        }
        if (p == RYU) {
            add_slide_moves(fx, fy, fz, HI_DIRS, 6, out);
            for (int dx = -1; dx <= 1; ++dx) for (int dy2 = -1; dy2 <= 1; ++dy2) for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy2 == 0 && dz == 0) continue;
                int tx = fx + dx, ty = fy + dy2, tz = fz + dz;
                if (!in_board(tx, ty, tz)) continue;
                if (o(tx, ty, tz) != player) out.push_back({tx, ty, tz});
            }
            return;
        }
    };

    auto is_move_safe = [&](int fx, int fy, int fz, int tx, int ty, int tz) {
        int sb = b(fx, fy, fz);
        int so = o(fx, fy, fz);
        int cb = b(tx, ty, tz);
        int co = o(tx, ty, tz);

        b(tx, ty, tz) = sb;
        o(tx, ty, tz) = so;
        b(fx, fy, fz) = 0;
        o(fx, fy, fz) = 0;

        bool safe = !is_king_in_check_ray_view(b, o, player);

        b(fx, fy, fz) = sb;
        o(fx, fy, fz) = so;
        b(tx, ty, tz) = cb;
        o(tx, ty, tz) = co;
        return safe;
    };

    auto is_drop_safe = [&](int ptype, int tx, int ty, int tz) {
        int cb = b(tx, ty, tz);
        int co = o(tx, ty, tz);
        b(tx, ty, tz) = ptype;
        o(tx, ty, tz) = player;
        bool safe = !is_king_in_check_ray_view(b, o, player);
        b(tx, ty, tz) = cb;
        o(tx, ty, tz) = co;
        return safe;
    };

    bool currently_in_check = is_king_in_check_ray_view(b, o, player);

    // Python版と同様にピン駒を推定（王手中でない場合のみ利用）
    bool pinned[729];
    for (int i = 0; i < 729; ++i) pinned[i] = false;

    int king_x = -1, king_y = -1, king_z = -1;
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 9; ++y) {
            for (int z = 0; z < 9; ++z) {
                if (o(x, y, z) == player && b(x, y, z) == OU) {
                    king_x = x;
                    king_y = y;
                    king_z = z;
                    goto king_found3;
                }
            }
        }
    }
king_found3:

    if (!currently_in_check && king_x >= 0) {
        int opp = (player == SENTE) ? GOTE : SENTE;
        int dy_opp = (opp == SENTE) ? 1 : -1;

        // HI dirs + KAKU dirs
        for (int i = 0; i < 6 + 12; ++i) {
            int dx, dy, dz;
            if (i < 6) {
                dx = HI_DIRS[i][0]; dy = HI_DIRS[i][1]; dz = HI_DIRS[i][2];
            } else {
                int j = i - 6;
                dx = KAKU_DIRS[j][0]; dy = KAKU_DIRS[j][1]; dz = KAKU_DIRS[j][2];
            }

            int tx = king_x + dx, ty = king_y + dy, tz = king_z + dz;
            int first_x = -1, first_y = -1, first_z = -1;
            while (in_board(tx, ty, tz)) {
                int own = o(tx, ty, tz);
                if (own != NONE) {
                    if (own == player) {
                        if (first_x < 0) {
                            first_x = tx; first_y = ty; first_z = tz;
                        } else {
                            break;
                        }
                    } else {
                        if (first_x >= 0) {
                            int p = b(tx, ty, tz);
                            bool is_pinned = false;
                            if ((p == HI || p == RYU) && i < 6) is_pinned = true;
                            else if ((p == KAKU || p == UMA) && i >= 6) is_pinned = true;
                            else if (p == KYO && dx == 0 && dy == -dy_opp && dz == 0) is_pinned = true;
                            if (is_pinned) pinned[idx3(first_x, first_y, first_z)] = true;
                        }
                        break;
                    }
                }
                tx += dx;
                ty += dy;
                tz += dz;
            }
        }
    }

    std::vector<int> flat;
    flat.reserve(50000);

    // Board moves
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 9; ++y) {
            for (int z = 0; z < 9; ++z) {
                if (o(x, y, z) != player) continue;
                int p = b(x, y, z);
                std::vector<std::array<int, 3>> pm;
                pm.reserve(64);
                piece_moves(x, y, z, pm);
                bool seen_dst[729];
                for (int i = 0; i < 729; ++i) seen_dst[i] = false;
                bool need_safety = true;
                for (auto mv : pm) {
                    int tx = mv[0], ty = mv[1], tz = mv[2];
                    int didx = idx3(tx, ty, tz);
                    if (seen_dst[didx]) continue;
                    seen_dst[didx] = true;
                    if (need_safety && !is_move_safe(x, y, z, tx, ty, tz)) continue;
                    flat.push_back(x);
                    flat.push_back(y);
                    flat.push_back(z);
                    flat.push_back(tx);
                    flat.push_back(ty);
                    flat.push_back(tz);
                    flat.push_back(0);
                    flat.push_back(0);
                    flat.push_back(0);

                    bool promotable = (p == FU || p == KYO || p == KEI || p == GIN || p == KAKU || p == HI);
                    if (promotable && (is_promo_zone(y, player) || is_promo_zone(ty, player))) {
                        flat.push_back(x);
                        flat.push_back(y);
                        flat.push_back(z);
                        flat.push_back(tx);
                        flat.push_back(ty);
                        flat.push_back(tz);
                        flat.push_back(1);
                        flat.push_back(0);
                        flat.push_back(0);
                    }
                }
            }
        }
    }

    // Nifu quick map
    bool nifu[9][9];
    for (int x = 0; x < 9; ++x) for (int z = 0; z < 9; ++z) nifu[x][z] = false;
    for (int x = 0; x < 9; ++x) {
        for (int y = 0; y < 9; ++y) {
            for (int z = 0; z < 9; ++z) {
                if (o(x, y, z) == player && b(x, y, z) == FU) nifu[x][z] = true;
            }
        }
    }

    // Drops by hand counts index FU,KYO,KEI,GIN,KIN,KAKU,HI
    int piece_from_dt[7] = {FU, KYO, KEI, GIN, KIN, KAKU, HI};
    for (int dt = 0; dt < 7; ++dt) {
        if (hc(dt) <= 0) continue;
        int ptype = piece_from_dt[dt];
        for (int x = 0; x < 9; ++x) {
            for (int z = 0; z < 9; ++z) {
                if (ptype == FU && nifu[x][z]) continue;
                for (int y = 0; y < 9; ++y) {
                    if (b(x, y, z) != 0) continue;
                    bool forward_limit = (player == SENTE && y == 0) || (player == GOTE && y == 8);
                    bool kei_limit = (player == SENTE && y <= 1) || (player == GOTE && y >= 7);
                    if (ptype == FU && forward_limit) continue;
                    if (ptype == KYO && forward_limit) continue;
                    if (ptype == KEI && kei_limit) continue;

                    if (currently_in_check && !is_drop_safe(ptype, x, y, z)) continue;

                    flat.push_back(0);
                    flat.push_back(0);
                    flat.push_back(0);
                    flat.push_back(x);
                    flat.push_back(y);
                    flat.push_back(z);
                    flat.push_back(0);
                    flat.push_back(1);
                    flat.push_back(dt);
                }
            }
        }
    }

    py::ssize_t n = static_cast<py::ssize_t>(flat.size() / 9);
    py::array_t<int> out({n, static_cast<py::ssize_t>(9)});
    auto outm = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (int j = 0; j < 9; ++j) {
            outm(i, j) = flat[static_cast<size_t>(i) * 9 + static_cast<size_t>(j)];
        }
    }
    return out;
}

PYBIND11_MODULE(shogi3d_core, m) {
    m.doc() = "3D shogi core helpers";
    m.def("is_king_in_check_ray_cpp", &is_king_in_check_ray_cpp, py::arg("board"), py::arg("owner"), py::arg("player"));
    m.def("generate_legal_moves_cpp", &generate_legal_moves_cpp, py::arg("board"), py::arg("owner"), py::arg("hand_counts"), py::arg("player"));
}
