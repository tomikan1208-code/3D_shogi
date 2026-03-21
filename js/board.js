// board.js - 盤面データ管理 (9x9x9)

'use strict';

// 駒の種類
export const PieceType = {
  NONE: 'none',
  FU: 'fu',       // 歩
  KYO: 'kyo',     // 香車
  KEI: 'kei',     // 桂馬
  GIN: 'gin',     // 銀
  KIN: 'kin',     // 金
  KAKU: 'kaku',   // 角
  HI: 'hi',       // 飛車
  OU: 'ou',       // 王/玉
  // 成り駒
  NARI_FU: 'nari_fu',
  NARI_KYO: 'nari_kyo',
  NARI_KEI: 'nari_kei',
  NARI_GIN: 'nari_gin',
  RYU: 'ryu',     // 竜（成飛車）
  UMA: 'uma',     // 馬（成角）
};

// 駒の表示名
export const PieceName = {
  [PieceType.NONE]: '',
  [PieceType.FU]: '歩',
  [PieceType.KYO]: '香',
  [PieceType.KEI]: '桂',
  [PieceType.GIN]: '銀',
  [PieceType.KIN]: '金',
  [PieceType.KAKU]: '角',
  [PieceType.HI]: '飛',
  [PieceType.OU]: '王',
  [PieceType.NARI_FU]: '金',
  [PieceType.NARI_KYO]: '金',
  [PieceType.NARI_KEI]: '金',
  [PieceType.NARI_GIN]: '金',
  [PieceType.RYU]: '竜',
  [PieceType.UMA]: '馬',
};

/**
 * 駒の表示名（先手の王=玉、後手=王、成り歩=と）
 */
export function getDisplayName(cell) {
  if (!cell) return '';
  switch (cell.type) {
    case PieceType.OU:      return cell.player === Player.SENTE ? '玉' : '王';
    case PieceType.NARI_FU: return 'と';
    default:                return PieceName[cell.type] || '';
  }
}

/** 成り駒かどうか */
export function isPromotedType(type) {
  return [PieceType.NARI_FU, PieceType.NARI_KYO, PieceType.NARI_KEI,
          PieceType.NARI_GIN, PieceType.UMA, PieceType.RYU].includes(type);
}

/** 駒のグレード（0:歩 〜 4:王） — 形の大きさに使用 */
export function getPieceGrade(type) {
  switch (type) {
    case PieceType.FU:                          return 0;
    case PieceType.KYO: case PieceType.KEI:    return 1;
    case PieceType.NARI_FU:                    return 1;
    case PieceType.GIN: case PieceType.KIN:    return 2;
    case PieceType.NARI_KYO: case PieceType.NARI_KEI: case PieceType.NARI_GIN: return 2;
    case PieceType.KAKU: case PieceType.HI:    return 3;
    case PieceType.UMA:  case PieceType.RYU:   return 3;
    case PieceType.OU:                          return 4;
    default:                                    return 2;
  }
}

// 成り変換マップ
export const PromotionMap = {
  [PieceType.FU]: PieceType.NARI_FU,
  [PieceType.KYO]: PieceType.NARI_KYO,
  [PieceType.KEI]: PieceType.NARI_KEI,
  [PieceType.GIN]: PieceType.NARI_GIN,
  [PieceType.KAKU]: PieceType.UMA,
  [PieceType.HI]: PieceType.RYU,
};

// 成り → 元の駒（捕獲時に戻す）
export const DemotionMap = {
  [PieceType.NARI_FU]:  PieceType.FU,
  [PieceType.NARI_KYO]: PieceType.KYO,
  [PieceType.NARI_KEI]: PieceType.KEI,
  [PieceType.NARI_GIN]: PieceType.GIN,
  [PieceType.RYU]:      PieceType.HI,
  [PieceType.UMA]:      PieceType.KAKU,
};

export function demoteType(type) {
  return DemotionMap[type] || type;
}

// 成れる駒かチェック
export function canPromote(type) {
  return type in PromotionMap;
}

// プレイヤー
export const Player = { SENTE: 1, GOTE: 2 };

let pieceIdCounter = 0;
function newId() { return ++pieceIdCounter; }

/**
 * セルオブジェクト
 * { type: PieceType, player: Player|null, id: number|null, promoted: bool }
 */
function emptyCell() {
  return { type: PieceType.NONE, player: Player.NONE, id: -1, promoted: false }; // Modified to use Player.NONE and id: -1
}

function makeCell(type, player) {
  return { type, player, id: newId(), promoted: false };
}

/** 盤上に置くための新規セルを作る（打ち用） */
export function makeCellForPlayer(type, player) {
  return makeCell(type, player);
}

/**
 * Board クラス
 * board[x][y][z]  (各 0-indexed、UI では 1-indexed)
 */
export class Board {
  constructor() {
    // 3D 配列を初期化 (9x9x9)
    this.cells = [];
    for (let x = 0; x < 9; x++) {
      this.cells[x] = [];
      for (let y = 0; y < 9; y++) {
        this.cells[x][y] = [];
        for (let z = 0; z < 9; z++) {
          this.cells[x][y][z] = emptyCell();
        }
      }
    }
  }

  get(x, y, z) {
    if (x < 0 || x > 8 || y < 0 || y > 8 || z < 0 || z > 8) return null;
    return this.cells[x][y][z];
  }

  set(x, y, z, cell) {
    this.cells[x][y][z] = cell;
  }

  // (x,y,z) が範囲内か
  inBounds(x, y, z) {
    return x >= 0 && x <= 8 && y >= 0 && y <= 8 && z >= 0 && z <= 8;
  }

  // 空セルを設定
  clear(x, y, z) {
    this.cells[x][y][z] = emptyCell();
  }

  // 駒を移動（取った駒を返す）
  move(fx, fy, fz, tx, ty, tz) {
    const from = this.get(fx, fy, fz);
    const to = this.get(tx, ty, tz);
    const captured = to.type !== PieceType.NONE ? { ...to } : null;
    this.cells[tx][ty][tz] = { ...from };
    this.cells[fx][fy][fz] = emptyCell();
    return captured;
  }

  getHash(currentPlayer, hands) {
    // 盤面
    let s = "";
    for(let x=0; x<9; x++) for(let y=0; y<9; y++) for(let z=0; z<9; z++) {
      const c = this.cells[x][y][z];
      if (c.type !== 'none') s += `${x}${y}${z}${c.player}${c.type},`;
    }
    // 持ち駒 (ソートして一意にする)
    const hSente = (hands[Player.SENTE] || []).map(p => p.type).sort().join('');
    const hGote  = (hands[Player.GOTE]  || []).map(p => p.type).sort().join('');
    return `${s}|${hSente}|${hGote}|${currentPlayer}`;
  }
}

// ============================================================
// 初期配置
// ============================================================

/**
 * xz 断面の配置表から Board にセットする
 * grid[z][x] = PieceType (z=0 が手前)
 * y: 0-indexed
 */
function placePlane(board, grid, y, player) {
  for (let z = 0; z < 9; z++) {
    for (let x = 0; x < 9; x++) {
      const type = grid[z][x];
      if (type !== PieceType.NONE) {
        board.set(x, y, z, makeCell(type, player));
      }
    }
  }
}

// y=1(0-indexed y=0), y=7(0-indexed y=6) の XZ断面
// 先手の配置（y=0）/ 後手の配置（y=6）ミラー
// ユーザー指定: 行がz方向、列がx方向
const _ = PieceType.NONE;
const K = PieceType.KYO;
const N = PieceType.KEI;
const S = PieceType.GIN;
const G = PieceType.KIN;
const O = PieceType.OU;

const PLANE_1_7 = [
  [K, K, K, K, K, K, K, K, K],
  [K, N, N, N, N, N, N, N, K],
  [K, N, S, S, G, S, S, N, K],
  [K, N, S, G, G, G, S, N, K],
  [K, N, G, G, O, G, G, N, K],
  [K, N, S, G, G, G, S, N, K],
  [K, N, S, S, G, S, S, N, K],
  [K, N, N, N, N, N, N, N, K],
  [K, K, K, K, K, K, K, K, K],
];

const A = PieceType.KAKU;
const H = PieceType.HI;

const PLANE_2_8 = [
  [_, _, _, _, _, _, _, _, _],
  [_, A, _, _, H, _, _, A, _],
  [_, _, _, _, _, _, _, _, _],
  [_, _, _, _, _, _, _, _, _],
  [_, H, _, _, _, _, _, H, _],
  [_, _, _, _, _, _, _, _, _],
  [_, _, _, _, _, _, _, _, _],
  [_, A, _, _, H, _, _, A, _],
  [_, _, _, _, _, _, _, _, _],
];

const F = PieceType.FU;
const PLANE_3_9_FU = Array.from({ length: 9 }, () => Array(9).fill(F));

export function initBoard() {
  pieceIdCounter = 0;
  const board = new Board();

  // 後手 (y=0,1,2 = 王側)
  placePlane(board, PLANE_1_7, 0, Player.GOTE);
  placePlane(board, PLANE_2_8, 1, Player.GOTE);
  placePlane(board, PLANE_3_9_FU, 2, Player.GOTE);

  // 先手 (y=8,7,6 = 玉側)
  placePlane(board, PLANE_1_7, 8, Player.SENTE);
  placePlane(board, PLANE_2_8, 7, Player.SENTE);
  placePlane(board, PLANE_3_9_FU, 6, Player.SENTE);

  return board;
}

// 成り可能エリア判定 (0-indexed)
export function isPromotionZone(y, player) {
  if (player === Player.SENTE) return y <= 2; // y=1,2,3 (奥側/後手陣)
  if (player === Player.GOTE) return y >= 6;  // y=7,8,9 (手前側/先手陣)
  return false;
}
