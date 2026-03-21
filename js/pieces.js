// pieces.js - 駒の移動ロジック

'use strict';

import { PieceType, Player } from './board.js';

// ローマ数字（z座標用）
export const ROMAN = ['I','II','III','IV','V','VI','VII','VIII','IX'];
// 漢数字（y座標用）
export const KANJIDIGIT = ['一','二','三','四','五','六','七','八','九'];

/**
 * 全方位1マス（王）の相対座標を返す
 */
function allAdjacent() {
  const moves = [];
  for (let dy = -1; dy <= 1; dy++)
    for (let dx = -1; dx <= 1; dx++)
      for (let dz = -1; dz <= 1; dz++)
        if (dx !== 0 || dy !== 0 || dz !== 0)
          moves.push([dx, dy, dz]);
  return moves;
}

/**
 * 金の相対座標
 * 前方(dy=+1): 3×3 全9マス
 * 同一面(dy=0): 周囲8マス
 * 後方(dy=-1): 真後ろ1マスのみ (dx=0, dz=0)
 */
function kinMoves(dir) {
  const moves = [];
  // dy=+dir (前方)
  for (let dx = -1; dx <= 1; dx++)
    for (let dz = -1; dz <= 1; dz++)
      moves.push([dx, dir, dz]);
  // dy=0 (同一面)
  for (let dx = -1; dx <= 1; dx++)
    for (let dz = -1; dz <= 1; dz++)
      if (dx !== 0 || dz !== 0)
        moves.push([dx, 0, dz]);
  // dy=-dir (真後ろ)
  moves.push([0, -dir, 0]);
  return moves;
}

/**
 * 銀の相対座標
 * 前方(dy=+1): 3×3 全9マス
 * 後方(dy=-1): 中央除く8マス
 */
function ginMoves(dir) {
  const moves = [];
  // dy=+dir 前方 全9
  for (let dx = -1; dx <= 1; dx++)
    for (let dz = -1; dz <= 1; dz++)
      moves.push([dx, dir, dz]);
  // dy=-dir 後方 中央以外
  for (let dx = -1; dx <= 1; dx++)
    for (let dz = -1; dz <= 1; dz++)
      if (dx !== 0 || dz !== 0)
        moves.push([dx, -dir, dz]);
  return moves;
}

/**
 * 桂馬: y+2 確定、かつ (x±1 または z±1) の 4マス
 */
function keiMoves(dir) {
  return [
    [ 1, 2*dir,  0],
    [-1, 2*dir,  0],
    [ 0, 2*dir,  1],
    [ 0, 2*dir, -1],
  ];
}

/**
 * 歩: y+1のみ
 */
function fuMoves(dir) {
  return [[0, dir, 0]];
}

/**
 * 角の方向ベクトル（12方向）
 */
const KAKU_DIRS = [];
for (const sx of [-1, 1])
  for (const sy of [-1, 1])
    for (const sz of [-1, 1]) {
      // (±x, ±z)
      KAKU_DIRS.push([sx, 0, sz]);
      // (±x, ±y)
      KAKU_DIRS.push([sx, sy, 0]);
      // (±y, ±z)
      KAKU_DIRS.push([0, sy, sz]);
    }
// 重複除去
const KAKU_DIRS_UNIQUE = [
  ...new Set(KAKU_DIRS.map(d => d.join(',')))
].map(s => s.split(',').map(Number));

/**
 * 飛車の方向ベクトル（6方向）
 */
const HI_DIRS = [
  [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
];

/**
 * 飛び駒の移動先を計算（障害物で止まる、取れる）
 */
function slidingMoves(board, fx, fy, fz, dirs, player) {
  const moves = [];
  for (const [dx, dy, dz] of dirs) {
    let x = fx + dx, y = fy + dy, z = fz + dz;
    while (board.inBounds(x, y, z)) {
      const cell = board.get(x, y, z);
      if (cell.type !== 'none') {
        if (cell.player !== player) moves.push([x, y, z]); // 取れる
        break;
      }
      moves.push([x, y, z]);
      x += dx; y += dy; z += dz;
    }
  }
  return moves;
}

/**
 * 香車（+y方向のみ。後手は-y）
 */
function kyoMoves(board, fx, fy, fz, player) {
  const dir = player === Player.SENTE ? -1 : 1;
  return slidingMoves(board, fx, fy, fz, [[0, dir, 0]], player);
}

/**
 * 成り竜（飛車+王）
 */
function ryuMoves(board, fx, fy, fz, player) {
  const sliding = slidingMoves(board, fx, fy, fz, HI_DIRS, player);
  const adj = allAdjacent()
    .map(([dx, dy, dz]) => [fx+dx, fy+dy, fz+dz])
    .filter(([x, y, z]) => board.inBounds(x, y, z) && board.get(x, y, z).player !== player);
  const all = [...sliding, ...adj];
  return dedup(all);
}

/**
 * 成り馬（角+王）
 */
function umaMoves(board, fx, fy, fz, player) {
  const sliding = slidingMoves(board, fx, fy, fz, KAKU_DIRS_UNIQUE, player);
  const adj = allAdjacent()
    .map(([dx, dy, dz]) => [fx+dx, fy+dy, fz+dz])
    .filter(([x, y, z]) => board.inBounds(x, y, z) && board.get(x, y, z).player !== player);
  const all = [...sliding, ...adj];
  return dedup(all);
}

function dedup(moves) {
  const seen = new Set();
  return moves.filter(([x, y, z]) => {
    const k = `${x},${y},${z}`;
    if (seen.has(k)) return false;
    seen.add(k); return true;
  });
}

/**
 * 歩・金系（ステップ系）の moves をフィルタ
 */
function filterMoves(board, fx, fy, fz, relMoves, player) {
  return relMoves
    .map(([dx, dy, dz]) => [fx+dx, fy+dy, fz+dz])
    .filter(([x, y, z]) =>
      board.inBounds(x, y, z) && board.get(x, y, z).player !== player
    );
}

/**
 * 指定した駒の移動可能マスリストを返す
 * returns: [[x, y, z], ...]  (0-indexed)
 */
export function getLegalMoves(board, fx, fy, fz) {
  const cell = board.get(fx, fy, fz);
  if (!cell || cell.type === PieceType.NONE) return [];

  const { type, player } = cell;
  const dir = player === Player.SENTE ? -1 : 1;

  switch (type) {
    case PieceType.OU:
      return filterMoves(board, fx, fy, fz, allAdjacent(), player);

    case PieceType.KIN:
    case PieceType.NARI_FU:
    case PieceType.NARI_KYO:
    case PieceType.NARI_KEI:
    case PieceType.NARI_GIN:
      return filterMoves(board, fx, fy, fz, kinMoves(dir), player);

    case PieceType.GIN:
      return filterMoves(board, fx, fy, fz, ginMoves(dir), player);

    case PieceType.KEI:
      return filterMoves(board, fx, fy, fz, keiMoves(dir), player);

    case PieceType.FU:
      return filterMoves(board, fx, fy, fz, fuMoves(dir), player);

    case PieceType.KYO:
      return kyoMoves(board, fx, fy, fz, player);

    case PieceType.KAKU:
      return slidingMoves(board, fx, fy, fz, KAKU_DIRS_UNIQUE, player);

    case PieceType.HI:
      return slidingMoves(board, fx, fy, fz, HI_DIRS, player);

    case PieceType.RYU:
      return ryuMoves(board, fx, fy, fz, player);

    case PieceType.UMA:
      return umaMoves(board, fx, fy, fz, player);

    default:
      return [];
  }
}
