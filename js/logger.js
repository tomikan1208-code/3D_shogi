// logger.js - 棋譜ログ管理

'use strict';

import { PieceName } from './board.js';
import { ROMAN, KANJIDIGIT } from './pieces.js';

// 0-indexed → ログ表示用
export function toDisplayX(x) { return x + 1; }           // アラビア数字
export function toDisplayY(y) { return KANJIDIGIT[y]; }   // 漢数字
export function toDisplayZ(z) { return ROMAN[z]; }         // ローマ数字

const PLAYER_SYMBOL = { 1: '▲', 2: '△' };

export class Logger {
  constructor() {
    this.logs = [];
  }

   /**
   * 移動ログを追加
   * @param {number} player - Player.SENTE(1) or GOTE(2)
   * @param {number} tx,ty,tz - 移動先 (0-indexed)
   * @param {string} pieceType - 移動後の駒タイプ
   * @param {number} id - 駒 ID
   * @param {boolean} promoted - 今回成ったか
   * @param {string} capturedType - 取った駒のタイプ (なければ null)
   */
  addMove(player, tx, ty, tz, pieceType, id, promoted, capturedType = null) {
    const sym = PLAYER_SYMBOL[player] || '?';
    const px = toDisplayX(tx);
    const py = toDisplayY(ty);
    const pz = toDisplayZ(tz);
    const name = PieceName[pieceType] || '?';
    const capName = capturedType ? PieceName[capturedType] : '';
    const capStr = capName ? `${capName}取り` : '';
    const promoStr = promoted ? '成' : '';
    
    // ユーザー要望の形式: y-x-z (例: 一3IV歩) + 取った駒
    const entry = `${sym}${py}${px}${pz}${name}${promoStr}${capStr}`;
    this.logs.push({ text: entry, player, turn: this.logs.length + 1 });
    return entry;
  }

  getAll() { return this.logs; }

  getLast() { return this.logs[this.logs.length - 1] || null; }

  clear() { this.logs = []; }
}
