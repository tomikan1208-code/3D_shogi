// game.js - ローカル専用ゲームメインロジック・状態管理

'use strict';

import { initBoard, Player, isPromotionZone, canPromote, PromotionMap, PieceType, makeCellForPlayer, demoteType, PieceName } from './board.js';
import { getLegalMoves } from './pieces.js';
import { Logger } from './logger.js';

export const GameMode = { LOCAL: 'local', NPC: 'npc' };
export const GamePhase = {
  IDLE: 'idle',
  SELECT: 'select',
  MOVE: 'move',
  CONFIRM: 'confirm',
  DROP: 'drop',
  DROP_CONFIRM: 'drop_confirm',
  PROMOTE: 'promote',
  DONE: 'done',
};

const AI_PIECE_TYPES = [
  PieceType.NONE,
  PieceType.FU,
  PieceType.KYO,
  PieceType.KEI,
  PieceType.GIN,
  PieceType.KIN,
  PieceType.KAKU,
  PieceType.HI,
  PieceType.OU,
  PieceType.NARI_FU,
  PieceType.NARI_KYO,
  PieceType.NARI_KEI,
  PieceType.NARI_GIN,
  PieceType.UMA,
  PieceType.RYU,
];

const AI_DROP_TYPES = [
  PieceType.FU,
  PieceType.KYO,
  PieceType.KEI,
  PieceType.GIN,
  PieceType.KIN,
  PieceType.KAKU,
  PieceType.HI,
];

export class GameState {
  constructor(mode = GameMode.LOCAL) {
    this.mode = mode;
    this.board = null;
    this.currentPlayer = Player.SENTE;
    this.phase = GamePhase.IDLE;
    this.selectedPos = null;
    this.legalMoves = [];
    this.confirmedPos = null;
    this.selectedHandType = null;
    this.lastFrom = null;
    this.lastTo = null;
    this.logger = new Logger();
    this.capturedPieces = { [Player.SENTE]: [], [Player.GOTE]: [] };
    this.winnerPlayer = null;
    this.onStateChange = null;
    this.highlightedSlice = null; // { axis: 'x'|'y'|'z', index: 0-8 }
    this.piecePositions = { [Player.SENTE]: new Set(), [Player.GOTE]: new Set() };
    this.kingPos = { [Player.SENTE]: null, [Player.GOTE]: null };
    this.npcPlayer = Player.GOTE;
    this.humanPlayer = Player.SENTE;
  }

  setNpcPlayer(npcPlayer = Player.GOTE) {
    this.npcPlayer = npcPlayer;
    this.humanPlayer = npcPlayer === Player.SENTE ? Player.GOTE : Player.SENTE;
  }

  startGame() {
    this.board = initBoard();
    this.currentPlayer = Player.SENTE;
    this.phase = GamePhase.SELECT;
    this.selectedPos = null;
    this.legalMoves = [];
    this.confirmedPos = null;
    this.selectedHandType = null;
    this.highlightedSlice = null;
    this.lastFrom = null;
    this.lastTo = null;
    this.logger.clear();
    this.capturedPieces = { [Player.SENTE]: [], [Player.GOTE]: [] };
    this.winnerPlayer = null;
    this._rebuildPieceCaches();
    this._notify();
  }

  getHandSummary(player = this.currentPlayer) {
    const mine = this.capturedPieces[player] || [];
    const counts = new Map();
    for (const p of mine) counts.set(p.type, (counts.get(p.type) || 0) + 1);
    const order = [
      PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN,
      PieceType.KIN, PieceType.KAKU, PieceType.HI
    ];
    const res = [];
    for (const t of order) {
      const c = counts.get(t) || 0;
      res.push({ type: t, name: PieceName[t] || '?', count: c });
    }
    return res;
  }

  toggleHandSelect(type) {
    if (this.phase === GamePhase.IDLE || this.phase === GamePhase.DONE || this.phase === GamePhase.PROMOTE) return;
    if (this.mode === GameMode.NPC && this.currentPlayer !== this.humanPlayer) return;
    if (this.selectedHandType === type) {
      this._cancelSelect();
      return;
    }
    this._cancelSelect(false);
    this.selectedHandType = type;
    this.legalMoves = this._getDropTargets(type);
    this.phase = GamePhase.DROP;
    this._notify();
  }

  handleLabelClick(axis, index) {
    if (this.highlightedSlice && this.highlightedSlice.axis === axis && this.highlightedSlice.index === index) {
      this.highlightedSlice = null;
    } else {
      this.highlightedSlice = { axis, index };
    }
    // 断面が切り替わっても選択を維持し、移動可能範囲を再計算する
    this._updateLegalMovesAfterSliceChange();
    this._notify();
  }

  resetHighlight() {
    if (this.highlightedSlice) {
      this.highlightedSlice = null;
      this._updateLegalMovesAfterSliceChange();
      this._notify();
    }
  }

  // 断面変更後に現在の選択内容（駒または持ち駒）の移動可能範囲を更新する
  _updateLegalMovesAfterSliceChange() {
    if (this.selectedPos) {
      this.legalMoves = this._getLegalMovesSafe(this.selectedPos.x, this.selectedPos.y, this.selectedPos.z);
    } else if (this.selectedHandType) {
      this.legalMoves = this._getDropTargets(this.selectedHandType);
    }
  }

  handleClick(x, y, z) {
    if (this.phase === GamePhase.IDLE || this.phase === GamePhase.DONE || this.phase === GamePhase.PROMOTE) return;
    if (this.mode === GameMode.NPC && this.currentPlayer !== this.humanPlayer) return;

    // 断面制限: 座標が断面内のみ入力を許可（他層への移動例外は廃止）
    if (this.highlightedSlice) {
      const { axis, index } = this.highlightedSlice;
      const isInSlice = (axis === 'x' && x === index) || (axis === 'y' && y === index) || (axis === 'z' && z === index);
      if (!isInSlice) return;
    }

    const cell = this.board.get(x, y, z);

    if (this.phase === GamePhase.DROP) {
      if (cell && cell.player === this.currentPlayer) {
        this.selectedHandType = null;
        this.selectedPos = { x, y, z };
        this.legalMoves = this._getLegalMovesSafe(x, y, z);
        this.phase = GamePhase.MOVE;
        this._notify();
        return;
      }
      if (this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z)) {
        this.confirmedPos = { x, y, z };
        this.phase = GamePhase.DROP_CONFIRM;
        this._notify();
        return;
      }
      this._cancelSelect();
      return;
    }

    if (this.phase === GamePhase.DROP_CONFIRM) {
      if (x === this.confirmedPos?.x && y === this.confirmedPos?.y && z === this.confirmedPos?.z) {
        this._executeDrop(x, y, z);
        return;
      }
      if (cell && cell.player === this.currentPlayer) {
        this.selectedHandType = null;
        this.selectedPos = { x, y, z };
        this.legalMoves = this._getLegalMovesSafe(x, y, z);
        this.confirmedPos = null;
        this.phase = GamePhase.MOVE;
        this._notify();
        return;
      }
      if (this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z)) {
        this.confirmedPos = { x, y, z };
        this._notify();
        return;
      }
      this._cancelSelect();
      return;
    }

    if (this.phase === GamePhase.SELECT) {
      if (cell && cell.player === this.currentPlayer) {
        this.selectedPos = { x, y, z };
        this.legalMoves = this._getLegalMovesSafe(x, y, z);
        this.phase = GamePhase.MOVE;
        this._notify();
      }
      return;
    }

    if (this.phase === GamePhase.MOVE) {
      if (cell && cell.player === this.currentPlayer) {
        this.selectedPos = { x, y, z };
        this.legalMoves = this._getLegalMovesSafe(x, y, z);
        this._notify();
        return;
      }
      if (this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z)) {
        this.confirmedPos = { x, y, z };
        this.phase = GamePhase.CONFIRM;
        this._notify();
        return;
      }
      this._cancelSelect();
      return;
    }

    if (this.phase === GamePhase.CONFIRM) {
      if (x === this.confirmedPos?.x && y === this.confirmedPos?.y && z === this.confirmedPos?.z) {
        this._executeMove(x, y, z);
        return;
      }
      if (cell && cell.player === this.currentPlayer) {
        this.selectedPos = { x, y, z };
        this.legalMoves = this._getLegalMovesSafe(x, y, z);
        this.confirmedPos = null;
        this.phase = GamePhase.MOVE;
        this._notify();
        return;
      }
      if (this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z)) {
        this.confirmedPos = { x, y, z };
        this._notify();
        return;
      }
      this._cancelSelect();
    }
  }

  handlePromotion(doPromote) {
    if (this.phase !== GamePhase.PROMOTE) return;
    if (doPromote) {
      const cell = this.board.get(this.lastTo.x, this.lastTo.y, this.lastTo.z);
      const newType = PromotionMap[cell.type];
      if (newType) { cell.type = newType; cell.promoted = true; }
    }
    const cell = this.board.get(this.lastTo.x, this.lastTo.y, this.lastTo.z);
    this.logger.addMove(this.currentPlayer, this.lastTo.x, this.lastTo.y, this.lastTo.z, cell.type, cell.id, doPromote, this.lastCapturedType);
    this.lastCapturedType = null;
    this._endTurn();
  }

  _executeMove(tx, ty, tz) {
    const { x: fx, y: fy, z: fz } = this.selectedPos;
    const captured = this.board.move(fx, fy, fz, tx, ty, tz);

    this._removePiecePos(this.currentPlayer, fx, fy, fz);
    this._addPiecePos(this.currentPlayer, tx, ty, tz);

    if (captured) {
      const opponent = this.currentPlayer === Player.SENTE ? Player.GOTE : Player.SENTE;
      this._removePiecePos(opponent, tx, ty, tz);
      captured.type = demoteType(captured.type);
      captured.promoted = false;
      captured.player = this.currentPlayer;
      this.capturedPieces[this.currentPlayer].push(captured);
      if (captured.type === PieceType.OU) {
        this.lastFrom = { x: fx, y: fy, z: fz }; this.lastTo = { x: tx, y: ty, z: tz };
        this.winnerPlayer = this.currentPlayer;
        this.phase = GamePhase.DONE;
        this._notify();
        return;
      }
    }
    this.lastFrom = { x: fx, y: fy, z: fz };
    this.lastTo = { x: tx, y: ty, z: tz };

    const movedCell = this.board.get(tx, ty, tz);
    if (movedCell && movedCell.type === PieceType.OU) {
      this.kingPos[this.currentPlayer] = { x: tx, y: ty, z: tz };
    }

    this.selectedPos = null; this.confirmedPos = null; this.legalMoves = []; this.selectedHandType = null;

    const cell = movedCell;
    const promoBy = isPromotionZone(fy, this.currentPlayer) || isPromotionZone(ty, this.currentPlayer);
    if (promoBy && canPromote(cell.type)) {
      this.lastCapturedType = captured ? captured.type : null;
      this.phase = GamePhase.PROMOTE;
      this._notify();
    } else {
      this.logger.addMove(this.currentPlayer, tx, ty, tz, cell.type, cell.id, false, captured ? captured.type : null);
      this._endTurn();
    }
  }

  _getDropTargets(type) {
    const res = [];
    for (let x = 0; x < 9; x++)
      for (let y = 0; y < 9; y++)
        for (let z = 0; z < 9; z++) {
          if (this.board.get(x, y, z).type !== PieceType.NONE) continue;
          if (type === PieceType.FU && this._isNiFuColumn(x, z, this.currentPlayer)) continue;
          if (!this._isDropSafe(type, x, y, z, this.currentPlayer)) continue;
          
          // 断面制限
          if (this.highlightedSlice) {
            const { axis, index } = this.highlightedSlice;
            if (axis === 'x' && x !== index) continue;
            if (axis === 'y' && y !== index) continue;
            if (axis === 'z' && z !== index) continue;
          }
          
          res.push([x, y, z]);
        }
    return res;
  }

  _executeDrop(tx, ty, tz) {
    const type = this.selectedHandType;
    const hand = this.capturedPieces[this.currentPlayer];
    const idx = hand.findIndex(p => p.type === type);
    if (idx < 0) return;
    hand.splice(idx, 1);
    this.board.set(tx, ty, tz, makeCellForPlayer(type, this.currentPlayer));
    this._addPiecePos(this.currentPlayer, tx, ty, tz);
    this.lastFrom = null; this.lastTo = { x: tx, y: ty, z: tz };
    const cell = this.board.get(tx, ty, tz);
    this.logger.addMove(this.currentPlayer, tx, ty, tz, type, -1, false, null); // No captured piece for a drop
    this._endTurn();
  }

  exportForAI() {
    const board = Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => Array(9).fill(0)));
    const owner = Array.from({ length: 9 }, () => Array.from({ length: 9 }, () => Array(9).fill(0)));

    for (let x = 0; x < 9; x++) {
      for (let y = 0; y < 9; y++) {
        for (let z = 0; z < 9; z++) {
          const cell = this.board.get(x, y, z);
          if (!cell || cell.type === PieceType.NONE) continue;
          board[x][y][z] = AI_PIECE_TYPES.indexOf(cell.type);
          owner[x][y][z] = cell.player;
        }
      }
    }

    const countHand = (player) => {
      const counts = Array(7).fill(0);
      for (const p of this.capturedPieces[player] || []) {
        const idx = AI_DROP_TYPES.indexOf(p.type);
        if (idx >= 0) counts[idx] += 1;
      }
      return counts;
    };

    return {
      board,
      owner,
      hand_sente: countHand(Player.SENTE),
      hand_gote: countHand(Player.GOTE),
      current_player: this.currentPlayer,
    };
  }

  applyAIMove(move) {
    if (!move || this.phase === GamePhase.DONE || this.phase === GamePhase.PROMOTE) return false;
    if (this.mode === GameMode.NPC && this.currentPlayer !== this.npcPlayer) return false;

    const tx = Number(move.tx);
    const ty = Number(move.ty);
    const tz = Number(move.tz);
    const isDrop = Boolean(move.is_drop);
    const promote = Boolean(move.promote);

    if (isDrop) {
      const dropType = AI_DROP_TYPES[Number(move.drop_type)];
      if (!dropType) return false;

      const hand = this.capturedPieces[this.currentPlayer] || [];
      const idx = hand.findIndex((p) => p.type === dropType);
      if (idx < 0) return false;

      hand.splice(idx, 1);
      this.board.set(tx, ty, tz, makeCellForPlayer(dropType, this.currentPlayer));
      this._addPiecePos(this.currentPlayer, tx, ty, tz);
      this.lastFrom = null;
      this.lastTo = { x: tx, y: ty, z: tz };
      this.logger.addMove(this.currentPlayer, tx, ty, tz, dropType, -1, false, null);
      this._endTurn();
      return true;
    }

    const fx = Number(move.fx);
    const fy = Number(move.fy);
    const fz = Number(move.fz);
    const fromCell = this.board.get(fx, fy, fz);
    if (!fromCell || fromCell.type === PieceType.NONE || fromCell.player !== this.currentPlayer) return false;

    const captured = this.board.move(fx, fy, fz, tx, ty, tz);
    this._removePiecePos(this.currentPlayer, fx, fy, fz);
    this._addPiecePos(this.currentPlayer, tx, ty, tz);

    let loggedCapturedType = null;
    if (captured) {
      const opponent = this.currentPlayer === Player.SENTE ? Player.GOTE : Player.SENTE;
      this._removePiecePos(opponent, tx, ty, tz);

      captured.type = demoteType(captured.type);
      captured.promoted = false;
      captured.player = this.currentPlayer;
      this.capturedPieces[this.currentPlayer].push(captured);
      loggedCapturedType = captured.type;

      if (captured.type === PieceType.OU) {
        this.lastFrom = { x: fx, y: fy, z: fz };
        this.lastTo = { x: tx, y: ty, z: tz };
        this.winnerPlayer = this.currentPlayer;
        this.phase = GamePhase.DONE;
        this._notify();
        return true;
      }
    }

    const movedCell = this.board.get(tx, ty, tz);
    if (movedCell && movedCell.type === PieceType.OU) {
      this.kingPos[this.currentPlayer] = { x: tx, y: ty, z: tz };
    }

    const promoBy = isPromotionZone(fy, this.currentPlayer) || isPromotionZone(ty, this.currentPlayer);
    if (promote && movedCell && promoBy && canPromote(movedCell.type)) {
      const promotedType = PromotionMap[movedCell.type];
      if (promotedType) {
        movedCell.type = promotedType;
        movedCell.promoted = true;
      }
    }

    this.lastFrom = { x: fx, y: fy, z: fz };
    this.lastTo = { x: tx, y: ty, z: tz };
    this.selectedPos = null;
    this.confirmedPos = null;
    this.legalMoves = [];
    this.selectedHandType = null;

    this.logger.addMove(this.currentPlayer, tx, ty, tz, movedCell.type, movedCell.id, promote, loggedCapturedType);
    this._endTurn();
    return true;
  }

  _endTurn() {
    const next = this.currentPlayer === Player.SENTE ? Player.GOTE : Player.SENTE;
    this.currentPlayer = next;
    this.phase = GamePhase.SELECT;
    this.selectedHandType = null; this.confirmedPos = null; this.legalMoves = [];
    if (this._isKingInCheck(next) && !this._hasAnyLegalMove(next)) {
      this.winnerPlayer = (next === Player.SENTE ? Player.GOTE : Player.SENTE);
      this.phase = GamePhase.DONE;
    }
    this._notify();
  }

  cancelSelection() {
    this._cancelSelect();
  }

  _cancelSelect(notify = true) {
    this.selectedPos = null; this.legalMoves = []; this.confirmedPos = null; this.selectedHandType = null;
    this.phase = GamePhase.SELECT;
    if (notify) this._notify();
  }

  _notify() { if (typeof this.onStateChange === 'function') this.onStateChange(this); }

  _findKing(player) {
    return this.kingPos[player];
  }

  _isKingInCheck(player) {
    const k = this._findKing(player);
    if (!k) return false;
    const attacker = player === Player.SENTE ? Player.GOTE : Player.SENTE;
    for (const key of this.piecePositions[attacker]) {
      const [x, y, z] = key.split(',').map(Number);
      if (getLegalMoves(this.board, x, y, z).some(m => m[0] === k.x && m[1] === k.y && m[2] === k.z)) return true;
    }
    return false;
  }

  _hasAnyLegalMove(player) {
    for (const key of this.piecePositions[player]) {
      const [fx, fy, fz] = key.split(',').map(Number);
      if (getLegalMoves(this.board, fx, fy, fz).some(([tx, ty, tz]) => this._isMoveSafe(fx, fy, fz, tx, ty, tz, player))) return true;
    }
    const hand = this.capturedPieces[player];
    for (const type of new Set(hand.map(p => p.type))) {
      for (let x = 0; x < 9; x++)
        for (let y = 0; y < 9; y++)
          for (let z = 0; z < 9; z++) {
            if (this.board.get(x, y, z).type !== PieceType.NONE) continue;
            if (type === PieceType.FU && this._isNiFuColumn(x, z, player)) continue;
            if (this._isDropSafe(type, x, y, z, player)) return true;
          }
    }
    return false;
  }

  _getLegalMovesSafe(fx, fy, fz) {
    let moves = getLegalMoves(this.board, fx, fy, fz).filter(([tx, ty, tz]) => this._isMoveSafe(fx, fy, fz, tx, ty, tz, this.currentPlayer));
    
    // 断面制限: 移動先も断面内に限定
    if (this.highlightedSlice) {
      const { axis, index } = this.highlightedSlice;
      moves = moves.filter(([tx, ty, tz]) => {
        if (axis === 'x') return tx === index;
        if (axis === 'y') return ty === index;
        if (axis === 'z') return tz === index;
        return true;
      });
    }
    return moves;
  }

  _isMoveSafe(fx, fy, fz, tx, ty, tz, player) {
    const from = this.board.get(fx, fy, fz); const to = this.board.get(tx, ty, tz);
    const sf = { ...from }; const st = { ...to };
    const opponent = player === Player.SENTE ? Player.GOTE : Player.SENTE;
    const prevKing = this.kingPos[player] ? { ...this.kingPos[player] } : null;

    this._removePiecePos(player, fx, fy, fz);
    this._addPiecePos(player, tx, ty, tz);
    if (st.type !== PieceType.NONE && st.player === opponent) {
      this._removePiecePos(opponent, tx, ty, tz);
    }
    if (sf.type === PieceType.OU) {
      this.kingPos[player] = { x: tx, y: ty, z: tz };
    }

    this.board.set(tx, ty, tz, { ...from }); this.board.clear(fx, fy, fz);
    const ok = !this._isKingInCheck(player);
    this.board.set(fx, fy, fz, sf); this.board.set(tx, ty, tz, st);

    this._removePiecePos(player, tx, ty, tz);
    this._addPiecePos(player, fx, fy, fz);
    if (st.type !== PieceType.NONE && st.player === opponent) {
      this._addPiecePos(opponent, tx, ty, tz);
    }
    if (sf.type === PieceType.OU) {
      this.kingPos[player] = prevKing;
    }

    return ok;
  }

  _isDropSafe(type, tx, ty, tz, player) {
    const to = this.board.get(tx, ty, tz); const st = { ...to };
    this._addPiecePos(player, tx, ty, tz);
    this.board.set(tx, ty, tz, { type, player, id: -1, promoted: false });
    const ok = !this._isKingInCheck(player);
    this.board.set(tx, ty, tz, st);
    this._removePiecePos(player, tx, ty, tz);
    return ok;
  }

  _isNiFuColumn(x, z, player) {
    for (let y = 0; y < 9; y++) {
      const c = this.board.get(x, y, z);
      if (c && c.player === player && c.type === PieceType.FU) return true;
    }
    return false;
  }

  _posKey(x, y, z) {
    return `${x},${y},${z}`;
  }

  _addPiecePos(player, x, y, z) {
    if (player !== Player.SENTE && player !== Player.GOTE) return;
    this.piecePositions[player].add(this._posKey(x, y, z));
  }

  _removePiecePos(player, x, y, z) {
    if (player !== Player.SENTE && player !== Player.GOTE) return;
    this.piecePositions[player].delete(this._posKey(x, y, z));
  }

  _rebuildPieceCaches() {
    this.piecePositions = { [Player.SENTE]: new Set(), [Player.GOTE]: new Set() };
    this.kingPos = { [Player.SENTE]: null, [Player.GOTE]: null };
    for (let x = 0; x < 9; x++)
      for (let y = 0; y < 9; y++)
        for (let z = 0; z < 9; z++) {
          const c = this.board.get(x, y, z);
          if (!c || c.type === PieceType.NONE) continue;
          this._addPiecePos(c.player, x, y, z);
          if (c.type === PieceType.OU) this.kingPos[c.player] = { x, y, z };
        }
  }
}
