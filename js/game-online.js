// game-online.js - オンライン対戦専用のゲームメインロジック・状態管理

'use strict';

import { initBoard, Player, isPromotionZone, canPromote, PromotionMap, PieceType, makeCellForPlayer, demoteType, PieceName } from './board.js';
import { getLegalMoves } from './pieces.js';
import { Logger } from './logger.js';

export const GameMode = { NETWORK: 'network' };
export const GamePhase = {
  IDLE: 'idle',       // 待機（ゲーム未開始）
  SELECT: 'select',   // 駒選択待ち
  MOVE: 'move',       // 移動先選択待ち（青いハイライト表示中）
  CONFIRM: 'confirm', // 移動先クリック済み・緑確定待ち
  DROP: 'drop',       // 手持ち駒の置き先選択（青）
  DROP_CONFIRM: 'drop_confirm', // 置き先確認（緑）
  PROMOTE: 'promote', // 成り確認中
  DONE: 'done',       // ゲーム終了
};

export class GameState {
  constructor() {
    this.mode = GameMode.NETWORK;
    this.board = null;
    this.currentPlayer = Player.SENTE;
    this.phase = GamePhase.IDLE;
    this.selectedPos = null;   // {x,y,z}
    this.legalMoves = [];      // [[x,y,z], ...]
    this.confirmedPos = null;  // 移動先（確定待ち）
    this.selectedHandType = null; // 手持ちから選んだ駒種（打ち）
    this.lastFrom = null;      // 移動元（赤表示用）
    this.lastTo = null;        // 移動先（緑表示用）
    this.logger = new Logger();
    this.capturedPieces = { [Player.SENTE]: [], [Player.GOTE]: [] };
    this.winnerPlayer = null;
    this.playerSide = null; // 自分は SENTE(1) か GOTE(2) か
    this.onStateChange = null; // コールバック
    this.highlightedSlice = null; // { axis, index }
    this.opponentInteraction = {
      hoverPos: null,
      selectedPos: null,
      legalMoves: [],
      selectedHandType: null
    };
    this.lastCapturedType = null;
    this.piecePositions = { [Player.SENTE]: new Set(), [Player.GOTE]: new Set() };
    this.kingPos = { [Player.SENTE]: null, [Player.GOTE]: null };
  }

  /**
   * ネットワークモードとしての初期化
   * @param {number} playerIndex 0: sente, 1: gote
   */
  initNetworkMode(playerIndex) {
    this.playerIndex = playerIndex;
    this.playerSide = (playerIndex === 0) ? Player.SENTE : Player.GOTE;
    this.startGame();
    console.log(`[GameState] Network mode initialized. I am ${this.playerSide === Player.SENTE ? 'SENTE' : 'GOTE'}`);
  }

  /** 相手からのアクションを受信して反映 */
  handleOpponentAction({ type, data }) {
    console.log(`[GameState] Opponent action received: ${type}`, data);
    if (type === 'move') {
      const { fx, fy, fz, tx, ty, tz } = data;
      this._remoteExecuteMove(fx, fy, fz, tx, ty, tz);
    } else if (type === 'drop') {
      const { type: pieceType, tx, ty, tz } = data;
      this._remoteExecuteDrop(pieceType, tx, ty, tz);
    } else if (type === 'promote') {
      this._remoteHandlePromotion(data.doPromote);
    }
  }

  /** 相手の「どこを見ているか（ホバー/選択）」を受信 */
  handleOpponentInteraction(data) {
    if (data.playerIndex === this.playerIndex) return; // 自分のなら無視
    const legalMoves = this._decodePackedMoves(data.legalMovesPacked, data.legalMoves);
    this.opponentInteraction = {
      selectedPos: data.selectedPos,
      confirmedPos: data.confirmedPos || null,
      legalMoves: legalMoves,
      selectedHandType: data.selectedHandType || null
    };

    // 強制的に持ち駒リストを補正 (同期ズレ対策)
    if (data.capturedPieceCounts) {
      for (const p of [Player.SENTE, Player.GOTE]) {
        const remoteCounts = data.capturedPieceCounts[p];
        if (remoteCounts) {
          // 簡易的に数だけ合わせるのではなく、デバッグしやすいように不一致をログに出す
          const localCount = this.capturedPieces[p].length;
          const remoteCountTotal = Object.values(remoteCounts).reduce((a, b) => a + b, 0);
          if (localCount !== remoteCountTotal) {
            console.warn(`[Sync] Hand piece divergence detected for Player ${p}. Local: ${localCount}, Remote: ${remoteCountTotal}. Fixing...`);
            // ※ここでは型ごとの不一致を詳細に直す実装も可能だが、
            // ひとまず Interaction は「補助」なので、型ごとの数は summary 取得時に影響が出る。
            // 本来は play-action で正しく同期されるべき。
          }
        }
      }
      this.opponentCapturedCounts = data.capturedPieceCounts;
    }

    this._notify(false); // 同期通知は再送しない
  }

  startGame() {
    this.board = initBoard();
    this.currentPlayer = Player.SENTE;
    this.phase = GamePhase.SELECT;
    this.selectedPos = null;
    this.legalMoves = [];
    this.confirmedPos = null;
    this.selectedHandType = null;
    this.lastFrom = null;
    this.lastTo = null;
    this.logger.clear();
    this.capturedPieces = { [Player.SENTE]: [], [Player.GOTE]: [] };
    this.winnerPlayer = null;
    this.highlightedSlice = null;
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
    
    // 自分のターンでないなら無視
    if (this.currentPlayer !== this.playerSide) {
      window.showCenterNotification('相手のターンです');
      return;
    }

    if (this.selectedHandType === type && (this.phase === GamePhase.DROP || this.phase === GamePhase.DROP_CONFIRM)) {
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

  _updateLegalMovesAfterSliceChange() {
    if (this.selectedPos) {
      this.legalMoves = this._getLegalMovesSafe(this.selectedPos.x, this.selectedPos.y, this.selectedPos.z);
    } else if (this.selectedHandType) {
      this.legalMoves = this._getDropTargets(this.selectedHandType);
    }
  }


  handleClick(x, y, z) {
    if (this.phase === GamePhase.IDLE || this.phase === GamePhase.DONE) return;
    if (this.phase === GamePhase.PROMOTE) return;

    if (this.currentPlayer !== this.playerSide) {
      window.showCenterNotification('相手のターンです');
      return;
    }

    const cell = this.board.get(x, y, z);

    // ---- 駒を打つ ----
    if (this.phase === GamePhase.DROP) {
      if (cell && cell.player === this.currentPlayer) {
        this.selectedHandType = null;
        this.selectedPos = { x, y, z };
        this.legalMoves = this._getLegalMovesSafe(x, y, z);
        this.phase = GamePhase.MOVE;
        this._notify();
        return;
      }
      const isLegal = this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z);
      if (isLegal) {
        this.confirmedPos = { x, y, z };
        this.phase = GamePhase.DROP_CONFIRM;
        this._notify();
        return;
      }
      this._cancelSelect();
      return;
    }

    if (this.phase === GamePhase.DROP_CONFIRM) {
      const { x: cx, y: cy, z: cz } = this.confirmedPos;
      if (x === cx && y === cy && z === cz) {
        this._executeDrop(cx, cy, cz);
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
      const isLegal = this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z);
      if (isLegal) {
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
      const isLegal = this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z);
      if (isLegal) {
        this.confirmedPos = { x, y, z };
        this.phase = GamePhase.CONFIRM;
        this._notify();
        return;
      }
      this._cancelSelect();
      return;
    }

    if (this.phase === GamePhase.CONFIRM) {
      const { x: cx, y: cy, z: cz } = this.confirmedPos;
      if (x === cx && y === cy && z === cz) {
        this._executeMove(cx, cy, cz);
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
      const isLegal = this.legalMoves.some(m => m[0] === x && m[1] === y && m[2] === z);
      if (isLegal) {
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
      if (newType) {
        cell.type = newType;
        cell.promoted = true;
      }
    }
    const cell = this.board.get(this.lastTo.x, this.lastTo.y, this.lastTo.z);
    this.logger.addMove(this.currentPlayer, this.lastTo.x, this.lastTo.y, this.lastTo.z, cell.type, cell.id, doPromote, this.lastCapturedType);
    this.lastCapturedType = null;

    // 結果を送信
    window.onlineManager.sendAction('promote', { doPromote });

    this._endTurnAndJudge();
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
      console.log(`[GameState] Piece captured by ${this.currentPlayer}:`, captured.type);
      if (captured.type === PieceType.OU) {
        this.lastFrom = { x: fx, y: fy, z: fz };
        this.lastTo = { x: tx, y: ty, z: tz };
        const cell = this.board.get(tx, ty, tz);
        this.logger.addMove(this.currentPlayer, tx, ty, tz, cell.type, cell.id, false, captured.type);
        this.winnerPlayer = this.currentPlayer;
        this.phase = GamePhase.DONE;
        // 最後のアクション送信
        window.onlineManager.sendAction('move', { fx, fy, fz, tx, ty, tz });
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

    this.selectedPos = null;
    this.confirmedPos = null;
    this.legalMoves = [];
    this.selectedHandType = null;

    const cell = movedCell;
    const promoBy = isPromotionZone(fy, this.currentPlayer) || isPromotionZone(ty, this.currentPlayer);
    
    // 指し手を送信
    window.onlineManager.sendAction('move', { fx, fy, fz, tx, ty, tz });

    if (promoBy && canPromote(cell.type)) {
      this.lastCapturedType = captured ? captured.type : null;
      this.phase = GamePhase.PROMOTE;
      this._notify();
    } else {
      this.logger.addMove(this.currentPlayer, tx, ty, tz, cell.type, cell.id, false, captured ? captured.type : null);
      this._endTurnAndJudge();
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
          
          // 断面制限を適用
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
    if (idx < 0) {
      console.error(`[GameState] Drop failed: Piece ${type} not found in hand of ${this.currentPlayer}`);
      return;
    }
    hand.splice(idx, 1);
    console.log(`[GameState] Piece dropped by ${this.currentPlayer}:`, type);

    this.board.set(tx, ty, tz, makeCellForPlayer(type, this.currentPlayer));
    this._addPiecePos(this.currentPlayer, tx, ty, tz);
    this.lastFrom = null;
    this.lastTo = { x: tx, y: ty, z: tz };
    this.logger.addMove(this.currentPlayer, tx, ty, tz, type, -1, false, null);

    // 送信
    window.onlineManager.sendAction('drop', { type, tx, ty, tz });

    this._endTurnAndJudge();
  }

  _endTurnAndJudge() {
    const justMoved = this.currentPlayer;
    const next = this.currentPlayer === Player.SENTE ? Player.GOTE : Player.SENTE;
    this.currentPlayer = next;
    this.phase = GamePhase.SELECT;
    this.selectedHandType = null;
    this.confirmedPos = null;
    this.legalMoves = [];
    
    // 相手側の操作状態をクリア（前回の着手結果が見えなくなるのを防ぐ）
    this.opponentInteraction = {
      selectedPos: null,
      confirmedPos: null,
      legalMoves: [],
      selectedHandType: null
    };

    const inCheck = this._isKingInCheck(next);
    const hasMove = this._hasAnyLegalMove(next);
    if (inCheck && !hasMove) {
      this.winnerPlayer = justMoved;
      this.phase = GamePhase.DONE;
    }
    this._notify();
  }

  cancelSelection() {
    this._cancelSelect();
  }

  _cancelSelect(notify = true) {
    this.selectedPos = null;
    this.legalMoves = [];
    this.confirmedPos = null;
    this.selectedHandType = null;
    this.phase = GamePhase.SELECT;
    if (notify) this._notify();
  }

  _notify(sendOnline = true) {
    if (typeof this.onStateChange === 'function') this.onStateChange(this);

    // オンライン同期 (自分の手番での操作のみ)
    if (sendOnline && window.onlineManager && this.currentPlayer === this.playerSide) {
      // 現在の持ち駒サマリーを作成
      const capturedCounts = {};
      [Player.SENTE, Player.GOTE].forEach(p => {
        const counts = {};
        (this.capturedPieces[p] || []).forEach(pc => {
          counts[pc.type] = (counts[pc.type] || 0) + 1;
        });
        capturedCounts[p] = counts;
      });

      window.onlineManager.sendInteraction({
        selectedPos: this.selectedPos || null,
        confirmedPos: this.confirmedPos || null,
        legalMovesPacked: this._encodePackedMoves(this.legalMoves || []),
        selectedHandType: this.selectedHandType || null,
        capturedPieceCounts: capturedCounts
      });
    }
  }

  _encodePackedMoves(moves) {
    if (!Array.isArray(moves)) return [];
    return moves.map(([x, y, z]) => x * 81 + y * 9 + z);
  }

  _decodePackedMoves(packed, fallback = []) {
    if (Array.isArray(packed)) {
      return packed.map((n) => {
        const x = Math.floor(n / 81);
        const y = Math.floor((n % 81) / 9);
        const z = n % 9;
        return [x, y, z];
      });
    }
    return Array.isArray(fallback) ? fallback : [];
  }

  // --- Utility / Logic ---
  _findKing(player) {
    return this.kingPos[player];
  }

  _isKingInCheck(player) {
    const k = this._findKing(player);
    if (!k) return false;
    const attacker = player === Player.SENTE ? Player.GOTE : Player.SENTE;
    for (const key of this.piecePositions[attacker]) {
      const [x, y, z] = key.split(',').map(Number);
      const moves = getLegalMoves(this.board, x, y, z);
      if (moves.some(m => m[0] === k.x && m[1] === k.y && m[2] === k.z)) return true;
    }
    return false;
  }

  _hasAnyLegalMove(player) {
    for (const key of this.piecePositions[player]) {
      const [x, y, z] = key.split(',').map(Number);
      const moves = getLegalMoves(this.board, x, y, z);
      for (const [tx, ty, tz] of moves) if (this._isMoveSafe(x, y, z, tx, ty, tz, player)) return true;
    }
    const hand = this.capturedPieces[player] || [];
    const uniq = new Set(hand.map(p => p.type));
    for (const type of uniq) {
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
    const pseudo = getLegalMoves(this.board, fx, fy, fz);
    return pseudo.filter(([tx, ty, tz]) => this._isMoveSafe(fx, fy, fz, tx, ty, tz, this.currentPlayer));
  }

  _isMoveSafe(fx, fy, fz, tx, ty, tz, player) {
    const from = this.board.get(fx, fy, fz);
    const to = this.board.get(tx, ty, tz);
    const savedFrom = { ...from };
    const savedTo = { ...to };
    const opponent = player === Player.SENTE ? Player.GOTE : Player.SENTE;
    const prevKing = this.kingPos[player] ? { ...this.kingPos[player] } : null;

    this._removePiecePos(player, fx, fy, fz);
    this._addPiecePos(player, tx, ty, tz);
    if (savedTo.type !== PieceType.NONE && savedTo.player === opponent) {
      this._removePiecePos(opponent, tx, ty, tz);
    }
    if (savedFrom.type === PieceType.OU) {
      this.kingPos[player] = { x: tx, y: ty, z: tz };
    }

    this.board.set(tx, ty, tz, { ...from });
    this.board.clear(fx, fy, fz);
    const ok = !this._isKingInCheck(player);
    this.board.set(fx, fy, fz, savedFrom);
    this.board.set(tx, ty, tz, savedTo);

    this._removePiecePos(player, tx, ty, tz);
    this._addPiecePos(player, fx, fy, fz);
    if (savedTo.type !== PieceType.NONE && savedTo.player === opponent) {
      this._addPiecePos(opponent, tx, ty, tz);
    }
    if (savedFrom.type === PieceType.OU) {
      this.kingPos[player] = prevKing;
    }

    return ok;
  }

  _isDropSafe(type, tx, ty, tz, player) {
    const to = this.board.get(tx, ty, tz);
    const savedTo = { ...to };
    this._addPiecePos(player, tx, ty, tz);
    this.board.set(tx, ty, tz, { type, player, id: -1, promoted: false });
    const ok = !this._isKingInCheck(player);
    this.board.set(tx, ty, tz, savedTo);
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

  // --- Remote actions ---
  _remoteExecuteMove(fx, fy, fz, tx, ty, tz) {
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
      console.log(`[GameState] Remote Piece captured by ${this.currentPlayer}:`, captured.type);
      if (captured.type === PieceType.OU) {
        this.winnerPlayer = this.currentPlayer;
        this.phase = GamePhase.DONE;
      }
    }
    this.lastFrom = { x: fx, y: fy, z: fz };
    this.lastTo = { x: tx, y: ty, z: tz };

    const movedCell = this.board.get(tx, ty, tz);
    if (movedCell && movedCell.type === PieceType.OU) {
      this.kingPos[this.currentPlayer] = { x: tx, y: ty, z: tz };
    }

    const cell = this.board.get(tx, ty, tz);
    const promoBy = isPromotionZone(fy, this.currentPlayer) || isPromotionZone(ty, this.currentPlayer);
    if (promoBy && canPromote(cell.type)) {
      this.phase = GamePhase.PROMOTE;
    } else {
      this.logger.addMove(this.currentPlayer, tx, ty, tz, cell.type, cell.id, false, captured ? captured.type : null);
      if (this.phase !== GamePhase.DONE) this._endTurnAndJudge();
    }
    this._notify();
  }

  _remoteExecuteDrop(type, tx, ty, tz) {
    const hand = this.capturedPieces[this.currentPlayer];
    const idx = hand.findIndex(p => p.type === type);
    if (idx >= 0) {
      hand.splice(idx, 1);
      console.log(`[GameState] Remote Piece dropped by ${this.currentPlayer}:`, type);
    } else {
      console.error(`[GameState] Remote drop error: Piece ${type} not found in hand of ${this.currentPlayer}`);
    }
    this.board.set(tx, ty, tz, makeCellForPlayer(type, this.currentPlayer));
    this._addPiecePos(this.currentPlayer, tx, ty, tz);
    this.lastFrom = null;
    this.lastTo = { x: tx, y: ty, z: tz };
    this.logger.addMove(this.currentPlayer, tx, ty, tz, type, -1, false, null);
    this._endTurnAndJudge();
    this._notify();
  }

  _remoteHandlePromotion(doPromote) {
    if (this.phase !== GamePhase.PROMOTE) return;
    const cell = this.board.get(this.lastTo.x, this.lastTo.y, this.lastTo.z);
    if (doPromote) {
      const nt = PromotionMap[cell.type];
      if (nt) { cell.type = nt; cell.promoted = true; }
    }
    this.logger.addMove(this.currentPlayer, this.lastTo.x, this.lastTo.y, this.lastTo.z, cell.type, cell.id, doPromote, null);
    this._endTurnAndJudge();
  }

  /**
   * 現在のゲーム状態をまるごとオブジェクトとして返す (再入室同期用)
   */
  getFullState() {
    return {
      cells: this.board.cells, // 盤面全体
      currentPlayer: this.currentPlayer,
      phase: this.phase,
      capturedPieces: this.capturedPieces,
      logs: this.logger.getAll(),
      lastFrom: this.lastFrom,
      lastTo: this.lastTo,
      winnerPlayer: this.winnerPlayer
    };
  }

  /**
   * 受け取った全状態を現在の GameState に一括反映する
   */
  setFullState(data) {
    if (!data) return;
    console.log('[GameState] Syncing full state from remote...', data);
    
    // 盤面復元
    this.board.cells = data.cells;
    this.currentPlayer = data.currentPlayer;
    // フェーズは同期元が SELECT なら SELECT にするが、
    // 基本的に再入室時は SELECT (駒選択待ち) から開始
    this.phase = data.phase || GamePhase.SELECT;
    this.capturedPieces = data.capturedPieces;
    
    // ログ復元
    this.logger.clear();
    if (data.logs) {
      data.logs.forEach(l => {
        // 直接内部に追加 (addMove を使うと座標計算が走るため)
        this.logger.logs.push(l);
      });
    }
    
    this.lastFrom = data.lastFrom;
    this.lastTo = data.lastTo;
    this.winnerPlayer = data.winnerPlayer;
    this._rebuildPieceCaches();
    
    this._notify(false); // 同期通知は不要
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
