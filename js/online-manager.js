/**
 * online-manager.js
 * Socket.io を使ったサーバーとの通信管理クラス。
 * MDD2 のロジックをベースに三次元将棋用に最適化。
 */

class OnlineManager {
  constructor() {
    this.socket = null;
    this.roomCode = null;
    this.playerIndex = null; // 0: sente, 1: gote
    this.playerName = 'Player';
    this.isHost = false;
    this.callbacks = {};
    this.interactionIntervalMs = 50;
    this._lastInteractionAt = 0;
    this._lastInteractionSig = '';
    this._pendingInteraction = null;
    this._interactionTimer = null;
  }

  init(callbacks) {
    this.callbacks = callbacks;
  }

  connect() {
    if (this.socket) return;
    // @ts-ignore
    this.socket = io();
    this._setupListeners();
  }

  createRoom() {
    this.connect();
    this.isHost = true;
    this.socket.emit('create-room', { playerName: '' });
  }

  joinRoom(roomCode) {
    this.connect();
    this.roomCode = roomCode.toUpperCase();
    this.isHost = false;
    this.socket.emit('join-room', { roomCode, playerName: '' });
  }

  /** 指し手を送信 (Move, Drop, Promotion等) */
  sendAction(actionType, data) {
    if (!this.socket || !this.roomCode) return;
    this.socket.emit('play-action', {
      type: actionType,
      playerIndex: this.playerIndex,
      data: data
    });
  }

  /** 操作状態を送信 (Hover, Select, LegalMoves等) */
  sendInteraction(data) {
    if (!this.socket || !this.roomCode) return;

    const payload = {
      playerIndex: this.playerIndex,
      ...data
    };
    const sig = JSON.stringify(payload);

    // 同一状態の連打は送らない
    if (sig === this._lastInteractionSig) return;

    const now = Date.now();
    const elapsed = now - this._lastInteractionAt;

    const flush = (entry) => {
      if (!entry || !this.socket || !this.roomCode) return;
      // 補助同期は欠落しても致命的ではないため volatile で詰まりを回避
      this.socket.volatile.emit('sync-interaction', entry.payload);
      this._lastInteractionAt = Date.now();
      this._lastInteractionSig = entry.sig;
    };

    if (elapsed >= this.interactionIntervalMs) {
      flush({ payload, sig });
      return;
    }

    // 送信間隔内は最新状態のみ保持
    this._pendingInteraction = { payload, sig };
    if (this._interactionTimer) return;

    const waitMs = Math.max(0, this.interactionIntervalMs - elapsed);
    this._interactionTimer = setTimeout(() => {
      this._interactionTimer = null;
      const pending = this._pendingInteraction;
      this._pendingInteraction = null;
      flush(pending);
    }, waitMs);
  }

  /** 自分の盤面状態を特定の相手に送信 (再入室対応) */
  sendFullState(requesterId, gameState) {
    if (!this.socket) return;
    this.socket.emit('send-full-state', { requesterId, gameState });
  }

  _setupListeners() {
    this.socket.on('room-created', ({ roomCode, playerIndex }) => {
      this.roomCode = roomCode;
      this.playerIndex = playerIndex;
      if (this.callbacks.onRoomCreated) this.callbacks.onRoomCreated({ roomCode, playerIndex });
    });

    this.socket.on('both-joined', ({ players }) => {
      if (this.callbacks.onBothJoined) this.callbacks.onBothJoined({ players });
    });

    this.socket.on('game-started', ({ playerIndex, gameState }) => {
      if (this.playerIndex === null) this.playerIndex = playerIndex;
      if (this.callbacks.onGameStarted) this.callbacks.onGameStarted({ playerIndex, gameState });
    });

    this.socket.on('opponent-action', (action) => {
      if (this.callbacks.onOpponentAction) this.callbacks.onOpponentAction(action);
    });

    this.socket.on('opponent-interaction', (data) => {
      if (this.callbacks.onOpponentInteraction) this.callbacks.onOpponentInteraction(data);
    });

    // 再入室者への状態提供要求
    this.socket.on('request-full-state', ({ requesterId }) => {
      if (this.callbacks.onRequestFullState) this.callbacks.onRequestFullState(requesterId);
    });

    // 再入室者自身が受け取る同期データ
    this.socket.on('sync-full-state', ({ gameState }) => {
      if (this.callbacks.onSyncFullState) this.callbacks.onSyncFullState(gameState);
    });

    this.socket.on('opponent-disconnected', ({ playerName }) => {
      if (this.callbacks.onOpponentDisconnected) this.callbacks.onOpponentDisconnected(playerName);
    });

    this.socket.on('join-error', (msg) => {
      if (this.callbacks.onJoinError) this.callbacks.onJoinError(msg);
    });

    this.socket.on('disconnect', () => {
      if (this._interactionTimer) {
        clearTimeout(this._interactionTimer);
        this._interactionTimer = null;
      }
      this._pendingInteraction = null;
      this._lastInteractionSig = '';
      this._lastInteractionAt = 0;
    });
  }

  /** 自分の番かどうか判定 */
  isMyTurn(currentPlayerPlayer) {
    const myPlayer = (this.playerIndex === 0) ? 1 : 2; // Board.js Player.SENTE=1, GOTE=2
    return myPlayer === currentPlayerPlayer;
  }
}

// グローバルに公開
window.onlineManager = new OnlineManager();
export default window.onlineManager;
