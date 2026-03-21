/**
 * online-lobby.js
 * ロビー画面のUI制御ロジック。
 */

import onlineManager from './online-manager.js';

window.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(window.location.search);
  if (params.get('mode') === 'local') return; // ローカルモードなら何もしない

  const lobbyScreen   = document.getElementById('lobby-screen');
  const gameScreen    = document.getElementById('game-screen');
  const lobbyMain     = document.getElementById('lobby-main');
  const lobbyWaiting  = document.getElementById('lobby-waiting');
  
  const btnCreate     = document.getElementById('btn-create-room');
  const btnJoin       = document.getElementById('btn-join-room');
  const btnCancel     = document.getElementById('btn-cancel-room');
  
  const nameInput     = document.getElementById('player-name');
  const codeInput     = document.getElementById('room-code-input');
  const codeDisplay   = document.getElementById('room-code-display');
  const statusMsg     = document.getElementById('lobby-status');

  const showStatus = (msg, isError = false) => {
    statusMsg.textContent = msg;
    statusMsg.style.color = isError ? '#e85555' : '#f5c842';
  };

  const commonCallbacks = {
    onRoomCreated: ({ roomCode }) => {
      lobbyMain.classList.add('hidden');
      lobbyWaiting.classList.remove('hidden');
      codeDisplay.textContent = roomCode;
      showStatus('対戦相手を待っています...');
    },
    onBothJoined: ({ players }) => {
      showStatus('相手が参加しました！準備中...');
    },
    onGameStarted: ({ playerIndex }) => {
      lobbyScreen.classList.remove('active');
      lobbyScreen.classList.add('hidden');
      gameScreen.classList.remove('hidden');
      gameScreen.classList.add('active');
      
      // 指標表示
      const roomPill = document.getElementById('online-room-indicator');
      const roomText = document.getElementById('room-id-text');
      if (roomPill) roomPill.classList.remove('hidden');
      if (roomText) roomText.textContent = onlineManager.roomCode;

      // ゲームオブジェクトの準備を待つ (Module の読み込み順や DOMContentLoaded のタイミング対策)
      const tryInit = () => {
        if (window.shogiGame) {
          window.shogiGame.initNetworkMode(playerIndex);
          console.log('[Lobby] Game initialized with playerIndex:', playerIndex);
          return true;
        }
        return false;
      };

      if (!tryInit()) {
        console.warn('[Lobby] Game object not ready, retrying...');
        const timer = setInterval(() => {
          if (tryInit()) clearInterval(timer);
        }, 100);
        // 5秒経ってもダメなら諦める（エラー表示など）
        setTimeout(() => clearInterval(timer), 5000);
      }
      
      // 画面表示後にリサイズイベントを発火させ、3Dレンダラーのサイズを確定させる
      setTimeout(() => window.dispatchEvent(new Event('resize')), 100);
    },
    onJoinError: (msg) => {
      // サーバーからのエラーメッセージを日本語に変換
      const jpMsg = msg === 'Room is full' ? '部屋が満員です' : 
                    msg === 'Room not found' ? '部屋が見つかりません' : msg;
      showStatus(jpMsg, true);
    },
    onOpponentAction: (action) => {
      if (window.shogiGame) window.shogiGame.handleOpponentAction(action);
    },
    onOpponentInteraction: (interaction) => {
      if (window.shogiGame) window.shogiGame.handleOpponentInteraction(interaction);
    },
    onOpponentDisconnected: (name) => {
      window.showCenterNotification(`相手(${name})が切断しました`);
    },
    // 再入室者へのデータ提供
    onRequestFullState: (requesterId) => {
      if (window.shogiGame) {
        console.log(`[Lobby] Providing full state to requester: ${requesterId}`);
        const state = window.shogiGame.getFullState();
        onlineManager.sendFullState(requesterId, state);
      }
    },
    // 再入室した自分へのデータ反映
    onSyncFullState: (gameState) => {
      if (window.shogiGame) {
        console.log('[Lobby] Received full state sync.');
        window.shogiGame.setFullState(gameState);
      }
    }
  };

  onlineManager.init(commonCallbacks);

  btnCreate.addEventListener('click', () => {
    showStatus('接続中...');
    onlineManager.createRoom();
  });

  btnJoin.addEventListener('click', () => {
    const code = codeInput.value.trim().toUpperCase();
    if (code.length !== 4) return showStatus('4文字の部屋コードを入力してください', true);
    showStatus('参加中...');
    onlineManager.joinRoom(code);
  });

  btnCancel.addEventListener('click', () => {
    location.reload(); 
  });
});

// 中央通知表示関数をグローバルに
window.showCenterNotification = (msg) => {
  const notify = document.getElementById('center-notification');
  if (!notify) return;
  notify.textContent = msg;
  notify.classList.remove('hidden');
  setTimeout(() => notify.classList.add('hidden'), 3000);
};
