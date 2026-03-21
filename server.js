// server.js - 三次元将棋サーバー
// 現在: 静的ファイル配信 + Socket.io（インターネット対戦の基盤）

'use strict';

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: '*', methods: ['GET', 'POST'] }
});

const PORT = process.env.PORT || 3000;
const PROXY_VERBOSE = process.env.SHOGI_PROXY_VERBOSE === '1';

// ============================================================
// 静的ファイル配信
// ============================================================
app.use(express.static(path.join(__dirname, 'public')));

// public/ にないファイルは旧パスもフォールバック
app.use(express.static(__dirname));

// ============================================================
// ルート
// ============================================================
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/game', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'game.html'));
});

// ============================================================
// AIサーバーのプロキシ (Port 5000 -> 3000)
// ============================================================
app.use(express.json()); // 解析用

app.post('/predict', (req, res) => {
  if (PROXY_VERBOSE) {
    console.log('[Proxy] Forwarding /predict request to AI Server...');
  }
  const options = {
    hostname: '127.0.0.1',
    port: 5000,
    path: '/predict',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    }
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.status(proxyRes.statusCode);
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (e) => {
    console.error(`[AI Proxy Error] ${e.message}`);
    res.status(500).json({ error: 'AIサーバーへの接続に失敗しました。AI.batが動いているか確認してください。' });
  });

  proxyReq.write(JSON.stringify(req.body));
  proxyReq.end();
});

// ============================================================
// Socket.io - インターネット対戦基盤（将来拡張）
// ============================================================

// ルームの管理
const rooms = new Map();

function createRoom(roomId) {
  const room = {
    id: roomId,
    players: [], // {id, name, role}
    started: false,
    history: [],
  };
  rooms.set(roomId, room);
  return room;
}

function genRoomId() {
  // MDD2風 4桁大文字英数字
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // 紛らわしい 0,1,I,O を除く
  let res = '';
  for (let i = 0; i < 4; i++) res += chars.charAt(Math.floor(Math.random() * chars.length));
  // 重複チェック
  if (rooms.has(res)) return genRoomId();
  return res;
}

io.on('connection', (socket) => {
  console.log(`[接続] ${socket.id}`);

  // ===== ルーム作成 =====
  socket.on('create-room', ({ playerName }, callback) => {
    const roomId = genRoomId();
    const room = createRoom(roomId);
    
    const name = playerName || '先手';
    const player = { id: socket.id, name: name, role: 'sente' };
    room.players.push(player);
    
    socket.join(roomId);
    socket.data.roomId = roomId;
    socket.data.role = 'sente';
    
    console.log(`[ルーム作成] ${roomId} by ${socket.id}`);
    if (callback) callback({ ok: true, roomCode: roomId, playerIndex: 0 });
    
    // 作成者に通知（自分一人）
    socket.emit('room-created', { roomCode: roomId, playerIndex: 0 });
  });

  // ===== ルーム参加 =====
  socket.on('join-room', ({ roomCode, playerName }, callback) => {
    const roomId = roomCode ? roomCode.toUpperCase() : '';
    const room = rooms.get(roomId);
    
    if (!room) {
      if (callback) callback({ ok: false, error: 'ルームが見つかりません' });
      socket.emit('join-error', 'ルームが見つかりません');
      return;
    }
    if (room.players.length >= 2) {
      if (callback) callback({ ok: false, error: 'ルームが満員です' });
      socket.emit('join-error', 'ルームが満員です');
      return;
    }

    const name = playerName || (room.players.length === 0 ? '先手' : '後手');
    
    // すでにゲームが開始されている場合、既存の役職を維持する
    let role = 'gote';
    if (room.started) {
      const existingRole = room.players[0].role;
      role = (existingRole === 'sente') ? 'gote' : 'sente';
    } else {
      // 最初期の参加なら暫定的に gote (後でシャッフルされる)
      role = (room.players.length === 0) ? 'sente' : 'gote';
    }

    const player = { id: socket.id, name: name, role: role };
    room.players.push(player);
    
    socket.join(roomId);
    socket.data.roomId = roomId;
    socket.data.role = role;
    
    console.log(`[ルーム参加] ${roomId} by ${socket.id} as ${role}`);
    if (callback) callback({ ok: true, roomCode: roomId, playerIndex: (role === 'sente' ? 0 : 1) });

    if (room.players.length === 2 && !room.started) {
      // 初回開始時のみランダムに先後を決定
      const senteIdx = Math.random() < 0.5 ? 0 : 1;
      room.players[senteIdx].role = 'sente';
      room.players[1 - senteIdx].role = 'gote';
      room.started = true;
      
      room.players.forEach((p, idx) => {
        io.to(p.id).emit('game-started', { 
          playerIndex: p.role === 'sente' ? 0 : 1, 
          gameState: { currentPlayerIndex: 0 } 
        });
      });
      console.log(`[ゲーム開始] ${roomId}: 初回先後決定`);
    } else if (room.players.length === 2 && room.started) {
      // 再入室時
      const p1 = room.players[0];
      const p2 = room.players[1];
      io.to(p1.id).emit('opponent-rejoined', { playerName: p2.name });
      io.to(p2.id).emit('game-started', { 
        playerIndex: p2.role === 'sente' ? 0 : 1,
        isRejoin: true
        // 注意: gameState の同期は別途 interaction 等で行う必要があるが、
        // 現状の interaction 同期ロジックである程度カバー可能
      });
      console.log(`[ルーム再入室] ${roomId}: ${p2.name} が復帰`);

      // 状態同期要求: 既存のプレイヤー(p1)に最新の状態を要求
      socket.to(roomId).emit('request-full-state', { requesterId: socket.id });
    }
  });

  // ===== 再入室時のフルステート同期転送 =====
  socket.on('send-full-state', ({ requesterId, gameState }) => {
    console.log(`[Sync] Forwarding full state from ${socket.id} to ${requesterId}`);
    io.to(requesterId).emit('sync-full-state', { gameState });
  });

  // ===== 指し手の同期 (Move / Drop / EndTurn) =====
  socket.on('play-action', (actionData) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;
    // 相手に転送
    socket.to(roomId).emit('opponent-action', actionData);
  });

  // ===== 操作状態の同期 (Hover / Select / LegalMoves) =====
  socket.on('sync-interaction', (interactionData) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;
    // 補助情報は取りこぼし許容のため volatile で転送し、遅延時の詰まりを防ぐ
    socket.to(roomId).volatile.emit('opponent-interaction', interactionData);
  });

  // ===== チャット =====
  socket.on('chat', ({ message }) => {
    const roomId = socket.data.roomId;
    if (!roomId) return;
    const room = rooms.get(roomId);
    if (!room) return;
    const player = room.players.find(p => p.id === socket.id);
    io.to(roomId).emit('chat', { from: player ? player.name : '未知', message });
  });

  // ===== 切断 =====
  socket.on('disconnect', () => {
    const roomId = socket.data.roomId;
    if (!roomId) return;
    const room = rooms.get(roomId);
    if (!room) return;
    
    const playerIndex = room.players.findIndex(p => p.id === socket.id);
    if (playerIndex >= 0) {
      const player = room.players[playerIndex];
      console.log(`[切断] ${socket.id} (${player.name}) from ${roomId}`);
      socket.to(roomId).emit('opponent-disconnected', { playerName: player.name });
      
      // プレイヤーをリストから削除（再入室を可能にするため完全に消さない選択肢もあるが、
      // ここではIDが変わるため一旦削除し、join時の人数チェックで空きがあれば入れるようにする）
      room.players.splice(playerIndex, 1);
    }
    
    // ルームが完全に空になったら削除
    if (room.players.length === 0) {
      console.log(`[ルーム削除] ${roomId} (空になりました)`);
      rooms.delete(roomId);
    }
  });
});

// ============================================================
// サーバー起動
// ============================================================
server.listen(PORT, () => {
  console.log(`\n╔════════════════════════════════╗`);
  console.log(`║  三次元将棋サーバー起動             ║`);
  console.log(`║  http://localhost:${PORT}           ║`);
  console.log(`╚════════════════════════════════╝\n`);
});
