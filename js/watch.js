// watch.js - NPC対戦観戦専用ロジック
'use strict';

import { Board, Player, initBoard, PieceType, DemotionMap, PromotionMap, canPromote, isPromotionZone, makeCellForPlayer, getPieceGrade } from './board.js';
import { View3D } from './view3d.js';
import { View2D, SliceType } from './view2d.js';
import { Logger } from './logger.js';

// 基本設定
const AI_SERVER_URL = '/predict';
const REPLAY_LIST_URL = '/watch-replays';
const REPLAY_LOAD_URL_BASE = '/watch-replays/';

class WatchManager {
  constructor() {
    this.board = initBoard();
    this.currentPlayer = Player.SENTE;
    this.turnCount = 1;
    this.history = []; // { boardData, ownerData, hands, player, log }
    this.capturedPieces = { [Player.SENTE]: [], [Player.GOTE]: [] };
    this.legalMoves = [];
    this.lastFrom = null;
    this.lastTo = null;
    this.repetitionMap = new Map(); // hash -> count
    this.logger = new Logger();
    this.isPlaying = false;
    this.isGenerating = false;
    this.isPrecomputed = false;
    this.thinkTime = 0.5;
    this.timerId = null;
    this.playbackIndex = 0;
    this.maxAutoPlies = 5000;
    this.openingNoisePlies = 24;
    this.repeatStochasticThreshold = 2;

    // View初期化
    this.view3d = new View3D(document.getElementById('container-3d'), this);
    this.view2d = new View2D(document.getElementById('wrapper-2d'), this);
    this.viewMode = '3d';

    this.initUI();
    this.saveHistory();
    this.refresh();
  }

  initUI() {
    // ボタン等
    const btnPlay = document.getElementById('btn-play');
    const btnPause = document.getElementById('btn-pause');
    const inputSpeed = document.getElementById('input-speed');
    const seekSlider = document.getElementById('seek-slider');
    const seekInput = document.getElementById('seek-input');
    const btnSeekGo = document.getElementById('btn-seek-go');
    const btnReplayRefresh = document.getElementById('btn-replay-refresh');
    const btnReplayLoad = document.getElementById('btn-replay-load');

    btnPlay.onclick = () => this.start();
    btnPause.onclick = () => this.stop();
    inputSpeed.onchange = (e) => this.thinkTime = parseFloat(e.target.value) || 0.5;
    seekSlider.oninput = (e) => this.seek(parseInt(e.target.value));
    seekInput.onchange = (e) => this.seek(parseInt(e.target.value));
    btnSeekGo.onclick = () => this.seek(parseInt(seekInput.value));
    if (btnReplayRefresh) btnReplayRefresh.onclick = () => this.refreshReplayList();
    if (btnReplayLoad) btnReplayLoad.onclick = () => this.loadSelectedReplay();

    // 視点切り替え
    document.getElementById('btn-3d-view').onclick = () => this.toggleView('3d');
    document.getElementById('btn-2d-view').onclick = () => this.toggleView('2d');

    this.refreshReplayList();
  }

  async refreshReplayList() {
    const select = document.getElementById('replay-select');
    if (!select) return;
    select.innerHTML = '';
    const optLoading = document.createElement('option');
    optLoading.value = '';
    optLoading.textContent = '読み込み中...';
    select.appendChild(optLoading);

    try {
      const resp = await fetch(REPLAY_LIST_URL);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      const files = Array.isArray(data.files) ? data.files : [];

      select.innerHTML = '';
      if (files.length === 0) {
        const empty = document.createElement('option');
        empty.value = '';
        empty.textContent = '保存棋譜がありません';
        select.appendChild(empty);
        return;
      }
      files.forEach((name) => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      });
    } catch (e) {
      select.innerHTML = '';
      const err = document.createElement('option');
      err.value = '';
      err.textContent = '一覧取得失敗';
      select.appendChild(err);
      console.error('Failed to load replay list:', e);
    }
  }

  async loadSelectedReplay() {
    const select = document.getElementById('replay-select');
    if (!select || !select.value) return;

    try {
      const resp = await fetch(`${REPLAY_LOAD_URL_BASE}${encodeURIComponent(select.value)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const replay = await resp.json();
      this.applyReplayData(replay, select.value);
    } catch (e) {
      console.error('Failed to load replay:', e);
      alert('保存棋譜の読み込みに失敗しました。');
    }
  }

  applyReplayData(replay, replayName = '') {
    const moves = Array.isArray(replay?.moves) ? replay.moves : [];
    if (moves.length === 0) {
      alert('この棋譜には手が含まれていません。');
      return;
    }

    this.stop();
    this.isGenerating = false;
    this.isPlaying = false;
    this.isPrecomputed = false;
    this.resetMatchState();

    for (const m of moves) {
      const move = {
        fx: Number(m.fx) || 0,
        fy: Number(m.fy) || 0,
        fz: Number(m.fz) || 0,
        tx: Number(m.tx) || 0,
        ty: Number(m.ty) || 0,
        tz: Number(m.tz) || 0,
        promote: Boolean(m.promote),
        is_drop: Boolean(m.is_drop),
        drop_type: Number(m.drop_type) || 0,
      };
      this.executeMove(move);
      this.saveHistory();
    }

    this.isPrecomputed = this.history.length > 1;
    this.playbackIndex = 0;
    this.applySnapshot(0);
    this.refresh();
    this.updateSeekUI();
    this.updateControls();

    const reason = replay?.reason || 'unknown';
    const winner = Number(replay?.winner) || 0;
    const winnerText = winner === Player.SENTE ? '先手' : (winner === Player.GOTE ? '後手' : '引き分け/不明');
    const label = replayName ? ` (${replayName})` : '';
    document.getElementById('phase-info').textContent = `棋譜読込${label}: ${moves.length}手 / 理由=${reason} / 勝者=${winnerText}`;
  }

  async start() {
    if (this.isPlaying || this.isGenerating) return;

    if (!this.isPrecomputed) {
      await this.precomputeGame();
      if (!this.isPrecomputed) {
        this.stop();
        return;
      }
    }

    if (this.playbackIndex >= this.history.length - 1) {
      this.playbackIndex = 0;
      this.applySnapshot(0);
      this.refresh();
      this.updateSeekUI();
    }

    this.isPlaying = true;
    this.updateControls();
    this.replayLoop();
  }

  stop() {
    this.isPlaying = false;
    if (this.timerId) clearTimeout(this.timerId);
    this.timerId = null;
    this.updateControls();
  }

  async precomputeGame() {
    this.isGenerating = true;
    this.updateControls();
    document.getElementById('phase-info').textContent = '対局生成中...';

    this.resetMatchState();

    let generated = 0;
    while (generated < this.maxAutoPlies) {
      // 現在の局面のハッシュを確認
      const hash = this.board.getHash(this.currentPlayer, this.capturedPieces);
      const count = (this.repetitionMap.get(hash) || 0) + 1;
      this.repetitionMap.set(hash, count);

      // 同一局面の再出現が早い段階で見えたら即ノイズ投入
      const stochastic = count >= this.repeatStochasticThreshold;

      const move = await this.fetchAIMove(stochastic);
      if (!move) break;

      const ended = this.executeMove(move);
      this.saveHistory();
      generated += 1;

      if (ended) break;
    }

    this.isGenerating = false;
    this.isPrecomputed = this.history.length > 1;
    this.playbackIndex = 0;
    this.applySnapshot(0);
    this.refresh();
    this.updateSeekUI();
    this.updateControls();

    if (!this.isPrecomputed) {
      alert('対局生成に失敗しました。AIサーバーの状態を確認してください。');
      document.getElementById('phase-info').textContent = '生成失敗';
    } else {
      document.getElementById('phase-info').textContent = `生成完了: ${this.history.length - 1} 手`;
    }
  }

  replayLoop() {
    if (!this.isPlaying) return;

    if (this.playbackIndex >= this.history.length - 1) {
      this.stop();
      return;
    }

    this.playbackIndex += 1;
    this.applySnapshot(this.playbackIndex);
    this.refresh();
    this.updateSeekUI();

    this.timerId = setTimeout(() => this.replayLoop(), this.thinkTime * 1000);
  }

  resetMatchState() {
    this.board = initBoard();
    this.currentPlayer = Player.SENTE;
    this.turnCount = 1;
    this.history = [];
    this.capturedPieces = { [Player.SENTE]: [], [Player.GOTE]: [] };
    this.lastFrom = null;
    this.lastTo = null;
    this.repetitionMap = new Map();
    this.logger = new Logger();
    this.playbackIndex = 0;
    this.saveHistory();
  }

  async fetchAIMove(stochastic = false) {
    // 盤面を配列形式に変換
    const bData = Array(9).fill().map(() => Array(9).fill().map(() => Array(9).fill(0)));
    const oData = Array(9).fill().map(() => Array(9).fill().map(() => Array(9).fill(0)));
    
    for(let x=0; x<9; x++) for(let y=0; y<9; y++) for(let z=0; z<9; z++) {
        const c = this.board.get(x,y,z);
        if (c.type !== 'none') {
            const types = [PieceType.NONE, PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI, PieceType.OU, 
                           PieceType.NARI_FU, PieceType.NARI_KYO, PieceType.NARI_KEI, PieceType.NARI_GIN, PieceType.UMA, PieceType.RYU];
            bData[x][y][z] = types.indexOf(c.type);
            oData[x][y][z] = c.player;
        }
    }

    // 持ち駒カウント
    const getHandCounts = (p) => {
        const counts = [0,0,0,0,0,0,0]; // f,k,k,g,k,k,h
        const types = [PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI];
        this.capturedPieces[p].forEach(pc => {
            const idx = types.indexOf(pc.type);
            if(idx >= 0) counts[idx]++;
        });
        return counts;
    };

    try {
      const openingNoise = this.turnCount <= this.openingNoisePlies;
        const resp = await fetch(AI_SERVER_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board: bData,
                owner: oData,
                hand_sente: getHandCounts(Player.SENTE),
                hand_gote: getHandCounts(Player.GOTE),
                current_player: this.currentPlayer,
          stochastic: stochastic,
          opening_noise: openingNoise
            })
        });
        if (!resp.ok) {
            const err = await resp.json();
            alert(`AIサーバーエラー: ${err.error || '不明なエラー'}`);
            return null;
        }
        const res = await resp.json();
        return res.move;
    } catch(e) {
        console.error("AI Fetch error:", e);
        return null;
    }
  }

  executeMove(m) {
    let ended = false;
    if (m.is_drop) {
        // 打ち
        const pieceMap = [PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI];
        const type = pieceMap[m.drop_type];
        const hand = this.capturedPieces[this.currentPlayer];
        const idx = hand.findIndex(p => p.type === type);
        if (idx >= 0) hand.splice(idx, 1);
        this.board.set(m.tx, m.ty, m.tz, makeCellForPlayer(type, this.currentPlayer));
        this.logger.addMove(this.currentPlayer, m.tx, m.ty, m.tz, type, -1, false);
    } else {
        // 移動
        const captured = this.board.move(m.fx, m.fy, m.fz, m.tx, m.ty, m.tz);
        const cell = this.board.get(m.tx, m.ty, m.tz);
        if (m.promote && PromotionMap[cell.type]) {
            cell.type = PromotionMap[cell.type];
            cell.promoted = true;
        }
        if (captured) {
            const raw = DemotionMap[captured.type] || captured.type;
            this.capturedPieces[this.currentPlayer].push({ type: raw, player: this.currentPlayer });
          if (captured.type === PieceType.OU) ended = true; // 終局
        }
        this.logger.addMove(this.currentPlayer, m.tx, m.ty, m.tz, cell.type, cell.id, m.promote, captured ? captured.type : null);
    }
    
    this.lastFrom = m.is_drop ? null : { x: m.fx, y: m.fy, z: m.fz };
    this.lastTo = { x: m.tx, y: m.ty, z: m.tz };
    
    this.currentPlayer = this.currentPlayer === Player.SENTE ? Player.GOTE : Player.SENTE;
    this.turnCount++;
    return ended;
  }

  saveHistory() {
    // 盤面状態をディープコピーして保存
    const snapshot = {
        board: JSON.stringify(this.board.cells),
        currentPlayer: this.currentPlayer,
        capturedPieces: JSON.stringify(this.capturedPieces),
        logs: JSON.stringify(this.logger.getAll()),
        turnCount: this.turnCount
    };
    this.history.push(snapshot);
    this.updateSeekUI();
  }

  seek(index) {
    if (!Number.isFinite(index)) return;
    if (index < 0 || index >= this.history.length) return;
    this.stop();
    this.playbackIndex = index;
    this.applySnapshot(index);
    this.refresh();
    this.updateSeekUI();
  }

  applySnapshot(index) {
    const snap = this.history[index];
    if (!snap) return;
    this.board.cells = JSON.parse(snap.board);
    this.currentPlayer = snap.currentPlayer;
    this.capturedPieces = JSON.parse(snap.capturedPieces);
    this.logger.logs = JSON.parse(snap.logs);
    this.turnCount = snap.turnCount;
  }

  refresh() {
    // UI更新
    document.getElementById('turn-info').textContent = this.currentPlayer === Player.SENTE ? '▲ 先手' : '△ 後手';
    document.getElementById('turn-info').className = 'turn-display ' + (this.currentPlayer === Player.SENTE ? 'sente' : 'gote');
    if (this.isPrecomputed) {
      const max = Math.max(this.history.length - 1, 0);
      document.getElementById('phase-info').textContent = `手数: ${this.playbackIndex} / ${max}`;
    } else {
      document.getElementById('phase-info').textContent = `手数: ${this.turnCount}`;
    }
    
    this.updateLogs();
    this.updateHands();
    
    if (this.viewMode === '3d') this.view3d.refresh();
    else this.view2d.render();
  }

  updateLogs() {
    const list = document.getElementById('log-list');
    list.innerHTML = '';
    const logs = this.logger.getAll();
    for (const log of [...logs].reverse()) {
        const li = document.createElement('li');
        li.textContent = `${log.turn}. ${log.text}`;
        li.className = log.player === Player.SENTE ? 'log-sente' : 'log-gote';
        list.appendChild(li);
    }
  }

  updateHands() {
    const senteRow = document.getElementById('hand-row-sente');
    const goteRow = document.getElementById('hand-row-gote');
    if (!senteRow || !goteRow) return;

    this._renderHand(senteRow, Player.SENTE);
    this._renderHand(goteRow, Player.GOTE);
  }

  _renderHand(container, player) {
    container.innerHTML = '';
    const summary = this.getHandSummary(player);
    const playerClass = player === Player.SENTE ? 'sente' : 'gote';

    summary.forEach(s => {
      const item = document.createElement('div');
      item.className = 'hand-item';
      
      const btn = document.createElement('button');
      const grade = getPieceGrade(s.type);
      btn.className = `hand-btn grade-${grade} ${playerClass}` + (s.count <= 0 ? ' empty' : '');
      btn.disabled = s.count <= 0;
      btn.innerHTML = `<span class="name">${s.name}</span>`;
      
      const countBadge = document.createElement('span');
      countBadge.className = 'count' + (s.count === 0 ? ' zero' : '');
      countBadge.textContent = s.count;

      item.appendChild(btn);
      item.appendChild(countBadge);
      container.appendChild(item);
    });
  }

  getHandSummary(player) {
    const mine = this.capturedPieces[player] || [];
    const counts = new Map();
    mine.forEach(p => counts.set(p.type, (counts.get(p.type) || 0) + 1));
    const types = [PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI];
    const names = { [PieceType.FU]:'歩', [PieceType.KYO]:'香', [PieceType.KEI]:'桂', [PieceType.GIN]:'銀', [PieceType.KIN]:'金', [PieceType.KAKU]:'角', [PieceType.HI]:'飛' };
    return types.map(t => ({ type: t, name: names[t], count: counts.get(t) || 0 }));
  }

  updateControls() {
    document.getElementById('btn-play').disabled = this.isPlaying || this.isGenerating;
    document.getElementById('btn-pause').disabled = !this.isPlaying;
  }

  updateSeekUI() {
    const slider = document.getElementById('seek-slider');
    const seekInput = document.getElementById('seek-input');
    const max = this.history.length - 1;
    slider.max = max;
    slider.value = this.playbackIndex;
    seekInput.max = max;
    seekInput.value = this.playbackIndex;
    document.getElementById('seek-val').textContent = this.playbackIndex;
    document.getElementById('max-seek').textContent = max;
  }

  toggleView(mode) {
    this.viewMode = mode;
    const btn3d = document.getElementById('btn-3d-view');
    const btn2d = document.getElementById('btn-2d-view');
    const con3d = document.getElementById('container-3d');
    const wrp2d = document.getElementById('wrapper-2d');

    if (mode === '3d') {
        btn3d.classList.add('active');
        btn2d.classList.remove('active');
        con3d.classList.remove('hidden');
        wrp2d.classList.add('hidden');
        wrp2d.style.display = 'none';
        this.view3d.refresh();
    } else {
        btn3d.classList.remove('active');
        btn2d.classList.add('active');
        con3d.classList.add('hidden');
        wrp2d.classList.remove('hidden');
        wrp2d.style.display = 'block';
        this.view2d.render();
    }
  }
}

// 起動
document.addEventListener('DOMContentLoaded', () => {
    window.wm = new WatchManager();
});
