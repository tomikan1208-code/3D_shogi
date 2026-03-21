// ui-online.js - オンライン対戦専用のUI制御

'use strict';

import { GameState, GamePhase } from './game-online.js';
import { View3D } from './view3d.js';
import { View2D, SliceType } from './view2d.js';
import { Player, PieceType } from './board.js';
import { ROMAN, KANJIDIGIT } from './pieces.js';

// DOM要素
const container3d    = document.getElementById('container-3d');
const wrapper2d      = document.getElementById('wrapper-2d');
const btn3dView      = document.getElementById('btn-3d-view');
const btn2dView      = document.getElementById('btn-2d-view');

// 断面コントロール
const sliceBtns    = document.querySelectorAll('.slice-btn');
const sliceSlider  = document.getElementById('slice-slider');
const sliceLabel   = document.getElementById('slice-label');
const zoomSlider   = document.getElementById('zoom-slider');
const zoomVal      = document.getElementById('zoom-val');

// 透明度 (3Dのみ)
const transparencySlider = document.getElementById('transparency-slider');
const transparencyVal    = document.getElementById('transparency-val');
const panelTransparency  = document.getElementById('panel-transparency');
const panelSlice         = document.getElementById('panel-slice');
const panelHand          = document.getElementById('panel-hand');
const sidePanel          = document.querySelector('.side-panel');

// 情報パネル
const turnInfo     = document.getElementById('turn-info');
const phaseInfo    = document.getElementById('phase-info');
const logList      = document.getElementById('log-list');

// 成りダイアログ
const promoteDialog  = document.getElementById('promote-dialog');
const btnPromote     = document.getElementById('btn-promote');
const btnNoPromote   = document.getElementById('btn-no-promote');

// ゲームオーバー
const gameOverBanner = document.getElementById('game-over');
const gameOverText   = document.getElementById('game-over-text');
const btnRestart     = document.getElementById('btn-restart');

let gs, view3d, view2d;
let currentViewMode = '3d';
let currentSliceType = SliceType.XZ;

// ============================================================
// 初期化
// ============================================================
export function initUI() {
  gs = new GameState();
  gs.onStateChange = onStateChange;
  window.shogiGame = gs;

  // 3Dビュー
  view3d = new View3D(container3d, gs);

  // 2Dビュー
  view2d = new View2D(wrapper2d, gs);

  // 3D/2D切り替え
  document.getElementById('btn-3d-view')?.addEventListener('click', () => { if (currentViewMode !== '3d') toggleView(); });
  document.getElementById('btn-2d-view')?.addEventListener('click', () => { if (currentViewMode !== '2d') toggleView(); });

  // 断面ボタン
  sliceBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      sliceBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentSliceType = btn.dataset.slice;
      updateSliceLabel();
      view2d.setSlice(currentSliceType, parseInt(sliceSlider.value));
    });
  });

  // 断面スライダー
  sliceSlider?.addEventListener('input', () => {
    updateSliceLabel();
    view2d.setSlice(currentSliceType, parseInt(sliceSlider.value));
  });

  // ズームスライダー
  zoomSlider?.addEventListener('input', () => {
    const v = parseInt(zoomSlider.value);
    if (zoomVal) zoomVal.textContent = v;
    view2d.setZoom(v);
  });

  // 透明度 (3D)
  transparencySlider?.addEventListener('input', () => {
    const v = parseFloat(transparencySlider.value);
    view3d.setTransparency(v);
    if (transparencyVal) transparencyVal.textContent = Math.round(v * 100) + '%';
  });

  // 成り
  btnPromote?.addEventListener('click', () => {
    gs.handlePromotion(true);
  });
  btnNoPromote?.addEventListener('click', () => {
    gs.handlePromotion(false);
  });

  // 再スタート (オンラインではロビーに戻る等の処理が必要だが、一旦ページリロードで対応)
  btnRestart?.addEventListener('click', () => {
    location.reload();
  });

  updateSliceLabel();
  _applyPanelState('3d');

  // エスケープキーでキャンセル
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      gs.cancelSelection();
    }
  });
}

function refresh() {
  if (currentViewMode === '3d') view3d.refresh();
  else view2d.render();
}

// ============================================================
// 状態変更コールバック
// ============================================================
function onStateChange(state) {
  const isMyTurn = (state.currentPlayer === state.playerSide);
  const playerName = state.currentPlayer === Player.SENTE ? '▲ 先手' : '△ 後手';
  const turnSuffix = isMyTurn ? ' (あなた)' : ' (相手)';

  if (turnInfo) {
    turnInfo.textContent = playerName + turnSuffix;
    turnInfo.className = 'turn-display ' + (state.currentPlayer === Player.SENTE ? 'sente' : 'gote');
  }

  const phaseNames = {
    [GamePhase.IDLE]:    '待機中',
    [GamePhase.SELECT]:  isMyTurn ? '駒を選択してください' : '相手の番です',
    [GamePhase.MOVE]:    '移動先を選択してください',
    [GamePhase.CONFIRM]: 'ここへ動かしますか？',
    [GamePhase.DROP]:    '打ち先を選択してください',
    [GamePhase.DROP_CONFIRM]: 'ここへ打ちますか？',
    [GamePhase.PROMOTE]: '成り確認中...',
    [GamePhase.DONE]:    '終局',
  };
  if (phaseInfo) phaseInfo.textContent = phaseNames[state.phase] || '';

  refresh();

  // 成りダイアログ表示（自分の時のみ）
  if (state.phase === GamePhase.PROMOTE && isMyTurn) {
    promoteDialog.classList.remove('hidden');
    promoteDialog.style.display = 'flex';
  } else {
    promoteDialog.classList.add('hidden');
    promoteDialog.style.display = 'none';
  }

  // ゲーム終了
  if (state.phase === GamePhase.DONE) {
    const winner = state.winnerPlayer === Player.SENTE ? '先手' : '後手';
    const isWin = (state.winnerPlayer === state.playerSide);
    if (gameOverText) gameOverText.textContent = `${winner} の勝ち！${isWin ? ' おめでとう！' : ''}`;
    gameOverBanner.classList.remove('hidden');
    gameOverBanner.style.display = 'flex';
  } else {
    gameOverBanner.classList.add('hidden');
    gameOverBanner.style.display = 'none';
  }

  refreshLog(state);
  updateHandUI(state);
}

function updateHandUI(state) {
  const rowMe = document.getElementById('hand-row');
  const rowOpp = document.getElementById('hand-row-opponent');
  if (!rowMe || !rowOpp) return;

  const mySide = state.playerSide;
  const oppSide = (mySide === Player.SENTE) ? Player.GOTE : Player.SENTE;

  _renderHand(rowMe, mySide, true);
  _renderHand(rowOpp, oppSide, false);

  const handHint = document.getElementById('hand-hint');
  if (handHint) {
    if (state.selectedHandType) handHint.textContent = 'もう一度クリックで確定';
    else handHint.textContent = '持ち駒を選択して配置できます';
  }
}

function _renderHand(container, player, isMe) {
  const allTypes = [
    PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN,
    PieceType.KIN, PieceType.KAKU, PieceType.HI
  ];
  
  const summary = gs.getHandSummary(player);
  const counts = new Map();
  summary.forEach(s => counts.set(s.type, s.count));
  
  // 自分の場合のみ選択を考慮
  let sel = isMe ? gs.selectedHandType : null;
  // 相手の選択状態を同期
  const oppSel = !isMe ? gs.opponentInteraction?.selectedHandType : null;

  const isMyTurn = (gs.currentPlayer === gs.playerSide);

  container.innerHTML = '';
  
  for (const type of allTypes) {
    const count = counts.get(type) || 0;
    const name = PieceName[type];
    const grade = getPieceGrade(type);
    
    // スタイル調整のためのラッパー
    const item = document.createElement('div');
    item.className = 'hand-item';
    
    const btn = document.createElement('button');
    const playerClass = player === Player.SENTE ? 'sente' : 'gote';
    let btnClass = `hand-btn grade-${grade} ${playerClass}`;
    if (isMe && sel === type) btnClass += ' active';
    if (!isMe && oppSel === type) btnClass += ' active-opponent'; // 相手の選択（赤）
    if (count <= 0) btnClass += ' empty';
    
    btn.className = btnClass;
    // 駒がある場合は色を出すため、disabled は count <= 0 の時だけに限定（または成りの時など）
    btn.disabled = count <= 0 || gs.phase === GamePhase.PROMOTE || gs.phase === GamePhase.DONE;
    
    btn.innerHTML = `<span class="name">${name}</span>`;
    if (isMe) {
      btn.onclick = () => gs.toggleHandSelect(type);
    }
    
    const countBadge = document.createElement('span');
    countBadge.className = 'count' + (count === 0 ? ' zero' : '');
    countBadge.textContent = count;
    
    item.appendChild(btn);
    item.appendChild(countBadge);
    container.appendChild(item);
  }
}

// 駒の強さ（大きさ）判定用
function getPieceGrade(type) {
  if (type === PieceType.HI || type === PieceType.KAKU) return 'major';
  if (type === PieceType.KIN || type === PieceType.GIN) return 'middle';
  return 'minor';
}

const PieceName = {
  [PieceType.FU]:   '歩',
  [PieceType.KYO]:  '香',
  [PieceType.KEI]:  '桂',
  [PieceType.GIN]:  '銀',
  [PieceType.KIN]:  '金',
  [PieceType.KAKU]: '角',
  [PieceType.HI]:   '飛',
  [PieceType.OU]:   '玉'
};

function refreshLog(state) {
  if (!logList) return;
  const logs = state.logger.getAll();
  logList.innerHTML = '';
  for (const log of [...logs].reverse()) {
    const li = document.createElement('li');
    li.textContent = `${log.turn}. ${log.text}`;
    li.className = log.player === Player.SENTE ? 'log-sente' : 'log-gote';
    logList.appendChild(li);
  }
}

function toggleView() {
  if (currentViewMode === '3d') {
    currentViewMode = '2d';
    container3d.classList.add('hidden');
    container3d.style.display = 'none';
    wrapper2d.classList.remove('hidden');
    wrapper2d.style.display = 'block'; 
    view2d.show();
    btn2dView?.classList.add('active');
    btn3dView?.classList.remove('active');
    view2d.setSlice(currentSliceType, parseInt(sliceSlider?.value || 4));
    view2d.setZoom(parseInt(zoomSlider?.value || 3));
    _applyPanelState('2d');
  } else {
    currentViewMode = '3d';
    container3d.classList.remove('hidden');
    container3d.style.display = 'block';
    wrapper2d.classList.add('hidden');
    wrapper2d.style.display = 'none';
    view2d.hide();
    btn3dView?.classList.add('active');
    btn2dView?.classList.remove('active');
    view3d.refresh();
    _applyPanelState('3d');
  }
}

function _applyPanelState(mode) {
  sidePanel?.classList.remove('mode-2d', 'mode-3d');
  if (mode === '2d') {
    panelSlice?.classList.remove('hidden');
    panelSlice?.classList.add('active');
    panelHand?.classList.add('active');
    sidePanel?.classList.add('mode-2d');
    if (panelTransparency) panelTransparency.classList.add('hidden');
  } else {
    panelSlice?.classList.add('hidden');
    panelSlice?.classList.remove('active');
    panelHand?.classList.add('active');
    sidePanel?.classList.add('mode-3d');
    if (panelTransparency) panelTransparency.classList.remove('hidden');
  }
}

function updateSliceLabel() {
  const val = parseInt(sliceSlider?.value || 4);
  if (!sliceLabel) return;
  switch (currentSliceType) {
    case SliceType.XZ: sliceLabel.textContent = `y = ${KANJIDIGIT[val]} (${val + 1})`; break;
    case SliceType.XY: sliceLabel.textContent = `z = ${ROMAN[val]} (${val + 1})`; break;
    case SliceType.YZ: sliceLabel.textContent = `x = ${val + 1}`; break;
  }
}

document.addEventListener('DOMContentLoaded', initUI);
