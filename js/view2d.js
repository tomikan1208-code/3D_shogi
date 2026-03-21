// view2d.js - 2D断面ビュー（マルチスライス縦並び + ズーム + スクロール）
// 表示基準:
//   上から見て玉（先手）を手前にしたとき「右上奥」を原点とする（2Dも右上が原点）
//   - x軸: 右→左に 1..9（bx=0 が x=1）
//   - y軸: 先手側（手前）を y=1（by=0）として、奥（後手側）に向かって増える
//   - z軸: 上→下に IX..I（bz=8 が IX）

'use strict';

import { PieceType, Player, getDisplayName, getPieceGrade, isPromotedType } from './board.js';

// 3D側 GRADE_PARAMS から導出した 2D用比率
// [scale, bwFrac, twFrac, h1Frac]
// bwFrac=底面幅/全高, twFrac=肩幅/全高, h1Frac=台形部分の高さ比率 (肩までの割合)
const PIECE_PROPS_2D = [
  [0.68, 0.865, 0.615, 0.846],  // grade 0: 歩
  [0.76, 0.865, 0.615, 0.846],  // grade 1: 香・桂
  [0.84, 0.902, 0.647, 0.824],  // grade 2: 銀・金
  [0.92, 0.940, 0.680, 0.800],  // grade 3: 角・飛
  [1.00, 0.980, 0.714, 0.796],  // grade 4: 王
];

const COLOR = {
  sente:     '#e8d5a3',
  gote:      '#5a7a9c',
  outline:   '#2a3040',
  blue:      'rgba(68,153,255,0.45)',
  green:     'rgba(68,204,68,0.65)',
  confirm:   'rgba(0,255,153,0.70)',
  red:       'rgba(255,44,44,0.45)',
  hover:     'rgba(255,0,0,0.45)',
  grid:      '#2e3550',
  bg:        '#12161f',
  curBorder: '#f5c842',
  headerBg:  '#1c2238',
  labelX:    '#e8d5a3',
  labelY:    '#88ccff',
  labelZ:    '#aaffaa',
};

export const SliceType = { XZ: 'XZ', XY: 'XY', YZ: 'YZ' };

// 1-indexed 表示用配列
const ARABIC = ['1','2','3','4','5','6','7','8','9'];
const KANJI  = ['一','二','三','四','五','六','七','八','九'];
const ROMAN  = ['I','II','III','IV','V','VI','VII','VIII','IX'];

// 逆順（描画順などの調整用）
const revARABIC = [...ARABIC].reverse();
const revKANJI  = [...KANJI].reverse();
const revROMAN  = [...ROMAN].reverse();

// 表示順序（2D: 右上が原点）
// col は左(0)→右(8) なので、右が1になるよう [9,8..1] を並べる
const xLabels = revARABIC;   
// y は上→下に 1..9（奥→手前）
const yLabels = KANJI;       
// z は上→下に I..IX
const zLabels = ROMAN;       

const HEADER_H  = 24;
const SLICE_GAP = 8;
const LABEL_W   = 22;
const MIN_VIS   = 1;
const MAX_VIS   = 9;

// ================================================================
// 座標マッピング補助（2D: 右上が原点）
//   col → bx = 8-col  (右がx=1)
//   row → bz = 8-row  (上がz=IX)
//   XY では row → by = row (上がy=1)
// ================================================================

export class View2D {
  constructor(wrapper, gs) {
    this.wrapper = wrapper;
    this.canvas  = document.createElement('canvas');
    this.canvas.style.display = 'block';
    this.wrapper.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');

    this.gs          = gs;
    this.sliceType   = SliceType.XZ;
    this.sliceIndex  = 4;   // board座標でのスライス値
    this.visibleSlices = 3;
    this._hovered    = null;

    this._resize();
    window.addEventListener('resize', () => this._resize());
    this.canvas.addEventListener('click',     (e) => this._onClick(e));
    this.canvas.addEventListener('mousemove', (e) => this._onMouseMove(e));
    this.canvas.addEventListener('mouseleave',()  => { this._hovered = null; this.render(); });
  }

  // ---- 寸法 ----
  get _wrapH() { return this.wrapper.clientHeight || 600; }
  get _wrapW() { return this.wrapper.clientWidth  || 800; }
  get _sliceBlockH() { return Math.max(64, this._wrapH / this.visibleSlices - SLICE_GAP); }
  get _cellSize()    { return Math.max(4, Math.floor(Math.min(this._sliceBlockH - HEADER_H, this._wrapW - LABEL_W * 4) / 9)); }
  get _boardPx()     { return this._cellSize * 9; }
  get _boardOffX()   { return Math.floor((this._wrapW - this._boardPx) / 2) + LABEL_W; }
  get _totalCanvasH(){ return 9 * (this._sliceBlockH + SLICE_GAP); }

  // XZ断面: col 0(左)=9, col 8(右)=1 / row 0(上)=IX, row 8(下)=I
  _canvasI(boardI) { return boardI; }
  _boardI(canvasI) { return canvasI; }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this._wrapW;
    const h = this._totalCanvasH;
    this.canvas.style.width  = w + 'px';
    this.canvas.style.height = h + 'px';
    this.canvas.width  = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
    this._scrollToCurrent();
  }

  _scrollToCurrent() {
    const ci = this._canvasI(this.sliceIndex);
    const bh = this._sliceBlockH + SLICE_GAP;
    this.wrapper.scrollTop = ci * bh + bh / 2 - this._wrapH / 2;
  }

  // ---- 描画 ----
  render() {
    if (!this.gs || !this.gs.board) {
      this.ctx.fillStyle = COLOR.bg;
      this.ctx.fillRect(0, 0, this._wrapW, this._totalCanvasH);
      return;
    }
    const ctx = this.ctx;
    ctx.fillStyle = COLOR.bg;
    ctx.fillRect(0, 0, this._wrapW, this._totalCanvasH);
    const cs = this._cellSize;
    const ox = this._boardOffX;
    for (let ci = 0; ci < 9; ci++) {
      this._drawSlice(ci, this._boardI(ci), ox, ci * (this._sliceBlockH + SLICE_GAP), cs);
    }
  }

  _drawSlice(canvasI, boardI, ox, oy, cs) {
    const ctx = this.ctx;
    const cur = (boardI === this.sliceIndex);

    // ヘッダー
    ctx.fillStyle = cur ? '#2a3060' : COLOR.headerBg;
    ctx.fillRect(0, oy, this._wrapW, HEADER_H);
    ctx.fillStyle = cur ? COLOR.curBorder : '#667788';
    ctx.font = `${cur ? 'bold ' : ''}12px sans-serif`;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(this._sliceLabel(boardI), this._wrapW / 2, oy + HEADER_H / 2);

    const bt = oy + HEADER_H; // board top

    if (cur) {
      ctx.strokeStyle = COLOR.curBorder; ctx.lineWidth = 2;
      ctx.strokeRect(ox - 2, bt - 2, this._boardPx + 4, this._boardPx + 4);
    }

    // ハイライト
    this._drawHighlights(boardI, ctx, cs, ox, bt);

    // ホバー
    if (this._hovered && this._hovered.canvasI === canvasI) {
      ctx.fillStyle = COLOR.hover;
      ctx.fillRect(ox + this._hovered.col * cs, bt + this._hovered.row * cs, cs, cs);
    }

    // グリッド
    ctx.strokeStyle = COLOR.grid; ctx.lineWidth = 0.5;
    for (let k = 0; k <= 9; k++) {
      ctx.beginPath(); ctx.moveTo(ox + k*cs, bt); ctx.lineTo(ox + k*cs, bt + 9*cs); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ox, bt + k*cs); ctx.lineTo(ox + 9*cs, bt + k*cs); ctx.stroke();
    }

    // 軸ラベル
    const [cLbl, rLbl] = this._axisLabels();
    const fSz = Math.max(8, cs * 0.38);
    ctx.font = `${fSz}px sans-serif`; ctx.textBaseline = 'middle';
    ctx.fillStyle = cLbl.color; ctx.textAlign = 'center';
    for (let k = 0; k < 9; k++) ctx.fillText(cLbl.arr[k], ox + k*cs + cs/2, bt - 10);
    ctx.fillStyle = rLbl.color; ctx.textAlign = 'left';
    for (let k = 0; k < 9; k++) ctx.fillText(rLbl.arr[k], ox + 9*cs + 4, bt + k*cs + cs/2);

    // 駒
    const slice = this._getSlice(boardI);
    for (let row = 0; row < 9; row++)
      for (let col = 0; col < 9; col++) {
        const { cell } = slice[row][col];
        if (!cell || cell.type === PieceType.NONE) continue;
        this._drawPiece(ctx, ox + col*cs, bt + row*cs, cs, cell);
      }
  }

  // ---- 軸ラベル ----
  // _axisLabels() → [列ラベル, 行ラベル]
  // XZ: 列=x(左9→右1), 行=z(上IX→下I)
  // XY: 列=x(左9→右1), 行=y(上一→下九)
  // YZ: 列=y(左九→右一), 行=z(上IX→下I)
  _axisLabels() {
    switch (this.sliceType) {
      case SliceType.XZ: return [{ arr: xLabels,   color: COLOR.labelX }, { arr: zLabels,   color: COLOR.labelZ }];
      case SliceType.XY: return [{ arr: xLabels,   color: COLOR.labelX }, { arr: yLabels,   color: COLOR.labelY }];
      case SliceType.YZ: return [{ arr: revKANJI,  color: COLOR.labelY }, { arr: zLabels,   color: COLOR.labelZ }];
    }
    return [{ arr: ARABIC, color: '#fff' }, { arr: ARABIC, color: '#fff' }];
  }

  _sliceLabel(boardI) {
    switch (this.sliceType) {
      case SliceType.XZ: return `y = ${KANJI[boardI]} (${boardI + 1})`;
      case SliceType.XY: return `z = ${ROMAN[boardI]} (${boardI + 1})`;
      case SliceType.YZ: return `x = ${ARABIC[boardI]}`;
    }
  }

  // ---- スライスデータ ----
  // 座標マッピング (MDD: 右上天原点):
  //   col 0→左=x=9, col 8→右=x=1  → bx = 8-col
  //   行方向: XZ/YZ row 0→上=z=0(I), row 8→下=z=8(IX) → bz = row
  //          XY    row 0→上=y=0(一), row 8→下=y=8(九) → by = row
  _getSlice(boardI) {
    const res = [];
    for (let row = 0; row < 9; row++) {
      res[row] = [];
      for (let col = 0; col < 9; col++) {
        let bx, by, bz;
        switch (this.sliceType) {
          case SliceType.XZ: bx = 8-col;   by = boardI;  bz = row;   break;  // 列=x(反転), 行=z
          case SliceType.XY: bx = 8-col;   by = row;     bz = boardI; break;  // 列=x(反転), 行=y
          case SliceType.YZ: bx = boardI;  by = 8-col;   bz = row;   break;  // 列=y(反転), 行=z
        }
        res[row][col] = { cell: this.gs.board.get(bx, by, bz), bx, by, bz };
      }
    }
    return res;
  }

  // board座標 → (col, row) 変換
  _toColRow(bx, by, bz, boardI) {
    switch (this.sliceType) {
      case SliceType.XZ: return by === boardI ? [8-bx, bz] : [null, null];
      case SliceType.XY: return bz === boardI ? [8-bx, by] : [null, null];
      case SliceType.YZ: return bx === boardI ? [8-by, bz] : [null, null];
    }
    return [null, null];
  }

  // board座標 → (bx, by, bz) from (col, row, boardI)
  _colRowToBoard(col, row, boardI) {
    switch (this.sliceType) {
      case SliceType.XZ: return { bx: 8-col,   by: boardI,  bz: row   };
      case SliceType.XY: return { bx: 8-col,   by: row,     bz: boardI };
      case SliceType.YZ: return { bx: boardI,  by: 8-col,   bz: row   };
    }
  }

  // ---- ハイライト ----
  _drawHighlights(boardI, ctx, cs, ox, bt) {
    const hl = (bx, by, bz, color) => {
      const [c, r] = this._toColRow(bx, by, bz, boardI);
      if (c === null) return;
      ctx.fillStyle = color;
      ctx.fillRect(ox + c*cs, bt + r*cs, cs, cs);
    };

    if (this.gs.lastFrom) hl(this.gs.lastFrom.x, this.gs.lastFrom.y, this.gs.lastFrom.z, COLOR.red);
    if (this.gs.lastTo)   hl(this.gs.lastTo.x,   this.gs.lastTo.y,   this.gs.lastTo.z,   COLOR.green);

    for (const [mx,my,mz] of (this.gs.legalMoves || [])) {
      // 自分側の移動可能場所: 常に青
      hl(mx, my, mz, COLOR.blue);
    }

    if (this.gs.selectedPos)  hl(this.gs.selectedPos.x,  this.gs.selectedPos.y,  this.gs.selectedPos.z,  COLOR.blue);
    if (this.gs.confirmedPos) hl(this.gs.confirmedPos.x, this.gs.confirmedPos.y, this.gs.confirmedPos.z, COLOR.confirm);

    // 相手の操作 (マゼンタ/赤ホバー)
    if (this.gs.opponentInteraction) {
      const opp = this.gs.opponentInteraction;
      const MAGENTA = 'rgba(255,0,255,0.45)';
      const RED     = COLOR.red;
      const BLUE    = COLOR.blue;

      if (opp.selectedPos)  hl(opp.selectedPos.x,  opp.selectedPos.y,  opp.selectedPos.z,  MAGENTA);
      if (opp.confirmedPos) hl(opp.confirmedPos.x, opp.confirmedPos.y, opp.confirmedPos.z, MAGENTA);
      
      if (opp.legalMoves) {
        const oh = opp.hoverPos;
        for (const [mx,my,mz] of opp.legalMoves) {
          const isHovered = oh && oh.x === mx && oh.y === my && oh.z === mz;
          hl(mx, my, mz, isHovered ? RED : BLUE);
        }
      }
    }
  }

  // ---- 駒描画 ----
  // 全断面で3Dと同じ比率の将棋五角形を描画
  // XZ断面: sente=上向き五角形(y軸が画面奥なので表が最前面)
  // XY断面: sente=上向き(敵gote側)
  // YZ断面: sente=左向き(敵gote側 = y=8 = col小)
  _drawPiece(ctx, px, py, cs, cell) {
    const sente = cell.player === Player.SENTE;
    const grade = Math.min(getPieceGrade(cell.type), 4);
    const [sc, bwf, twf, h1f] = PIECE_PROPS_2D[grade];
    const promoted = isPromotedType(cell.type);

    const cx = px + cs / 2;
    const cy = py + cs / 2;

    if (this.sliceType === SliceType.YZ) {
      // 横向き五角形: sente=右向き (y=1 側へ)
      this._drawHorizontalPentagon(ctx, cx, cy, cs, sc, bwf, twf, h1f, !sente, sente);
    } else {
      // 縦向き五角形: sente=上向き
      this._drawVerticalPentagon(ctx, cx, cy, cs, sc, bwf, twf, h1f, sente, sente);
    }

    if (cs >= 14) {
      const textColor = promoted ? '#aa1800' : (sente ? '#2a1000' : '#0a1a2a');
      ctx.fillStyle = textColor;
      ctx.font = `bold ${Math.max(8, cs * 0.34)}px serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(getDisplayName(cell), cx, cy);
    }
  }

  // 縦向き将棋五角形（XZ, XY断面）
  // cx,cy: セル中心  sc: スケール(0〜1)  bwf,twf: 底面・肩の幅比率(全高基準)
  // h1f: 肩の高さ比率(台形部分)  tipTop: 先端が上かどうか  isSente: 先手かどうか
  _drawVerticalPentagon(ctx, cx, cy, cs, sc, bwf, twf, h1f, tipTop, isSente) {
    const h    = cs * sc;          // 駒の高さ
    const half = h / 2;            // 上下の半分
    const bw   = h * bwf / 2;      // 底面の半幅
    const sw   = h * twf / 2;      // 肩の半幅
    // shoulderDist: 先端から肩までの距離
    const sd   = h * (1 - h1f);

    const tipY  = tipTop ? cy - half : cy + half;
    const baseY = tipTop ? cy + half : cy - half;
    const shY   = tipTop ? cy - half + sd : cy + half - sd;

    ctx.beginPath();
    ctx.moveTo(cx,       tipY);    // 先端
    ctx.lineTo(cx + sw,  shY);     // 右肩
    ctx.lineTo(cx + bw,  baseY);   // 右底
    ctx.lineTo(cx - bw,  baseY);   // 左底
    ctx.lineTo(cx - sw,  shY);     // 左肩
    ctx.closePath();
    ctx.fillStyle = isSente ? COLOR.sente : COLOR.gote; ctx.fill();
    ctx.strokeStyle = COLOR.outline; ctx.lineWidth = 1; ctx.stroke();
  }

  // 横向き将棋五角形（YZ断面）
  // tipLeft: 先端が左かどうか
  _drawHorizontalPentagon(ctx, cx, cy, cs, sc, bwf, twf, h1f, tipLeft, isSente) {
    const w    = cs * sc;          // 駒の（横方向の）幅
    const half = w / 2;
    const bh   = w * bwf / 2;      // 底辺の半高さ
    const sh   = w * twf / 2;      // 肩の半高さ
    const sd   = w * (1 - h1f);    // 先端から肩までの距離（横方向）

    const tipX  = tipLeft ? cx - half : cx + half;
    const baseX = tipLeft ? cx + half : cx - half;
    const shX   = tipLeft ? cx - half + sd : cx + half - sd;

    ctx.beginPath();
    ctx.moveTo(tipX,   cy);        // 先端
    ctx.lineTo(shX,    cy - sh);   // 上肩
    ctx.lineTo(baseX,  cy - bh);   // 右上底
    ctx.lineTo(baseX,  cy + bh);   // 右下底
    ctx.lineTo(shX,    cy + sh);   // 下肩
    ctx.closePath();
    ctx.fillStyle = isSente ? COLOR.sente : COLOR.gote; ctx.fill();
    ctx.strokeStyle = COLOR.outline; ctx.lineWidth = 1; ctx.stroke();
  }

  // ---- クリック / ホバー ----
  _hitTest(e) {
    const rect = this.canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const cs = this._cellSize;
    const ox = this._boardOffX;
    for (let ci = 0; ci < 9; ci++) {
      const bt = ci * (this._sliceBlockH + SLICE_GAP) + HEADER_H;
      if (my < bt || my >= bt + 9*cs) continue;
      const col = Math.floor((mx - ox) / cs);
      const row = Math.floor((my - bt) / cs);
      if (col < 0 || col > 8 || row < 0 || row > 8) continue;
      const boardI = this._boardI(ci);
      const { bx, by, bz } = this._colRowToBoard(col, row, boardI);
      return { canvasI: ci, boardI, col, row, bx, by, bz };
    }
    return null;
  }

  _onClick(e) {
    const h = this._hitTest(e);
    if (!h) return;
    if (h.boardI !== this.sliceIndex) this.sliceIndex = h.boardI;
    this.gs.handleClick(h.bx, h.by, h.bz);
  }

  _onMouseMove(e) {
    const h    = this._hitTest(e);
    const prev = this._hovered;
    if (!h) { if (prev) { this._hovered = null; this.render(); } return; }
    if (!prev || prev.canvasI !== h.canvasI || prev.col !== h.col || prev.row !== h.row) {
      this._hovered = h; this.render();
    }
  }

  // ---- 公開 API ----
  setSlice(type, idx) { this.sliceType = type; this.sliceIndex = idx; this.render(); this._scrollToCurrent(); }
  setZoom(v)  { this.visibleSlices = Math.max(MIN_VIS, Math.min(MAX_VIS, Math.round(v))); this._resize(); }
  show()      { this.wrapper.style.display = 'block'; this._resize(); }
  hide()      { this.wrapper.style.display = 'none'; }
}
