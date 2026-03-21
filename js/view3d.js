// view3d.js - Three.js 3Dビュー + ナビキューブ
//
// 座標マッピング (将棋準拠):
//   board.x → Three.js X (反転: 右がx=1=bx=0, 左がx=9=bx=8)
//   board.y → Three.js Z (手前=先手y=1, 奥=後手y=9)
//   board.z → Three.js Y (高さ)
//
// 駒形状: 9面体（将棋駒の立体化）
//   底面1 + 4台形側面 + 先端への4三角面
//   先手はTip方向が+Z（奥/後手側）、後手はY軸180°回転

'use strict';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PieceType, Player, getDisplayName, isPromotedType, getPieceGrade } from './board.js';

// ============================================================
// 定数
// ============================================================
const CELL = 1.2;
const GRID = 9;
const OFFSET = (GRID - 1) / 2 * CELL;

const COLOR = {
  grid:    0x445566,
  sente:   0xdec88a,
  gote:    0x4a6a8c,
  senteDark: '#1a0800',
  goteDark:  '#0a1a2a',
  promoted:  '#cc2200',  // 成り駒のテキスト色
  blue:    0x4499ff,
  green:   0x44cc44,
  red:     0xff4444,
  confirm: 0x00ff99,
};

// ============================================================
// 座標変換
// ============================================================
const toW  = (i) => i * CELL - OFFSET;
const toWX = (bx) => OFFSET - bx * CELL;   // x=1(bx=0) -> 画面右端
const toWY = (by) => -OFFSET + by * CELL;  // y=1(by=0) -> 画面奥端 (王側)
const toWZ = (bz) => OFFSET - bz * CELL;   // z=I(bz=0) -> 画面上端 (天)

const boardToW = (bx, by, bz) => ({
  tx: toWX(bx), // board.x -> Three.x (右1 -> 左9)
  ty: toWZ(bz), // board.z -> Three.y (高さ: 上I -> 下IX)
  tz: toWY(by), // board.y -> Three.z (奥行き: 奥1 -> 手前9)
});

// ============================================================
// 将棋駒ジオメトリ（9面体: 底1 + 4側面 + 4頂三角）
//
// 形状設計（自然座標: tip=+Y, front=+Z, piece body in XY plane）:
//   Bottom 4 vertices (y=0):  ±bw/2, ±bd/2
//   Body top 4 vertices (y=h1): ±tw/2, ±td/2  (台形にする)
//   Tip 1 vertex (y=h2, x=0, z=0)
//
// 完成後 rotateX(-PI/2) して tip を +Z 方向へ向ける
// ============================================================
const GRADE_PARAMS = [
  // bw=横幅(X), bd=厚み(Y), tw=先端横幅, td=先端厚み
  // h1r=台形部分の高さ(比率大きく)、h2r=全体の高さ
  // grade 0: 歩
  { scale: 0.68, bw: 0.90, bd: 0.36, tw: 0.64, td: 0.24, h1r: 0.88, h2r: 1.04 },
  // grade 1: 香・桂・成歩
  { scale: 0.76, bw: 0.90, bd: 0.36, tw: 0.64, td: 0.24, h1r: 0.88, h2r: 1.04 },
  // grade 2: 銀・金・成香・成桂・成銀
  { scale: 0.84, bw: 0.92, bd: 0.38, tw: 0.66, td: 0.26, h1r: 0.84, h2r: 1.02 },
  // grade 3: 角・飛・馬・竜
  { scale: 0.92, bw: 0.94, bd: 0.40, tw: 0.68, td: 0.28, h1r: 0.80, h2r: 1.00 },
  // grade 4: 王・玉
  { scale: 1.00, bw: 0.96, bd: 0.42, tw: 0.70, td: 0.30, h1r: 0.78, h2r: 0.98 },
];

/**
 * 将棋駒の9面体BufferGeometry を生成
 * @param {number} grade 0〜4
 */
/**
 * 将棋駒の7面体BufferGeometry を生成 (稜線あり)
 * フロント面とバック面が五角形、その間を側面でつなぐ
 * @param {number} grade 0〜4
 */
function createPieceGeo(grade) {
  const p  = GRADE_PARAMS[grade] ?? GRADE_PARAMS[2];
  const s  = p.scale * CELL;
  
  const bw = p.bw * s * 0.5;   // 底面半幅
  const tw = p.tw * s * 0.5;   // 肩半幅
  const h1 = p.h1r * s;        // 肩までの高さ
  const h2 = p.h2r * s;        // 先端（稜線）までの高さ
  const bth = p.bd * s * 0.5;  // 底面の厚み半分
  const tth = p.td * s * 0.5;  // 先端の厚み半分 (テーパ)

  // 10頂点 (Front五角形 5点 + Back五角形 5点)
  const verts = new Float32Array([
    // Front face (z = +depth)
    -bw, 0,  bth,  // 0: BL
     bw, 0,  bth,  // 1: BR
     tw, h1, tth,  // 2: TR (shoulder) - thin
      0, h2, tth,  // 3: Tip - thin
    -tw, h1, tth,  // 4: TL (shoulder) - thin
    // Back face (z = -depth)
    -bw, 0, -bth,  // 5: BL
     bw, 0, -bth,  // 6: BR
     tw, h1,-tth,  // 7: TR (shoulder) - thin
      0, h2,-tth,  // 8: Tip - thin
    -tw, h1,-tth,  // 9: TL (shoulder) - thin
  ]);

  // 面インデックス
  const idx = [
    // Front face
    0,1,2, 0,2,4, 2,3,4,
    // Back face (反転)
    5,7,6, 5,9,7, 7,9,8,
    // Bottom
    0,5,6, 0,6,1,
    // Left side
    0,4,9, 0,9,5,
    // Right side
    1,6,7, 1,7,2,
    // Shoulder Right
    2,7,8, 2,8,3,
    // Shoulder Left
    4,3,8, 4,8,9,
  ];

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();

  // rotateX(+PI/2): Tip(+Y) -> +Z 方向へ向ける
  geo.rotateX(Math.PI / 2);
  // Z方向中央揃え
  geo.translate(0, 0, -h2 / 2);

  return geo;
}

// ============================================================
// テクスチャ生成
// ============================================================
function makeNameTexture(cell) {
  const name     = getDisplayName(cell);
  const promoted = isPromotedType(cell.type);
  const isSente  = cell.player === Player.SENTE;

  const c = document.createElement('canvas');
  c.width = 128; c.height = 128;
  const ctx = c.getContext('2d');
  // 背景なし（透明）
  ctx.clearRect(0, 0, 128, 128);

  // 文字色（先手:濃茶, 後手:白、成り駒:赤）
  const baseColor = isSente ? '#2a1000' : '#ffffff';
  ctx.fillStyle = promoted ? COLOR.promoted : baseColor;
  ctx.shadowBlur = 0; // 影を消す

  const italic   = promoted && cell.type !== PieceType.NARI_FU;
  const fontSize = name.length > 1 ? 46 : 62;
  ctx.font = `${italic ? 'italic ' : ''}bold ${fontSize}px serif`;
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(name, 64, 68);

  return new THREE.CanvasTexture(c);
}

function makeLabelSprite(text, color) {
  const c = document.createElement('canvas');
  c.width = 80; c.height = 64;
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, 80, 64);
  ctx.fillStyle = color;
  ctx.font = `bold ${text.length > 2 ? 20 : 26}px serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, 40, 32);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false,
  }));
  sp.scale.set(text.length > 2 ? 0.65 : 0.5, 0.45, 1);
  return sp;
}

// ============================================================
// View3D
// ============================================================
export class View3D {
  constructor(container, gameState) {
    this.container = container;
    this.gs        = gameState;
    this.pieceMeshes    = new Map();
    this.pieceBodies    = [];
    this.highlightMeshes = [];
    this.labelTargets    = []; // 静的なラベル判定用ターゲット
    this.labelSprites    = { x: [], y: [], z: [] }; // スケール更新用
    this.transparency    = 0.85;
    this._hoverPos       = null;
    this._hoverMesh      = null;
    this._lastPieceStateKey = '';

    // グレード別ジオメトリをキャッシュ（5段階）
    this._geoCache = [0,1,2,3,4].map(g => createPieceGeo(g));

    this._initScene();
    this._initGrid();
    this._initAxisLabels();
    this._initNaviCube();
    this._initHoverMarker();
    this._bindEvents();
    this._startLoop();
  }

  // ----------------------------------------------------------
  // シーン
  // ----------------------------------------------------------
  _initScene() {
    let w = this.container.clientWidth;
    let h = this.container.clientHeight;

    // 非表示状態で初期化される場合の NaN クラッシュ対策
    if (w === 0 || h === 0) {
      w = window.innerWidth;
      h = window.innerHeight;
    }

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setSize(w, h);
    this.renderer.autoClear = false;
    this.container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1e2a);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 500);
    this.camera.position.set(-12, 14, 20);
    this.camera.lookAt(0, 0, 0);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(-10, 20, 10);
    this.scene.add(dir);

    this.raycaster   = new THREE.Raycaster();
    this.mouse       = new THREE.Vector2();
    this.clickTargets = [];
    this.clickMap    = new Map();
  }

  // ----------------------------------------------------------
  // グリッド（将棋式 10×10×10 セル枠）
  // ----------------------------------------------------------
  _initGrid() {
    // depthWrite: false で奥の透明なオブジェクト(駒など)が欠けないようにする
    const mat  = new THREE.LineBasicMaterial({ color: COLOR.grid, transparent: true, opacity: 0.3, depthWrite: false });
    const gMin = -OFFSET - CELL * 0.5;
    const gMax =  OFFSET + CELL * 0.5;
    const lp   = (j) => (j - 0.5) * CELL - OFFSET;

    for (let a = 0; a <= GRID; a++) {
      const pa = lp(a);
      for (let b = 0; b <= GRID; b++) {
        const pb = lp(b);
        const add = (pts) => this.scene.add(new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(pts), mat));
        add([new THREE.Vector3(gMin, pa, pb), new THREE.Vector3(gMax, pa, pb)]);
        add([new THREE.Vector3(pa, gMin, pb), new THREE.Vector3(pa, gMax, pb)]);
        add([new THREE.Vector3(pa, pb, gMin), new THREE.Vector3(pa, pb, gMax)]);
      }
    }
  }

  // ----------------------------------------------------------
  // 軸ラベル
  // ----------------------------------------------------------
  _initAxisLabels() {
    const ARABIC = ['1','2','3','4','5','6','7','8','9'];
    const KANJI  = ['一','二','三','四','五','六','七','八','九'];
    const ROMAN  = ['I','II','III','IV','V','VI','VII','VIII','IX'];

    const gMax = OFFSET + CELL * 0.5;
    const PAD  = CELL * 0.9;

    const addLabel = (sprite, axis, index) => {
      this.scene.add(sprite);
      // 見えない当たり判定用ボックスを追加 (Raycaster が検知できるよう透明度0の Mesh にする)
      const size = CELL * 0.8;
      const hitBox = new THREE.Mesh(
        new THREE.BoxGeometry(size, size, size),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 })
      );
      this.scene.add(hitBox);
      this.labelTargets.push(hitBox);
      this.clickMap.set(hitBox.uuid, { isLabel: true, axis, index });
      return { sprite, hitBox };
    };

    const posBack = -OFFSET - CELL * 0.5 - PAD;
    const posTop  =  OFFSET + CELL * 0.5 + PAD;
    const posLeft =  OFFSET + CELL * 0.5 + PAD;

    for (let i = 0; i < 9; i++) {
      const wx = toWX(i);
      const wz = toWY(i);
      const wh = toWZ(i);
      
      // X軸ラベル
      const lx = addLabel(makeLabelSprite(ARABIC[i], '#e8d5a3'), 'x', i);
      lx.sprite.position.set(wx, posTop, posBack);
      lx.hitBox.position.set(wx, posTop, posBack);
      this.labelSprites.x.push(lx.sprite);

      // Y軸ラベル
      const ly = addLabel(makeLabelSprite(KANJI[i], '#88ccff'), 'y', i);
      ly.sprite.position.set(posLeft, posTop, wz);
      ly.hitBox.position.set(posLeft, posTop, wz);
      this.labelSprites.y.push(ly.sprite);

      // Z軸ラベル
      const lz = addLabel(makeLabelSprite(ROMAN[i], '#aaffaa'), 'z', i);
      lz.sprite.position.set(posLeft, wh, posBack);
      lz.hitBox.position.set(posLeft, wh, posBack);
      this.labelSprites.z.push(lz.sprite);
    }

    // 軸名ラベル
    const posL = OFFSET + CELL * 1.5;
    [
      ['x', [0,    posL, -posL], '#e8d5a3'],
      ['y', [posL, posL, 0],     '#88ccff'],
      ['z', [posL, 0, -posL],    '#aaffaa'],
    ].forEach(([t, pos, col]) => {
      const sp = makeLabelSprite(t, col);
      sp.position.set(...pos); sp.scale.multiplyScalar(1.5);
      this.scene.add(sp);
    });

    // 原点リセットマーカー (ラベルの交点)
    const oSize = CELL / 3;
    const originMarker = new THREE.Mesh(
      new THREE.BoxGeometry(oSize, oSize, oSize),
      new THREE.MeshBasicMaterial({ color: COLOR.red })
    );
    // ラベルが並ぶ直線の交点
    originMarker.position.set(posLeft, posTop, posBack);
    originMarker.visible = false; // 初期状態（強調なし）は非表示
    this.scene.add(originMarker);
    this.originMarker = originMarker;
    this.labelTargets.push(originMarker);
    this.clickMap.set(originMarker.uuid, { isOrigin: true });
  }

  // ----------------------------------------------------------
  // ナビキューブ
  // ----------------------------------------------------------
  _initNaviCube() {
    this.naviScene = new THREE.Scene();
    this.naviCamera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    this.naviCamera.position.set(-3.5, 3.5, 3.5);
    this.naviCamera.lookAt(0, 0, 0);

    this.naviClickTargets = [];
    this.naviClickDirs = new Map();

    const HALF = 0.5, EDGE_W = 0.10, FACE_H = HALF - EDGE_W;

    const faceLabels = [
      { label: '前', dir: [0, 0, 1],  color: '#1a3a8a' },
      { label: '後', dir: [0, 0,-1],  color: '#1a3a8a' },
      { label: '上', dir: [0, 1, 0],  color: '#0d6640' },
      { label: '下', dir: [0,-1, 0],  color: '#0d6640' },
      { label: '右', dir: [-1,0, 0],  color: '#7a2a10' },
      { label: '左', dir: [1, 0, 0],  color: '#7a2a10' },
    ];

    for (const { label, dir, color } of faceLabels) {
      const c = document.createElement('canvas');
      c.width = 128; c.height = 128;
      const ctx = c.getContext('2d');
      ctx.fillStyle = color; ctx.fillRect(0, 0, 128, 128);
      const b = 128 * 0.20;
      ctx.strokeStyle = 'rgba(255,255,255,0.40)'; ctx.lineWidth = 1.5;
      ctx.beginPath();
      [b, 128-b].forEach(v => { ctx.moveTo(v,0); ctx.lineTo(v,128); });
      [b, 128-b].forEach(v => { ctx.moveTo(0,v); ctx.lineTo(128,v); });
      ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 30px sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(label, 64, 64);
      const faceSize = HALF * 2 * 0.60;
      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(faceSize, faceSize),
        new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), side: THREE.DoubleSide })
      );
      mesh.position.set(...dir.map(v => v * HALF));
      if (dir[0] !== 0) mesh.rotateY(Math.PI / 2);
      else if (dir[1] !== 0) mesh.rotateX(Math.PI / 2);
      this.naviScene.add(mesh);
      this.naviClickTargets.push(mesh);
      this.naviClickDirs.set(mesh.uuid, { type: 'face', dir });
    }

    this.naviScene.add(new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1,1,1)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 })
    ));

    for (const sx of [-1,1]) for (const sy of [-1,1]) for (const sz of [-1,1]) {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(EDGE_W*2, EDGE_W*2, EDGE_W*2),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      m.position.set(sx*FACE_H, sy*FACE_H, sz*FACE_H);
      this.naviScene.add(m);
      this.naviClickTargets.push(m);
      this.naviClickDirs.set(m.uuid, { type: 'corner', dir: [sx,sy,sz] });
    }

    const edgeDirList = [];
    for (const a of [-1,1]) for (const b of [-1,1]) edgeDirList.push([a,b,0],[a,0,b],[0,a,b]);
    for (const [ex,ey,ez] of edgeDirList) {
      const m = new THREE.Mesh(
        new THREE.BoxGeometry(
          ex===0 ? HALF*2 : EDGE_W*2,
          ey===0 ? HALF*2 : EDGE_W*2,
          ez===0 ? HALF*2 : EDGE_W*2
        ),
        new THREE.MeshBasicMaterial({ visible: false })
      );
      m.position.set(
        ex===0 ? 0 : ex*FACE_H,
        ey===0 ? 0 : ey*FACE_H,
        ez===0 ? 0 : ez*FACE_H
      );
      this.naviScene.add(m);
      this.naviClickTargets.push(m);
      this.naviClickDirs.set(m.uuid, { type: 'edge', dir: [ex,ey,ez] });
    }
  }

  // ----------------------------------------------------------
  // イベント
  // ----------------------------------------------------------
  _bindEvents() {
    this.renderer.domElement.addEventListener('mousedown', (e) => {
      this._mouseDownPos = { x: e.clientX, y: e.clientY };
    });
    this.renderer.domElement.addEventListener('click', (e) => this._onClick(e));
    this.renderer.domElement.addEventListener('mousemove', (e) => this._onMouseMove(e));
    this.renderer.domElement.addEventListener('mouseleave', () => this._clearHover());
    window.addEventListener('resize', () => this._onResize());

    // エスケープキーでハイライト解除
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this.gs.resetHighlight();
    });
  }

  _onClick(e) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const w = rect.width, h = rect.height;

    // ドラッグ（回転）した場合はクリックを無視
    if (this._mouseDownPos) {
      const dx = e.clientX - this._mouseDownPos.x;
      const dy = e.clientY - this._mouseDownPos.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > 5) return; 
    }

    if (cx > w - 120 && cy < 120) { this._onNaviClick(cx - (w-120), cy, 120); return; }

    this.mouse.x = (cx/w)*2-1; this.mouse.y = -(cy/h)*2+1;
    this.raycaster.setFromCamera(this.mouse, this.camera);
    // ピース用とラベル用両方をマージして判定
    const hits = this.raycaster.intersectObjects([...this.clickTargets, ...this.labelTargets]);
    if (hits.length > 0) {
      const info = this.clickMap.get(hits[0].object.uuid);
      if (info) {
        if (info.isOrigin) {
          this.gs.resetHighlight();
        } else if (info.isLabel) {
          this.gs.handleLabelClick(info.axis, info.index);
        } else {
          this.gs.handleClick(info.x, info.y, info.z);
        }
      }
    }
  }

  _onMouseMove(e) {
    if (!this.gs?.board) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
    const w = rect.width, h = rect.height;

    if (cx > w - 120 && cy < 120) { this._clearHover(); return; }

    this.mouse.x = (cx / w) * 2 - 1;
    this.mouse.y = -(cy / h) * 2 + 1;
    this.raycaster.setFromCamera(this.mouse, this.camera);


    // 判定対象を「現在クリック可能なもの（強調断面内または全表示）」に限定
    const checkTargets = [...this.clickTargets];

    const hits = this.raycaster.intersectObjects(checkTargets);
    if (!hits.length) { this._clearHover(); return; }
    
    const obj = hits[0].object;
    const info = this.clickMap.get(obj.uuid);
    if (!info) { this._clearHover(); return; }

    // 断面強調時は範囲外のホバーを無効化
    const highlight = this.gs.highlightedSlice;
    if (highlight) {
      const { axis, index } = highlight;
      const x = info.x, y = info.y, z = info.z;
      const isInSlice = (axis === 'x' && x === index) || (axis === 'y' && y === index) || (axis === 'z' && z === index);
      if (!isInSlice) { this._clearHover(); return; }
    }

    // 持ち駒配置用マーカーの青→緑変更
    if (this._emptyMarkers && this._emptyMarkers.includes(obj)) {
      if (this._hoveredMarker && this._hoveredMarker !== obj) {
        this._hoveredMarker.material.color.set(COLOR.blue);
      }
      if (obj.material.color.getHex() === COLOR.blue) {
        obj.material.color.set(COLOR.green);
        this._hoveredMarker = obj;
      }
    } else {
      if (this._hoveredMarker) {
        this._hoveredMarker.material.color.set(COLOR.blue);
        this._hoveredMarker = null;
      }
    }
    this._setHover(info.x, info.y, info.z);
  }

  _initHoverMarker() {
    const mat = new THREE.MeshBasicMaterial({ color: COLOR.blue, transparent: true, opacity: 0.22, depthWrite: false });
    const m   = new THREE.Mesh(new THREE.BoxGeometry(CELL, CELL, CELL), mat);
    m.visible = false;
    this.scene.add(m);
    this._hoverMesh = m;
  }

  _updateOpponentInteraction() {
  }

  _setHover(bx, by, bz) {
    if (this._hoverPos && this._hoverPos.x === bx && this._hoverPos.y === by && this._hoverPos.z === bz) return;
    this._hoverPos = { x: bx, y: by, z: bz };
    if (!this._hoverMesh) return;
    this._hoverMesh.position.set(...this._pos(bx, by, bz));
    this._hoverMesh.visible = true;
  }

  _clearHover() {
    this._hoverPos = null;
    if (this._hoverMesh) this._hoverMesh.visible = false;
    if (this._hoveredMarker) {
      this._hoveredMarker.material.color.set(COLOR.blue);
      this._hoveredMarker = null;
    }
  }

  _onNaviClick(lx, ly, size) {
    const ray = new THREE.Raycaster();
    ray.setFromCamera(new THREE.Vector2((lx/size)*2-1, -(ly/size)*2+1), this.naviCamera);
    const hits = ray.intersectObjects(this.naviClickTargets);
    if (hits.length > 0) {
      const info = this.naviClickDirs.get(hits[0].object.uuid);
      if (info) this._setViewFromDir(info.dir);
    }
  }

  _setViewFromDir(dir) {
    const dist = 28;
    const [dx,dy,dz] = dir;
    const len = Math.sqrt(dx*dx+dy*dy+dz*dz);
    this.camera.position.set(dx/len*dist, dy/len*dist, dz/len*dist);
    this.camera.lookAt(0, 0, 0);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  _onResize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.camera.aspect = w/h; this.camera.updateProjectionMatrix();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    this.renderer.setSize(w, h);
  }

  // ----------------------------------------------------------
  // 盤面描画
  // ----------------------------------------------------------
  _pos(bx, by, bz) {
    const { tx, ty, tz } = boardToW(bx, by, bz);
    return [tx, ty, tz];
  }

  refresh() {
    if (!this.gs.board) return;
    const nextKey = this._makePieceStateKey();
    const pieceChanged = nextKey !== this._lastPieceStateKey;

    if (pieceChanged) {
      this._clearPieces();
      this._drawPieces();
      this._lastPieceStateKey = nextKey;
    }

    this._clearHighlights();
    this._drawHighlights();
    this._updateLabelScales();
    this._updateOpponentInteraction();
  }

  _makePieceStateKey() {
    let s = '';
    const board = this.gs.board;
    for (let x = 0; x < 9; x++)
      for (let y = 0; y < 9; y++)
        for (let z = 0; z < 9; z++) {
          const c = board.get(x, y, z);
          if (!c || c.type === PieceType.NONE) continue;
          s += `${x}${y}${z}${c.player}${c.type}${c.promoted ? 1 : 0}|`;
        }
    const hl = this.gs.highlightedSlice;
    if (hl) s += `@${hl.axis}${hl.index}`;
    return s;
  }

  _updateLabelScales() {
    const highlight = this.gs.highlightedSlice;
    // 強調中のみリセットマーカーを表示
    if (this.originMarker) this.originMarker.visible = !!highlight;

    ['x', 'y', 'z'].forEach(axis => {
      this.labelSprites[axis].forEach((sp, idx) => {
        const isHighlighted = highlight && highlight.axis === axis && highlight.index === idx;
        const baseScale = axis === 'x' ? 0.5 : 0.5; // Xだけ微調整してた時期もありましたが基本 0.5
        // Xのデフォルトが [0.5, 0.45, 1] だったので、比率を維持
        const factor = isHighlighted ? 2.0 : 1.0;
        const xLenScale = (this.gs.ROMAN?.length > 2) ? 0.65 : 0.5; // axis ではなく text 内容に依るが簡略化
        sp.scale.set(0.5 * factor, 0.45 * factor, 1);
      });
    });
  }

  _clearPieces() {
    for (const { body, sprite } of this.pieceMeshes.values()) {
      this.scene.remove(body); this.scene.remove(sprite);
      this.clickMap.delete(body.uuid);
    }
    this.clickTargets = []; this.pieceMeshes.clear(); this.pieceBodies = [];
  }

  _clearHighlights() {
    for (const m of this.highlightMeshes) {
      this.scene.remove(m);
      this.clickMap.delete(m.uuid);
    }
    this.highlightMeshes = [];
  }

  _drawPieces() {
    const board = this.gs.board;
    const highlight = this.gs.highlightedSlice;

    for (let bx = 0; bx < 9; bx++) for (let by = 0; by < 9; by++) for (let bz = 0; bz < 9; bz++) {
      const cell = board.get(bx, by, bz);
      if (!cell || cell.type === PieceType.NONE) continue;

      const grade   = getPieceGrade(cell.type);
      const geo     = this._geoCache[grade];
      const isSente = cell.player === Player.SENTE;

      let opacity = this.transparency;
      if (highlight) {
        const { axis, index } = highlight;
        const isOn = (axis === 'x' && bx === index) || (axis === 'y' && by === index) || (axis === 'z' && bz === index);
        opacity = isOn ? 1.0 : 0.25;
      }

      const mat  = new THREE.MeshPhongMaterial({
        color: isSente ? COLOR.sente : COLOR.gote,
        transparent: true, opacity: opacity,
        shininess: 30, flatShading: true
      });
      const body = new THREE.Mesh(geo, mat);
      if (isSente) body.rotation.y = Math.PI;
      body.position.set(...this._pos(bx, by, bz));
      this.scene.add(body);

      const tex    = makeNameTexture(cell);
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, opacity: opacity }));
      const [tx, ty, tz] = this._pos(bx, by, bz);
      sprite.position.set(tx, ty + 0.22, tz);
      sprite.scale.set(0.65, 0.65, 0.65);
      this.scene.add(sprite);

      // 強調断面内、または強調なしの場合のみクリック対象にする
      if (!highlight || (highlight && ((highlight.axis === 'x' && bx === highlight.index) || (highlight.axis === 'y' && by === highlight.index) || (highlight.axis === 'z' && bz === highlight.index)))) {
        this.clickTargets.push(body);
      }
      
      this.clickMap.set(body.uuid, { x: bx, y: by, z: bz });
      this.pieceMeshes.set(`${bx},${by},${bz}`, { body, sprite });
      this.pieceBodies.push(body);
    }
  }

  _drawHighlights() {
    const gs = this.gs;
    this._emptyMarkers = [];

    const addMarker = (bx, by, bz, color, alpha = 0.45, isClickable = false) => {
      const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: alpha, depthWrite: false });
      const m   = new THREE.Mesh(new THREE.BoxGeometry(CELL, CELL, CELL), mat);
      m.position.set(...this._pos(bx, by, bz));
      this.scene.add(m);
      this.highlightMeshes.push(m);
      
      if (isClickable) {
        // 断面強調時は断面内のみクリック可能
        const highlight = this.gs.highlightedSlice;
        let canClick = true;
        if (highlight) {
          const { axis, index } = highlight;
          canClick = (axis === 'x' && bx === index) || (axis === 'y' && by === index) || (axis === 'z' && bz === index);
        }
        
        if (canClick) {
          this.clickTargets.push(m);
        }
        this.clickMap.set(m.uuid, { x: bx, y: by, z: bz });
        this._emptyMarkers.push(m);
      }
    };

    if (gs.lastFrom)    addMarker(gs.lastFrom.x, gs.lastFrom.y, gs.lastFrom.z, COLOR.red,   0.55);
    
    // 前回着手: 自分も相手も緑で表示するように統一
    if (gs.lastTo)      addMarker(gs.lastTo.x,   gs.lastTo.y,   gs.lastTo.z,   COLOR.green, 0.55);
    
    // 現在のホバー座標（自分）
    const myHover = this._hoverPos;

    for (const [mx,my,mz] of gs.legalMoves) {
      // 自分側の移動可能場所: 常に青 (ローカル同様)
      addMarker(mx, my, mz, COLOR.blue, 0.45, true);
    }
    
    // 選択中の自分の駒 (青 - ローカル回帰)
    if (gs.selectedPos) addMarker(gs.selectedPos.x, gs.selectedPos.y, gs.selectedPos.z, COLOR.blue, 0.45);
    
    // 確定待ち (ターコイズ - ローカル回帰)
    if (gs.confirmedPos) addMarker(gs.confirmedPos.x, gs.confirmedPos.y, gs.confirmedPos.z, COLOR.confirm, 0.7);

    // --- 相手の操作の可視化 ---
    if (this.gs.opponentInteraction) {
      const opp = this.gs.opponentInteraction;
      const MAGENTA = 0xff00ff;
      const RED     = 0xff0000;
      const BLUE    = COLOR.blue;

      // 相手の選択駒 (マゼンタ)
      if (opp.selectedPos) {
        addMarker(opp.selectedPos.x, opp.selectedPos.y, opp.selectedPos.z, MAGENTA, 0.4);
      }
      // 相手の確定待ち (マゼンタ)
      if (opp.confirmedPos) {
        addMarker(opp.confirmedPos.x, opp.confirmedPos.y, opp.confirmedPos.z, MAGENTA, 0.6);
      }
      // 相手の検討中移動先 (青、ホバー時は赤)
      if (opp.legalMoves) {
        const oh = opp.hoverPos;
        for (const [mx, my, mz] of opp.legalMoves) {
          const isHovered = oh && oh.x === mx && oh.y === my && oh.z === mz;
          addMarker(mx, my, mz, isHovered ? RED : BLUE, 0.25);
        }
      }
    }
  }

  setTransparency(val) {
    this.transparency = val;
    for (const { body } of this.pieceMeshes.values()) body.material.opacity = val;
  }

  // ----------------------------------------------------------
  // レンダリングループ
  // ----------------------------------------------------------
  _startLoop() {
    const loop = () => {
      requestAnimationFrame(loop);
      this.controls.update();

      const dpr = window.devicePixelRatio;
      const w = this.renderer.domElement.width  / dpr;
      const h = this.renderer.domElement.height / dpr;
      if (w <= 0 || h <= 0) return;

      this.renderer.clear();
      this.renderer.render(this.scene, this.camera);

      // ナビキューブの描画（サイズが十分な場合のみ）
      if (w >= 120 && h >= 120) {
        this.naviCamera.position.copy(this.camera.position).normalize().multiplyScalar(5);
        this.naviCamera.lookAt(0, 0, 0);

        const s = 120 * dpr;
        this.renderer.setScissorTest(true);
        this.renderer.setScissor(  (w-120)*dpr, (h-120)*dpr, s, s);
        this.renderer.setViewport( (w-120)*dpr, (h-120)*dpr, s, s);
        this.renderer.render(this.naviScene, this.naviCamera);
        this.renderer.setScissorTest(false);
        this.renderer.setViewport(0, 0, w*dpr, h*dpr);
      }
    };
    loop();
  }

  show()    { this.renderer.domElement.style.display = 'block'; }
  hide()    { this.renderer.domElement.style.display = 'none'; }
  dispose() { this.renderer.dispose(); this.container.removeChild(this.renderer.domElement); }
}
