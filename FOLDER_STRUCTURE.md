# 3D Shogi フォルダ構造ガイド

このドキュメントは、3d_shogi フォルダを編集しやすくするための構造説明です。

## 1. まず最初に見る場所（目的別）

- 画面デザインを変えたい
  - css/style.css
  - 各ページHTML: index.html, game.html, online.html, watch.html, pieces.html
- ローカル対戦のルールや挙動を変えたい
  - js/game.js
  - js/board.js
  - js/pieces.js
- オンライン対戦の挙動を変えたい
  - js/game-online.js
  - js/online-manager.js
  - js/online-lobby.js
  - server.js
- 3D/2Dの描画を変えたい
  - js/view3d.js
  - js/view2d.js
- NPC観戦（AI vs AI）の挙動を変えたい
  - js/watch.js
  - ai_training/ai_server.py
- 学習そのものを変えたい
  - ai_training/shogi_env.py
  - ai_training/model.py
  - ai_training/train.py

---

## 2. ルート直下の構成

- index.html
  - モード選択トップ画面。
- game.html
  - ローカル対戦画面。
- online.html
  - オンライン対戦画面（ロビー + 対局UI）。
- watch.html
  - NPC観戦画面。
- pieces.html
  - 駒の動き可視化ページ。
- server.js
  - Node.jsサーバー本体。
  - 静的配信、Socket.io通信、AI推論プロキシ（/predict -> 5000）を担当。
- package.json
  - Nodeサーバーの依存関係と起動スクリプト。
- rules.md
  - ルール文書。
- AI.bat
  - PythonのAI推論サーバー起動。
- start.bat
  - Nodeサーバー + AIサーバーをまとめて起動。
- n.bat
  - 補助バッチ（用途は中身確認推奨）。

補足:
- node_modules はnpm依存の展開先。通常は編集しない。
- server.log はサーバーログ出力先。

---

## 3. フロントエンド層（js/ と css/）

### 3.1 コアロジック

- js/board.js
  - 盤面データ構造（9x9x9）、駒種定義、初期配置、成り関連。
  - ルールの土台。最重要ファイルの1つ。
- js/pieces.js
  - 各駒の移動生成ロジック。
  - 駒の動き仕様を変えるならここ。
- js/logger.js
  - 棋譜ログ処理。

### 3.2 ローカル対戦

- js/game.js
  - ローカル対戦の状態遷移（SELECT, MOVE, DROP, PROMOTEなど）。
- js/ui.js
  - ローカル画面のDOM操作とイベント配線。

### 3.3 オンライン対戦

- js/game-online.js
  - オンライン用ゲーム状態。
  - 自分ターン制約、相手アクション反映、同期処理。
- js/online-manager.js
  - Socket.ioの送受信ラッパー。
- js/online-lobby.js
  - ロビーUI、入室、ゲーム開始、再入室同期。
- js/ui-online.js
  - オンライン対戦画面のUI制御。

### 3.4 描画

- js/view3d.js
  - Three.jsでの3D表示。
- js/view2d.js
  - 断面2D表示。

### 3.5 観戦

- js/watch.js
  - AIサーバーへ /predict を投げて手を取得し、局面を進行。
  - 履歴シークや再生速度もここで制御。

### 3.6 スタイル

- css/style.css
  - 全ページ共通の見た目。

---

## 4. サーバー層

### 4.1 Nodeサーバー（対局・配信）

- server.js
  - 静的ファイル配信。
  - Socket.ioでルーム作成・参加・指し手同期・再入室同期。
  - /predict へのPOSTをPython AIサーバーへプロキシ。

### 4.2 Python AI推論/学習

- ai_training/ai_server.py
  - Flask API (/predict)。
  - 受け取った局面を env に復元してモデル推論。
- ai_training/shogi_env.py
  - Gym環境（合法手生成、報酬、終了判定）。
  - 学習ルールの中心。
- ai_training/model.py
  - 3D-CNN + MLPの特徴抽出器。
- ai_training/train.py
  - MaskablePPOで自己対戦学習。
- ai_training/models/
  - 学習済みモデル保存先。
- ai_training/logs/
  - TensorBoardログ。

---

## 5. 実行フロー（重要）

### 5.1 ローカル対戦

1. ブラウザで game.html を表示
2. js/ui.js が js/game.js を使って状態管理
3. js/view3d.js / js/view2d.js が描画

### 5.2 オンライン対戦

1. online.html でロビー表示
2. js/online-lobby.js が js/online-manager.js 経由でSocket.io接続
3. server.js がルーム・同期を仲介
4. 対局中は js/game-online.js がローカル状態 + 相手同期を制御

### 5.3 観戦（AI vs AI）

1. watch.html + js/watch.js を起動
2. js/watch.js が /predict を呼ぶ
3. server.js が AIサーバー(5000)へ転送
4. ai_training/ai_server.py がモデル推論結果を返す

---

## 6. 編集時のおすすめ順序

- ルール変更の場合
  1. js/board.js, js/pieces.js を先に修正
  2. js/game.js と js/game-online.js の整合を確認
  3. 必要なら ai_training/shogi_env.py も同じルールに合わせる

- UI変更の場合
  1. 対象ページHTMLを修正
  2. css/style.css を修正
  3. 必要に応じて ui.js / ui-online.js / watch.js のDOM参照を更新

- オンライン同期変更の場合
  1. js/online-manager.js（送受信フォーマット）
  2. server.js（Socketイベント）
  3. js/game-online.js（反映ロジック）

---

## 7. よくある改修ポイント

- 駒の動きを変える: js/pieces.js
- 初期配置を変える: js/board.js
- 成り条件を変える: js/board.js と js/game.js / js/game-online.js
- ロビー挙動を変える: js/online-lobby.js
- ルーム仕様を変える: server.js
- AIの強さを変える: ai_training/train.py, ai_training/model.py
- AIの行動ルールを変える: ai_training/shogi_env.py

---

## 8. 起動の目安

- Nodeサーバーのみ
  - npm start
- Node監視起動
  - npm run dev
- AIサーバー
  - AI.bat
- まとめて起動（Windows）
  - start.bat

---

## 9. 変更時の注意

- フロント側ルールと学習環境側ルールがずれると、AIの手が不正になる。
  - 対象: js/board.js, js/pieces.js, ai_training/shogi_env.py
- オンライン同期イベント名を変える場合は、必ずクライアントとserver.jsを同時に変更する。
- 画面IDやclass名を変更した場合、対応するUIスクリプト内のquerySelector系も同時更新する。

---

## 10. 簡易ツリー

- 3d_shogi/
  - index.html
  - game.html
  - online.html
  - watch.html
  - pieces.html
  - server.js
  - package.json
  - css/
    - style.css
  - js/
    - board.js
    - pieces.js
    - logger.js
    - game.js
    - ui.js
    - game-online.js
    - online-manager.js
    - online-lobby.js
    - ui-online.js
    - view3d.js
    - view2d.js
    - watch.js
  - ai_training/
    - ai_server.py
    - shogi_env.py
    - model.py
    - train.py
    - test_env.py
    - benchmark_env.py
    - models/
    - logs/
