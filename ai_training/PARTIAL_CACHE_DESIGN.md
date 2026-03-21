# 部分的キャッシュ設計書（Shogi3DEnv）

対象: ai_training/shogi_env.py

## 1. 目的

`get_legal_moves` と `_is_king_in_check` の再計算コストを下げるため、局面差分に応じて必要最小限の再計算だけを行う。

狙い:
- 同一局面内で同じ駒の疑似合法手を何度も計算しない
- 1手更新後に「影響がある駒だけ」再計算する
- 王手判定を毎回全探索しない

## 2. 現状実装の前提

現状で既にある高速化基盤:
- `piece_positions` を保持
- `king_positions` を保持

主な計算ホットスポット:
- `get_legal_moves(player)`
- `_is_king_in_check(player)`
- `_is_move_safe(...)` / `_is_drop_safe(...)` 内の王手判定

## 3. キャッシュ対象

### 3.1 駒ごとの疑似合法手キャッシュ

定義:
- `piece_move_cache[(x,y,z,ptype,owner)] = [ [tx,ty,tz], ... ]`

意味:
- 「王手回避を考えない」疑似合法手（今の `_get_piece_moves` 相当）

### 3.2 利きマップ（被攻撃マス）キャッシュ

定義:
- `attack_map[player][x][y][z] = attack_count`

意味:
- そのマスを player 側が何枚で攻撃しているか
- 王手判定は `attack_map[opp][king_pos] > 0` で判定可能

### 3.3 スライディング依存インデックス（任意だが推奨）

定義:
- `ray_watchers[square] = {piece_key, ...}`

意味:
- そのマスの占有状態が変わると影響を受ける飛び駒（飛/角/香/竜/馬）

## 4. 無効化ルール一覧（重要）

以下のイベント発生時に、該当キャッシュを無効化する。

### 4.1 通常移動（capture なし）

入力:
- `from = (fx,fy,fz)`
- `to = (tx,ty,tz)`
- `moved_piece`

無効化対象:
1. moved_piece 自身の `piece_move_cache`
2. `from` または `to` を通るスライディング駒の `piece_move_cache`
3. `from` / `to` の `ray_watchers` 登録更新
4. `attack_map` は「差分更新」
   - moved_piece の旧手集合を減算
   - moved_piece の新手集合を加算
   - 影響スライダーの旧新差分を反映
5. moved_piece が `OU` の場合は `king_positions` 更新

### 4.2 通常移動（capture あり）

4.1 に追加で:
1. captured_piece の `piece_move_cache` を削除
2. captured_piece の攻撃寄与を `attack_map` から減算
3. captured_piece の `ray_watchers` を削除

### 4.3 成り（promotion）

4.1/4.2 に追加で:
1. 成り前ptypeのキャッシュを破棄
2. 成り後ptypeで再計算（駒種変化のため必須）

### 4.4 打ち（drop）

入力:
- `to = (tx,ty,tz)`
- `dropped_piece`

無効化対象:
1. dropped_piece の `piece_move_cache` 新規作成
2. `to` を通るスライディング駒の `piece_move_cache`
3. `to` の `ray_watchers` 更新
4. `attack_map` 差分更新
   - dropped_piece の攻撃を加算
   - 影響スライダーの旧新差分を反映

### 4.5 王手安全確認用の仮適用（_is_move_safe / _is_drop_safe）

ルール:
- 本体キャッシュを壊さない

方式:
1. 変更対象キャッシュの差分ログを積む
2. 判定後に差分ログでロールバック

注意:
- ここでフル再構築すると逆に遅くなるため、差分適用/差分巻戻し必須

### 4.6 reset / 局面復元

無効化対象:
- 全キャッシュ初期化

再構築順:
1. `piece_positions`, `king_positions`
2. 全駒 `piece_move_cache`
3. `ray_watchers`
4. `attack_map`

## 5. 再計算対象の決定方法

入力: `dirty_squares = {from, to, capture_square, drop_square}`

手順:
1. dirty にいる駒本体
2. `ray_watchers[dirty]` に登録されたスライダー
3. 王移動/王捕獲時は king 周辺の近接駒

これを `affected_pieces` とし、ここだけ再計算する。

## 6. 王中心判定との整合

現状 `_is_king_in_check` は王中心のレイ探索に寄っている。

選択肢:
1. 既存王中心判定を維持し、`piece_move_cache` を `get_legal_moves` のみに使う
2. 王手判定を `attack_map` 参照に完全移行

推奨:
- 段階導入として 1 を先に実装
- 安定後に 2 へ移行

## 7. 実装フェーズ

### Phase 1（低リスク）

- `piece_move_cache` のみ導入
- invalidation は conservatively 広め（from/to 直線全体）
- 正しさ優先

### Phase 2（本命）

- `ray_watchers` 導入
- affected 駒の絞り込み
- `_is_move_safe` / `_is_drop_safe` を差分ログ方式に変更

### Phase 3（拡張）

- `attack_map` 導入
- 王手判定を O(1) 参照へ

## 8. テスト要件

最低限の回帰テスト:
1. 初期局面で合法手数一致（導入前後）
2. ランダム対局 N 手で合法手集合一致（導入前後）
3. 王手局面/詰み局面で判定一致
4. 打ち（特に歩打ち・二歩）一致
5. 成り局面一致

性能テスト:
1. `get_legal_moves` calls/sec
2. `step` calls/sec
3. `_is_move_safe` 1回あたり時間

## 9. リスク

- 無効化漏れで不正合法手が混入する
- 影響範囲を狭めすぎるとバグる

対策:
- 開発中はデバッグモードで「キャッシュ版とフル再計算版を比較」して差分検知

## 10. 期待効果

実装後の目標:
- `get_legal_moves` の平均時間 25-45% 短縮
- 終盤（駒密度低）でさらに短縮
- 学習時の env step throughput 向上
