# Colab Training (Google Drive)

## 1. Google Drive に ai_training フォルダを置く
- 例: `MyDrive/ai_training/`

## 2. Colab ノートブックで実行
以下をセルで順に実行してください。

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
cd /content
cp -r /content/drive/MyDrive/ai_training ./ai_training
cd /content/ai_training
python -m pip install --upgrade pip
python -m pip install torch gymnasium stable-baselines3 sb3-contrib flask flask-cors
```

```bash
cd /content/ai_training
python train_colab.py \
  --drive-root /content/drive/MyDrive/3d_shogi_train \
  --n-envs 2 \
  --n-steps 4096 \
  --n-epochs 4 \
  --batch-size 256 \
  --max-mask-gb 8 \
  --device cuda \
  --force-cuda \
  --fixed-action-space
```

## 挙動
- 学習開始時: `drive-root/models` に既存モデルがあれば最新を自動ロード
- 学習中: チェックポイントと最新モデルを `drive-root/models` に保存
- TensorBoardログ: `drive-root/logs` に保存

### TensorBoard が "No dashboards are active" のとき
- 学習セルを先に実行して、少なくとも 1 世代進めてください。
- イベントファイルが無い場合は表示されません（`events.out.tfevents.*` が必要）。
- `COLAB_TRAINING.ipynb` の TensorBoard セルは、Drive/ローカル両方の候補パスを探索するようにしてあります。

## アップロード必須ファイル（ai_training 単体）
- 必須:
  - `train_colab.py`
  - `train.py`
  - `shogi_env.py`
  - `model.py`
  - `requirements_colab.txt`（任意だが推奨）
- 任意:
  - `cpp_kingcheck.py`, `shogi3d_core*.pyd`（高速化拡張を使う場合）
- 不要:
  - `__pycache__/`, `logs/`, `models/` の一時ファイル

## 3. 再開方法
同じコマンドをもう一度実行すると、Drive 上の最新モデルから継続します。
