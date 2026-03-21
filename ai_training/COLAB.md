# Google Colab Setup

## 1) Clone repository

```bash
git clone <YOUR_REPO_URL>
cd 3d_shogi
```

## 2) Install dependencies

```bash
pip install -r ai_training/requirements_colab.txt
```

## 3) (Optional) Mount Google Drive for model persistence

```python
from google.colab import drive
drive.mount('/content/drive')
```

Drive root example:
- `/content/drive/MyDrive`

## 4) Start training (Colab launcher)

Without Drive:

```bash
python ai_training/colab_train.py --n-envs 2 --n-steps 128 --total-steps 200000 --opening-random-plies 2
```

GPU mandatory:

```bash
python ai_training/colab_train.py --force-cuda --device cuda --n-envs 2 --n-steps 128 --total-steps 200000 --opening-random-plies 2
```

With Drive:

```bash
python ai_training/colab_train.py \
  --drive-root /content/drive/MyDrive \
  --force-cuda \
  --device cuda \
  --n-envs 2 \
  --n-steps 128 \
  --total-steps 200000 \
  --opening-random-plies 2
```

## 5) Notes

- If CPU-only runtime is selected, training is still possible but slower.
- To resume, run the same command again; latest checkpoint in model directory will be loaded automatically.
- `SHOGI_MODEL_DIR` and `SHOGI_LOG_DIR` are now supported by `train.py`.
