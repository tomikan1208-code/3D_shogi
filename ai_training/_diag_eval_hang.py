import os
import time

print('[DIAG] start', flush=True)


t0 = time.time()
import torch
print(f"[DIAG] torch import ok in {time.time() - t0:.2f}s", flush=True)

t1 = time.time()
from sb3_contrib import MaskablePPO
print(f"[DIAG] sb3_contrib import ok in {time.time() - t1:.2f}s", flush=True)

t2 = time.time()
from ai_training.eval_vs_random import _resolve_model_path, _load_model_for_inference
print(f"[DIAG] eval module import ok in {time.time() - t2:.2f}s", flush=True)

model_path = _resolve_model_path('./ai_training/models/shogi3d_ppo_latest.zip')
print(f"[DIAG] resolved: {model_path}", flush=True)

for device in ['auto', 'cpu']:
    t3 = time.time()
    print(f"[DIAG] loading model device={device} ...", flush=True)
    model = _load_model_for_inference(model_path, device)
    print(f"[DIAG] model loaded device={device} in {time.time() - t3:.2f}s", flush=True)
    del model

print('[DIAG] done', flush=True)
