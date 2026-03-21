import os
import torch
import numpy as np
import threading
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS
from sb3_contrib import MaskablePPO
from shogi_env import Shogi3DEnv, Player, PieceType

app = Flask(__name__)
CORS(app)

# モデルのパス検索
BASE_DIR = os.path.dirname(__file__)
MODEL_SEARCH_DIRS = [
    os.getenv("SHOGI_MODEL_DIR", "").strip(),
    os.path.join(BASE_DIR, "models"),
    os.path.join(os.path.dirname(BASE_DIR), "models"),
]

def find_latest_model():
    # ステップ数（数字部分）が含まれる場合は、最大のものを探す
    # 例: shogi3d_checkpoint_120043_steps.zip
    import re

    def get_steps(fn):
        m = re.search(r'checkpoint_(\d+)_steps', fn)
        if m:
            return int(m.group(1))
        nums = re.findall(r'\d+', fn)
        return int(nums[-1]) if nums else 0

    candidates = []
    for d in MODEL_SEARCH_DIRS:
        if not d:
            continue
        if not os.path.exists(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith(".zip")]
        for f in files:
            candidates.append((get_steps(f), os.path.join(d, f)))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

env = None
model = None
model_device = "cuda" if torch.cuda.is_available() else "cpu"
model_loaded_path = None
model_load_error = None
predict_lock = threading.Lock()
# ルーム/セッションごとの繰り返し管理 (簡易版)
session_repetition = {} 
SESSION_REPETITION_MAX = 50000
AI_VERBOSE = os.getenv("SHOGI_AI_VERBOSE", "0") == "1"
AI_USE_REP_LOOKAHEAD = os.getenv("SHOGI_AI_REP_LOOKAHEAD", "0") == "1"

def get_pos_hash(env):
    return env._get_pos_hash()


def _init_env_for_model(loaded_model):
    global env
    model_action_n = int(getattr(getattr(loaded_model, "action_space", None), "n", 0))
    desired_fixed = 1 if model_action_n > 1500 else 0
    os.environ["SHOGI_FIXED_ACTION_SPACE"] = str(desired_fixed)
    env = Shogi3DEnv()
    env_action_n = int(getattr(env.action_space, "n", 0))
    if model_action_n > 0 and env_action_n != model_action_n:
        raise RuntimeError(
            f"Action-space mismatch after env init: model={model_action_n}, env={env_action_n}."
        )
    print(
        f"[AI Server] action_space_n={env_action_n} (model={model_action_n}, SHOGI_FIXED_ACTION_SPACE={desired_fixed})"
    )

def _snapshot_env_state(e):
    return {
        "board": e.board.copy(),
        "owner": e.owner.copy(),
        "hands": {
            Player.SENTE: dict(e.hands[Player.SENTE]),
            Player.GOTE: dict(e.hands[Player.GOTE]),
        },
        "piece_positions": {
            Player.SENTE: set(e.piece_positions[Player.SENTE]),
            Player.GOTE: set(e.piece_positions[Player.GOTE]),
        },
        "king_positions": {
            Player.SENTE: e.king_positions[Player.SENTE],
            Player.GOTE: e.king_positions[Player.GOTE],
        },
        "current_player": e.current_player,
        "turn_count": e.turn_count,
        "position_version": e.position_version,
        "repetition_map": dict(e.repetition_map),
    }

def _restore_env_state(e, st):
    e.board = st["board"]
    e.owner = st["owner"]
    e.hands = {
        Player.SENTE: Counter(st["hands"][Player.SENTE]),
        Player.GOTE: Counter(st["hands"][Player.GOTE]),
    }
    e.piece_positions = {
        Player.SENTE: st["piece_positions"][Player.SENTE],
        Player.GOTE: st["piece_positions"][Player.GOTE],
    }
    e.king_positions = {
        Player.SENTE: st["king_positions"][Player.SENTE],
        Player.GOTE: st["king_positions"][Player.GOTE],
    }
    e.current_player = st["current_player"]
    e.turn_count = st["turn_count"]
    e.position_version = st["position_version"]
    e.repetition_map = st["repetition_map"]
    e._rebuild_all_caches()
    e.legal_moves_cache.clear()
    e.legal_action_cache.clear()

def _estimate_next_repetition(e, action_idx):
    st = _snapshot_env_state(e)
    try:
        _, _, _, _, _ = e.step(int(action_idx))
        nxt = get_pos_hash(e)
        return session_repetition.get(nxt, 0)
    finally:
        _restore_env_state(e, st)

def _weighted_sample(indices, probs):
    p = np.asarray(probs, dtype=np.float64)
    p_sum = p.sum()
    if p_sum <= 0:
        return int(indices[0])
    p = p / p_sum
    return int(np.random.choice(indices, p=p))

def _pick_action_adaptive(e, legal_indices, legal_probs, request):
    deterministic = bool(request.get("deterministic", False))
    opening_noise = bool(request.get("opening_noise", False))
    forced_stochastic = bool(request.get("stochastic", False))

    order = np.argsort(legal_probs)[::-1]
    sorted_idx = legal_indices[order]
    sorted_probs = legal_probs[order]
    p0 = float(sorted_probs[0]) if len(sorted_probs) else 0.0

    if deterministic:
        return int(sorted_idx[0]), "deterministic_top1"

    # 確信度が非常に高い局面は最善手固定（終盤の読み切りを尊重）
    if (not opening_noise) and (not forced_stochastic) and p0 >= 0.97:
        return int(sorted_idx[0]), "high_conf_top1"

    if opening_noise:
        top_n = min(7, len(sorted_idx))
        base = sorted_probs[:top_n]
        # 序盤は温度を上げて多様性を確保
        temp = 1.35
    elif forced_stochastic or p0 < 0.75:
        top_n = min(6, len(sorted_idx))
        base = sorted_probs[:top_n]
        temp = 1.20
    else:
        top_n = min(3, len(sorted_idx))
        base = sorted_probs[:top_n]
        temp = 1.05

    cand_idx = sorted_idx[:top_n]
    scaled = np.power(np.maximum(base, 1e-12), 1.0 / temp)

    # 1手先の局面が既出ほど重く減衰して千日手を回避
    penalties = []
    if AI_USE_REP_LOOKAHEAD:
        for a in cand_idx:
            rep = _estimate_next_repetition(e, int(a))
            penalties.append(np.exp(-1.2 * rep))
    else:
        penalties = [1.0] * len(cand_idx)

    final_probs = scaled * np.asarray(penalties, dtype=np.float64)
    picked = _weighted_sample(cand_idx, final_probs)
    return picked, f"adaptive_top{top_n}_p0={p0:.3f}"

def get_model():
    global model, model_loaded_path, model_load_error
    if model is None:
        path = find_latest_model()
        if path and os.path.exists(path):
            try:
                model = MaskablePPO.load(path, device=model_device)
                _init_env_for_model(model)
                model_loaded_path = os.path.abspath(path)
                model_load_error = None
                print("[AI Server] Model load OK")
                print(f"[AI Server] name   : {os.path.basename(model_loaded_path)}")
                print(f"[AI Server] path   : {model_loaded_path}")
                print(f"[AI Server] device : {model_device}")
            except Exception as e:
                model_load_error = str(e)
                model_loaded_path = None
                print(f"[AI Server] Model load ERROR: {e}")
        else:
            model_load_error = "No model file found in models directory."
            model_loaded_path = None
            print("[AI Server] No model file found in models directory.")
    return model


@app.route("/model_status", methods=["GET"])
def model_status():
    m = get_model()
    return jsonify({
        "ok": m is not None,
        "device": model_device,
        "model_name": os.path.basename(model_loaded_path) if model_loaded_path else None,
        "model_path": model_loaded_path,
        "error": model_load_error,
    })

@app.route("/predict", methods=["POST"])
def predict():
    global env
    data = request.json
    # data format: { board: [9][9][9], hand_sente: [7], hand_gote: [7], current_player: 1 or 2 }
    
    if not data:
        return jsonify({"error": "No data"}), 400

    m = get_model()
    if m is None:
        return jsonify({"error": "Model not found"}), 500
    if env is None:
        return jsonify({"error": "Environment is not initialized"}), 500

    with predict_lock:
        # 環境の状態を復元
        board_data = np.array(data["board"], dtype=np.int32)
        owner_data = np.array(data["owner"], dtype=np.int8)
        curr_p = int(data["current_player"])

        env.reset()
        env.board = board_data
        env.owner = owner_data
        env.current_player = curr_p

        # 持ち駒の復元
        env.hands[Player.SENTE] = Counter()
        env.hands[Player.GOTE] = Counter()
        piece_map = [PieceType.FU, PieceType.KYO, PieceType.KEI, PieceType.GIN, PieceType.KIN, PieceType.KAKU, PieceType.HI]

        for i, count in enumerate(data["hand_sente"]):
            c = int(count)
            if c > 0:
                env.hands[Player.SENTE][piece_map[i]] = c
        for i, count in enumerate(data["hand_gote"]):
            c = int(count)
            if c > 0:
                env.hands[Player.GOTE][piece_map[i]] = c

        # 駒の位置セットを再構築
        env.piece_positions[Player.SENTE] = set()
        env.piece_positions[Player.GOTE] = set()
        env.king_positions[Player.SENTE] = None
        env.king_positions[Player.GOTE] = None
        for x in range(9):
            for y in range(9):
                for z in range(9):
                    o = env.owner[x, y, z]
                    if o != Player.NONE:
                        env.piece_positions[o].add((x, y, z))
                        if env.board[x, y, z] == PieceType.OU:
                            env.king_positions[o] = (x, y, z)

        env._rebuild_all_caches()
        in_check_before = bool(env._is_king_in_check(curr_p))

        # 推論
        obs = env._get_obs()
        masks = env.action_masks()

        pos_hash = get_pos_hash(env)
        session_repetition[pos_hash] = session_repetition.get(pos_hash, 0) + 1
        if len(session_repetition) > SESSION_REPETITION_MAX:
            session_repetition.clear()

        stochastic = bool(data.get("stochastic", False))
        # サーバー側でも3回以上なら強制的に確率的にする
        if session_repetition.get(pos_hash, 0) >= 3:
            stochastic = True
            if AI_VERBOSE:
                print(f"[AI Server] Repetition detected for {pos_hash}. Forcing stochastic.")

        # 共通の推論処理 (確率配分)
        with torch.no_grad():
            obs_tensor, _ = m.policy.obs_to_tensor(obs)
            act_masks = torch.as_tensor(masks).unsqueeze(0).to(m.device)
            distribution = m.policy.get_distribution(obs_tensor, action_masks=act_masks)
            probs = distribution.distribution.probs[0].cpu().numpy()

            legal_indices = np.where(masks)[0]
            legal_probs = probs[legal_indices]

            if len(legal_indices) > 0:
                request_opts = {
                    "deterministic": bool(data.get("deterministic", False)),
                    "opening_noise": bool(data.get("opening_noise", False)),
                    "stochastic": stochastic,
                }
                action, mode = _pick_action_adaptive(env, legal_indices, legal_probs, request_opts)
                if AI_VERBOSE:
                    print(f"[AI Server] action_mode={mode}")
            else:
                action, _ = m.predict(obs, action_masks=masks, deterministic=True)

        # 選ばれた手を取得
        legal_action_map = env._get_legal_action_map(curr_p)
        move = legal_action_map.get(int(action))
        if move is None:
            if not legal_action_map:
                return jsonify({"error": "No legal actions"}), 500
            # 予測が不正インデックスだった場合は先頭合法手へフォールバック
            action, move = next(iter(legal_action_map.items()))

        # 念のため、王手中はサーバー側で最終安全性を再検証する。
        # これにより、万一のマスク不整合や座標不整合でも王手放置手を返さない。
        if in_check_before:
            is_drop = bool(move[7])
            is_safe = True
            if is_drop:
                ptype = piece_map[int(move[8])]
                is_safe = bool(env._is_drop_safe(ptype, int(move[3]), int(move[4]), int(move[5]), curr_p))
            else:
                is_safe = bool(env._is_move_safe(int(move[0]), int(move[1]), int(move[2]), int(move[3]), int(move[4]), int(move[5]), curr_p))

            if not is_safe:
                fallback = None
                for a_idx, cand in legal_action_map.items():
                    if bool(cand[7]):
                        ptype = piece_map[int(cand[8])]
                        ok = bool(env._is_drop_safe(ptype, int(cand[3]), int(cand[4]), int(cand[5]), curr_p))
                    else:
                        ok = bool(env._is_move_safe(int(cand[0]), int(cand[1]), int(cand[2]), int(cand[3]), int(cand[4]), int(cand[5]), curr_p))
                    if ok:
                        fallback = (a_idx, cand)
                        break

                if fallback is not None:
                    action, move = fallback
                    if AI_VERBOSE:
                        print("[AI Server] Unsafe predicted move under check. Fallback to safe legal move.")
                else:
                    return jsonify({"error": "No safe response while in-check"}), 500
    
    return jsonify({
        "action_idx": int(action),
        "move": {
            "fx": int(move[0]), "fy": int(move[1]), "fz": int(move[2]),
            "tx": int(move[3]), "ty": int(move[4]), "tz": int(move[5]),
            "promote": bool(move[6]),
            "is_drop": bool(move[7]),
            "drop_type": int(move[8])
        }
    })

if __name__ == "__main__":
    # Eager-load once so startup log always shows model status.
    get_model()
    app.run(port=5000)
