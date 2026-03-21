from shogi_env import Shogi3DEnv, Player
import numpy as np

def test():
    print("Initializing environment...")
    env = Shogi3DEnv()
    
    print("Resetting environment...")
    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Board shape: {obs['board'].shape}")
    print(f"Hand SENTE: {obs['hand_sente']}")
    print(f"Current Player (Obs): {obs['current_player']}")

    print("\nChecking action masks...")
    masks = env.action_masks()
    print(f"Action masks shape: {masks.shape}")
    print(f"Number of enabled steps (from masks): {np.sum(masks)}")

    print("\nTaking a legal step...")
    # 合法手を取得（現在は Discrete のインデックスを直接選ぶ）
    legal_moves = env.get_legal_moves(Player.SENTE)
    print(f"Found {len(legal_moves)} legal moves for SENTE.")
    
    if len(legal_moves) > 0:
        # 最初の合法手のインデックスは 0
        action = 0
        
        print(f"Taking action index: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step 1 successful. Reward: {reward}")
        print(f"Current Player (Obs) after Step 1 (GOTE expected): {obs['current_player']}")
        
        # 2手目 (後手の番)
        legal_moves_gote = env.get_legal_moves(Player.GOTE)
        if legal_moves_gote:
            action_g = 0 # 最初の合法手
            obs, reward, terminated, truncated, info = env.step(action_g)
            print(f"Step 2 (GOTE) successful. Reward: {reward}")
            print(f"Current Player (Obs) after Step 2 (SENTE expected): {obs['current_player']}")
    else:
        print("No legal moves found. Check initialization.")

    print("Test finished successfully.")

if __name__ == "__main__":
    test()
