import time
from shogi_env import Shogi3DEnv, Player

def benchmark():
    env = Shogi3DEnv()
    env.reset()
    
    # テスト回数
    iterations = 1
    
    print(f"Starting benchmark: {iterations} calls of get_legal_moves...")
    start_time = time.time()
    
    for _ in range(iterations):
        env.get_legal_moves(Player.SENTE)
        env.get_legal_moves(Player.GOTE)
        
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Elapsed time for {iterations * 2} calls: {elapsed:.4f} seconds")
    print(f"Average time per call: {elapsed / (iterations * 2):.6f} seconds")
    print(f"Estimated FPS (calls per second): {(iterations * 2) / elapsed:.2f}")

if __name__ == "__main__":
    benchmark()
