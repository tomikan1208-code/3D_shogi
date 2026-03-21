import torch
import torch.nn as nn
import os
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Shogi3DFeaturesExtractor(BaseFeaturesExtractor):
    """
    3D-CNN + MLP features extractor for 3D Shogi (Dict Observation)
    Input: board(30, 9, 9, 9), hand_sente(7,), hand_gote(7,)
    """
    def __init__(self, observation_space, features_dim=512):
        # BaseFeaturesExtractor は Dict 空間全体を受け取る
        super(Shogi3DFeaturesExtractor, self).__init__(observation_space, features_dim)
        
        # 1. 盤面用 3D-CNN
        board_space = observation_space.spaces["board"]
        n_input_channels = board_space.shape[0]
        c1 = int(os.getenv("SHOGI_CNN_C1", "64"))
        c2 = int(os.getenv("SHOGI_CNN_C2", "128"))
        c3 = int(os.getenv("SHOGI_CNN_C3", "128"))
        self.cnn = nn.Sequential(
            nn.Conv3d(n_input_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # CNN出力次元の計算
        with torch.no_grad():
            sample_board = torch.as_tensor(board_space.sample()[None]).float()
            n_flatten = self.cnn(sample_board).shape[1]

        # 2. 持ち駒用 MLP (+ 手番情報 1次元)
        self.hand_mlp = nn.Sequential(
            nn.Linear(7 + 7 + 1, 32),
            nn.ReLU()
        )

        # 3. 結合後の統合レイヤー
        self.combined_linear = nn.Sequential(
            nn.Linear(n_flatten + 32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # 盤面
        board_features = self.cnn(observations["board"])
        # 持ち駒 + 手番情報を結合してMLPへ
        hand_turn = torch.cat([
            observations["hand_sente"], 
            observations["hand_gote"], 
            observations["current_player"]
        ], dim=1)
        hand_features = self.hand_mlp(hand_turn)
        
        # 結合
        combined = torch.cat([board_features, hand_features], dim=1)
        return self.combined_linear(combined)
