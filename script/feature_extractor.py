"""
PPO용 Transformer Feature Extractor.

SB3의 BaseFeaturesExtractor를 상속하여 policy_kwargs로 주입됨.
train.py의 config.train.feature_extractor == "transformer"일 때 활성화.

입력 obs (shape=(860,)) 구성 (portfolio_env_hmm.py _get_obs 참조):
    [weights(n_assets=14) |
     past_ret_flat(n_assets × window = 14 × 60 = 840) |
     vol(3) + regime(n_regimes=3) = 6]

처리:
    past_ret → (window, n_assets) → Linear(n_assets, d_model) + learnable PE
           → TransformerEncoder(num_layers, nhead)
           → Global Average Pool over time → (d_model,)
    concat with (weights + vol + regime) → LayerNorm → Linear → GELU → features_dim
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        window: int = 60,
        n_assets: int = 14,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model({d_model})은 nhead({nhead})의 배수여야 합니다."
            )

        self.window = window
        self.n_assets = n_assets
        self.d_model = d_model

        obs_dim = int(observation_space.shape[0])
        # non-sequence 부분: weights + vol + regime
        self.non_seq_dim = obs_dim - n_assets * window
        if self.non_seq_dim < n_assets:
            raise ValueError(
                f"obs_dim({obs_dim})이 너무 작습니다. "
                f"최소 n_assets({n_assets}) + n_assets*window 필요."
            )

        # (window, n_assets) 시퀀스 각 타임스텝을 d_model로 임베딩
        self.token_embed = nn.Linear(n_assets, d_model)

        # 학습 가능한 Positional Encoding (seq_len 고정이므로 충분)
        self.pos_embed = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(d_model + self.non_seq_dim),
            nn.Linear(d_model + self.non_seq_dim, features_dim),
            nn.GELU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        A, W = self.n_assets, self.window

        weights = obs[:, :A]                              # (B, A)
        past_ret_flat = obs[:, A : A + A * W]             # (B, A*W)
        rest = obs[:, A + A * W:]                         # (B, non_seq_dim - A)

        # (B, A*W) → (B, W, A): t=0:asset0..A-1, t=1:..., ...
        seq = past_ret_flat.view(B, W, A)

        x = self.token_embed(seq) + self.pos_embed       # (B, W, d_model)
        x = self.encoder(x)                               # (B, W, d_model)
        seq_feat = x.mean(dim=1)                          # (B, d_model)

        non_seq = torch.cat([weights, rest], dim=1)       # (B, non_seq_dim)
        fused = torch.cat([seq_feat, non_seq], dim=1)     # (B, d_model+non_seq_dim)
        return self.fusion(fused)


def build_policy_kwargs_transformer(cfg: dict) -> dict:
    """train.py / hpo.py에서 호출. config 값만 읽어서 policy_kwargs 구성."""
    t = cfg["transformer"]
    from config import parse_net_arch
    return dict(
        net_arch=parse_net_arch(cfg["ppo"]["net_arch"]),
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=t["features_dim"],
            window=cfg["data"]["window"],
            n_assets=len(cfg["data"]["asset_cols"]),
            d_model=t["d_model"],
            nhead=t["nhead"],
            num_layers=t["num_layers"],
            dropout=t["dropout"],
        ),
    )


if __name__ == "__main__":
    # 형태 점검: config의 env.obs_dim으로 forward 시험
    from config import load_config

    cfg = load_config()
    obs_dim = (
        len(cfg["data"]["asset_cols"])
        + len(cfg["data"]["asset_cols"]) * cfg["data"]["window"]
        + len(cfg["data"]["vol_cols"])
        + cfg["data"]["n_regimes"]
    )
    print(f"obs_dim = {obs_dim}")

    space = gym.spaces.Box(low=-1e6, high=1e6, shape=(obs_dim,))
    fe = TransformerFeatureExtractor(
        space,
        features_dim=cfg["transformer"]["features_dim"],
        window=cfg["data"]["window"],
        n_assets=len(cfg["data"]["asset_cols"]),
        d_model=cfg["transformer"]["d_model"],
        nhead=cfg["transformer"]["nhead"],
        num_layers=cfg["transformer"]["num_layers"],
        dropout=cfg["transformer"]["dropout"],
    )
    x = torch.randn(4, obs_dim)
    y = fe(x)
    print(f"forward OK: {x.shape} → {y.shape}")
    print(f"params: {sum(p.numel() for p in fe.parameters()):,}")
