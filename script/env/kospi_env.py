"""
Single-agent portfolio environment for KOSPI sector ETFs.

Observation (dict):
    asset  : (M, L, N)  per-asset z-scored window features (row 0 = cash = zeros)
    market : (L, K)     market-wide window features (vkospi, vol20_kospi)
    weight : (M,)       previous-step post-trade weights

Action:
    (M,) simplex vector (M = num_assets + 1, index 0 is cash).
    The env will project any numpy array to the simplex so the agent need not.

Reward:
    log(1 + w·r - cost·turnover)

Tensors are precomputed up-front (mirrors the reference repo's approach).
Everything is numpy so it stays framework-agnostic and easy to port to
multi-agent / vectorized rollouts later.
"""
import numpy as np


class KospiPortfolioEnv:
    def __init__(self, df, cols, mean, std, window=20, cost=0.0005):
        self.cols = cols
        self.window = window
        self.cost = cost
        self.M = len(cols["returns"]) + 1  # +1 for cash
        self.L = window
        self.N = 3                          # return, vol20, vol60
        self.K = len(cols["market"])

        norm = (df - mean) / std
        self._build_tensors(df, norm)
        self.reset()

    def _build_tensors(self, df, norm):
        ret_raw = df[self.cols["returns"]].values.astype(np.float32)
        z_ret = norm[self.cols["returns"]].values.astype(np.float32)
        z_v20 = norm[self.cols["vol20"]].values.astype(np.float32)
        z_v60 = norm[self.cols["vol60"]].values.astype(np.float32)
        z_mkt = norm[self.cols["market"]].values.astype(np.float32)

        T = len(df)
        asset_states, market_states, r_next = [], [], []

        for t in range(self.L, T - 1):
            # (L, A, N) -> (A, L, N)
            window_feats = np.stack(
                [z_ret[t - self.L:t], z_v20[t - self.L:t], z_v60[t - self.L:t]],
                axis=-1,
            ).transpose(1, 0, 2)

            cash_row = np.zeros((1, self.L, self.N), dtype=np.float32)
            asset_state = np.concatenate([cash_row, window_feats], axis=0)  # (M, L, N)
            market_state = z_mkt[t - self.L:t]                              # (L, K)
            r_t1 = np.concatenate([[0.0], ret_raw[t]]).astype(np.float32)   # (M,)

            asset_states.append(asset_state)
            market_states.append(market_state)
            r_next.append(r_t1)

        self.states_asset = np.stack(asset_states)   # (T', M, L, N)
        self.states_market = np.stack(market_states) # (T', L, K)
        self.r_next = np.stack(r_next)               # (T', M)
        self.T = len(self.states_asset)

    # ------------------------------------------------------------------
    # gym-like API
    # ------------------------------------------------------------------
    def reset(self):
        self.t = 0
        self.w = np.zeros(self.M, dtype=np.float32)
        self.w[0] = 1.0  # start fully in cash
        return self._obs()

    def _obs(self):
        return {
            "asset":  self.states_asset[self.t],
            "market": self.states_market[self.t],
            "weight": self.w.copy(),
        }

    def step(self, action):
        w_new = self._project_simplex(action)
        r_vec = self.r_next[self.t]

        turnover = float(np.abs(w_new[1:] - self.w[1:]).sum())
        mu = self.cost * turnover
        r_p = float(np.dot(w_new, r_vec) - mu)
        reward = float(np.log1p(r_p))

        # post-trade weight drift after one day of returns
        denom = 1.0 + r_p + 1e-8
        w_post = w_new * (1.0 + r_vec) / denom
        w_post = w_post / (w_post.sum() + 1e-8)
        self.w = w_post.astype(np.float32)

        self.t += 1
        done = self.t >= self.T
        info = {"portfolio_return": r_p, "turnover": turnover}
        obs = self._obs() if not done else None
        return obs, reward, done, info

    # ------------------------------------------------------------------
    @staticmethod
    def _project_simplex(a):
        a = np.asarray(a, dtype=np.float32)
        a = np.clip(a, 0.0, None)
        s = a.sum()
        if s <= 1e-8:
            out = np.zeros_like(a)
            out[0] = 1.0
            return out
        return a / s
