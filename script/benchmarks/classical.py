"""Classical portfolio baselines evaluated through KospiPortfolioEnv.

Policies expose a single ``act(obs)`` method returning a weight vector of
shape (M,). ``rollout(env, policy)`` plays one full episode and returns a
metric dict that can be compared directly with the RL agent's evaluate().
"""
import numpy as np
from scipy.optimize import minimize


class EqualWeightPolicy:
    """Static 1/(M-1) across all non-cash assets."""

    def __init__(self, M, include_cash=False):
        self.M = M
        if include_cash:
            self.w = np.ones(M, dtype=np.float32) / M
        else:
            self.w = np.zeros(M, dtype=np.float32)
            self.w[1:] = 1.0 / (M - 1)

    def act(self, obs):
        return self.w


class MinVariancePolicy:
    """Long-only minimum-variance using the asset return window from obs."""

    def __init__(self, M):
        self.M = M

    def act(self, obs):
        # asset state: (M, L, N) -> channel 0 is z-scored return window
        x = obs["asset"][1:, :, 0]  # (M-1, L)
        cov = np.cov(x) + 1e-6 * np.eye(self.M - 1)

        w0 = np.ones(self.M - 1) / (self.M - 1)
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
        bnds = tuple((0.0, 1.0) for _ in range(self.M - 1))
        res = minimize(
            lambda w: w @ cov @ w, w0,
            method="SLSQP", bounds=bnds, constraints=cons,
        )
        w = np.zeros(self.M, dtype=np.float32)
        w[1:] = res.x if res.success else w0
        return w


def rollout(env, policy):
    obs = env.reset()
    rewards, returns, turnovers = [], [], []
    done = False
    while not done:
        action = policy.act(obs)
        obs, r, done, info = env.step(action)
        rewards.append(r)
        returns.append(info["portfolio_return"])
        turnovers.append(info["turnover"])

    rewards = np.array(rewards)
    returns = np.array(returns)
    return {
        "reward_sum":   float(rewards.sum()),
        "cum_return":   float(np.exp(rewards.sum()) - 1.0),
        "mean_return":  float(returns.mean()),
        "sharpe":       float(returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)),
        "avg_turnover": float(np.mean(turnovers)),
    }
