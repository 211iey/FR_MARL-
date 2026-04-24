"""Minimal PyTorch PPO agent with a Dirichlet simplex policy.

Architecture is intentionally small so that hyper-params / layers can be
iterated on later without rewriting the training loop. Observation handling
mirrors KospiPortfolioEnv's dict obs and keeps things framework-agnostic.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, M, L, N, K, hidden=64):
        super().__init__()
        self.M, self.L, self.N, self.K = M, L, N, K

        # 1xL conv collapses the time window per asset
        self.asset_conv = nn.Sequential(
            nn.Conv2d(N, hidden, kernel_size=(1, L)),
            nn.ReLU(),
        )
        self.market_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(L * K, hidden),
            nn.ReLU(),
        )

        feat_dim = hidden * M + hidden + M
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, M)
        self.critic_head = nn.Linear(hidden, 1)

    def _features(self, obs):
        a = obs["asset"]     # (B, M, L, N)
        m = obs["market"]    # (B, L, K)
        w = obs["weight"]    # (B, M)

        a = a.permute(0, 3, 1, 2)          # (B, N, M, L)
        x = self.asset_conv(a)             # (B, hidden, M, 1)
        x = x.squeeze(-1).transpose(1, 2)  # (B, M, hidden)
        x = x.reshape(x.size(0), -1)       # (B, M*hidden)
        mfeat = self.market_mlp(m)         # (B, hidden)
        return torch.cat([x, mfeat, w], dim=-1)

    def forward(self, obs):
        h = self.shared(self._features(obs))
        alpha = F.softplus(self.actor_head(h)) + 1e-3   # Dirichlet concentration
        value = self.critic_head(h).squeeze(-1)
        return alpha, value

    def act(self, obs):
        alpha, value = self.forward(obs)
        dist = Dirichlet(alpha)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value


# ----------------------------------------------------------------------
# Obs helpers
# ----------------------------------------------------------------------
def to_tensor(obs, device):
    return {
        "asset":  torch.as_tensor(obs["asset"],  dtype=torch.float32, device=device).unsqueeze(0),
        "market": torch.as_tensor(obs["market"], dtype=torch.float32, device=device).unsqueeze(0),
        "weight": torch.as_tensor(obs["weight"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def stack_obs(obs_list, device):
    return {
        "asset":  torch.as_tensor(np.stack([o["asset"]  for o in obs_list]), dtype=torch.float32, device=device),
        "market": torch.as_tensor(np.stack([o["market"] for o in obs_list]), dtype=torch.float32, device=device),
        "weight": torch.as_tensor(np.stack([o["weight"] for o in obs_list]), dtype=torch.float32, device=device),
    }


# ----------------------------------------------------------------------
# PPO
# ----------------------------------------------------------------------
class PPOAgent:
    def __init__(self, env, hidden=64, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=4, batch_size=64, device="cpu"):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size

        self.net = ActorCritic(env.M, env.L, env.N, env.K, hidden=hidden).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ---------- rollout ----------
    def collect(self):
        obs = self.env.reset()
        obs_buf, act_buf, logp_buf = [], [], []
        rew_buf, val_buf, done_buf = [], [], []
        done = False
        while not done:
            t_obs = to_tensor(obs, self.device)
            with torch.no_grad():
                action, logp, value = self.net.act(t_obs)
            a = action.squeeze(0).cpu().numpy()
            next_obs, r, done, _ = self.env.step(a)

            obs_buf.append(obs)
            act_buf.append(a)
            logp_buf.append(logp.item())
            rew_buf.append(r)
            val_buf.append(value.item())
            done_buf.append(done)

            obs = next_obs if not done else obs
        return obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf

    # ---------- advantage ----------
    def compute_gae(self, rewards, values, dones):
        advs = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        next_v = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_v * mask - values[t]
            gae = delta + self.gamma * self.lam * mask * gae
            advs[t] = gae
            next_v = values[t]
        returns = advs + np.array(values, dtype=np.float32)
        return advs, returns

    # ---------- update ----------
    def update(self, obs_buf, act_buf, logp_buf, advs, returns):
        N = len(obs_buf)
        idx = np.arange(N)

        actions = torch.as_tensor(np.stack(act_buf), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(np.array(logp_buf), dtype=torch.float32, device=self.device)
        advs_t = (advs - advs.mean()) / (advs.std() + 1e-8)
        advs_t = torch.as_tensor(advs_t, dtype=torch.float32, device=self.device)
        rets_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, N, self.batch_size):
                mb = idx[s:s + self.batch_size]
                mb_obs = stack_obs([obs_buf[i] for i in mb], self.device)

                alpha, value = self.net(mb_obs)
                dist = Dirichlet(alpha)

                a_mb = actions[mb].clamp_min(1e-6)
                a_mb = a_mb / a_mb.sum(dim=-1, keepdim=True)
                logp = dist.log_prob(a_mb)

                ratio = torch.exp(logp - old_logp[mb])
                surr1 = ratio * advs_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advs_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(value, rets_t[mb])
                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

    # ---------- loops ----------
    def train(self, n_iters):
        history = []
        for it in range(n_iters):
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = self.collect()
            advs, returns = self.compute_gae(rew_buf, val_buf, done_buf)
            self.update(obs_buf, act_buf, logp_buf, advs, returns)

            ep_reward = float(np.sum(rew_buf))
            cum_ret = float(np.exp(ep_reward) - 1.0)
            history.append({"iter": it, "reward": ep_reward, "cum_return": cum_ret})
            print(f"[iter {it:3d}] reward={ep_reward:+.4f}  cum_return={cum_ret*100:+.2f}%")
        return history

    def evaluate(self, env):
        """Deterministic rollout using the Dirichlet mean as the action."""
        obs = env.reset()
        rewards, returns, turnovers = [], [], []
        done = False
        while not done:
            t_obs = to_tensor(obs, self.device)
            with torch.no_grad():
                alpha, _ = self.net(t_obs)
                a = (alpha / alpha.sum(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()
            obs, r, done, info = env.step(a)
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
