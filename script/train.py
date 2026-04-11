"""PPO baseline training entrypoint.

Usage:
    python script/train.py --config config.json --save runs/ppo_baseline.pt
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from data.preprocess import load_dataset, normalizer, split
from env.kospi_env import KospiPortfolioEnv
from agents.ppo_agent import PPOAgent


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_envs(cfg):
    df, cols = load_dataset(cfg["data"]["csv_path"])
    splits = split(df, cfg["data"]["train_end"], cfg["data"]["valid_end"])
    mean, std = normalizer(splits["train"], cols)
    return {
        name: KospiPortfolioEnv(
            part, cols, mean, std,
            window=cfg["env"]["window"],
            cost=cfg["env"]["cost"],
        )
        for name, part in splits.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--save", default="runs/ppo_baseline.pt")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    set_seed(cfg["train"]["seed"])
    envs = make_envs(cfg)

    print(f"[data] train={envs['train'].T}  valid={envs['valid'].T}  test={envs['test'].T}")
    print(f"[env]  M={envs['train'].M}  L={envs['train'].L}  N={envs['train'].N}  K={envs['train'].K}")

    agent = PPOAgent(
        envs["train"],
        hidden=cfg["agent"]["hidden"],
        lr=cfg["agent"]["lr"],
        gamma=cfg["agent"]["gamma"],
        lam=cfg["agent"]["lam"],
        clip=cfg["agent"]["clip"],
        epochs=cfg["agent"]["epochs"],
        batch_size=cfg["agent"]["batch_size"],
        device=cfg["train"]["device"],
    )
    agent.train(cfg["train"]["n_iters"])

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.net.state_dict(), args.save)
    print(f"\nSaved model to {args.save}")

    print("\n[Validation]")
    print(agent.evaluate(envs["valid"]))


if __name__ == "__main__":
    main()
