"""Evaluate PPO + classical baselines on the same split.

Usage:
    python script/evaluate.py --split test --ckpt runs/ppo_baseline.pt
"""
import argparse
import json
from pathlib import Path

import torch

from agents.ppo_agent import PPOAgent
from benchmarks.classical import EqualWeightPolicy, MinVariancePolicy, rollout
from train import make_envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--ckpt", default="runs/ppo_baseline.pt")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    envs = make_envs(cfg)
    env = envs[args.split]

    print(f"=== Split: {args.split} | T={env.T} steps | M={env.M} ===\n")

    print("[Equal Weight]")
    print(rollout(env, EqualWeightPolicy(env.M, include_cash=False)))

    print("\n[Min Variance]")
    print(rollout(env, MinVariancePolicy(env.M)))

    ckpt_path = Path(args.ckpt)
    if ckpt_path.exists():
        print("\n[PPO]")
        agent = PPOAgent(
            env,
            hidden=cfg["agent"]["hidden"],
            lr=cfg["agent"]["lr"],
            gamma=cfg["agent"]["gamma"],
            lam=cfg["agent"]["lam"],
            clip=cfg["agent"]["clip"],
            epochs=cfg["agent"]["epochs"],
            batch_size=cfg["agent"]["batch_size"],
            device=cfg["train"]["device"],
        )
        agent.net.load_state_dict(torch.load(ckpt_path, map_location=cfg["train"]["device"]))
        print(agent.evaluate(env))
    else:
        print(f"\n[PPO] skipped — checkpoint not found at {ckpt_path}")


if __name__ == "__main__":
    main()
