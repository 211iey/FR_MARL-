"""
Multi-Agent용 포트폴리오 환경.

DiversityPortfolioEnv:
  PortfolioEnv를 상속하여 다양성 보너스(diversity bonus)만 추가.
  diversity_registry(공유 dict)를 통해 에이전트 간 현재 비중을 실시간 참조.

다양성 지표:
  Correlation 모드: bonus = -mean(corr(my_w, other_w))   → 상관↓ = bonus↑
  TV 모드:          bonus = mean(0.5 * |my_w - other_w|) → 거리↑ = bonus↑

최종 보상:
  reward = DSR + lambda * diversity_bonus
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from portfolio_env_hmm import PortfolioEnv


class DiversityPortfolioEnv(PortfolioEnv):
    """
    다양성 페널티를 지원하는 단일 에이전트 환경.
    N개 인스턴스가 동일한 diversity_registry를 공유하여
    서로의 포트폴리오 비중을 참조한다.
    """

    def __init__(
        self,
        *args,
        diversity_registry: dict,
        agent_id: int,
        diversity_lambda: float = 0.0,
        diversity_mode: str = "correlation",  # "correlation" | "tv"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.diversity_registry = diversity_registry
        self.agent_id = agent_id
        self.diversity_lambda = diversity_lambda
        self.diversity_mode = diversity_mode

    def step(self, action: np.ndarray):
        obs, reward, terminated, truncated, info = super().step(action)

        # 현재 비중을 레지스트리에 등록
        self.diversity_registry[self.agent_id] = info["weights"].copy()

        # 다른 에이전트가 한 명 이상 등록됐을 때만 보너스 계산
        if self.diversity_lambda > 0.0 and len(self.diversity_registry) > 1:
            bonus = self._diversity_bonus()
            reward = float(reward) + self.diversity_lambda * bonus
            info["diversity_bonus"] = bonus
        else:
            info["diversity_bonus"] = 0.0

        return obs, reward, terminated, truncated, info

    def _diversity_bonus(self) -> float:
        my_w = self.diversity_registry[self.agent_id]
        others = [
            w for aid, w in self.diversity_registry.items()
            if aid != self.agent_id and len(w) > 0
        ]
        if not others:
            return 0.0

        if self.diversity_mode == "correlation":
            corrs = []
            for w in others:
                c = np.corrcoef(my_w, w)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
            return float(-np.mean(corrs)) if corrs else 0.0
        else:  # tv
            tvs = [0.5 * float(np.abs(my_w - w).sum()) for w in others]
            return float(np.mean(tvs))
