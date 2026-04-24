"""
Multi-Agent IPPO 학습 — λ sweep 실험.

각 에이전트는 독립 PPO(SB3)로 학습하되,
DiversityPortfolioEnv를 통해 매 스텝 다른 에이전트의 비중을 참조하고
다양성 보너스(lambda * diversity)를 보상에 반영한다.

보상 스케일 안정화:
  DiversityPortfolioEnv가 내보내는 reward = DSR + λ * diversity_bonus.
  λ 값에 따라 raw reward 범위가 변하므로 train env만 VecNormalize로 감싸
  running μ,σ 로 동적 정규화한다. (train.py::build_ppo 가 자동 래핑)
  val/test 평가 시에는 raw env 사용 → 실제 DSR sum 및 성과지표로 best 선정.

실험 구조:
  단일 기간 고정 (config.marl 기준)
  λ ∈ [0.0, 0.1, ..., 0.9] × diversity_mode ∈ [correlation, tv]

사용 예:
    python3 marl_train.py --mode smoke
    python3 marl_train.py --mode full
    python3 marl_train.py --lambda-val 0.3 --diversity-mode tv
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from HMM import train_market_hmm
from config import load_config
from marl_env import DiversityPortfolioEnv
from portfolio_env_hmm import PortfolioEnv
from rolling_window import slice_with_context
from train import build_ppo, evaluate_deterministic


# ─────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────

def make_diversity_env(
    data_df: pd.DataFrame,
    cfg: dict,
    hmm_model,
    hmm_scaler,
    diversity_registry: dict,
    agent_id: int,
    diversity_lambda: float,
    diversity_mode: str,
) -> DiversityPortfolioEnv:
    return DiversityPortfolioEnv(
        data=data_df,
        asset_cols=cfg["data"]["asset_cols"],
        vol_cols=cfg["data"]["vol_cols"],
        hmm_model=hmm_model,
        hmm_scaler=hmm_scaler,
        window=cfg["data"]["window"],
        eta=cfg["env"]["eta"],
        cost=cfg["env"]["cost"],
        diversity_registry=diversity_registry,
        agent_id=agent_id,
        diversity_lambda=diversity_lambda,
        diversity_mode=diversity_mode,
    )


def make_plain_env(data_df: pd.DataFrame, cfg: dict, hmm_model, hmm_scaler) -> PortfolioEnv:
    return PortfolioEnv(
        data=data_df,
        asset_cols=cfg["data"]["asset_cols"],
        vol_cols=cfg["data"]["vol_cols"],
        hmm_model=hmm_model,
        hmm_scaler=hmm_scaler,
        window=cfg["data"]["window"],
        eta=cfg["env"]["eta"],
        cost=cfg["env"]["cost"],
    )


def compute_agent_correlation(weights_list: list[list]) -> float:
    """에이전트 간 평균 포트폴리오 상관계수 (test 기간 전체 time-step 평균 비중 기준)."""
    mean_ws = [np.mean(np.array(w), axis=0) for w in weights_list if len(w) > 0]
    if len(mean_ws) < 2:
        return float("nan")
    corrs = []
    for i in range(len(mean_ws)):
        for j in range(i + 1, len(mean_ws)):
            c = np.corrcoef(mean_ws[i], mean_ws[j])[0, 1]
            if np.isfinite(c):
                corrs.append(c)
    return float(np.mean(corrs)) if corrs else float("nan")


# ─────────────────────────────────────────────────────────────────
# 단일 실험 (하나의 lambda + mode 조합)
# ─────────────────────────────────────────────────────────────────

def run_experiment(
    cfg: dict,
    df: pd.DataFrame,
    lambda_val: float,
    diversity_mode: str,
    total_timesteps: int,
    seed: int = 0,
) -> dict:
    marl = cfg["marl"]
    window = cfg["data"]["window"]
    n_agents = marl["n_agents"]
    eval_freq = cfg["train"]["eval_freq"]

    # 데이터 슬라이스
    train_start = pd.Timestamp(marl["train_start"])
    train_end   = pd.Timestamp(marl["train_end"])
    val_start   = pd.Timestamp(f"{marl['val_year']}-01-01")
    val_end     = pd.Timestamp(f"{marl['val_year']}-12-31")
    test_start  = pd.Timestamp(f"{marl['test_year']}-01-01")
    test_end    = pd.Timestamp(f"{marl['test_year']}-12-31")

    train_df = df.loc[train_start:train_end]
    val_df   = slice_with_context(df, val_start, val_end, window)
    test_df  = slice_with_context(df, test_start, test_end, window)

    print(f"  데이터: train={len(train_df)}일, val={len(val_df)}일, test={len(test_df)}일")

    # HMM (train only)
    hmm_model, hmm_scaler = train_market_hmm(train_df, n_regimes=cfg["data"]["n_regimes"])

    # 저장 경로
    exp_dir = (
        Path(cfg["paths"]["models_dir"])
        / "marl"
        / diversity_mode
        / f"lambda_{lambda_val:.1f}"
    )
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 에이전트별 환경 + PPO 생성
    diversity_registry: dict = {}
    train_envs, val_envs, models = [], [], []

    for i in range(n_agents):
        set_random_seed(seed + i)
        t_env = make_diversity_env(
            train_df, cfg, hmm_model, hmm_scaler,
            diversity_registry, i, lambda_val, diversity_mode,
        )
        v_env = make_plain_env(val_df, cfg, hmm_model, hmm_scaler)
        model = build_ppo(t_env, seed=seed + i, cfg=cfg)
        train_envs.append(t_env)
        val_envs.append(v_env)
        models.append(model)

    # 동기식 학습 루프: eval_freq 단위로 모든 에이전트를 번갈아 학습
    n_chunks = max(1, total_timesteps // eval_freq)
    best_val_dsr = [-np.inf] * n_agents

    print(f"  학습 시작: {n_agents}명 × {total_timesteps:,} steps × λ={lambda_val}")
    for chunk in range(n_chunks):
        for i, (model, t_env, v_env) in enumerate(zip(models, train_envs, val_envs)):
            model.learn(
                total_timesteps=eval_freq,
                reset_num_timesteps=(chunk == 0),
                progress_bar=False,
            )
            dsr, _ = evaluate_deterministic(model, v_env)
            if dsr > best_val_dsr[i]:
                best_val_dsr[i] = dsr
                model.save(str(exp_dir / f"agent_{i}.zip"))

    # fallback 저장 (eval_freq > total_timesteps인 경우)
    for i, model in enumerate(models):
        p = exp_dir / f"agent_{i}.zip"
        if not p.exists():
            model.save(str(p))

    # Test rollout
    test_results = []
    all_weights = []
    for i in range(n_agents):
        m = PPO.load(str(exp_dir / f"agent_{i}.zip"))
        t_env = make_plain_env(test_df, cfg, hmm_model, hmm_scaler)
        dsr, metrics = evaluate_deterministic(m, t_env)
        test_results.append({"agent": i, **metrics, "test_dsr_sum": dsr})
        all_weights.append(t_env.weight_history)

    inter_corr = compute_agent_correlation(all_weights)
    mean_sharpe = float(np.mean([r["sharpe_ratio"] for r in test_results]))
    mean_mdd    = float(np.mean([r["mdd"]         for r in test_results]))
    mean_turn   = float(np.mean([r["turnover"]     for r in test_results]))

    print(
        f"  → λ={lambda_val} [{diversity_mode}] "
        f"Sharpe={mean_sharpe:.3f}, MDD={mean_mdd*100:.1f}%, "
        f"Turn={mean_turn:.4f}, inter_corr={inter_corr:.3f}"
    )

    # 일별 시계열 저장
    daily_dir = Path(cfg["paths"]["results_dir"]) / "marl" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily = {
        "lambda_val": lambda_val,
        "diversity_mode": diversity_mode,
        "agent_results": test_results,
        "agent_weights": [w for w in all_weights],
        "inter_agent_correlation": inter_corr,
    }
    fname = f"{diversity_mode}_lambda_{lambda_val:.1f}.pkl"
    with open(daily_dir / fname, "wb") as f:
        pickle.dump(daily, f)

    return {
        "lambda_val": lambda_val,
        "diversity_mode": diversity_mode,
        "mean_sharpe": mean_sharpe,
        "mean_mdd": mean_mdd,
        "mean_turnover": mean_turn,
        "inter_agent_correlation": inter_corr,
        "best_val_dsr": best_val_dsr,
        "agent_results": test_results,
    }


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--config", default=None)
    parser.add_argument("--lambda-val", type=float, default=None,
                        help="단일 λ 값만 실험 (sweep 생략)")
    parser.add_argument("--diversity-mode", choices=["correlation", "tv", "both"],
                        default="both", help="다양성 지표 선택")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    marl = cfg["marl"]

    total_timesteps = (
        cfg["train"]["total_timesteps_smoke"]
        if args.mode == "smoke"
        else cfg["train"]["total_timesteps"]
    )

    lambdas = (
        [args.lambda_val]
        if args.lambda_val is not None
        else (marl["smoke_lambdas"] if args.mode == "smoke" else marl["lambdas"])
    )

    modes = (
        marl["diversity_modes"]
        if args.diversity_mode == "both"
        else [args.diversity_mode]
    )

    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()

    print("=" * 70)
    print(f" MARL IPPO λ sweep  (mode={args.mode})")
    print("=" * 70)
    print(f"  에이전트 수:  {marl['n_agents']}")
    print(f"  λ 목록:       {lambdas}")
    print(f"  다양성 지표:  {modes}")
    print(f"  timesteps:    {total_timesteps:,}")
    print(f"  기간: train {marl['train_start']}~{marl['train_end']} "
          f"| val {marl['val_year']} | test {marl['test_year']}")

    all_results = []
    for mode in modes:
        for lam in lambdas:
            print(f"\n{'='*70}")
            print(f" λ={lam:.1f}  mode={mode}")
            print("=" * 70)
            result = run_experiment(cfg, df, lam, mode, total_timesteps, seed=args.seed)
            all_results.append(result)

    # CSV 저장
    rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k not in ("agent_results", "best_val_dsr")}
        rows.append(row)
    sweep_df = pd.DataFrame(rows)

    out_dir = Path(cfg["paths"]["results_dir"]) / "marl"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / "sweep_results.csv"
    sweep_df.to_csv(sweep_path, index=False)

    print("\n" + "=" * 70)
    print(f" 완료 → {sweep_path}")
    print("=" * 70)
    print(sweep_df.to_string(index=False))


if __name__ == "__main__":
    main()
