"""
Rolling Window + PPO 학습.

사용 예:
    python3 train.py --mode smoke       # 파이프라인 검증 (축소 timesteps)
    python3 train.py --mode full        # 본 학습
    python3 train.py --mode smoke --rounds 1 --n-seeds 2   # 초간략 테스트
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from HMM import train_market_hmm
from config import load_config, parse_net_arch
from portfolio_env_hmm import PortfolioEnv
from rolling_window import RoundSpec, generate_rounds, slice_with_context


# ─────────────────────────────────────────────────────────────────
# Callback
# ─────────────────────────────────────────────────────────────────

class ValidationCallback(BaseCallback):
    """eval_freq 스텝마다 val_env에서 deterministic rollout 후 best DSR 기준 저장."""

    def __init__(self, val_env: PortfolioEnv, eval_freq: int, save_path: str):
        super().__init__(verbose=0)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_val_dsr = -np.inf
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            dsr_sum, metrics = evaluate_deterministic(self.model, self.val_env)
            record = {
                "step": self.n_calls,
                "val_dsr_sum": dsr_sum,
                "val_sharpe": metrics.get("sharpe_ratio", float("nan")),
                "val_cum_return": metrics.get("cumulative_return", float("nan")),
                "val_mdd": metrics.get("mdd", float("nan")),
                "val_turnover": metrics.get("turnover", float("nan")),
            }
            self.history.append(record)
            if dsr_sum > self.best_val_dsr:
                self.best_val_dsr = dsr_sum
                self.model.save(self.save_path)
        return True


# ─────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────

def make_env(data_df, cfg, hmm_model, hmm_scaler) -> PortfolioEnv:
    return PortfolioEnv(
        data=data_df,
        asset_cols=cfg["data"]["asset_cols"],
        vol_cols=cfg["data"]["vol_cols"],
        hmm_model=hmm_model,
        hmm_scaler=hmm_scaler,
        window=cfg["data"]["window"],
        eta=cfg["env"]["eta"],
        cost=cfg["env"]["cost"],
        max_weight=cfg["env"].get("max_weight"),
    )


def make_run_tag(cfg: dict) -> str | None:
    """
    실험 변형 태그 생성. 기본 설정(HMM-on, cap 없음)은 None을 반환하여
    기존 출력 경로(results/rolling_window_results.csv 등)를 그대로 사용.
    """
    use_hmm = cfg["train"].get("use_hmm", True)
    mw = cfg["env"].get("max_weight")
    cap_active = (mw is not None) and (float(mw) < 1.0)
    if use_hmm and not cap_active:
        return None
    parts = ["hmm" if use_hmm else "nohmm"]
    parts.append(f"cap{int(round(float(mw) * 100))}" if cap_active else "nocap")
    return "_".join(parts)


def get_run_paths(cfg: dict) -> dict:
    """태그에 따라 모델/결과/일별 디렉토리 경로 결정."""
    tag = make_run_tag(cfg)
    base_models = Path(cfg["paths"]["models_dir"])
    base_results = Path(cfg["paths"]["results_dir"])
    if tag is None:
        return {
            "tag": None,
            "models_root": base_models,
            "results_csv": base_results / "rolling_window_results.csv",
            "daily_dir":   base_results / "daily",
        }
    return {
        "tag": tag,
        "models_root": base_models / tag,
        "results_csv": base_results / f"rolling_window_results_{tag}.csv",
        "daily_dir":   base_results / "daily" / tag,
    }


def wrap_train_env(env, cfg: dict) -> VecEnv:
    """
    학습용 env만 VecNormalize로 래핑 (reward 동적 정규화).

    - norm_obs=False: PortfolioEnv의 obs는 이미 설계된 스케일
    - norm_reward=True: DSR 분포에 따라 running μ,σ 로 정규화 → PPO 안정화
    - val/test는 raw env 그대로 사용 → 실제 DSR sum 기준으로 best 모델 선정
    """
    vn_cfg = cfg["env"].get("vecnormalize", {})
    if not vn_cfg.get("enabled", False):
        return env
    venv = DummyVecEnv([lambda e=env: e])
    return VecNormalize(
        venv,
        norm_obs=False,
        norm_reward=True,
        clip_reward=float(vn_cfg.get("clip_reward", 10.0)),
        gamma=float(cfg["ppo"]["gamma"]),
    )


def _build_policy_kwargs(cfg: dict) -> dict:
    """config.train.feature_extractor 값에 따라 policy_kwargs 구성."""
    kind = cfg["train"].get("feature_extractor", "mlp").lower()
    if kind == "mlp":
        return dict(net_arch=parse_net_arch(cfg["ppo"]["net_arch"]))
    if kind == "transformer":
        from feature_extractor import build_policy_kwargs_transformer
        return build_policy_kwargs_transformer(cfg)
    raise ValueError(f"알 수 없는 feature_extractor: {kind}")


def build_ppo(env, seed: int, cfg: dict, prev_params=None) -> PPO:
    p = cfg["ppo"]
    # raw gym.Env가 들어오면 자동으로 VecNormalize 래핑
    if not isinstance(env, VecEnv):
        env = wrap_train_env(env, cfg)
    model = PPO(
        policy=p["policy"],
        env=env,
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        gae_lambda=p["gae_lambda"],
        clip_range=p["clip_range"],
        ent_coef=p["ent_coef"],
        n_steps=p["n_steps"],
        batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],
        policy_kwargs=_build_policy_kwargs(cfg),
        verbose=p["verbose"],
        seed=seed,
    )
    if prev_params is not None:
        # 이전 Round best의 정책/가치함수 파라미터만 인계 (optimizer state는 새로 시작)
        model.set_parameters(prev_params, exact_match=False)
    return model


def evaluate_deterministic(model, env: PortfolioEnv) -> tuple[float, dict]:
    """env 1 에피소드 deterministic rollout → (DSR 합, metrics)."""
    obs, _ = env.reset()
    done = False
    dsr_sum = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        dsr_sum += float(reward)
        done = terminated or truncated
    return dsr_sum, env.get_metrics()


# ─────────────────────────────────────────────────────────────────
# Round 학습
# ─────────────────────────────────────────────────────────────────

def train_round(
    spec: RoundSpec,
    full_df: pd.DataFrame,
    cfg: dict,
    args,
    prev_best_params=None,
) -> dict:
    window = cfg["data"]["window"]
    n_regimes = cfg["data"]["n_regimes"]

    train_df = full_df.loc[spec.train_start : spec.train_end]
    val_df = slice_with_context(full_df, spec.val_start, spec.val_end, window)
    test_df = slice_with_context(full_df, spec.test_start, spec.test_end, window)

    print(f"  데이터: train={len(train_df)}일, val={len(val_df)}일, "
          f"test={len(test_df)}일")

    # HMM 학습 (use_hmm=False면 skip → ablation)
    if cfg["train"].get("use_hmm", True):
        print("  HMM 학습 중...")
        hmm_model, hmm_scaler = train_market_hmm(train_df, n_regimes=n_regimes)
    else:
        print("  HMM 사용 안 함 (ablation: --no-hmm)")
        hmm_model, hmm_scaler = None, None

    paths = get_run_paths(cfg)
    round_dir = paths["models_root"] / f"round_{spec.round_index + 1:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    seed_results = []
    for seed in range(args.n_seeds):
        print(f"\n  ── Seed {seed} (timesteps={args.total_timesteps:,}) ──")
        set_random_seed(seed)

        train_env = make_env(train_df, cfg, hmm_model, hmm_scaler)
        val_env = make_env(val_df, cfg, hmm_model, hmm_scaler)

        seed_path = round_dir / f"seed_{seed}.zip"
        model = build_ppo(train_env, seed, cfg, prev_params=prev_best_params)
        callback = ValidationCallback(
            val_env=val_env,
            eval_freq=cfg["train"]["eval_freq"],
            save_path=str(seed_path),
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=False,
        )

        # eval_freq보다 짧게 학습된 경우를 대비해 fallback 저장
        if not seed_path.exists():
            model.save(str(seed_path))

        seed_results.append({
            "seed": seed,
            "best_val_dsr": float(callback.best_val_dsr),
            "history": callback.history,
        })
        print(f"    best val DSR: {callback.best_val_dsr:.4f}")

    # best seed 선정 (val DSR 최고)
    best = max(seed_results, key=lambda x: x["best_val_dsr"])
    best_seed = best["seed"]
    (round_dir / "best_seed.txt").write_text(str(best_seed))

    # 시드별 val history 저장 (evaluate.py에서 쓰일 수 있음)
    history_path = round_dir / "val_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(seed_results, f)

    print(f"\n  → Best seed: {best_seed} (val DSR={best['best_val_dsr']:.4f})")

    # Test rollout
    best_model = PPO.load(str(round_dir / f"seed_{best_seed}.zip"))
    test_env = make_env(test_df, cfg, hmm_model, hmm_scaler)
    test_dsr, test_metrics = evaluate_deterministic(best_model, test_env)

    print(f"  Test → Sharpe={test_metrics['sharpe_ratio']:.4f}, "
          f"CumRet={test_metrics['cumulative_return']*100:.2f}%, "
          f"MDD={test_metrics['mdd']*100:.2f}%, "
          f"Turnover={test_metrics['turnover']:.4f}")

    avg_weights = np.mean(test_env.weight_history[1:], axis=0)
    print("  평균 투자 비중:")
    for col, w in zip(cfg["data"]["asset_cols"], avg_weights):
        bar = "█" * int(w * 40)
        print(f"    {col:<14} {w*100:5.1f}%  {bar}")

    # 일별 시계열 저장 (evaluate.py용)
    daily_dir = paths["daily_dir"]
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily = {
        "round_index": spec.round_index,
        # test_env.dates는 test_df 전체(컨텍스트 포함)의 인덱스
        # 실제 거래 시작 시점은 window 인덱스부터
        "dates": [str(d.date()) for d in test_env.dates[window : test_env.T]],
        "portfolio_values": list(map(float, test_env.value_history)),
        "weights": [w.tolist() for w in test_env.weight_history],
        "returns": list(map(float, test_env.return_history)),
        "spec": {
            "train_start": str(spec.train_start.date()),
            "train_end": str(spec.train_end.date()),
            "val_start": str(spec.val_start.date()),
            "val_end": str(spec.val_end.date()),
            "test_start": str(spec.test_start.date()),
            "test_end": str(spec.test_end.date()),
        },
    }
    with open(daily_dir / f"round_{spec.round_index + 1:02d}.pkl", "wb") as f:
        pickle.dump(daily, f)

    return {
        "round": spec.round_index + 1,
        "best_seed": best_seed,
        "best_val_dsr": best["best_val_dsr"],
        "test_dsr_sum": float(test_dsr),
        "test_sharpe": float(test_metrics["sharpe_ratio"]),
        "test_cum_return": float(test_metrics["cumulative_return"]),
        "test_mdd": float(test_metrics["mdd"]),
        "test_turnover": float(test_metrics["turnover"]),
        "test_n_steps": int(test_metrics["n_steps"]),
        "spec": spec,
        "best_params": best_model.get_parameters() if args.transfer_weights else None,
    }


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--config", default=None)
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="config의 train.n_seeds를 덮어씀")
    parser.add_argument("--rounds", type=int, default=None,
                        help="첫 N개 Round만 실행 (디버그)")
    parser.add_argument("--no-transfer", action="store_true",
                        help="Round 간 weight 인계 끄기")
    parser.add_argument("--feature-extractor", choices=["mlp", "transformer"],
                        default=None, help="config 값 덮어쓰기")
    parser.add_argument("--no-hmm", action="store_true",
                        help="HMM 비활성 (ablation). 결과는 별도 디렉토리에 저장")
    parser.add_argument("--max-weight", type=float, default=None,
                        help="단일 자산 최대 비중 cap (예: 0.5). 기본=cap 없음")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.feature_extractor:
        cfg["train"]["feature_extractor"] = args.feature_extractor
    if args.no_hmm:
        cfg["train"]["use_hmm"] = False
    if args.max_weight is not None:
        cfg["env"]["max_weight"] = args.max_weight

    args.total_timesteps = (
        cfg["train"]["total_timesteps_smoke"]
        if args.mode == "smoke"
        else cfg["train"]["total_timesteps"]
    )
    args.n_seeds = args.n_seeds or cfg["train"]["n_seeds"]
    args.transfer_weights = (
        cfg["train"]["transfer_weights"] and not args.no_transfer
    )

    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()

    rounds = generate_rounds(df, cfg)
    if args.rounds:
        rounds = rounds[: args.rounds]

    paths = get_run_paths(cfg)
    use_hmm = cfg["train"].get("use_hmm", True)
    mw = cfg["env"].get("max_weight")
    cap_str = f"{mw}" if (mw is not None and float(mw) < 1.0) else "없음"

    print("=" * 70)
    print(f" Rolling Window PPO 학습  (mode={args.mode})")
    print("=" * 70)
    print(f"  데이터: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"  라운드: {len(rounds)}개")
    print(f"  시드:   {args.n_seeds}")
    print(f"  steps:  {args.total_timesteps:,}")
    print(f"  weight 인계: {args.transfer_weights}")
    print(f"  HMM 사용:    {use_hmm}")
    print(f"  max_weight:  {cap_str}")
    print(f"  변형 태그:   {paths['tag'] or '(기본 baseline)'}")
    print(f"  결과 저장:   {paths['results_csv']}")

    results = []
    prev_best_params = None
    for spec in rounds:
        print("\n" + "=" * 70)
        print(" " + spec.describe())
        print("=" * 70)
        r = train_round(spec, df, cfg, args, prev_best_params=prev_best_params)
        results.append(r)
        prev_best_params = r["best_params"]

    # 결과 CSV
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k not in ("spec", "best_params")}
        s = r["spec"]
        row.update({
            "train_start": s.train_start.date(),
            "train_end":   s.train_end.date(),
            "val_start":   s.val_start.date(),
            "val_end":     s.val_end.date(),
            "test_start":  s.test_start.date(),
            "test_end":    s.test_end.date(),
        })
        rows.append(row)
    results_df = pd.DataFrame(rows)

    results_path = paths["results_csv"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print(f" 학습 완료 → {results_path}")
    print("=" * 70)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
