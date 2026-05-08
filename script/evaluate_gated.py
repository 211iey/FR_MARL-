"""
Mahalanobis-gated rollout 평가.

아이디어:
  - HMM-on (cap=0.3) 모델과 HMM-off (cap=0.3) 모델 둘 다 이미 학습되어 있음.
  - 매 test 일 t에서 [kospi_t, vkospi_t]의 학습창 대비 Mahalanobis distance 계산.
  - 최근 ROLLING_WINDOW일 평균 distance가 THRESHOLD 초과 → HMM-off 모델 사용.
  - 그렇지 않으면 HMM-on 모델 사용.
  - 두 env에 동일한 action을 동시에 적용해 동기 유지 (portfolio 상태 동일).

이는 추가 학습 없이 동적 gating의 사전적(ex-ante) 검증이 가능합니다.

산출물:
  results/rolling_window_results_gated.csv
  results/daily/gated/round_NN.pkl
  results/figures/gated/...

사용:
    python3 evaluate_gated.py
    python3 evaluate_gated.py --threshold 1.0 --window 5
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from HMM import train_market_hmm
from config import load_config
from portfolio_env_hmm import PortfolioEnv
from rolling_window import generate_rounds, slice_with_context

mpl.rcParams["font.family"] = "AppleGothic"
mpl.rcParams["axes.unicode_minus"] = False


def mahalanobis(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    diff = x - mu
    return float(np.sqrt(diff @ cov_inv @ diff))


def make_env(test_df, cfg, hmm_model, hmm_scaler) -> PortfolioEnv:
    return PortfolioEnv(
        data=test_df,
        asset_cols=cfg["data"]["asset_cols"],
        vol_cols=cfg["data"]["vol_cols"],
        hmm_model=hmm_model,
        hmm_scaler=hmm_scaler,
        window=cfg["data"]["window"],
        eta=cfg["env"]["eta"],
        cost=cfg["env"]["cost"],
        max_weight=0.3,
    )


def gated_rollout(
    hmm_policy, nohmm_policy, hmm_env, nohmm_env,
    macro_arr, mu, cov_inv,
    threshold: float, rolling_window: int,
    decision_freq: int = 1, warmup: int = 5,
):
    """
    Gated rollout.

    decision_freq=1   → 매일 의사결정 (daily mode)
    decision_freq=63  → 분기별 (~매 분기 거래일 약 63일)
    decision_freq>1   → 그 주기마다만 모델 갈아끼움, 사이엔 commit

    warmup: 첫 의사결정 전 최소로 누적해야 할 일 수 (HMM-on 기본값 유지).
    """
    obs_hmm, _ = hmm_env.reset()
    obs_nohmm, _ = nohmm_env.reset()

    daily_dists, decisions = [], []
    use_off = False                # 초기 commit: HMM-on
    last_decision_step = -decision_freq  # 첫 결정이 warmup 이후 즉시 가능하도록
    step_count = 0
    done = False

    while not done:
        t = hmm_env.t
        x_t = macro_arr[t]
        d = mahalanobis(x_t, mu, cov_inv)
        daily_dists.append(d)

        days_since = step_count - last_decision_step
        if step_count >= warmup and days_since >= decision_freq:
            recent = daily_dists[-rolling_window:]
            avg = float(np.mean(recent))
            use_off = avg > threshold
            last_decision_step = step_count

        decisions.append(use_off)

        if use_off:
            action, _ = nohmm_policy.predict(obs_nohmm, deterministic=True)
        else:
            action, _ = hmm_policy.predict(obs_hmm, deterministic=True)

        obs_hmm,   _, term1, trunc1, _ = hmm_env.step(action)
        obs_nohmm, _, term2, trunc2, _ = nohmm_env.step(action)
        done = term1 or trunc1 or term2 or trunc2
        step_count += 1

    return hmm_env.get_metrics(), hmm_env, daily_dists, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Mahalanobis 거리 threshold (기본 1.0)")
    parser.add_argument("--window", type=int, default=60,
                        help="Rolling 평균 윈도우 (기본 60일, env vol window 매칭)")
    parser.add_argument("--decision-freq", type=int, default=63,
                        help="의사결정 주기 (1=daily, 63=quarterly 등, 기본 63)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="첫 의사결정 전 최소 누적일 (기본 5)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()
    rounds = generate_rounds(df, cfg)

    base_models = Path(cfg["paths"]["models_dir"])
    hmm_model_root   = base_models / "hmm_cap30"
    nohmm_model_root = base_models / "nohmm_cap30"

    mode_label = "daily" if args.decision_freq == 1 else f"every {args.decision_freq}d"
    print("=" * 70)
    print(f" Mahalanobis-gated 평가  (threshold={args.threshold}, "
          f"rolling={args.window}d, decision={mode_label}, warmup={args.warmup}d)")
    print("=" * 70)

    rows = []
    daily_records = []
    daily_pkls = []

    for spec in rounds:
        round_idx = spec.round_index + 1
        train_df = df.loc[spec.train_start:spec.train_end]
        test_df  = slice_with_context(df, spec.test_start, spec.test_end, cfg["data"]["window"])

        # HMM 재학습 (라운드별)
        hmm_model, hmm_scaler = train_market_hmm(train_df, n_regimes=cfg["data"]["n_regimes"])

        # train 분포 통계 (Mahalanobis용)
        train_X = train_df[["kospi", "vkospi"]].dropna().values
        mu = train_X.mean(axis=0)
        cov = np.cov(train_X, rowvar=False) + 1e-8 * np.eye(2)
        cov_inv = np.linalg.inv(cov)

        # 모델 로드
        hmm_policy = PPO.load(str(hmm_model_root / f"round_{round_idx:02d}" / "seed_0.zip"))
        nohmm_policy = PPO.load(str(nohmm_model_root / f"round_{round_idx:02d}" / "seed_0.zip"))

        # 두 env 준비
        hmm_env   = make_env(test_df, cfg, hmm_model, hmm_scaler)
        nohmm_env = make_env(test_df, cfg, None, None)

        macro_arr = hmm_env.macro_arr  # 전체 test_df의 [kospi, vkospi]

        metrics, env_used, distances, decisions = gated_rollout(
            hmm_policy, nohmm_policy, hmm_env, nohmm_env,
            macro_arr, mu, cov_inv,
            args.threshold, args.window,
            decision_freq=args.decision_freq, warmup=args.warmup,
        )

        n_off = int(sum(decisions))
        n_on = int(len(decisions) - n_off)

        row = {
            "round": round_idx,
            **metrics,
            "n_days_hmm_on":  n_on,
            "n_days_hmm_off": n_off,
            "frac_off":       n_off / len(decisions) if decisions else 0.0,
            "test_start":     str(spec.test_start.date()),
            "test_end":       str(spec.test_end.date()),
        }
        rows.append(row)
        print(
            f"  Round {round_idx}: Sharpe={metrics['sharpe_ratio']:+.3f}, "
            f"CumRet={metrics['cumulative_return']*100:+.1f}%, "
            f"MDD={metrics['mdd']*100:+.1f}%  |  HMM-off {n_off}일/{len(decisions)}일"
        )

        # 일별 결과 보존
        test_dates = pd.to_datetime(test_df.index[cfg["data"]["window"] : env_used.T])
        daily_pkls.append({
            "round_index": spec.round_index,
            "dates": [str(d.date()) for d in test_dates],
            "portfolio_values": list(map(float, env_used.value_history)),
            "weights": [w.tolist() for w in env_used.weight_history],
            "returns": list(map(float, env_used.return_history)),
            "distances": distances,
            "decisions_use_off": decisions,
            "spec": {
                "train_start": str(spec.train_start.date()),
                "train_end":   str(spec.train_end.date()),
                "test_start":  str(spec.test_start.date()),
                "test_end":    str(spec.test_end.date()),
            },
        })

        for date, dist, dec, ret in zip(
            test_dates,
            distances[: len(test_dates)],
            decisions[: len(test_dates)],
            env_used.return_history[: len(test_dates)],
        ):
            daily_records.append({
                "round": round_idx,
                "date":  str(date.date()),
                "distance": float(dist),
                "use_hmm_off": bool(dec),
                "log_return": float(ret),
            })

    # ── 결과 저장 ─────────────────────────────────
    results_dir = Path(cfg["paths"]["results_dir"])
    out_csv = results_dir / "rolling_window_results_gated.csv"
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    daily_df = pd.DataFrame(daily_records)
    daily_csv = results_dir / "mahalanobis" / "gated_daily.csv"
    daily_csv.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(daily_csv, index=False)

    daily_pkl_dir = results_dir / "daily" / "gated"
    daily_pkl_dir.mkdir(parents=True, exist_ok=True)
    for d in daily_pkls:
        with open(daily_pkl_dir / f"round_{d['round_index']+1:02d}.pkl", "wb") as f:
            pickle.dump(d, f)

    # ── 비교 표 ──────────────────────────────────
    hmm_df    = pd.read_csv(results_dir / "rolling_window_results_hmm_cap30.csv")
    nohmm_df  = pd.read_csv(results_dir / "rolling_window_results_nohmm_cap30.csv")

    cmp = pd.DataFrame({
        "round":         out_df["round"],
        "sharpe_hmm":    hmm_df["test_sharpe"],
        "sharpe_nohmm":  nohmm_df["test_sharpe"],
        "sharpe_gated":  out_df["sharpe_ratio"],
        "frac_off":      out_df["frac_off"].round(2),
    })
    cmp["best_fixed"]    = np.maximum(cmp["sharpe_hmm"], cmp["sharpe_nohmm"])
    cmp["gated_correct"] = cmp["sharpe_gated"] >= cmp["best_fixed"] - 0.02

    print("\n" + "=" * 90)
    print(" 라운드별 비교: 항상 HMM-on / 항상 HMM-off / Mahalanobis-gated / oracle")
    print("=" * 90)
    print(cmp.round(3).to_string(index=False))

    avg_hmm    = float(cmp["sharpe_hmm"].mean())
    avg_nohmm  = float(cmp["sharpe_nohmm"].mean())
    avg_gated  = float(cmp["sharpe_gated"].mean())
    avg_oracle = float(cmp["best_fixed"].mean())

    print("\n" + "=" * 90)
    print(" 8 라운드 평균 Sharpe")
    print("=" * 90)
    print(f"  항상 HMM-on              : {avg_hmm:+.3f}")
    print(f"  항상 HMM-off             : {avg_nohmm:+.3f}")
    print(f"  Mahalanobis-gated (실제) : {avg_gated:+.3f}")
    print(f"  Oracle (사후 최적)        : {avg_oracle:+.3f}")
    print(f"  Gated/Oracle 효율         : {avg_gated/avg_oracle*100:.1f}%")

    # ── 시각화 ───────────────────────────────────
    fig_dir = Path(cfg["paths"]["figures_dir"]) / "gated"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 막대그래프 비교
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.27
    rounds_x = cmp["round"].values
    ax.bar(rounds_x - width, cmp["sharpe_hmm"], width,
           color="#4a90c2", label="항상 HMM-on", edgecolor="black", linewidth=0.4)
    ax.bar(rounds_x,         cmp["sharpe_nohmm"], width,
           color="#cc0000", label="항상 HMM-off", edgecolor="black", linewidth=0.4)
    ax.bar(rounds_x + width, cmp["sharpe_gated"], width,
           color="#2ca663", label="Mahalanobis-gated", edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(rounds_x)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Sharpe")
    ax.set_title(
        f"Mahalanobis-gated rollout — threshold={args.threshold}, window={args.window}일",
        fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "round_sharpe_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\n  결과 CSV : {out_csv}")
    print(f"  일별 CSV : {daily_csv}")
    print(f"  daily pkl: {daily_pkl_dir}/")
    print(f"  그림     : {fig_dir}/round_sharpe_comparison.png")


if __name__ == "__main__":
    main()
