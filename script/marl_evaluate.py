"""
MARL λ sweep 결과 평가 및 시각화.

사용 예:
    python3 marl_evaluate.py
    python3 marl_evaluate.py --config <path>

전제: marl_train.py 실행 후 results/marl/sweep_results.csv 존재
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

from config import load_config


# ─────────────────────────────────────────────────────────────────
# 로드
# ─────────────────────────────────────────────────────────────────

def load_sweep(cfg: dict) -> pd.DataFrame:
    p = Path(cfg["paths"]["results_dir"]) / "marl" / "sweep_results.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} 없음. 먼저 marl_train.py를 실행하세요.")
    return pd.read_csv(p)


def load_daily(cfg: dict, diversity_mode: str, lambda_val: float) -> dict | None:
    p = (Path(cfg["paths"]["results_dir"]) / "marl" / "daily"
         / f"{diversity_mode}_lambda_{lambda_val:.1f}.pkl")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────

def plot_lambda_sweep(df: pd.DataFrame, out_path: Path):
    """λ vs Sharpe / 상관계수 / Turnover — 논문 핵심 결과 검증용."""
    modes = df["diversity_mode"].unique()
    colors = {"correlation": "#5b8dee", "tv": "#3ecf8e"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("MARL IPPO — λ sweep 결과", fontsize=14, fontweight="bold")

    metrics = [
        ("mean_sharpe",              "Sharpe Ratio (평균)",        "↑ 높을수록 좋음"),
        ("inter_agent_correlation",  "에이전트 간 평균 상관계수",   "↓ 낮을수록 다양성↑"),
        ("mean_turnover",            "평균 Turnover",               "↓ 낮을수록 좋음"),
    ]

    for ax, (col, ylabel, note) in zip(axes, metrics):
        for mode in modes:
            sub = df[df["diversity_mode"] == mode].sort_values("lambda_val")
            ax.plot(
                sub["lambda_val"], sub[col],
                marker="o", label=mode, color=colors.get(mode),
                linewidth=2, markersize=6,
            )
        ax.set_xlabel("λ (다양성 가중치)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(note, fontsize=10, color="gray")
        ax.set_xticks([round(x * 0.1, 1) for x in range(10)])
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weight_heatmap_compare(
    daily_0: dict | None,
    daily_best: dict | None,
    asset_cols: list[str],
    best_lambda: float,
    out_path: Path,
):
    """λ=0.0 vs λ=best 포트폴리오 비중 비교."""
    if daily_0 is None or daily_best is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("포트폴리오 비중 비교 (에이전트 평균)", fontsize=13, fontweight="bold")

    for ax, daily, title in zip(
        axes,
        [daily_0, daily_best],
        ["λ=0.0 (다양성 없음)", f"λ={best_lambda:.1f} (최적 다양성)"],
    ):
        agent_weights = [np.array(w) for w in daily["agent_weights"] if len(w) > 1]
        if not agent_weights:
            continue
        # step-by-step 비중 (초기값 제외)
        trimmed = []
        for w in agent_weights:
            w = w[1:] if len(w) > len(w) - 1 else w  # weight_history는 T+1개
            trimmed.append(w)
        min_T = min(len(w) for w in trimmed)
        mean_w = np.mean([w[:min_T] for w in trimmed], axis=0)  # (T, n_assets)
        sns.heatmap(
            mean_w.T,
            ax=ax,
            cmap="viridis",
            yticklabels=asset_cols,
            xticklabels=False,
            cbar_kws={"label": "Weight"},
            vmin=0, vmax=1,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Test 기간 (시간)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_sweep(cfg)

    asset_cols = cfg["data"]["asset_cols"]
    figures_dir = Path(cfg["paths"]["figures_dir"]) / "marl"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── 요약 출력 ───────────────────────────────────────────────
    print("=" * 70)
    print(" MARL λ sweep 결과 요약")
    print("=" * 70)
    cols = ["diversity_mode", "lambda_val", "mean_sharpe",
            "inter_agent_correlation", "mean_mdd", "mean_turnover"]
    print(df[cols].to_string(index=False))

    # 논문 검증: λ↑ → 상관↓ → Sharpe↑, Turnover↓
    print("\n" + "─" * 70)
    for mode in df["diversity_mode"].unique():
        sub = df[df["diversity_mode"] == mode].sort_values("lambda_val")
        corr_trend = np.corrcoef(sub["lambda_val"], sub["inter_agent_correlation"])[0, 1]
        sharpe_trend = np.corrcoef(sub["lambda_val"], sub["mean_sharpe"])[0, 1]
        turn_trend = np.corrcoef(sub["lambda_val"], sub["mean_turnover"])[0, 1]
        print(f" [{mode}]  λ↑→상관 r={corr_trend:.2f} | λ↑→Sharpe r={sharpe_trend:.2f} | "
              f"λ↑→Turnover r={turn_trend:.2f}")
    print("─" * 70)
    print(" 논문 예측: 상관 r<0, Sharpe r>0, Turnover r<0")

    # ── 시각화 ─────────────────────────────────────────────────
    plot_lambda_sweep(df, figures_dir / "lambda_sweep.png")

    # 최적 λ (각 mode별 Sharpe 최대)
    for mode in df["diversity_mode"].unique():
        sub = df[df["diversity_mode"] == mode]
        best_lam = float(sub.loc[sub["mean_sharpe"].idxmax(), "lambda_val"])
        d0 = load_daily(cfg, mode, 0.0)
        db = load_daily(cfg, mode, best_lam)
        plot_weight_heatmap_compare(
            d0, db, asset_cols, best_lam,
            figures_dir / f"weights_compare_{mode}.png",
        )

    # 최종 요약 저장
    summary_path = Path(cfg["paths"]["results_dir"]) / "marl" / "summary.csv"
    df.to_csv(summary_path, index=False)

    print(f"\n→ 시각화: {figures_dir}/")
    print(f"→ 요약:   {summary_path}")


if __name__ == "__main__":
    main()
