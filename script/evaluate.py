"""
Rolling Window 평가 및 시각화.

사용 예:
    python3 evaluate.py
    python3 evaluate.py --config <path>

전제:
  train.py 실행 후 생성된 파일들:
    - results/rolling_window_results.csv
    - results/daily/round_XX.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import load_config


# ─────────────────────────────────────────────────────────────
# 로드
# ─────────────────────────────────────────────────────────────

def load_daily_results(cfg: dict, n_rounds: int) -> list[dict]:
    daily_dir = Path(cfg["paths"]["results_dir"]) / "daily"
    out = []
    for i in range(n_rounds):
        p = daily_dir / f"round_{i + 1:02d}.pkl"
        if not p.exists():
            print(f"[warning] {p} 없음 — 건너뜀")
            continue
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


# ─────────────────────────────────────────────────────────────
# 시계열 구성
# ─────────────────────────────────────────────────────────────

def build_rl_series(daily_list: list[dict]) -> pd.Series:
    """각 Round test 일별 log return을 이어붙인 전체 시계열."""
    dates, rets = [], []
    for d in daily_list:
        dd = pd.to_datetime(d["dates"])
        rr = np.asarray(d["returns"], dtype=np.float64)
        n = min(len(dd), len(rr))
        dates.extend(dd[:n])
        rets.extend(rr[:n])
    s = pd.Series(rets, index=pd.DatetimeIndex(dates), name="rl")
    return s[~s.index.duplicated(keep="first")]


def build_weights_series(
    daily_list: list[dict],
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """모든 Round test의 step-by-step 비중을 이어붙인 (T, n_assets) 행렬."""
    all_w, all_d = [], []
    for d in daily_list:
        w = np.asarray(d["weights"], dtype=np.float64)   # (T+1, n_assets)
        dd = pd.to_datetime(d["dates"])                   # (T,)
        # weight_history는 초기값 포함하므로 [1:]로 step 이후 비중만
        if len(w) == len(dd) + 1:
            w = w[1:]
        n = min(len(w), len(dd))
        all_w.append(w[:n])
        all_d.extend(dd[:n])
    weights_arr = np.vstack(all_w)
    return weights_arr, pd.DatetimeIndex(all_d)


def metrics_from_logret(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"sharpe": np.nan, "cum_return": np.nan, "mdd": np.nan}
    cum = np.exp(r.cumsum())
    drawdowns = cum / np.maximum.accumulate(cum) - 1
    return {
        "sharpe":     float(r.mean() / (r.std() + 1e-8) * np.sqrt(252)),
        "cum_return": float(cum[-1] - 1.0),
        "mdd":        float(drawdowns.min()),
    }


# ─────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────

def plot_cumulative(
    rl_series: pd.Series,
    bench: dict[str, pd.Series],
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    rl_cum = np.exp(rl_series.cumsum())
    ax.plot(rl_cum.index, rl_cum.values, label="PPO (RL)", linewidth=2)
    for name, s in bench.items():
        cum = np.exp(s.dropna().cumsum())
        ax.plot(cum.index, cum.values, label=name, linewidth=1.3, alpha=0.75)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Cumulative Return (Rolling Window Test Periods)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Value (start=1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_round_bar(
    results_df: pd.DataFrame,
    col: str,
    title: str,
    out_path: Path,
    ylabel: str | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(results_df["round"], results_df[col])
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel or col)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(results_df["round"])
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_weights_heatmap(
    weights_arr: np.ndarray,
    dates: pd.DatetimeIndex,
    asset_cols: list[str],
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        weights_arr.T,
        ax=ax,
        cmap="viridis",
        yticklabels=asset_cols,
        xticklabels=False,
        cbar_kws={"label": "Weight"},
    )
    ax.set_title("Portfolio Weights Over Time")
    ax.set_xlabel(f"Time ({dates[0].date()} ~ {dates[-1].date()})")
    ax.set_ylabel("Asset")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    results_path = (
        Path(cfg["paths"]["results_dir"]) / "rolling_window_results.csv"
    )
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} 없음. 먼저 train.py를 실행하세요."
        )
    results_df = pd.read_csv(results_path)

    daily_list = load_daily_results(cfg, len(results_df))
    if not daily_list:
        raise RuntimeError("daily/*.pkl 없음.")

    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()

    # RL 시계열 + 벤치마크
    rl_series = build_rl_series(daily_list)
    common_idx = df.index.intersection(rl_series.index)
    bench: dict[str, pd.Series] = {}
    if "kospi" in df.columns:
        bench["KOSPI"] = df.loc[common_idx, "kospi"].dropna()
    asset_cols = cfg["data"]["asset_cols"]
    bench["Equal Weight"] = df.loc[common_idx, asset_cols].mean(axis=1).dropna()

    # 전체 기간 metrics
    rl_metrics = metrics_from_logret(rl_series.values)
    bench_metrics = {
        k: metrics_from_logret(v.values) for k, v in bench.items()
    }

    # 비중 heatmap용 데이터
    weights_arr, weight_dates = build_weights_series(daily_list)

    # ── 출력 ───────────────────────────────────────────
    print("=" * 70)
    print(" 전체 기간 성과 요약 (Rolling Window 전체 test를 이어붙인 결과)")
    print("=" * 70)
    summary = pd.DataFrame([
        {"strategy": "PPO (RL)", **rl_metrics},
        *[{"strategy": name, **m} for name, m in bench_metrics.items()],
    ])
    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print(" Round별 Test 성과")
    print("=" * 70)
    cols = ["round", "best_seed", "test_sharpe", "test_cum_return",
            "test_mdd", "test_turnover"]
    print(results_df[cols].to_string(index=False))

    # ── 저장 ───────────────────────────────────────────
    summary_path = Path(cfg["paths"]["results_dir"]) / "summary.csv"
    summary.to_csv(summary_path, index=False)

    figures_dir = Path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_cumulative(
        rl_series, bench,
        figures_dir / "cumulative_returns.png",
    )
    plot_round_bar(
        results_df, "test_sharpe",
        "Test Sharpe Ratio by Round",
        figures_dir / "round_sharpe.png",
        ylabel="Sharpe Ratio",
    )
    plot_round_bar(
        results_df, "test_mdd",
        "Test Max Drawdown by Round",
        figures_dir / "round_mdd.png",
        ylabel="MDD",
    )
    plot_weights_heatmap(
        weights_arr, weight_dates, asset_cols,
        figures_dir / "weights_heatmap.png",
    )

    print(f"\n→ summary.csv: {summary_path}")
    print(f"→ figures:     {figures_dir}/")


if __name__ == "__main__":
    main()
