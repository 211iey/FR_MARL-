"""
HMM posterior entropy를 paradigm-shift detector로 검증.

가설: Round 4(금리)·Round 8(AI)에서 HMM이 어느 체제인지 확신하지 못함
     (즉 posterior entropy가 다른 라운드보다 높음).
     이 entropy 신호로 사전적(ex-ante) HMM-on/off 게이팅이 가능한가?

산출물 (results/figures/entropy/):
  1. round_entropy_vs_advantage.png  : 라운드별 entropy vs HMM advantage
  2. daily_entropy_timeseries.png    : 일별 entropy 시계열 (전 라운드)
  3. entropy_scatter.png             : entropy ↔ HMM advantage 산점도
  4. threshold_simulation.png        : threshold별 가상 게이팅 성과

사용:
    python3 analyze_entropy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config

mpl.rcParams["font.family"] = "AppleGothic"
mpl.rcParams["axes.unicode_minus"] = False


def shannon_entropy(probs: np.ndarray) -> float:
    """3-체제 사후확률의 Shannon entropy (자연로그 기준, 단위: nat)."""
    p = probs[probs > 1e-12]
    return float(-np.sum(p * np.log(p)))


def main():
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    csv_dir = results_dir / "daily_csv"
    fig_dir = Path(cfg["paths"]["figures_dir"]) / "entropy"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──────────────────────────────────────────
    daily = pd.read_csv(csv_dir / "all_weights_with_regime.csv")
    daily["date"] = pd.to_datetime(daily["date"])
    prob_cols = [f"regime_prob_{i}" for i in range(3)]
    daily["entropy"] = daily[prob_cols].apply(
        lambda r: shannon_entropy(r.values), axis=1
    )
    # 기준값 — 3체제 균등분포의 max entropy = ln(3) ≈ 1.0986
    max_entropy = np.log(3)

    # 라운드별 Sharpe (cap=0.3 케이스로 비교 — cap 통제)
    hmm_path    = results_dir / "rolling_window_results_hmm_cap30.csv"
    nohmm_path  = results_dir / "rolling_window_results_nohmm_cap30.csv"
    hmm_df    = pd.read_csv(hmm_path).sort_values("round").reset_index(drop=True)
    nohmm_df  = pd.read_csv(nohmm_path).sort_values("round").reset_index(drop=True)

    # ── 라운드별 집계 ────────────────────────────────────────
    round_stats = daily.groupby("round").agg(
        entropy_mean=("entropy", "mean"),
        entropy_std=("entropy", "std"),
        entropy_max=("entropy", "max"),
        entropy_p90=("entropy", lambda x: float(np.percentile(x, 90))),
    ).reset_index()
    round_stats = round_stats.merge(
        hmm_df[["round", "test_sharpe"]].rename(columns={"test_sharpe": "sharpe_hmm"}),
        on="round",
    )
    round_stats = round_stats.merge(
        nohmm_df[["round", "test_sharpe"]].rename(columns={"test_sharpe": "sharpe_nohmm"}),
        on="round",
    )
    round_stats["hmm_advantage"] = round_stats["sharpe_hmm"] - round_stats["sharpe_nohmm"]

    print("=" * 80)
    print(" 라운드별 entropy & HMM advantage")
    print("=" * 80)
    print(round_stats.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 상관관계
    corr = round_stats[["entropy_mean", "hmm_advantage"]].corr().iloc[0, 1]
    print(f"\n  Pearson r (entropy_mean vs hmm_advantage) = {corr:+.3f}")
    print(f"  → 음(-)이면 'entropy 높을수록 HMM이 해롭다' = 가설 지지")

    # ── 시각화 1: 라운드별 dual-axis ──────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    color_e = "#4a90c2"
    ax1.bar(round_stats["round"], round_stats["entropy_mean"],
            color=color_e, alpha=0.75, label="평균 entropy", edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("HMM posterior entropy (평균)", color=color_e, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color_e)
    ax1.axhline(max_entropy, color="gray", linestyle=":", linewidth=1,
                label=f"max entropy ln(3)={max_entropy:.2f}")
    ax1.set_xticks(round_stats["round"])

    ax2 = ax1.twinx()
    color_a = "#cc0000"
    ax2.plot(round_stats["round"], round_stats["hmm_advantage"],
             color=color_a, marker="o", linewidth=2, markersize=8,
             label="HMM advantage (Sharpe_on - Sharpe_off)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("HMM advantage  (Δ Sharpe)", color=color_a, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color_a)

    # 실패 라운드 강조
    for r_idx in [4, 8]:
        ax1.axvspan(r_idx - 0.4, r_idx + 0.4, color="#ffe8e8", alpha=0.4, zorder=0)

    ax1.set_title("라운드별 HMM posterior entropy vs HMM advantage",
                  fontsize=12, fontweight="bold")

    # 범례 통합
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / "round_entropy_vs_advantage.png", dpi=150)
    plt.close(fig)

    # ── 시각화 2: 일별 entropy 시계열 ──────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))
    daily_sorted = daily.sort_values("date").reset_index(drop=True)
    ax.plot(daily_sorted["date"], daily_sorted["entropy"],
            color="#0a2540", linewidth=0.7, alpha=0.85)
    ax.fill_between(daily_sorted["date"], 0, daily_sorted["entropy"],
                    color="#4a90c2", alpha=0.25)
    ax.axhline(max_entropy, color="gray", linestyle=":", linewidth=1)

    # 라운드 경계
    for r in range(2, 9):
        sub = daily_sorted[daily_sorted["round"] == r]
        if not sub.empty:
            ax.axvline(sub["date"].iloc[0], color="gray",
                       linestyle=":", linewidth=0.5, alpha=0.6)
            ax.text(sub["date"].iloc[0], max_entropy * 1.02,
                    f"R{r}", fontsize=8, color="gray", ha="left")

    # 실패 라운드 음영
    for r in [4, 8]:
        sub = daily_sorted[daily_sorted["round"] == r]
        if not sub.empty:
            ax.axvspan(sub["date"].iloc[0], sub["date"].iloc[-1],
                       color="#cc0000", alpha=0.10, zorder=0)

    ax.set_xlabel("Date")
    ax.set_ylabel("HMM posterior entropy")
    ax.set_title("일별 HMM posterior entropy — 빨강 음영: HMM 실패 라운드 (R4, R8)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max_entropy * 1.1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "daily_entropy_timeseries.png", dpi=150)
    plt.close(fig)

    # ── 시각화 3: 산점도 ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(round_stats["entropy_mean"], round_stats["hmm_advantage"],
               s=160, c=round_stats["round"], cmap="viridis",
               edgecolor="black", linewidth=1.4, zorder=3)
    for _, r in round_stats.iterrows():
        ax.annotate(f"R{int(r['round'])}",
                    (r["entropy_mean"], r["hmm_advantage"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=11,
                    fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("라운드 평균 HMM posterior entropy", fontsize=11)
    ax.set_ylabel("HMM advantage (Sharpe_on - Sharpe_off)", fontsize=11)
    ax.set_title(f"Entropy vs HMM 효용  —  Pearson r = {corr:+.2f}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    # 회귀선
    coef = np.polyfit(round_stats["entropy_mean"], round_stats["hmm_advantage"], 1)
    xline = np.linspace(round_stats["entropy_mean"].min() - 0.05,
                        round_stats["entropy_mean"].max() + 0.05, 100)
    ax.plot(xline, np.polyval(coef, xline), color="#cc0000",
            linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"linear fit: y = {coef[0]:+.2f}x + {coef[1]:+.2f}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "entropy_scatter.png", dpi=150)
    plt.close(fig)

    # ── 시각화 4: Threshold gating simulation ────────────
    # entropy_mean > threshold면 HMM-off 사용, 아니면 HMM-on 사용한다고 가정
    # 라운드 단위 게이팅
    thresholds = np.linspace(
        round_stats["entropy_mean"].min(),
        round_stats["entropy_mean"].max() + 0.05, 50
    )
    gated_avg_sharpe = []
    for thr in thresholds:
        choose_off = round_stats["entropy_mean"] > thr
        sharpes = np.where(
            choose_off, round_stats["sharpe_nohmm"], round_stats["sharpe_hmm"]
        )
        gated_avg_sharpe.append(float(np.mean(sharpes)))

    baseline_hmm = float(round_stats["sharpe_hmm"].mean())
    baseline_nohmm = float(round_stats["sharpe_nohmm"].mean())
    best_thr_idx = int(np.argmax(gated_avg_sharpe))
    best_thr = thresholds[best_thr_idx]
    best_sharpe = gated_avg_sharpe[best_thr_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, gated_avg_sharpe, color="#0a2540", linewidth=2.2,
            label="Entropy-gated 평균 Sharpe")
    ax.axhline(baseline_hmm, color="#4a90c2", linestyle="--", linewidth=1.5,
               label=f"항상 HMM-on  (avg = {baseline_hmm:.3f})")
    ax.axhline(baseline_nohmm, color="#cc0000", linestyle="--", linewidth=1.5,
               label=f"항상 HMM-off (avg = {baseline_nohmm:.3f})")
    ax.axvline(best_thr, color="green", linestyle=":", linewidth=1.5, alpha=0.7,
               label=f"최적 threshold = {best_thr:.3f}\n→ avg = {best_sharpe:.3f}")
    ax.set_xlabel("Entropy threshold (이 값 초과 시 HMM 끔)", fontsize=11)
    ax.set_ylabel("8 라운드 평균 Test Sharpe", fontsize=11)
    ax.set_title("라운드 단위 Entropy Gating — 가상 성과", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "threshold_simulation.png", dpi=150)
    plt.close(fig)

    # ── 최종 요약 ────────────────────────────────────────
    print("\n" + "=" * 80)
    print(" 최종 요약")
    print("=" * 80)
    print(f"  Pearson r (entropy ↔ HMM 효용):  {corr:+.3f}")
    print(f"  항상 HMM-on  평균 Sharpe:         {baseline_hmm:.3f}")
    print(f"  항상 HMM-off 평균 Sharpe:         {baseline_nohmm:.3f}")
    print(f"  Entropy gating 최적 threshold:    {best_thr:.3f}")
    print(f"  Entropy gating 최적 Sharpe:       {best_sharpe:.3f}")

    if corr < -0.3:
        print("\n  → 가설 지지: entropy ↑ → HMM 효용 ↓")
    elif corr > 0.3:
        print("\n  → 가설 반대 결과: entropy ↑ → HMM 효용 ↑")
    else:
        print("\n  → 약한 신호: |r| < 0.3, entropy는 강한 predictor가 아님")

    print(f"\n  생성된 그림: {fig_dir}/")


if __name__ == "__main__":
    main()
