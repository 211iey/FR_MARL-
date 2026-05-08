"""
Mahalanobis distance 기반 OOD detection 검증.

가설: Round 4(금리)·Round 8(AI) test 기간은 HMM 학습창 분포에서 멀리 떨어진
     OOD 영역이며, Mahalanobis distance가 entropy보다 더 좋은 감지 신호다.

각 라운드별로 HMM 입력공간 [kospi, vkospi]에서:
  1. train 데이터로 μ, Σ 추정
  2. test 일별 Mahalanobis distance 계산: D = sqrt((x-μ)^T Σ^-1 (x-μ))
  3. 라운드별 통계 (mean/p90/max) 집계
  4. HMM advantage(Sharpe_on - Sharpe_off)와 상관관계
  5. Threshold gating 가상 성과

산출물 (results/figures/mahalanobis/):
  1. round_dist_vs_advantage.png    : 라운드별 distance vs HMM advantage
  2. daily_distance_timeseries.png  : 일별 distance 시계열
  3. distance_scatter.png           : distance ↔ HMM advantage 산점도
  4. threshold_simulation.png       : Mahalanobis threshold gating 가상 성과
  5. comparison_entropy_mahala.png  : entropy vs Mahalanobis 비교

사용:
    python3 analyze_mahalanobis.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import load_config

mpl.rcParams["font.family"] = "AppleGothic"
mpl.rcParams["axes.unicode_minus"] = False


FEATURES = ["kospi", "vkospi"]   # HMM이 보는 공간 그대로


def mahalanobis_distances(X: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    diff = X - mu
    # 각 행(샘플)별로 quadratic form 계산
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))


def main():
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    fig_dir = Path(cfg["paths"]["figures_dir"]) / "mahalanobis"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_out_dir = results_dir / "mahalanobis"
    csv_out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ────────────────────────────────────
    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()

    daily_dir = results_dir / "daily"
    pkl_files = sorted(daily_dir.glob("round_*.pkl"))

    # 라운드별 train·test 기간 → Mahalanobis distance 계산
    round_records = []
    daily_dist_records = []

    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        round_idx = d["round_index"] + 1
        spec = d["spec"]
        train_start = pd.Timestamp(spec["train_start"])
        train_end   = pd.Timestamp(spec["train_end"])
        test_dates  = pd.to_datetime(d["dates"])

        # train 통계
        train_X = df.loc[train_start:train_end, FEATURES].dropna().values
        mu = train_X.mean(axis=0)
        cov = np.cov(train_X, rowvar=False)
        # 수치 안정화
        cov += 1e-8 * np.eye(cov.shape[0])
        cov_inv = np.linalg.inv(cov)

        # test 일별 distance
        test_X = df.loc[test_dates, FEATURES].dropna()
        test_X_arr = test_X.values
        test_dists = mahalanobis_distances(test_X_arr, mu, cov_inv)

        # 비교용: train 자체의 distance 분포 (chi-square 기반 reference)
        train_dists = mahalanobis_distances(train_X, mu, cov_inv)

        round_records.append({
            "round": round_idx,
            "train_dist_mean":  float(np.mean(train_dists)),
            "train_dist_p95":   float(np.percentile(train_dists, 95)),
            "test_dist_mean":   float(np.mean(test_dists)),
            "test_dist_p90":    float(np.percentile(test_dists, 90)),
            "test_dist_max":    float(np.max(test_dists)),
            "frac_above_p95":   float(np.mean(test_dists > np.percentile(train_dists, 95))),
        })

        for date, dist in zip(test_X.index, test_dists):
            daily_dist_records.append({
                "round": round_idx,
                "date":  date,
                "mahalanobis": float(dist),
            })

    round_df = pd.DataFrame(round_records)
    daily_df = pd.DataFrame(daily_dist_records).sort_values("date").reset_index(drop=True)

    # ── HMM advantage 결합 ────────────────────────────
    hmm_df = pd.read_csv(
        results_dir / "rolling_window_results_hmm_cap30.csv"
    ).sort_values("round").reset_index(drop=True)
    nohmm_df = pd.read_csv(
        results_dir / "rolling_window_results_nohmm_cap30.csv"
    ).sort_values("round").reset_index(drop=True)

    round_df = round_df.merge(
        hmm_df[["round", "test_sharpe"]].rename(columns={"test_sharpe": "sharpe_hmm"}),
        on="round",
    )
    round_df = round_df.merge(
        nohmm_df[["round", "test_sharpe"]].rename(columns={"test_sharpe": "sharpe_nohmm"}),
        on="round",
    )
    round_df["hmm_advantage"] = round_df["sharpe_hmm"] - round_df["sharpe_nohmm"]

    print("=" * 90)
    print(" 라운드별 Mahalanobis distance & HMM advantage")
    print("=" * 90)
    pcols = ["round", "test_dist_mean", "test_dist_p90", "test_dist_max",
             "frac_above_p95", "sharpe_hmm", "sharpe_nohmm", "hmm_advantage"]
    print(round_df[pcols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # 상관관계 (mean과 p90 둘 다)
    corr_mean = round_df[["test_dist_mean", "hmm_advantage"]].corr().iloc[0, 1]
    corr_p90  = round_df[["test_dist_p90",  "hmm_advantage"]].corr().iloc[0, 1]
    corr_frac = round_df[["frac_above_p95", "hmm_advantage"]].corr().iloc[0, 1]

    print(f"\n  Pearson r (test_dist_mean ↔ HMM advantage):    {corr_mean:+.3f}")
    print(f"  Pearson r (test_dist_p90  ↔ HMM advantage):    {corr_p90:+.3f}")
    print(f"  Pearson r (frac_above_p95 ↔ HMM advantage):    {corr_frac:+.3f}")
    print(f"  → 음(-)일수록 'distance 높을수록 HMM이 해롭다' = 가설 지지")

    # ── 시각화 1: dual-axis 라운드별 ─────────────────
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.bar(round_df["round"], round_df["test_dist_mean"],
            color="#4a90c2", alpha=0.78, edgecolor="black", linewidth=0.5,
            label="Test mean Mahalanobis distance")
    ax1.set_xlabel("Round", fontsize=11)
    ax1.set_ylabel("Test Mahalanobis distance (mean)", color="#4a90c2", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#4a90c2")
    ax1.set_xticks(round_df["round"])

    # train 95th percentile reference
    train_p95_avg = round_df["train_dist_p95"].mean()
    ax1.axhline(train_p95_avg, color="gray", linestyle=":", linewidth=1,
                label=f"평균 train p95 = {train_p95_avg:.2f}")

    ax2 = ax1.twinx()
    ax2.plot(round_df["round"], round_df["hmm_advantage"],
             color="#cc0000", marker="o", linewidth=2.2, markersize=8,
             label="HMM advantage (Sharpe_on - Sharpe_off)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("HMM advantage  (Δ Sharpe)", color="#cc0000", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#cc0000")

    # 실패 라운드 음영
    for r_idx in [4, 8]:
        ax1.axvspan(r_idx - 0.4, r_idx + 0.4, color="#ffe8e8", alpha=0.4, zorder=0)

    ax1.set_title("Mahalanobis distance vs HMM advantage  —  라운드별",
                  fontsize=12, fontweight="bold")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "round_dist_vs_advantage.png", dpi=150)
    plt.close(fig)

    # ── 시각화 2: 일별 시계열 ──────────────────────
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(daily_df["date"], daily_df["mahalanobis"],
            color="#0a2540", linewidth=0.7, alpha=0.85)
    ax.fill_between(daily_df["date"], 0, daily_df["mahalanobis"],
                    color="#4a90c2", alpha=0.25)

    # train 95th percentile (라운드별로 다름) — 평균선 표시
    ax.axhline(train_p95_avg, color="gray", linestyle=":", linewidth=1,
               label=f"평균 train p95 = {train_p95_avg:.2f}")

    # 라운드 경계 + R4, R8 음영
    for r in range(2, 9):
        sub = daily_df[daily_df["round"] == r]
        if not sub.empty:
            ax.axvline(sub["date"].iloc[0], color="gray",
                       linestyle=":", linewidth=0.5, alpha=0.6)
            ax.text(sub["date"].iloc[0],
                    daily_df["mahalanobis"].max() * 1.02,
                    f"R{r}", fontsize=8, color="gray", ha="left")
    for r in [4, 8]:
        sub = daily_df[daily_df["round"] == r]
        if not sub.empty:
            ax.axvspan(sub["date"].iloc[0], sub["date"].iloc[-1],
                       color="#cc0000", alpha=0.10, zorder=0)

    ax.set_xlabel("Date")
    ax.set_ylabel("Mahalanobis distance")
    ax.set_title("일별 Mahalanobis distance — 빨강 음영: HMM 실패 라운드",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "daily_distance_timeseries.png", dpi=150)
    plt.close(fig)

    # ── 시각화 3: 산점도 ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(round_df["test_dist_mean"], round_df["hmm_advantage"],
               s=160, c=round_df["round"], cmap="viridis",
               edgecolor="black", linewidth=1.4, zorder=3)
    for _, r in round_df.iterrows():
        ax.annotate(f"R{int(r['round'])}",
                    (r["test_dist_mean"], r["hmm_advantage"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=11,
                    fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("라운드 평균 Mahalanobis distance", fontsize=11)
    ax.set_ylabel("HMM advantage (Sharpe_on - Sharpe_off)", fontsize=11)
    ax.set_title(f"Mahalanobis distance vs HMM 효용  —  Pearson r = {corr_mean:+.2f}",
                 fontsize=12, fontweight="bold")
    coef = np.polyfit(round_df["test_dist_mean"], round_df["hmm_advantage"], 1)
    xline = np.linspace(round_df["test_dist_mean"].min() - 0.05,
                        round_df["test_dist_mean"].max() + 0.1, 100)
    ax.plot(xline, np.polyval(coef, xline), color="#cc0000",
            linestyle="--", linewidth=1.5, alpha=0.7,
            label=f"linear fit: y = {coef[0]:+.2f}x + {coef[1]:+.2f}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "distance_scatter.png", dpi=150)
    plt.close(fig)

    # ── 시각화 4: Threshold gating simulation ─────
    thresholds = np.linspace(round_df["test_dist_mean"].min(),
                             round_df["test_dist_mean"].max() + 0.05, 60)
    gated_avg_sharpe = []
    for thr in thresholds:
        choose_off = round_df["test_dist_mean"] > thr
        sharpes = np.where(
            choose_off, round_df["sharpe_nohmm"], round_df["sharpe_hmm"]
        )
        gated_avg_sharpe.append(float(np.mean(sharpes)))

    baseline_hmm = float(round_df["sharpe_hmm"].mean())
    baseline_nohmm = float(round_df["sharpe_nohmm"].mean())
    best_idx = int(np.argmax(gated_avg_sharpe))
    best_thr = thresholds[best_idx]
    best_sharpe = gated_avg_sharpe[best_idx]
    oracle = float(np.maximum(round_df["sharpe_hmm"], round_df["sharpe_nohmm"]).mean())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, gated_avg_sharpe, color="#0a2540", linewidth=2.2,
            label="Mahalanobis-gated 평균 Sharpe")
    ax.axhline(baseline_hmm, color="#4a90c2", linestyle="--", linewidth=1.5,
               label=f"항상 HMM-on  ({baseline_hmm:.3f})")
    ax.axhline(baseline_nohmm, color="#cc0000", linestyle="--", linewidth=1.5,
               label=f"항상 HMM-off ({baseline_nohmm:.3f})")
    ax.axhline(oracle, color="green", linestyle="-.", linewidth=1.5, alpha=0.7,
               label=f"Oracle (사후 최적, {oracle:.3f})")
    ax.axvline(best_thr, color="green", linestyle=":", linewidth=1.5, alpha=0.7,
               label=f"최적 threshold = {best_thr:.3f}\n→ avg = {best_sharpe:.3f}")
    ax.set_xlabel("Mahalanobis threshold (이 값 초과 시 HMM 끔)", fontsize=11)
    ax.set_ylabel("8 라운드 평균 Test Sharpe", fontsize=11)
    ax.set_title("라운드 단위 Mahalanobis Gating — 가상 성과", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "threshold_simulation.png", dpi=150)
    plt.close(fig)

    # ── 시각화 5: entropy vs Mahalanobis 비교 ────
    csv_dir = results_dir / "daily_csv"
    daily_full = pd.read_csv(csv_dir / "all_weights_with_regime.csv")
    prob_cols = [f"regime_prob_{i}" for i in range(3)]
    # 벡터화된 entropy 계산 (행 단위 apply 대신)
    probs = daily_full[prob_cols].values.astype(np.float64)
    probs_safe = np.clip(probs, 1e-12, 1.0)
    daily_full["entropy"] = -np.sum(probs * np.log(probs_safe), axis=1)
    ent_round = daily_full.groupby("round")["entropy"].mean().reset_index()
    round_df = round_df.merge(ent_round, on="round").rename(
        columns={"entropy": "entropy_mean"}
    )
    corr_ent = round_df[["entropy_mean", "hmm_advantage"]].corr().iloc[0, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, signal, label, r_val in [
        (axes[0], "entropy_mean",   "HMM posterior entropy",       corr_ent),
        (axes[1], "test_dist_mean", "Mahalanobis distance (mean)", corr_mean),
    ]:
        ax.scatter(round_df[signal], round_df["hmm_advantage"],
                   s=140, c=round_df["round"], cmap="viridis",
                   edgecolor="black", linewidth=1.3, zorder=3)
        for _, r in round_df.iterrows():
            ax.annotate(f"R{int(r['round'])}",
                        (r[signal], r["hmm_advantage"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=10,
                        fontweight="bold")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.7)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("HMM advantage (Δ Sharpe)", fontsize=10)
        ax.set_title(f"{label}\nPearson r = {r_val:+.3f}",
                     fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
    fig.suptitle("Entropy vs Mahalanobis — paradigm shift detector 비교",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "comparison_entropy_mahala.png", dpi=150)
    plt.close(fig)

    # ── 최종 요약 ──────────────────────────────────
    print("\n" + "=" * 90)
    print(" 최종 요약")
    print("=" * 90)
    print(f"  Pearson r entropy   ↔ HMM 효용:  {corr_ent:+.3f}")
    print(f"  Pearson r dist_mean ↔ HMM 효용:  {corr_mean:+.3f}")
    print(f"  Pearson r dist_p90  ↔ HMM 효용:  {corr_p90:+.3f}")
    print()
    print(f"  항상 HMM-on  Sharpe avg:        {baseline_hmm:.3f}")
    print(f"  항상 HMM-off Sharpe avg:        {baseline_nohmm:.3f}")
    print(f"  Mahalanobis gating 최적:         {best_sharpe:.3f}  (thr={best_thr:.3f})")
    print(f"  Oracle (사후 최적):              {oracle:.3f}")

    if corr_mean < -0.3:
        print("\n  ✓ 가설 지지: Mahalanobis 높을수록 HMM 해롭다")
    elif corr_mean > 0.3:
        print("\n  ✗ 가설 반대 결과")
    else:
        print("\n  ~ 약한 신호: |r| < 0.3")

    # ── CSV 저장 ──────────────────────────────────────────
    # 1) 라운드별 종합 표 (entropy 비교 포함)
    round_summary_path = csv_out_dir / "round_summary.csv"
    round_df.round(4).to_csv(round_summary_path, index=False)

    # 2) 일별 Mahalanobis distance
    daily_csv_path = csv_out_dir / "daily_distance.csv"
    daily_df.assign(date=daily_df["date"].dt.strftime("%Y-%m-%d")) \
            .round(4).to_csv(daily_csv_path, index=False)

    # 3) Threshold sweep (gating 성과)
    sweep_path = csv_out_dir / "threshold_sweep.csv"
    pd.DataFrame({
        "threshold": np.round(thresholds, 4),
        "gated_avg_sharpe": np.round(gated_avg_sharpe, 4),
    }).to_csv(sweep_path, index=False)

    # 4) Gating 의사결정 표 (각 라운드별 어느 모델이 채택됐는지)
    decision_df = round_df[["round", "test_dist_mean", "sharpe_hmm", "sharpe_nohmm",
                            "hmm_advantage"]].copy()
    decision_df["gated_use_hmm_off"] = decision_df["test_dist_mean"] > best_thr
    decision_df["gated_sharpe"] = np.where(
        decision_df["gated_use_hmm_off"],
        decision_df["sharpe_nohmm"],
        decision_df["sharpe_hmm"],
    )
    decision_df["oracle_sharpe"] = np.maximum(
        decision_df["sharpe_hmm"], decision_df["sharpe_nohmm"]
    )
    decision_df["correct_decision"] = (
        decision_df["gated_sharpe"] == decision_df["oracle_sharpe"]
    )
    decision_path = csv_out_dir / "gating_decisions.csv"
    decision_df.round(4).to_csv(decision_path, index=False)

    print(f"\n  생성된 그림: {fig_dir}/")
    print(f"  생성된 CSV : {csv_out_dir}/")
    print(f"    ├── round_summary.csv     (라운드별 distance + Sharpe + entropy)")
    print(f"    ├── daily_distance.csv    (일별 Mahalanobis 거리)")
    print(f"    ├── threshold_sweep.csv   (threshold별 gated 평균 Sharpe)")
    print(f"    └── gating_decisions.csv  (라운드별 의사결정 + 정답 여부)")


if __name__ == "__main__":
    main()
