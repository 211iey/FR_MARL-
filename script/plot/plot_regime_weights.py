"""
Regime + 자산그룹별 비중 시각화 3종 (그룹 단순화 버전).

자산 14개를 3그룹으로 묶어 stacked area를 단순화:
  - 위험자산: semicon, bank, car, finance, ener&chem, iron, necessities, healthcare
  - 그 외:    wti, copper, silver
  - 안전자산: treasury3, treasury10, gold

각 패널 상단에 regime indicator bar(빨/회/초)를 두어 전환점 가시화,
전환 시점은 굵은 검은 수직선으로 강조.

산출물 (results/figures/regime/):
  1. regime_aligned_weights.png  : 8라운드 패널
  2. round8_closeup.png          : Round 8 집중 분석
  3. L1_vs_sharpe.png            : 라운드별 평균 L1 vs Sharpe

사용:
    python3 plot_regime_weights.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from config import load_config

# Korean font (Mac)
mpl.rcParams["font.family"] = "AppleGothic"
mpl.rcParams["axes.unicode_minus"] = False


# ─── 자산 그룹 정의 ─────────────────────────────────────────────
ASSET_GROUPS = {
    "위험자산": ["semicon", "bank", "car", "finance", "ener&chem",
              "iron", "necessities", "healthcare"],
    "그 외":   ["wti", "copper", "silver"],
    "안전자산": ["treasury3", "treasury10", "gold"],
}
# 스택 순서: 아래 → 위
GROUP_ORDER = ["위험자산", "그 외", "안전자산"]

# 블루 팔레트 (어두움→밝음)
GROUP_COLORS = {
    "위험자산": "#0a2540",   # deep navy (바닥)
    "그 외":   "#4a90c2",   # medium blue
    "안전자산": "#c5dff5",   # pale sky blue (꼭대기)
}

# Regime indicator (변동성 기반 분류)
REGIME_COLORS = {
    0: "#3aa163",  # Low-Vol  - 초록 (안정장)
    1: "#e8851a",  # Spike    - 주황 (단기 변동성 충격)
    2: "#c0392b",  # High-Vol - 적색 (고변동 위험장)
}
REGIME_NAMES = {0: "Low-Vol", 1: "Spike", 2: "High-Vol"}
REGIME_ABBR  = {0: "L",       1: "S",     2: "H"}

TRANSITION_COLOR = "#000000"
TRANSITION_LW = 2.0


# ─── 헬퍼 ──────────────────────────────────────────────────────
def build_regime_remap(summary: pd.DataFrame) -> dict[int, dict[int, int]]:
    """
    변동성 기반 regime 재매핑:
      0 = Low-Vol  (가장 낮은 mean_vkospi)
      1 = Spike    (나머지 둘 중 self_loop_prob 낮은 쪽 → 짧은 지속)
      2 = High-Vol (나머지 둘 중 self_loop_prob 높은 쪽 → 긴 지속)
    """
    out = {}
    for r, sub in summary.groupby("round"):
        sub_sorted = sub.sort_values("mean_vkospi").reset_index(drop=True)
        low_vol = int(sub_sorted.iloc[0]["regime"])
        rem = sub_sorted.iloc[1:].sort_values("self_loop_prob").reset_index(drop=True)
        spike    = int(rem.iloc[0]["regime"])
        high_vol = int(rem.iloc[1]["regime"])
        out[int(r)] = {low_vol: 0, spike: 1, high_vol: 2}
    return out


def get_regime_runs(regime_seq: np.ndarray) -> list[tuple[int, int, int]]:
    runs = []
    s, cur = 0, regime_seq[0]
    for i in range(1, len(regime_seq)):
        if regime_seq[i] != cur:
            runs.append((s, i, int(cur)))
            s, cur = i, regime_seq[i]
    runs.append((s, len(regime_seq), int(cur)))
    return runs


def aggregate_groups(daily: pd.DataFrame) -> pd.DataFrame:
    for g, cols in ASSET_GROUPS.items():
        daily[g] = daily[cols].sum(axis=1)
    return daily


def draw_regime_bar(ax_bar, dates, regime_seq, transitions=True):
    """상단 regime indicator bar."""
    for s, e, r in get_regime_runs(regime_seq):
        ax_bar.axvspan(dates[s], dates[min(e, len(dates) - 1)],
                       color=REGIME_COLORS[r], alpha=0.95, zorder=1)
    if transitions:
        for i in range(1, len(regime_seq)):
            if regime_seq[i] != regime_seq[i - 1]:
                ax_bar.axvline(dates[i], color=TRANSITION_COLOR,
                               linewidth=TRANSITION_LW, zorder=5)
    ax_bar.set_yticks([])
    ax_bar.set_xticks([])
    ax_bar.set_ylim(0, 1)
    for spine in ax_bar.spines.values():
        spine.set_linewidth(0.6)


def draw_stack(ax, sub, regime_seq, dates):
    """stacked area + 전환선."""
    weights_grouped = sub[GROUP_ORDER].values  # (T, 3)
    ax.stackplot(
        sub["date"], weights_grouped.T,
        labels=GROUP_ORDER,
        colors=[GROUP_COLORS[g] for g in GROUP_ORDER],
        alpha=1.0,
    )
    # 굵은 전환선
    for i in range(1, len(regime_seq)):
        if regime_seq[i] != regime_seq[i - 1]:
            ax.axvline(dates[i], color=TRANSITION_COLOR,
                       linewidth=TRANSITION_LW, alpha=0.85, zorder=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(dates[0], dates[-1])


# ─── Viz 1: 8라운드 패널 ──────────────────────────────────────
def plot_viz1_panels(daily, rolling, out_path):
    fig = plt.figure(figsize=(16, 19))
    outer = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.13,
                             top=0.95, bottom=0.05, left=0.05, right=0.86)

    for round_idx in range(1, 9):
        sub = daily[daily["round"] == round_idx].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        dates = sub["date"].values
        regime_seq = sub["regime_ranked"].values

        cell = outer[(round_idx - 1) // 2, (round_idx - 1) % 2]
        inner = cell.subgridspec(2, 1, height_ratios=[1, 14], hspace=0.04)
        ax_bar = fig.add_subplot(inner[0])
        ax = fig.add_subplot(inner[1], sharex=ax_bar)

        draw_regime_bar(ax_bar, dates, regime_seq)
        draw_stack(ax, sub, regime_seq, dates)

        sharpe = rolling.loc[rolling["round"] == round_idx, "test_sharpe"].values[0]
        n_trans = int((regime_seq[1:] != regime_seq[:-1]).sum())
        regime_dist = np.bincount(regime_seq, minlength=3).tolist()

        ax_bar.set_title(
            f"Round {round_idx}  ·  Sharpe={sharpe:+.2f}  ·  전환 {n_trans}회  "
            f"·  [L={regime_dist[0]} S={regime_dist[1]} H={regime_dist[2]}]",
            fontsize=10.5, pad=6,
        )
        ax.set_ylabel("Weight", fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(False)

    # ── Legend ─────────────────────────────────────
    asset_handles = [Patch(facecolor=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    regime_handles = [
        Patch(facecolor=REGIME_COLORS[r], label=f"{REGIME_NAMES[r]} (regime {r})")
        for r in [0, 1, 2]
    ]
    line_handle = [plt.Line2D([0], [0], color=TRANSITION_COLOR,
                              linewidth=TRANSITION_LW, label="Regime 전환")]

    fig.legend(asset_handles + regime_handles + line_handle,
               [h.get_label() for h in asset_handles + regime_handles + line_handle],
               loc="center right", fontsize=10, frameon=True,
               bbox_to_anchor=(0.99, 0.5), title="범례",
               title_fontsize=11)

    fig.suptitle(
        "변동성 체제(Low-Vol / Spike / High-Vol) 전환과 자산군별 포트폴리오 비중 — 8 라운드 (2019~2026 Q1)",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─── Viz 2: Round 8 집중 ──────────────────────────────────────
def plot_viz2_round8(daily, rolling, out_path):
    sub = daily[daily["round"] == 8].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    dates = sub["date"].values
    regime_seq = sub["regime_ranked"].values

    fig = plt.figure(figsize=(15, 7.5))
    outer = fig.add_gridspec(2, 1, height_ratios=[1, 14], hspace=0.04,
                             top=0.90, bottom=0.13, left=0.07, right=0.82)
    ax_bar = fig.add_subplot(outer[0])
    ax = fig.add_subplot(outer[1], sharex=ax_bar)

    draw_regime_bar(ax_bar, dates, regime_seq)
    draw_stack(ax, sub, regime_seq, dates)

    # 전환점에 라벨
    for i in range(1, len(regime_seq)):
        if regime_seq[i] != regime_seq[i - 1]:
            label = f"{REGIME_ABBR[regime_seq[i-1]]}→{REGIME_ABBR[regime_seq[i]]}"
            ax_bar.annotate(
                label, xy=(dates[i], 1.4), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=TRANSITION_COLOR,
                annotation_clip=False,
            )

    sharpe = rolling.loc[rolling["round"] == 8, "test_sharpe"].values[0]
    mdd = rolling.loc[rolling["round"] == 8, "test_mdd"].values[0]
    regime_dist = np.bincount(regime_seq, minlength=3).tolist()

    ax_bar.set_title(
        f"Round 8 (2026 Q1) — Sharpe={sharpe:+.3f}, MDD={mdd*100:.1f}%  ·  "
        f"regime 분포 [Low-Vol={regime_dist[0]}, Spike={regime_dist[1]}, High-Vol={regime_dist[2]}]",
        fontsize=12.5, fontweight="bold", pad=18,
    )

    ax.set_ylabel("Portfolio Weight", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    asset_handles = [Patch(facecolor=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    regime_handles = [
        Patch(facecolor=REGIME_COLORS[r], label=REGIME_NAMES[r]) for r in [0, 1, 2]
    ]
    line_handle = [plt.Line2D([0], [0], color=TRANSITION_COLOR,
                              linewidth=TRANSITION_LW, label="Regime 전환")]
    fig.legend(asset_handles + regime_handles + line_handle,
               [h.get_label() for h in asset_handles + regime_handles + line_handle],
               loc="center right", fontsize=10, frameon=True,
               bbox_to_anchor=(0.99, 0.5))

    fig.text(
        0.07, 0.025,
        "학습창(2020~2024)에서 추정된 HMM은 2026 Q1 거의 전체를 단일 체제로 식별. "
        "드물게 등장한 regime은 1일짜리 노이즈로, 전환 시 매번 corner-to-corner jump 발생.",
        fontsize=9, color="#444", style="italic",
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─── Viz 3: L1 vs Sharpe 산점도 ───────────────────────────────
def plot_viz3_scatter(trans, rolling, out_path):
    agg = trans.groupby("round").agg(
        mean_L1=("L1_change", "mean"),
        n_transitions=("L1_change", "count"),
        max_L1=("L1_change", "max"),
    ).reset_index()
    agg = agg.merge(rolling[["round", "test_sharpe"]], on="round")

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(
        agg["mean_L1"], agg["test_sharpe"],
        s=agg["n_transitions"] * 6 + 100,
        c=agg["round"], cmap="Blues", vmin=0, vmax=9,
        edgecolor="black", linewidth=1.4, alpha=0.92, zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, ticks=range(1, 9))
    cbar.set_label("Round", fontsize=10)

    for _, row in agg.iterrows():
        ax.annotate(
            f"R{int(row['round'])}\n(n={int(row['n_transitions'])})",
            (row["mean_L1"], row["test_sharpe"]),
            textcoords="offset points", xytext=(11, 7), fontsize=10,
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

    coef = np.polyfit(agg["mean_L1"], agg["test_sharpe"], 1)
    xline = np.linspace(agg["mean_L1"].min() - 0.05,
                        agg["mean_L1"].max() + 0.05, 100)
    ax.plot(xline, np.polyval(coef, xline), color="#0a2540",
            linestyle="--", linewidth=1.6,
            label=f"linear fit: y = {coef[0]:.2f}x + {coef[1]:.2f}", zorder=2)

    corr = agg[["mean_L1", "test_sharpe"]].corr().iloc[0, 1]
    ax.text(
        0.02, 0.98,
        f"Pearson r = {corr:.3f}\n(점 크기 ∝ 전환 횟수)",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="#0a2540"),
    )

    ax.set_xlabel("Regime 전환 시 평균 L1 비중변화 (이론적 max=2.0)", fontsize=11)
    ax.set_ylabel("Test Sharpe Ratio", fontsize=11)
    ax.set_title("Corner-solution Jump의 규모와 성과의 관계 — 라운드별 (n=8)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")
    return agg


# ─── Viz 4: 자산별 regime 평균 비중 막대그래프 ─────────────────
def plot_asset_regime_bars(daily, out_path):
    """14개 자산 × 3 regime 평균 비중 (8라운드 통합)."""
    ordered_assets = (ASSET_GROUPS["위험자산"] +
                      ASSET_GROUPS["그 외"] +
                      ASSET_GROUPS["안전자산"])

    grouped = daily.groupby("regime_ranked")[ordered_assets].mean()
    grouped = grouped.reindex([0, 1, 2])  # 순서 보장

    fig, axes = plt.subplots(4, 4, figsize=(17, 13))
    fig.suptitle(
        "자산별 Regime 평균 비중 — 8 라운드 통합  (8라운드 1,753일 평균)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    ymax = grouped.values.max() * 1.18

    for idx, asset in enumerate(ordered_assets):
        ax = axes.flat[idx]
        weights = grouped[asset].values
        bars = ax.bar(
            [REGIME_NAMES[r] for r in [0, 1, 2]], weights,
            color=[REGIME_COLORS[r] for r in [0, 1, 2]],
            edgecolor="black", linewidth=0.6, alpha=0.92,
        )
        for bar, w in zip(bars, weights):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.012,
                    f"{w*100:.1f}%", ha="center", fontsize=9, fontweight="bold")

        # 자산 그룹별 제목 색상
        if asset in ASSET_GROUPS["위험자산"]:
            title_color = GROUP_COLORS["위험자산"]
            group_label = "위험"
        elif asset in ASSET_GROUPS["안전자산"]:
            title_color = GROUP_COLORS["안전자산"]
            group_label = "안전"
        else:
            title_color = GROUP_COLORS["그 외"]
            group_label = "기타"

        ax.set_title(f"{asset}  [{group_label}]",
                     fontsize=11, color=title_color, fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    # 빈 패널 숨기기
    for idx in range(len(ordered_assets), 16):
        axes.flat[idx].axis("off")

    fig.text(0.99, 0.01,
             "안전자산 그룹: 진한 파랑 제목  ·  위험자산: navy  ·  그 외: 중간 파랑",
             ha="right", fontsize=9, style="italic", color="gray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─── Viz 5: 라운드별 자산×regime heatmap ──────────────────────
def plot_asset_regime_heatmap(daily, out_path):
    """8라운드 × (14자산 × 3regime) heatmap grid."""
    ordered_assets = (ASSET_GROUPS["위험자산"] +
                      ASSET_GROUPS["그 외"] +
                      ASSET_GROUPS["안전자산"])

    fig, axes = plt.subplots(2, 4, figsize=(20, 12), sharey=True)
    fig.suptitle(
        "Round × 자산 × Regime 평균 비중 — 라운드별 regime-conditional 자산 선호",
        fontsize=14, fontweight="bold", y=0.995,
    )

    # 모든 라운드 공통 색상 스케일
    all_max = 0.0
    pivots = {}
    for round_idx in range(1, 9):
        sub = daily[daily["round"] == round_idx]
        pivot = sub.groupby("regime_ranked")[ordered_assets].mean()
        pivot = pivot.reindex([0, 1, 2])
        pivots[round_idx] = pivot.T  # rows=assets, cols=regimes
        all_max = max(all_max, np.nanmax(pivot.values))

    vmax = min(1.0, all_max)

    for round_idx in range(1, 9):
        ax = axes.flat[round_idx - 1]
        mat = pivots[round_idx].values  # (14, 3)

        im = ax.imshow(mat, aspect="auto", cmap="Blues",
                       vmin=0, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(3))
        ax.set_xticklabels([REGIME_NAMES[r] for r in [0, 1, 2]],
                           fontsize=9, rotation=15)
        ax.set_yticks(range(len(ordered_assets)))
        ax.set_yticklabels(ordered_assets, fontsize=8)

        # 셀 값 텍스트
        for i in range(len(ordered_assets)):
            for j in range(3):
                v = mat[i, j]
                if np.isfinite(v) and v > 0.005:
                    color = "white" if v > vmax * 0.55 else "#0a2540"
                    ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                            color=color, fontsize=7.5, fontweight="bold")

        # 자산 그룹 구분선
        ax.axhline(7.5, color="#cc0000", linewidth=1.0, linestyle="--", alpha=0.7)
        ax.axhline(10.5, color="#cc0000", linewidth=1.0, linestyle="--", alpha=0.7)

        ax.set_title(f"Round {round_idx}", fontsize=11, fontweight="bold")
        ax.tick_params(length=0)

    # 컬러바
    cbar_ax = fig.add_axes([0.92, 0.08, 0.012, 0.82])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("평균 비중", fontsize=10)

    # 그룹 라벨 (왼쪽 첫 패널 옆에 표시)
    fig.text(0.02, 0.74, "위험\n자산", fontsize=10, color=GROUP_COLORS["위험자산"],
             fontweight="bold", ha="center", va="center")
    fig.text(0.02, 0.44, "그 외", fontsize=10, color=GROUP_COLORS["그 외"],
             fontweight="bold", ha="center", va="center")
    fig.text(0.02, 0.21, "안전\n자산", fontsize=10, color="#3a6a99",
             fontweight="bold", ha="center", va="center")

    fig.tight_layout(rect=[0.04, 0, 0.91, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ─── main ──────────────────────────────────────────────────────
def main():
    cfg = load_config()
    results_dir = Path(cfg["paths"]["results_dir"])
    csv_dir = results_dir / "daily_csv"
    fig_dir = Path(cfg["paths"]["figures_dir"]) / "regime"
    fig_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(csv_dir / "all_weights_with_regime.csv")
    trans = pd.read_csv(csv_dir / "regime_transitions.csv")
    summary = pd.read_csv(csv_dir / "regime_summary.csv")
    rolling = pd.read_csv(results_dir / "rolling_window_results.csv")

    remap = build_regime_remap(summary)
    daily["regime_ranked"] = daily.apply(
        lambda r: remap[int(r["round"])][int(r["regime"])], axis=1
    ).astype(int)

    daily = aggregate_groups(daily)

    print("Generating plots...")
    plot_viz1_panels(daily, rolling, fig_dir / "regime_aligned_weights.png")
    plot_viz2_round8(daily, rolling, fig_dir / "round8_closeup.png")
    agg = plot_viz3_scatter(trans, rolling, fig_dir / "L1_vs_sharpe.png")
    plot_asset_regime_bars(daily, fig_dir / "asset_regime_bars.png")
    plot_asset_regime_heatmap(daily, fig_dir / "asset_regime_heatmap.png")

    print("\n[자산 그룹 비중 평균 — 라운드별]")
    means = (daily.groupby("round")[GROUP_ORDER].mean() * 100).round(1)
    means.columns = [f"{c} (%)" for c in means.columns]
    print(means.to_string())


if __name__ == "__main__":
    main()
