"""
각 round 테스트 기간의 레짐을 low-vol / spike / high-vol 로 재분류해 시각화.

분류 기준 (regime_summary 기준):
  spike   : std_kospi_ret > 0.5          (급격한 충격)
  high-vol: std_kospi_ret <= 0.5 AND mean_vkospi > 0  (지속적 스트레스)
  low-vol : std_kospi_ret <= 0.5 AND mean_vkospi <= 0 (평온)
"""

import pickle, pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = pathlib.Path(__file__).resolve().parents[2]
DAILY_CSV  = ROOT / "results" / "daily_csv"
DAILY_PKL  = ROOT / "results" / "daily"
OUT_DIR    = ROOT / "results" / "figures" / "regime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 레짐 성격 분류 ─────────────────────────────────────────────────────────
rs = pd.read_csv(DAILY_CSV / "regime_summary.csv")

def classify(row):
    if row["std_kospi_ret"] > 0.5:
        return "spike"
    if row["mean_vkospi"] > 0:
        return "high-vol"
    return "low-vol"

rs["regime_type"] = rs.apply(classify, axis=1)
regime_type_map = rs.set_index(["round", "regime"])["regime_type"].to_dict()

# ── 라운드별 테스트 기간 레짐 비율 계산 ────────────────────────────────────
df = pd.read_csv(DAILY_CSV / "all_weights_with_regime.csv", parse_dates=["date"])
df["regime_type"] = df.apply(lambda r: regime_type_map[(r["round"], r["regime"])], axis=1)

# 라운드별 테스트 연도 추출 (pkl spec 활용)
test_years = {}
for pkl_path in sorted(DAILY_PKL.glob("round_*.pkl")):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    r = d["round_index"] + 1
    test_years[r] = pd.Timestamp(d["spec"]["test_start"]).year

ratio = (
    df.groupby(["round", "regime_type"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=["low-vol", "spike", "high-vol"], fill_value=0)
)
ratio_pct = ratio.div(ratio.sum(axis=1), axis=0) * 100

# x축 레이블: "R4\n2022" 형식
xlabels = [f"R{r}\n{test_years.get(r,'')}" for r in ratio_pct.index]

# ── 시각화 ─────────────────────────────────────────────────────────────────
COLORS = {"low-vol": "#4C9BE8", "spike": "#F5A623", "high-vol": "#E84C4C"}
ORDER  = ["low-vol", "spike", "high-vol"]

fig, ax = plt.subplots(figsize=(10, 5))

bottoms = np.zeros(len(ratio_pct))
for col in ORDER:
    vals = ratio_pct[col].values
    bars = ax.bar(range(len(ratio_pct)), vals, bottom=bottoms,
                  color=COLORS[col], label=col, width=0.6, edgecolor="white", linewidth=0.5)
    # 5% 이상인 경우만 레이블 표시
    for i, (v, b) in enumerate(zip(vals, bottoms)):
        if v >= 5:
            ax.text(i, b + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")
    bottoms += vals

ax.set_xticks(range(len(ratio_pct)))
ax.set_xticklabels(xlabels, fontsize=10)
ax.set_ylabel("Test Period (%)", fontsize=11)
ax.set_ylim(0, 105)
ax.set_title("Regime Type Distribution per Round (Test Period)", fontsize=13, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

legend_patches = [mpatches.Patch(color=COLORS[c], label=c) for c in ORDER]
ax.legend(handles=legend_patches, loc="upper right", fontsize=10, framealpha=0.9)

# round 4 강조 (2022)
ax.axvline(x=3, color="gray", linestyle="--", linewidth=1.2, alpha=0.6)
ax.text(3.08, 102, "2022", fontsize=8.5, color="gray")

plt.tight_layout()
out = OUT_DIR / "regime_type_ratio.png"
plt.savefig(out, dpi=150)
print(f"saved → {out}")

# ── 수치 출력 ─────────────────────────────────────────────────────────────
ratio_pct.index = [f"R{r}({test_years.get(r,'')})" for r in ratio_pct.index]
print("\n" + ratio_pct.round(1).to_string())
