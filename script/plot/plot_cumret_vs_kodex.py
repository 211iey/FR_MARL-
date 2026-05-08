"""
Baseline / HMM cap30 / noHMM cap30 의 테스트 기간 누적 수익률을
KODEX 200과 라운드별로 비교 시각화.
"""

import pickle, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = pathlib.Path(__file__).resolve().parents[2]
DAILY   = ROOT / "results" / "daily"
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "Baseline":    "",
    "HMM cap30":   "hmm_cap30",
    "noHMM cap30": "nohmm_cap30",
}
COLORS = {
    "Baseline":    "#4C9BE8",
    "HMM cap30":   "#E84C4C",
    "noHMM cap30": "#F5A623",
    "KODEX 200":   "#888888",
}

# ── KODEX 200 로드 ──────────────────────────────────────────────────────────
kodex = pd.read_csv(ROOT / "data" / "kodex200_log_returns.csv",
                    parse_dates=["Date"], index_col="Date")["kodex200"]


def load_rounds(subdir: str) -> dict[int, dict]:
    pkl_dir = DAILY if subdir == "" else DAILY / subdir
    result = {}
    for pkl_path in sorted(pkl_dir.glob("round_*.pkl")):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        r = d["round_index"] + 1
        dates = pd.to_datetime(d["dates"])
        pv    = np.array(d["portfolio_values"])   # len = n_dates + 1
        # dates[i] → pv[i+1]; 시작일을 dates[0]으로 맞춰 pv[0]=1.0 포함
        idx = pd.DatetimeIndex([dates[0] - pd.tseries.offsets.BDay(1)] + list(dates))
        result[r] = {"dates": idx, "pv": pv,
                     "test_start": d["spec"]["test_start"],
                     "test_end":   d["spec"]["test_end"]}
    return result


def kodex_cumret(dates: pd.DatetimeIndex) -> np.ndarray:
    """dates[0]을 기준(=1.0)으로 KODEX 200 누적 수익률 반환."""
    # dates[0]은 가상의 t-1일 (시작 전)이므로 dates[1:]의 log return을 누적
    log_rets = kodex.reindex(dates[1:]).fillna(0).values
    return np.concatenate([[1.0], np.exp(np.cumsum(log_rets))])


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
model_data = {name: load_rounds(sub) for name, sub in MODELS.items()}
n_rounds = max(len(v) for v in model_data.values())

# ── 그리기: 라운드별 서브플롯 ─────────────────────────────────────────────
ncols = 4
nrows = (n_rounds + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.8), squeeze=False)
fig.suptitle("Cumulative Return vs KODEX 200 (Test Period, per Round)",
             fontsize=14, fontweight="bold", y=1.01)

for r in range(1, n_rounds + 1):
    row, col = (r - 1) // ncols, (r - 1) % ncols
    ax = axes[row][col]

    ref = next(iter(model_data.values()))[r]
    k_cum = kodex_cumret(ref["dates"])
    ax.plot(ref["dates"], k_cum, color=COLORS["KODEX 200"],
            linewidth=1.4, linestyle="--", label="KODEX 200")

    for name, rounds in model_data.items():
        if r not in rounds:
            continue
        d = rounds[r]
        ax.plot(d["dates"], d["pv"], color=COLORS[name],
                linewidth=1.6, label=name)

    test_year = pd.Timestamp(ref["test_start"]).year
    ax.set_title(f"R{r}  ({test_year})", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # 최종 수익률 텍스트
    for name, rounds in model_data.items():
        if r not in rounds:
            continue
        final = rounds[r]["pv"][-1]
        color = COLORS[name]
        ax.annotate(f"{(final-1)*100:+.1f}%",
                    xy=(rounds[r]["dates"][-1], final),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=6.5, color=color, va="center")
    k_final = k_cum[-1]
    ax.annotate(f"{(k_final-1)*100:+.1f}%",
                xy=(ref["dates"][-1], k_final),
                xytext=(4, 0), textcoords="offset points",
                fontsize=6.5, color=COLORS["KODEX 200"], va="center")

# 빈 축 숨기기
for r in range(n_rounds, nrows * ncols):
    axes[r // ncols][r % ncols].set_visible(False)

# 공통 범례
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out = OUT_DIR / "cumret_vs_kodex200_hmm.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved → {out}")
