"""
Baseline vs KODEX 200 누적 수익률 비교 (라운드별 서브플롯).
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

COLORS = {
    "Baseline":  "#4C9BE8",
    "KODEX 200": "#888888",
}

kodex = pd.read_csv(ROOT / "data" / "kodex200_log_returns.csv",
                    parse_dates=["Date"], index_col="Date")["kodex200"]


def load_rounds() -> dict[int, dict]:
    result = {}
    for pkl_path in sorted(DAILY.glob("round_*.pkl")):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        r     = d["round_index"] + 1
        dates = pd.to_datetime(d["dates"])
        pv    = np.array(d["portfolio_values"])
        idx   = pd.DatetimeIndex([dates[0] - pd.tseries.offsets.BDay(1)] + list(dates))
        result[r] = {"dates": idx, "pv": pv,
                     "test_start": d["spec"]["test_start"]}
    return result


def kodex_cumret(dates: pd.DatetimeIndex) -> np.ndarray:
    log_rets = kodex.reindex(dates[1:]).fillna(0).values
    return np.concatenate([[1.0], np.exp(np.cumsum(log_rets))])


rounds = load_rounds()
n_rounds = len(rounds)
ncols = 4
nrows = (n_rounds + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.8), squeeze=False)
fig.suptitle("Cumulative Return: Baseline vs KODEX 200 (Test Period, per Round)",
             fontsize=14, fontweight="bold", y=1.01)

for r, d in rounds.items():
    row, col = (r - 1) // ncols, (r - 1) % ncols
    ax = axes[row][col]

    k_cum = kodex_cumret(d["dates"])
    ax.plot(d["dates"], k_cum, color=COLORS["KODEX 200"],
            linewidth=1.4, linestyle="--", label="KODEX 200")
    ax.plot(d["dates"], d["pv"], color=COLORS["Baseline"],
            linewidth=1.6, label="Baseline")

    test_year = pd.Timestamp(d["test_start"]).year
    ax.set_title(f"R{r}  ({test_year})", fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.tick_params(axis="x", labelsize=7, rotation=20)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    for label, vals in [("Baseline", d["pv"]), ("KODEX 200", k_cum)]:
        final = vals[-1]
        ax.annotate(f"{(final-1)*100:+.1f}%",
                    xy=(d["dates"][-1], final),
                    xytext=(4, 0), textcoords="offset points",
                    fontsize=7, color=COLORS[label], va="center")

for r in range(n_rounds, nrows * ncols):
    axes[r // ncols][r % ncols].set_visible(False)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
out = OUT_DIR / "cumret_vs_kodex200_baseline.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved → {out}")
