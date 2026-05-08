"""
Baseline / HMM cap30 / noHMM cap30 의 라운드별 평균 자산 비중 시각화.
각 모델 × 라운드 조합의 테스트 기간 일별 가중치 평균을 stacked bar로 표시.
"""

import pickle, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT     = pathlib.Path(__file__).resolve().parents[2]
DAILY    = ROOT / "results" / "daily"
OUT_DIR  = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = [
    "semicon", "bank", "car", "finance",
    "treasury3", "ener&chem", "iron",
    "wti", "gold", "copper", "silver",
    "necessities", "healthcare", "treasury10",
]

ASSET_GROUPS = {
    "semicon":    "equity",
    "bank":       "equity",
    "car":        "equity",
    "finance":    "equity",
    "iron":       "equity",
    "ener&chem":  "commodity",
    "wti":        "commodity",
    "gold":       "commodity",
    "copper":     "commodity",
    "silver":     "commodity",
    "necessities":"defensive",
    "healthcare": "defensive",
    "treasury3":  "bond",
    "treasury10": "bond",
}

# 자산 색상 (그룹별 색조 유지)
COLORS = {
    "semicon":    "#1f77b4",
    "bank":       "#4a90d9",
    "car":        "#74b3e8",
    "finance":    "#aed4f5",
    "iron":       "#c6e2fa",
    "ener&chem":  "#d62728",
    "wti":        "#e85555",
    "gold":       "#f5a623",
    "copper":     "#f7c96e",
    "silver":     "#fbe4b0",
    "necessities":"#2ca02c",
    "healthcare": "#76c576",
    "treasury3":  "#9467bd",
    "treasury10": "#c5b0d5",
}


def load_model(model_name: str):
    """pkl 파일에서 라운드별 (mean_weights, test_year) 반환.
    model_name이 빈 문자열이면 results/daily/ 루트를 사용 (baseline).
    """
    pkl_dir = DAILY if model_name == "" else DAILY / model_name
    records = []
    for pkl_path in sorted(pkl_dir.glob("round_*.pkl")):
        with open(pkl_path, "rb") as f:
            d = pickle.load(f)
        r_idx  = d["round_index"] + 1
        year   = pd.Timestamp(d["spec"]["test_start"]).year
        w_mean = np.array(d["weights"]).mean(axis=0)   # (14,)
        records.append({"round": r_idx, "year": year,
                        **dict(zip(ASSETS, w_mean))})
    return pd.DataFrame(records).set_index("round")


def draw_stacked(ax, df: pd.DataFrame, title: str):
    xlabels = [f"R{r}\n{df.loc[r,'year']}" for r in df.index]
    x = np.arange(len(df))
    bottoms = np.zeros(len(df))

    for asset in ASSETS:
        vals = df[asset].values
        ax.bar(x, vals, bottom=bottoms,
               color=COLORS[asset], width=0.65,
               edgecolor="white", linewidth=0.4, label=asset)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.05:
                ax.text(i, b + v / 2, f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)


# ── 데이터 로드 ──────────────────────────────────────────────────────────────
baseline = load_model("")
hmm      = load_model("hmm_cap30")
nohmm    = load_model("nohmm_cap30")

MODELS = [
    (baseline, "Baseline",    "weight_by_round_baseline.png"),
    (hmm,      "HMM cap30",   "weight_by_round_hmm_cap30.png"),
    (nohmm,    "noHMM cap30", "weight_by_round_nohmm_cap30.png"),
]

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[a]) for a in ASSETS]

# ── 개별 파일 저장 ────────────────────────────────────────────────────────────
for df, title, fname in MODELS:
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f"Average Asset Weights per Round (Test Period)\n{title}",
                 fontsize=13, fontweight="bold")
    draw_stacked(ax, df, title)
    ax.set_xlabel("Round (Test Year)", fontsize=10)
    fig.legend(legend_handles, ASSETS,
               loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=9, framealpha=0.9, ncol=1,
               title="Asset", title_fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / fname
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved → {out}")

# ── 합본 파일 저장 ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=False)
fig.suptitle("Average Asset Weights per Round (Test Period)",
             fontsize=14, fontweight="bold", y=1.005)

for ax, (df, title, _) in zip(axes, MODELS):
    draw_stacked(ax, df, title)

fig.legend(legend_handles, ASSETS,
           loc="center left", bbox_to_anchor=(1.0, 0.5),
           fontsize=9, framealpha=0.9, ncol=1,
           title="Asset", title_fontsize=9)
axes[2].set_xlabel("Round (Test Year)", fontsize=10)

plt.tight_layout()
out = OUT_DIR / "weight_by_round_all_models.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved → {out}")
