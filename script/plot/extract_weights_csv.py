"""
results/daily/round_NN.pkl → CSV 변환.

각 라운드의 일별 포트폴리오 비중을 CSV로 저장:
  results/daily_csv/round_NN_weights.csv  (라운드별)
  results/daily_csv/all_weights.csv       (8라운드 통합, round 컬럼 포함)

사용:
    python3 extract_weights_csv.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from config import load_config


def main():
    cfg = load_config()
    asset_cols = cfg["data"]["asset_cols"]

    daily_dir = Path(cfg["paths"]["results_dir"]) / "daily"
    out_dir = Path(cfg["paths"]["results_dir"]) / "daily_csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(daily_dir.glob("round_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"{daily_dir}에 pickle 없음")

    all_rows = []
    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            d = pickle.load(f)

        round_idx = d["round_index"] + 1
        dates = d["dates"]
        # weight_history는 초기 equal-weight + 매 step 후 비중 (T+1개)
        # dates(T개)와 정렬: weights[1:]가 dates[i] 종료 시점의 비중
        weights = d["weights"][1 : len(dates) + 1]
        returns = d["returns"]
        values = d["portfolio_values"][1 : len(dates) + 1]

        df = pd.DataFrame(weights, columns=asset_cols)
        df.insert(0, "date", dates)
        df.insert(1, "portfolio_value", values)
        df.insert(2, "return", returns)

        per_round_path = out_dir / f"round_{round_idx:02d}_weights.csv"
        df.to_csv(per_round_path, index=False)
        print(f"  saved: {per_round_path}  ({len(df)} rows)")

        df.insert(0, "round", round_idx)
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    combined_path = out_dir / "all_weights.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\n통합 CSV: {combined_path}  ({len(combined)} rows)")
    print(f"기간: {combined['date'].min()} ~ {combined['date'].max()}")


if __name__ == "__main__":
    main()
