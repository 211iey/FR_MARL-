"""
일별 HMM 체제(regime) + 포트폴리오 비중 결합 분석.

각 라운드의 train_df로 HMM 재학습 후 test 기간에 대해 일별 regime을 추론하고,
일별 weight과 결합한 CSV + regime 전환 시점만 모은 CSV를 생성.

산출물:
  results/daily_csv/all_weights_with_regime.csv
      → 일별 [round, date, regime, regime_prob_*, return, weights..]

  results/daily_csv/regime_transitions.csv
      → 전환 시점 [round, date, prev_regime, new_regime,
                   weight_before(...), weight_after(...), L1_change]

  results/daily_csv/regime_summary.csv
      → 라운드별 regime 해석용 [round, regime, mean_kospi_ret, mean_vkospi, n_days_train]

사용:
    python3 extract_regime_weights.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from HMM import train_market_hmm
from config import load_config


def main():
    cfg = load_config()
    asset_cols = cfg["data"]["asset_cols"]
    n_regimes = cfg["data"]["n_regimes"]

    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()

    daily_dir = Path(cfg["paths"]["results_dir"]) / "daily"
    out_dir = Path(cfg["paths"]["results_dir"]) / "daily_csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(daily_dir.glob("round_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"{daily_dir}에 pickle 없음")

    all_daily = []
    all_transitions = []
    regime_summaries = []

    for pkl in pkl_files:
        with open(pkl, "rb") as f:
            d = pickle.load(f)

        round_idx = d["round_index"] + 1
        spec = d["spec"]
        train_start = pd.Timestamp(spec["train_start"])
        train_end = pd.Timestamp(spec["train_end"])

        # ── 1. HMM 재학습 (train 기간만) ─────────────────────────
        train_df = df.loc[train_start:train_end]
        hmm_model, scaler = train_market_hmm(train_df, n_regimes=n_regimes)

        # ── 2. test 기간 일별 regime 추론 ────────────────────────
        dates = pd.to_datetime(d["dates"])
        feat = df.loc[dates, ["kospi", "vkospi"]].dropna()
        scaled = scaler.transform(feat.values)
        regime_seq = hmm_model.predict(scaled)             # (T,)
        regime_prob = hmm_model.predict_proba(scaled)      # (T, n_regimes)

        # ── 3. weights와 결합 ───────────────────────────────────
        weights = np.array(d["weights"][1 : len(dates) + 1])
        returns = d["returns"]

        daily = pd.DataFrame(weights, columns=asset_cols, index=feat.index)
        daily.insert(0, "round", round_idx)
        daily.insert(1, "date", feat.index.strftime("%Y-%m-%d"))
        daily.insert(2, "regime", regime_seq)
        for k in range(n_regimes):
            daily.insert(3 + k, f"regime_prob_{k}", regime_prob[:, k])
        daily.insert(3 + n_regimes, "return", returns[: len(feat)])

        all_daily.append(daily.reset_index(drop=True))

        # ── 4. regime 전환 시점 추출 ────────────────────────────
        change_idx = np.where(np.diff(regime_seq) != 0)[0] + 1  # 새 regime 첫날
        for i in change_idx:
            w_before = weights[i - 1]
            w_after = weights[i]
            row = {
                "round": round_idx,
                "date": feat.index[i].strftime("%Y-%m-%d"),
                "prev_regime": int(regime_seq[i - 1]),
                "new_regime": int(regime_seq[i]),
                "L1_change": float(np.abs(w_after - w_before).sum()),
            }
            for j, c in enumerate(asset_cols):
                row[f"before_{c}"] = float(w_before[j])
                row[f"after_{c}"] = float(w_after[j])
            all_transitions.append(row)

        # ── 5. regime 해석용 (train 기간 평균 KOSPI/VKOSPI + 지속성) ─
        train_feat = train_df[["kospi", "vkospi"]].dropna()
        train_scaled = scaler.transform(train_feat.values)
        train_regime = hmm_model.predict(train_scaled)
        # 자기 자신으로 머무를 확률 (지속성 → spike vs high-vol 구분)
        self_loop = np.diag(hmm_model.transmat_)
        for k in range(n_regimes):
            mask = train_regime == k
            slp = float(self_loop[k])
            exp_dur = (1.0 / (1.0 - slp)) if slp < 1.0 else float("inf")
            regime_summaries.append({
                "round": round_idx,
                "regime": k,
                "n_days_train": int(mask.sum()),
                "mean_kospi_ret": float(train_feat["kospi"].values[mask].mean()) if mask.any() else float("nan"),
                "mean_vkospi": float(train_feat["vkospi"].values[mask].mean()) if mask.any() else float("nan"),
                "std_kospi_ret": float(train_feat["kospi"].values[mask].std()) if mask.any() else float("nan"),
                "self_loop_prob": slp,
                "expected_duration": exp_dur,
            })

        n_trans = len(change_idx)
        print(f"  Round {round_idx}: {len(daily)}일, regime 전환 {n_trans}회, "
              f"regime 분포={np.bincount(regime_seq, minlength=n_regimes).tolist()}")

    # ── 저장 ──────────────────────────────────────────────────
    daily_df = pd.concat(all_daily, ignore_index=True)
    daily_path = out_dir / "all_weights_with_regime.csv"
    daily_df.to_csv(daily_path, index=False)

    trans_df = pd.DataFrame(all_transitions)
    trans_path = out_dir / "regime_transitions.csv"
    trans_df.to_csv(trans_path, index=False)

    summary_df = pd.DataFrame(regime_summaries)
    summary_path = out_dir / "regime_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n일별 결합:  {daily_path}  ({len(daily_df)} rows)")
    print(f"전환 시점:  {trans_path}  ({len(trans_df)} transitions)")
    print(f"regime 요약: {summary_path}  ({len(summary_df)} rows)")

    # ── 발표용 요약 출력 ───────────────────────────────────────
    print("\n" + "=" * 70)
    print(" 라운드별 regime 전환 시 평균 비중 변화 (L1 distance)")
    print("=" * 70)
    if len(trans_df) > 0:
        agg = (trans_df.groupby("round")["L1_change"]
               .agg(["count", "mean", "max"])
               .rename(columns={"count": "n_transitions",
                                "mean": "mean_L1",
                                "max": "max_L1"}))
        print(agg.to_string())


if __name__ == "__main__":
    main()
