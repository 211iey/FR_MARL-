"""
Rolling Window 스케줄러.

논문 Section 3.3.1 Figure 3.3:
  Round k: Train (train_years) / Val (val_years) / Test (test_years)
  stride_years만큼 연 단위로 슬라이딩.

데이터가 충분치 않은 마지막 Round는 부분 연도(예: test_end가 data_max로 clip)를 허용.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class RoundSpec:
    round_index: int          # 0-based
    train_start: pd.Timestamp
    train_end:   pd.Timestamp
    val_start:   pd.Timestamp
    val_end:     pd.Timestamp
    test_start:  pd.Timestamp
    test_end:    pd.Timestamp
    test_clipped: bool = False  # data_max로 잘린 경우 True

    def describe(self) -> str:
        clip = " (test clipped)" if self.test_clipped else ""
        return (
            f"Round {self.round_index + 1:02d}: "
            f"Train {self.train_start.date()} ~ {self.train_end.date()} | "
            f"Val {self.val_start.date()} ~ {self.val_end.date()} | "
            f"Test {self.test_start.date()} ~ {self.test_end.date()}{clip}"
        )


def generate_rounds(df: pd.DataFrame, cfg: dict) -> list[RoundSpec]:
    """
    설정값대로 Round 리스트 생성.

    중단 조건:
      - Val 기간이 데이터 범위를 완전히 벗어나면 그 Round부터 생성 안 함.
      - Test 기간이 데이터 범위 안에서 일부라도 시작되면 포함 (end만 clip).
    """
    rw = cfg["rolling_window"]
    start_year = rw["train_start_year"]
    train_years = rw["train_years"]
    val_years = rw["val_years"]
    test_years = rw["test_years"]
    stride = rw["stride_years"]

    data_min = df.index.min()
    data_max = df.index.max()

    rounds: list[RoundSpec] = []
    k = 0
    while True:
        train_start = pd.Timestamp(f"{start_year + k * stride}-01-01")
        train_end = pd.Timestamp(
            f"{start_year + k * stride + train_years - 1}-12-31"
        )
        val_start = pd.Timestamp(f"{train_end.year + 1}-01-01")
        val_end = pd.Timestamp(
            f"{train_end.year + val_years}-12-31"
        )
        test_start = pd.Timestamp(f"{val_end.year + 1}-01-01")
        test_end = pd.Timestamp(
            f"{val_end.year + test_years}-12-31"
        )

        # 완전한 val 기간 필요. val_end가 data_max를 넘으면 중단.
        if val_end > data_max:
            break

        # test는 부분 연도 허용: test_start가 data_max보다 앞이면 포함, end만 clip.
        if test_start > data_max:
            break

        test_clipped = test_end > data_max
        test_end_actual = min(test_end, data_max)

        rounds.append(
            RoundSpec(
                round_index=k,
                train_start=max(train_start, data_min),
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=test_start,
                test_end=test_end_actual,
                test_clipped=test_clipped,
            )
        )
        k += 1

    if not rounds:
        raise RuntimeError(
            f"Round 생성 실패: 데이터 범위 [{data_min.date()}, {data_max.date()}]와 "
            f"설정이 맞지 않습니다."
        )
    return rounds


def slice_with_context(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    context_days: int,
) -> pd.DataFrame:
    """
    [start, end] 구간을 슬라이스하되, 앞쪽에 거래일 기준 context_days만큼 더 포함.

    PortfolioEnv는 _get_obs()에서 과거 window일 수익률을 참조하므로,
    val/test의 첫 거래일에서 바로 액션할 수 있으려면 앞쪽 컨텍스트가 필요.
    (PortfolioEnv.reset() 직후 self.t = window부터 시작하므로, context_days = window이면
    val_start가 첫 액션 시점이 됨.)
    """
    target = df[(df.index >= start) & (df.index <= end)]
    if target.empty:
        raise ValueError(
            f"슬라이스 결과가 비었습니다: [{start.date()}, {end.date()}]"
        )
    first_pos = df.index.get_loc(target.index[0])
    last_pos = df.index.get_loc(target.index[-1])
    ctx_pos = max(0, first_pos - context_days)
    return df.iloc[ctx_pos : last_pos + 1]


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    df = pd.read_csv(
        cfg["paths"]["data_csv"], index_col=0, parse_dates=True
    ).sort_index()

    print("=" * 70)
    print(" Rolling Window 점검")
    print("=" * 70)
    print(f"데이터 범위: {df.index.min().date()} ~ {df.index.max().date()} "
          f"({len(df)}일)")
    print("-" * 70)

    rounds = generate_rounds(df, cfg)
    for r in rounds:
        print(f"  {r.describe()}")

    print("-" * 70)
    print(f"총 {len(rounds)} 라운드")

    # 슬라이스 크기 점검 (첫 Round 기준)
    r0 = rounds[0]
    window = cfg["data"]["window"]
    train_df = df.loc[r0.train_start : r0.train_end]
    val_df = slice_with_context(df, r0.val_start, r0.val_end, window)
    test_df = slice_with_context(df, r0.test_start, r0.test_end, window)
    print()
    print(f"Round 1 슬라이스 크기 (컨텍스트 {window}일 포함):")
    print(f"  train: {len(train_df):>5}일  [{train_df.index[0].date()} ~ "
          f"{train_df.index[-1].date()}]")
    print(f"  val  : {len(val_df):>5}일  [{val_df.index[0].date()} ~ "
          f"{val_df.index[-1].date()}]")
    print(f"  test : {len(test_df):>5}일  [{test_df.index[0].date()} ~ "
          f"{test_df.index[-1].date()}]")
