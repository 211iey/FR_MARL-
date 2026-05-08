"""
한국 시장 포트폴리오 강화학습 환경
====================================
논문: Multi-Agent RL for Portfolio Management (M. Choi, TU Delft 2024)
데이터: 코스피 ETF 14개 자산 (2013-08-06 ~ 2026-04-03)

State (857차원)
---------------
[이전 비중 14 | 과거 60일 로그수익률 14×60=840 | vol20, vol20/vol60, vkospi 3]

Action (14차원)
---------------
각 자산 logit 값 → softmax → 포트폴리오 비중 (합계=1)

Reward
------
Differential Sharpe Ratio (DSR) — 논문 수식 3.4~3.7
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from HMM import train_market_hmm


def _cap_weights(w: np.ndarray, max_w: float, max_iters: int = 20) -> np.ndarray:
    """
    단일 자산 비중을 max_w 이하로 cap 후 sum=1을 유지하도록 재정규화.
    초과분은 cap 미만 자산들에게 비중 비례로 분배 (반복).

    feasibility: max_w * n_assets >= 1 이어야 함. 아닐 경우 등가중으로 fallback.
    """
    n = len(w)
    if max_w >= 1.0:
        return w.astype(np.float32)
    if max_w * n < 1.0 - 1e-9:
        return np.ones(n, dtype=np.float32) / n

    w = w.astype(np.float64).copy()
    for _ in range(max_iters):
        over = w > max_w + 1e-9
        if not over.any():
            break
        excess = float((w[over] - max_w).sum())
        w[over] = max_w
        under = ~over
        if not under.any():
            break
        under_sum = float(w[under].sum())
        if under_sum > 1e-9:
            w[under] += excess * w[under] / under_sum
        else:
            # 모든 미초과 자산이 0인 경우: 균등 분배
            w[under] += excess / under.sum()

    w = np.maximum(w, 0.0)
    s = w.sum()
    if s > 0:
        w = w / s
    return w.astype(np.float32)


# ─────────────────────────────────────────────
# 데이터 로드 및 분할
# ─────────────────────────────────────────────

def load_and_split(csv_path: str, window: int = 60, n_regimes: int = 0) -> dict:
    """
    CSV 로드 후 훈련/검증/테스트로 분할.

    Returns:
        dict with keys: train, val, test, asset_cols, vol_cols, n_assets, window
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()

    asset_cols = ['semicon', 'bank', 'car', 'finance', 'treasury3', 'ener&chem',
                  'iron', 'wti', 'gold', 'copper', 'necessities', 'healthcare',
                  'silver', 'treasury10']
    vol_cols   = ['vol20', 'vol20_vol60', 'vkospi',]

    splits = {
        'train': df[df.index <= '2019-12-31'],
        'val':   df[(df.index >= '2020-01-01') & (df.index <= '2020-12-31')],
        'test':  df[df.index >= '2021-01-01'],
    }

    total_dim = len(asset_cols) + len(asset_cols)*window + len(vol_cols) + n_regimes

    print("=" * 50)
    print(" 데이터 분할 완료")
    print("=" * 50)
    for name, split in splits.items():
        print(f"  {name:5s}: {split.index[0].date()} ~ {split.index[-1].date()} ({len(split)}일)")
    print(f"  자산 수:   {len(asset_cols)}개")
    print(f"  State 차원: {len(asset_cols)} + {len(asset_cols)}×{window} + {len(vol_cols)}"
          f" + {n_regimes} = {total_dim}")

    return {
        **splits,
        'asset_cols': asset_cols,
        'vol_cols':   vol_cols,
        'n_assets':   len(asset_cols),
        'window':     window,
    }


# ─────────────────────────────────────────────
# 포트폴리오 환경
# ─────────────────────────────────────────────

class PortfolioEnv(gym.Env):
    """
    단일 에이전트 포트폴리오 환경.
    MARL 확장 시 이 클래스를 K개 병렬로 실행하고
    PettingZoo parallel_env로 감싸면 됩니다.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data:       pd.DataFrame,
        asset_cols: list,
        vol_cols:   list,
        hmm_model:  object = None,  # 학습된 HMM 모델 (None이면 HMM-off ablation)
        hmm_scaler: object = None,  # HMM용 StandardScaler
        window:     int   = 60,
        eta:        float = 1 / 252,
        cost:       float = 0.00015,
        max_weight: float = None,   # 단일 자산 최대 비중 cap (None이면 cap 없음)
    ):
        super().__init__()

        self.data       = data.reset_index(drop=True)
        self.dates      = data.index
        self.asset_cols = asset_cols
        self.vol_cols   = vol_cols
        self.n_assets   = len(asset_cols)
        self.window     = window
        self.eta        = eta
        self.cost       = cost
        self.T          = len(data)
        self.max_weight = max_weight  # None or float ∈ (1/n_assets, 1]

        # ── HMM 관련 설정 (추가)
        self.hmm_model  = hmm_model
        self.hmm_scaler = hmm_scaler
        self.n_regimes  = hmm_model.n_components if hmm_model is not None else 0

        # ── 수익률 / 변동성 배열 추출
        self.returns_arr = data[asset_cols].values.astype(np.float32)
        self.vol_arr     = data[vol_cols].values.astype(np.float32)
        
        # 만약 kospi_ret이 데이터에 있다면 HMM용으로 따로 추출해둡니다.
        if 'kospi' in data.columns:
            self.macro_arr = data[['kospi', 'vkospi']].values.astype(np.float32)

        # ── Action Space (기존 유지)
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # ── Observation Space 수정 (기존 857 + HMM 국면 수)
        # 14(비중) + 14*60(수익률) + 3(변동성) + n_regimes(국면확률)
        obs_dim = self.n_assets + (self.n_assets * self.window) + len(vol_cols) + self.n_regimes
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self._init_state()

    # ── 내부 상태 초기화 ─────────────────────────────────────────────
    def _init_state(self):
        self.t       = self.window          # 첫 번째 유효 시점
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.A = 0.0     # DSR 이동평균 (수익률)
        self.B = 1e-8    # DSR 이동평균 (수익률²)
        self.portfolio_value  = 1.0
        self.value_history    = [1.0]
        self.weight_history   = [self.weights.copy()]
        self.return_history   = []

    # ── State 구성 ───────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        # 1. 기존 데이터 추출
        # 과거 window일 수익률: (window, n_assets) → flatten
        past_ret = self.returns_arr[self.t - self.window: self.t]  # (60, 14)
        
        # 변동성 지표 (vol20, vol20/vol60, vkospi)
        vol = self.vol_arr[self.t]  # (3,)
        vol = np.nan_to_num(vol, nan=0.0)

        # 2. HMM 국면 확률 도출 (New!)
        # self.data에서 현재 시점(self.t)의 kospi_ret과 vkospi를 가져옵니다.
        # (주의: self.t는 데이터프레임의 index와 매칭되어야 함)
        if self.hmm_model is not None:
            current_macro = self.data.iloc[[self.t]][['kospi', 'vkospi']]  # DataFrame 유지 (feature names 보존)
            scaled_macro = self.hmm_scaler.transform(current_macro)
            regime_probs = self.hmm_model.predict_proba(scaled_macro)[0]  # e.g., [0.1, 0.7, 0.2]
        else:
            regime_probs = np.array([], dtype=np.float32)

        # 3. 모든 정보를 하나로 결합
        obs = np.concatenate([
            self.weights,        # (14,) 이전 비중
            past_ret.flatten(),  # (840,) 과거 수익률
            vol,                 # (3,)  변동성 지표
            regime_probs         # (3,)  HMM 국면 확률 (추가됨!)
        ]).astype(np.float32)

        return obs

    # ── DSR 계산 (논문 수식 3.4 ~ 3.7) ──────────────────────────────
    def _compute_dsr(self, r_t: float) -> float:
        delta_A = r_t - self.A
        delta_B = r_t ** 2 - self.B

        denom = max((self.B - self.A ** 2) ** 1.5, 1e-8)
        dsr   = (self.B * delta_A - 0.5 * self.A * delta_B) / denom

        # 이동평균 업데이트
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return float(np.clip(dsr, -10.0, 10.0))  # 10은 임의 값

    # ── reset ────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        obs = self._get_obs()
        info = {"date": str(self.dates[self.t].date()) if self.t < len(self.dates) else ""}
        return obs, info

    # ── step ─────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        assert self.t < self.T, "에피소드가 이미 종료됐습니다."

        # ── Action → 포트폴리오 비중 (softmax)
        # action_space 범위에 맞게 클리핑 (foresight 테스트 시 넓은 범위 허용)
        lo = float(self.action_space.low[0])
        hi = float(self.action_space.high[0])
        a  = np.clip(action, lo, hi).astype(np.float64)
        exp_a  = np.exp(a - a.max())          # 수치 안정성
        new_w  = (exp_a / exp_a.sum()).astype(np.float32)

        # ── 단일 자산 비중 cap (옵션)
        if self.max_weight is not None and self.max_weight < 1.0:
            new_w = _cap_weights(new_w, float(self.max_weight))

        # ── 거래비용 계산 (논문은 0으로 가정)
        turnover = float(np.abs(new_w - self.weights).sum())
        cost_penalty = self.cost * turnover

        # ── 포트폴리오 수익률
        r_t = float(np.dot(new_w, self.returns_arr[self.t])) - cost_penalty

        # ── DSR 보상
        reward = self._compute_dsr(r_t)

        # ── 포트폴리오 가치 업데이트 (로그수익률 → 지수 변환)
        self.portfolio_value *= np.exp(r_t)

        # ── 히스토리 기록
        self.weights = new_w
        self.value_history.append(self.portfolio_value)
        self.weight_history.append(new_w.copy())
        self.return_history.append(r_t)

        # ── 시점 이동
        self.t += 1
        terminated = (self.t >= self.T)
        truncated  = False

        # ── 다음 State
        obs  = self._get_obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32)

        info = {
            "portfolio_return":  r_t,
            "portfolio_value":   self.portfolio_value,
            "weights":           new_w,
            "turnover":          turnover,
            "date":              str(self.dates[self.t - 1].date()) if self.t - 1 < len(self.dates) else "",
        }

        return obs, reward, terminated, truncated, info

    # ── 성과 지표 계산 ───────────────────────────────────────────────
    def get_metrics(self) -> dict:
        """에피소드 종료 후 성과 지표 반환."""
        r = np.array(self.return_history)
        if len(r) == 0:
            return {}

        cum_return = float(self.portfolio_value - 1.0)
        sharpe     = float(r.mean() / (r.std() + 1e-8) * np.sqrt(252))

        cum_values = np.array(self.value_history)
        drawdowns  = cum_values / np.maximum.accumulate(cum_values) - 1
        mdd        = float(drawdowns.min())

        weights_arr = np.array(self.weight_history)
        turnover    = float(np.mean(
            np.abs(np.diff(weights_arr, axis=0)).sum(axis=1)
        )) if len(weights_arr) > 1 else 0.0

        return {
            "cumulative_return": cum_return,
            "sharpe_ratio":      sharpe,
            "mdd":               mdd,
            "turnover":          turnover,
            "n_steps":           len(r),
        }


# ─────────────────────────────────────────────
# Sanity Check (논문 Section 4.1.1)
# ─────────────────────────────────────────────

def sanity_check(data, hmm_model, hmm_scaler):
    print("\n" + "=" * 50)
    print(" Sanity Check (foresight agent)")
    print("=" * 50)
    
    # 2. 환경 생성 시 모델과 스케일러를 넣어줍니다.
    env = PortfolioEnv(
        data=data['test'],
        asset_cols=data['asset_cols'],
        vol_cols=data['vol_cols'],
        hmm_model=hmm_model,   # 추가
        hmm_scaler=hmm_scaler, # 추가
        window=data['window']
    )
    
    obs, _ = env.reset()
    done   = False
    correct_actions = 0
    total_steps     = 0

    while not done:
        # 미래 수익률을 직접 확인 (foresight)
        future_returns = env.returns_arr[env.t]

        # 최고 수익 자산에 극단적 logit 부여
        action = np.full(env.n_assets, -300.0, dtype=np.float32)
        best   = int(np.argmax(future_returns))
        action[best] = 300.0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # softmax 후 best 자산 비중이 99% 이상인지 확인
        w = info['weights']
        if w[best] > 0.99:
            correct_actions += 1
        total_steps += 1

    metrics = env.get_metrics()
    accuracy = correct_actions / total_steps * 100

    print(f"  최고 자산 집중 정확도: {accuracy:.1f}% ({correct_actions}/{total_steps})")
    print(f"  누적 수익률:          {metrics['cumulative_return']*100:.1f}%")
    print(f"  Sharpe:              {metrics['sharpe_ratio']:.4f}")
    print(f"  → {'PASS' if accuracy > 99 else 'FAIL'}: 환경 구현 정상" if accuracy > 99
          else f"  → FAIL: 환경 점검 필요")

    return accuracy > 99


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. 데이터 로드
    data_dict = load_and_split("ppo_with_hmm/etf_log_returns_cleaned.csv", window=60, n_regimes=3)

# 2. HMM 모델과 스케일러를 먼저 학습시킵니다.
# 여기서 나온 hmm_model과 hmm_scaler를 env에 넣어줄 겁니다.
    print("HMM 학습 중...")
    hmm_model, hmm_scaler = train_market_hmm(data_dict['train'], n_regimes=3)

# 3. 환경(env)을 생성할 때 위에서 만든 모델들을 인자로 전달합니다.
    train_env = PortfolioEnv(
        data       = data_dict['train'],
        asset_cols = data_dict['asset_cols'],
        vol_cols   = data_dict['vol_cols'],
        hmm_model  = hmm_model,   # ★ 중요: 위에서 학습시킨 모델 주입
        hmm_scaler = hmm_scaler,  # ★ 중요: 위에서 학습시킨 스케일러 주입
        window     = 60
    )

# 4. 이제 확인해보면 에러 없이 돌아갈 겁니다!
    print("환경 리셋 테스트 시작...")
    obs, info = train_env.reset()
    print(f"리셋 성공! obs shape: {obs.shape}")

    # ── 3. 기본 동작 확인
    print("\n" + "=" * 50)
    print(" 환경 기본 동작 확인")
    print("=" * 50)
    obs, info = train_env.reset()
    expected_dim = train_env.observation_space.shape[0]
    print(f"  obs shape:    {obs.shape}  (예상: ({expected_dim},))")
    print(f"  obs NaN 여부: {np.any(np.isnan(obs))}")
    print(f"  obs 범위:     [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"  초기 비중 합: {train_env.weights.sum():.6f}  (1이면 정상)")

    # 랜덤 액션 1스텝
    action = train_env.action_space.sample()
    obs2, reward, terminated, truncated, info = train_env.step(action)
    print(f"\n  랜덤 액션 후:")
    print(f"  reward:          {reward:.6f}")
    print(f"  비중 합:         {info['weights'].sum():.6f}")
    print(f"  포트폴리오 가치: {info['portfolio_value']:.6f}")
    print(f"  날짜:            {info['date']}")

    # 전체 에피소드 실행
    print("\n" + "=" * 50)
    print(" 전체 에피소드 (랜덤 정책) 실행")
    print("=" * 50)
    obs, _ = train_env.reset()
    done   = False
    while not done:
        action = train_env.action_space.sample()
        obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated

    metrics = train_env.get_metrics()
    print(f"  총 스텝:    {metrics['n_steps']}일")
    print(f"  누적수익률: {metrics['cumulative_return']*100:.2f}%")
    print(f"  Sharpe:    {metrics['sharpe_ratio']:.4f}")
    print(f"  MDD:       {metrics['mdd']*100:.2f}%")
    print(f"  Turnover:  {metrics['turnover']:.4f}")

    # ── 4. Sanity Check
    passed = sanity_check(data_dict, hmm_model, hmm_scaler)
