# 한국 시장 포트폴리오 강화학습 — Claude Code 인수인계

> 이 파일은 Claude Code가 프로젝트 컨텍스트를 즉시 파악하고 작업을 이어받기 위한 문서입니다.

---

## 프로젝트 한 줄 요약

PPO 기반 강화학습으로 14개 코스피 ETF 포트폴리오를 자동 운용하는 시스템.
TU Delft M. Choi (2024) 논문을 한국 시장 데이터에 적용하고, HMM 국면 감지를 추가한 버전.

---

## 현재 완료된 것 (Done)

### 1. 환경 구현 (`env.py`) — 완료, 검증됨
- `PortfolioEnv(gym.Env)` 구현 완료
- State: `(860,)` = 비중(14) + 과거 60일 수익률(840) + 변동성(3) + HMM 국면 확률(3)
- Action: logit(14) → softmax → 포트폴리오 비중
- Reward: Differential Sharpe Ratio (DSR), `[-10, 10]` clip
- 거래비용: `cost=0.00015` (turnover 기반)
- Sanity Check (foresight agent) PASS 확인

### 2. HMM 국면 감지 (`HMM.py`) — 완료
- `train_market_hmm(train_df, n_regimes=3)` → `(hmm_model, hmm_scaler)` 반환
- 훈련 데이터로만 학습 (Data Leakage 없음)
- `_get_obs()` 내부에서 State에 concat

### 3. 데이터 (`etf_log_returns_cleaned.csv`) — 완료
- 기간: 2013-08-06 ~ 2026-04-03
- 자산: 14개 코스피 ETF (semicon, bank, car, finance, treasury3, ener&chem, iron, wti, gold, copper, necessities, healthcare, silver, treasury10)
- 변동성 컬럼: `vol20`, `vol20_vol60`, `vkospi`
- 분할: train (≤2019-12-31) / val (2020) / test (≥2021-01-01)

---

## 지금 당장 해야 할 것 (Todo)

### Task 1: `train.py` 작성 — Rolling Window + PPO

**구현 스펙:**

```
Rolling Window 구조 (논문 Section 3.3.1 기준):
  Round 1: Train 2013-2017 (5년) / Val 2018 / Test 2019
  Round 2: Train 2014-2018 (5년) / Val 2019 / Test 2020
  Round 3: Train 2015-2019 (5년) / Val 2020 / Test 2021
  ...
  (데이터 끝까지 1년씩 슬라이딩)

각 Round마다:
  1. HMM을 해당 훈련 윈도우 데이터로 재학습 (Data Leakage 방지)
  2. PPO 에이전트 5개 시드(0~4)로 학습
  3. Val DSR 가장 높은 시드 선택
  4. 선택된 시드를 다음 Round 초기 모델로 인계 (논문 방식)
  5. Test 기간 성과 기록
```

**PPO 하이퍼파라미터 (논문 Table 3.2 + Appendix A 기준):**

```python
PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    gamma         = 0.9,
    gae_lambda    = 0.9,
    clip_range    = 0.25,
    ent_coef      = 0.035,
    n_steps       = 2048,
    batch_size    = 64,
    n_epochs      = 2,
    policy_kwargs = dict(net_arch=[64, 64]),
    verbose       = 0,
)
TOTAL_TIMESTEPS = 1_454_400  # 논문 Appendix A
N_SEEDS = 5
```

**Callback 스펙 (`ValidationCallback`):**

```python
# 매 eval_freq 스텝마다 val 환경에서 greedy 평가
# best Val DSR 기준으로 모델 저장
# 기록: sharpe, cumulative_return, step 번호
```

**저장 구조:**

```
models/
  round_01/
    seed_0.zip
    seed_1.zip
    seed_2.zip  ← best (예시)
    seed_3.zip
    seed_4.zip
    best_seed.txt  ← "2" 기록
  round_02/
    ...
results/
  rolling_window_results.csv  ← round별 test 성과 집계
```

---

### Task 2: `evaluate.py` 작성

**구현 스펙:**

```python
# 1. rolling_window_results.csv 로드
# 2. round별 test 성과 집계 (Sharpe, 누적수익률, MDD, Turnover)
# 3. KOSPI 벤치마크 성과와 나란히 비교
# 4. 시각화 4종:
#    (a) 누적 수익률 시계열 (RL vs KOSPI vs 균등가중)
#    (b) Round별 Sharpe Ratio 막대그래프
#    (c) Round별 MDD
#    (d) Round별 포트폴리오 비중 분포 (히트맵)
# 5. 최종 요약 테이블 출력 및 CSV 저장
```

---

### Task 3: Transformer 기반 데이터 표현 교체

**현재 방식 (MLP):**
```python
# env.py _get_obs():
past_ret = self.returns_arr[self.t - self.window: self.t]  # (60, 14)
obs = np.concatenate([..., past_ret.flatten(), ...])       # flatten → 순서 무시
```

**목표 방식 (Transformer):**

```
아이디어:
  60일 × 14자산 수익률 시퀀스를 Transformer Encoder로 압축
  → 압축된 벡터를 MLP Actor/Critic에 입력

구현 방법:
  Option A — Feature Extractor 교체 (권장)
    SB3의 policy_kwargs["features_extractor_class"]에
    커스텀 TransformerFeatureExtractor 주입
    → 나머지 PPO 코드 변경 없음

  Option B — 별도 전처리
    매 스텝 Transformer forward pass로 임베딩 생성
    → State를 임베딩 벡터로 교체
```

**TransformerFeatureExtractor 스펙:**

```python
class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    입력: obs (860,) → 분리 후
      - past_ret: (60, 14) 시퀀스로 reshape
      - 나머지: (17,) 비중 + 변동성 + HMM
    처리:
      - past_ret → Positional Encoding → Transformer Encoder (2 layer, 4 head)
      - Transformer 출력 → Global Average Pooling → (d_model,)
      - (d_model,) + (17,) → concat → 최종 feature 벡터
    출력: (features_dim,) → Actor/Critic MLP 입력
    """
    def __init__(self, observation_space, features_dim=128, d_model=64, nhead=4, num_layers=2):
        ...
```

**주의사항:**
- `d_model`은 반드시 `nhead`의 배수
- 입력 시퀀스 길이(60)가 고정이므로 Positional Encoding은 학습 가능한 파라미터로
- `features_dim`은 PPO `net_arch` 입력 차원과 맞춰야 함

---

## 추후 계획 (논문 확장)

### Phase 2: SAC + Replay Buffer (단일 에이전트)
- PPO → SAC로 알고리즘 교체
- Off-Policy 특성 활용: 과거 데이터(위기 구간 등) 반복 학습
- `env.py`는 그대로 재사용 가능 (SB3 SAC 호환)
- 비교 실험: PPO vs SAC (동일 데이터, 동일 평가 방식)

### Phase 3: Multi-Agent IPPO (다중 에이전트)
- `PettingZoo` parallel_env로 env.py 확장
- `BenchMARL` IPPO 에이전트 연결
- 다양성 페널티 두 가지 실험:
  - Correlation: `corr(w_A, w_B)` — 직관적, 수렴 빠름
  - Total Variation Distance: `0.5 × Σ|w_A,i - w_B,i|`
- 다양성 가중치 λ ∈ [0.0, 0.1, ..., 0.9] sweep
- 논문 결과: λ↑ → 에이전트 간 상관↓ → Sharpe↑, Turnover↓

---

## 파일 구조 (현재 + 예정)

```
project/
├── env.py                          ← 완료 (수정 금지)
├── HMM.py                          ← 완료 (수정 금지)
├── etf_log_returns_cleaned.csv     ← 완료
│
├── train.py                        ← 작성 필요 (Task 1)
├── evaluate.py                     ← 작성 필요 (Task 2)
├── feature_extractor.py            ← 작성 필요 (Task 3, Transformer)
│
├── models/                         ← train.py 실행 후 생성
│   ├── round_01/
│   └── round_02/
│
├── results/                        ← evaluate.py 실행 후 생성
│   ├── rolling_window_results.csv
│   └── figures/
│
└── CLAUDE.md                       ← 이 파일
```

---

## 핵심 설계 결정 사항 (변경 금지)

| 결정 사항 | 이유 |
|-----------|------|
| State에 시장 수익률 제거(KOSPI 차감)한 값 입력 | 섹터 고유 시그널만 학습하기 위해 |
| Reward: DSR (Differential Sharpe Ratio) | Sharpe를 매 스텝 보상으로 분해 가능한 유일한 방법 |
| DSR clip: `[-10, 10]` | 극단적 보상이 학습 불안정 유발 |
| Action space: `[-10, 10]` logit | softmax 온도 조절 효과, 논문 Section 4.1.1 참조 |
| HMM을 환경이 아닌 `_get_obs()`에 배치 | HMM은 국면 감지 도구이지 시장 시뮬레이터가 아님 |
| Val 기준 best 모델 선택 | Test 기간에 Data Leakage 방지 |
| HMM은 각 Rolling Window 훈련 데이터로 재학습 | 미래 데이터 누출 방지 |

---

## 의존성

```bash
pip install gymnasium stable-baselines3 hmmlearn pandas numpy torch
pip install pettingzoo benchmarl  # Phase 3 다중 에이전트 시
```

---

## 참고 논문 수식 위치

| 수식 | 논문 위치 | 내용 |
|------|----------|------|
| State 구성 | Section 3.2.1, Eq 3.1 | 관측 행렬 정의 |
| Softmax Action | Section 3.2.2, Eq 3.2 | 포트폴리오 비중 변환 |
| DSR | Section 3.2.3, Eq 3.4~3.7 | Differential Sharpe Ratio |
| PPO Clip | Section 2.1.4, Eq 2.15 | PPO 목적함수 |
| Diversity Penalty | Section 3.4.1, Eq 3.9~3.10 | 다양성 보상 설계 |
| TV Distance | Section 3.4.1, Eq 3.8 | 총변동 거리 |
| Rolling Window | Section 3.3.1, Figure 3.3 | 학습 방법론 |
