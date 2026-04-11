# FR_MARL — KOSPI 섹터 ETF 포트폴리오 강화학습

KOSPI 섹터 ETF 포트폴리오를 학습 기반으로 운용하는 프로젝트.
현재는 **단일 에이전트 PPO 베이스라인** 단계이고, 이후 멀티 에이전트(MARL)로 확장 예정.

---

## 1. 데이터

- **자산 14종** (`script/data/preprocess.py`의 `ASSETS`)
  semicon, bank, car, finance, treasury3, ener&chem, iron, wti, gold, copper,
  necessities, healthcare, silver, treasury10
- **시장 변수 2종**: `vkospi`, `vol20_kospi`
- **기간**: 2012-01 ~ 현재 (실데이터)
- **파일** (`data/`):
  - `etf_log_Returns(2012 start).csv` — 14 자산 일별 로그수익률
  - `excess_returns(2012 start).csv` — 초과수익률 + vol20/vol60 + vkospi/vol20_kospi (전처리 결과 통합본)
  - `vkospi.csv` — VKOSPI 원본

## 2. 전처리

`script/add_vol_vkospi.py` 가 한 번 실행되어 `excess_returns(...).csv` 에 다음을 합쳐둠:

1. 자산별 `vol20_*` (20일 표준편차), `vol60_*` (60일 표준편차)
2. VKOSPI 종가 (`vkospi`)
3. yfinance에서 받은 KOSPI 지수의 `vol20_kospi`

학습 시점 로딩은 `script/data/preprocess.py`:

- `load_dataset(csv_path)` — 필요한 컬럼만 뽑고 NaN 워밍업 구간(60일) 제거
- `split(df, train_end="2021-12-31", valid_end="2022-12-31")`
  - **train**: ~2021-12-31
  - **valid**: 2022-01 ~ 2022-12-31
  - **test**: 2023-01 ~
- `normalizer(train_df, cols)` — train 구간 기준 z-score 평균/표준편차

## 3. 환경 — `KospiPortfolioEnv`

`script/env/kospi_env.py`. gym-like 단일 에이전트 환경.

**Observation (dict)**

| key      | shape       | 내용                                              |
|----------|-------------|---------------------------------------------------|
| `asset`  | `(M, L, N)` | 자산별 z-score 윈도우 — row 0 은 cash(0 채움)     |
| `market` | `(L, K)`    | 시장 변수 윈도우 (`vkospi`, `vol20_kospi`)        |
| `weight` | `(M,)`      | 직전 step post-trade 가중치                       |

- `M = 15` (14 자산 + cash)
- `L = 20` (윈도우 길이)
- `N = 3` (return, vol20, vol60)
- `K = 2` (vkospi, vol20_kospi)

**Action**: `(M,)` 심플렉스 벡터. 어떤 numpy 배열을 넘겨도 환경 내부 `_project_simplex` 가 비음+합=1 로 사영.

**Reward**:
```
r_p    = w_new · r_t  -  cost · turnover
reward = log(1 + r_p)
turnover = sum(|w_new[1:] - w_prev[1:]|)   # cash 제외
```

**Cost**: `0.0005` (turnover 1당 5bp).

**Weight drift**: step 후 `w_post = w_new (1+r) / (1+r_p)` 로 수익에 따른 가중치 변화 반영.

성능상 이유로 `_build_tensors` 에서 모든 step 의 obs/return 텐서를 미리 만들어 둠 (참조 레포 방식).

## 4. 베이스라인

### 4.1 PPO 에이전트 — `script/agents/ppo_agent.py`

**모델 (`ActorCritic`)**

- **Asset encoder**: `Conv2d(N → hidden, kernel=(1, L))` 로 자산별 시계열을 한 번에 압축 → `(B, M, hidden)` → flatten
- **Market encoder**: `Linear(L*K → hidden)` 의 단순 MLP
- **Concat**: `[asset_feat, market_feat, prev_weight]` → shared MLP(`hidden`)
- **Actor head**: `Linear(hidden → M)` → `softplus + 1e-3` 로 **Dirichlet concentration α** 출력
- **Critic head**: `Linear(hidden → 1)` 스칼라 가치

행동 분포는 Dirichlet 이라 자연스럽게 simplex 위에 샘플이 떨어짐.
평가(`evaluate`) 시에는 Dirichlet 평균(α / Σα) 을 결정론적으로 사용.

**알고리즘**

- 표준 PPO (clip)
- GAE: `gamma=0.99`, `lam=0.95`
- Clip ratio: `0.2`
- 매 iter: full episode rollout → GAE → `epochs=4` × minibatch(`batch_size=64`) 업데이트
- Loss: `actor_loss + 0.5 * value_loss - 0.01 * entropy`
- 옵티마이저: Adam (`lr=3e-4`)

**기본 하이퍼파라미터** (`config.json`)

```json
{
  "data":  { "csv_path": "data/excess_returns(2012 start).csv",
             "train_end": "2021-12-31", "valid_end": "2022-12-31" },
  "env":   { "window": 20, "cost": 0.0005 },
  "agent": { "hidden": 64, "lr": 3e-4, "gamma": 0.99, "lam": 0.95,
             "clip": 0.2, "epochs": 4, "batch_size": 64 },
  "train": { "n_iters": 50, "device": "cpu", "seed": 0 }
}
```

### 4.2 클래식 베이스라인 — `script/benchmarks/classical.py`

같은 환경 위에서 `rollout(env, policy)` 로 평가.

- **`EqualWeightPolicy`** — 비현금 14 자산에 균등 가중치 `1/(M-1)`. (cash 포함 옵션 있음)
- **`MinVariancePolicy`** — obs 의 z-score 수익 윈도우 `(M-1, L)` 로 공분산을 추정하고, SLSQP 로 long-only 최소분산 가중치를 매 step 풂.

### 4.3 평가 지표

PPO `evaluate()` 와 `rollout()` 모두 동일 dict 반환:

| key            | 정의                                              |
|----------------|---------------------------------------------------|
| `reward_sum`   | Σ log-return                                      |
| `cum_return`   | `exp(reward_sum) - 1`                             |
| `mean_return`  | 일별 포트폴리오 수익률 평균                       |
| `sharpe`       | `mean / std × √252` (연율화, rf=0)                |
| `avg_turnover` | 일별 turnover 평균                                |

## 5. 프로젝트 구조 (source layout)

```
FR_MARL-/
├── README.md                   ← 본 문서
├── config.json                 ← 학습/환경 하이퍼파라미터
├── data/                       ← 원본/전처리된 CSV (코드 아님)
│   ├── etf_log_Returns(2012 start).csv
│   ├── excess_returns(2012 start).csv
│   └── vkospi.csv
├── reference/                  ← 참고용 외부 레포
└── script/                     ← 모든 .py 소스 (source layout)
    ├── add_vol_vkospi.py       ← 전처리 1회 실행 스크립트
    ├── Baseline(MARL).ipynb    ← 초기 노트북
    ├── data/preprocess.py      ← load_dataset / split / normalizer
    ├── env/kospi_env.py        ← KospiPortfolioEnv
    ├── benchmarks/classical.py ← EqualWeight / MinVariance / rollout
    ├── agents/ppo_agent.py     ← ActorCritic + PPOAgent
    ├── train.py                ← PPO 학습 엔트리포인트
    └── evaluate.py             ← PPO + 클래식 비교 엔트리포인트
```

`python script/train.py` 처럼 **프로젝트 루트에서** 실행하면
`sys.path[0] = script/` 가 되어 `from data.preprocess import …` 등의 import 가 그대로 동작함.
`config.json` 의 `csv_path` 는 루트 기준 상대경로이므로 같은 이유로 OK.

## 6. 사용법

학습:
```bash
python script/train.py --config config.json --save runs/ppo_baseline.pt
```
- `data/excess_returns(2012 start).csv` 로딩 → 환경 3종(train/valid/test) 생성
- `n_iters` 만큼 PPO 학습 → `runs/ppo_baseline.pt` 저장
- 학습 후 자동으로 valid 평가 출력

평가 (PPO + 클래식 베이스라인 동시 비교):
```bash
python script/evaluate.py --split test --ckpt runs/ppo_baseline.pt
```
- `--split` 은 `train | valid | test`
- 체크포인트가 없으면 PPO 부분만 스킵하고 클래식만 출력

## 7. 현재 진행 상황 (2026-04-11 기준)

- [x] 원본 데이터 수집 (KOSPI 14 섹터 ETF, KOSPI/VKOSPI)
- [x] `add_vol_vkospi.py` 로 vol20/vol60 + vkospi/vol20_kospi 통합 CSV 생성
- [x] 단일 에이전트 환경 `KospiPortfolioEnv` 구현 (precomputed tensor 방식)
- [x] PPO + Dirichlet 정책 베이스라인 (`PPOAgent`) 구현
- [x] 클래식 베이스라인 (Equal Weight, Min Variance) 구현
- [x] `train.py` / `evaluate.py` 엔트리포인트 + `config.json` 분리
- [x] source layout 정리 (`script/` 하위로 통합)
- [ ] PPO 학습 결과 vs 클래식 베이스라인 정량 비교 기록
- [ ] 멀티 에이전트(MARL) 환경/에이전트 설계
