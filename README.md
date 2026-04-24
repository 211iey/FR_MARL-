# MARL 

## 프로젝트 한 줄 요약

PPO 기반 강화학습으로 14개 코스피 ETF 포트폴리오를 자동 운용하는 시스템.
TU Delft M. Choi (2024) 논문을 한국 시장 데이터에 적용하고, HMM 국면 감지를 추가한 버전.

---

## 핵심 설계 결정 사항 

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
