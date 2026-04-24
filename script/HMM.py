import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

def train_market_hmm(train_data, n_regimes=3):
    # HMM이 관찰할 변수 선택 (KOSPI 수익률과 VKOSPI)
    features = train_data[['kospi', 'vkospi']].dropna()
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Gaussian HMM 모델 생성 및 학습
    model = GaussianHMM(n_components=n_regimes, covariance_type="full", n_iter=100)
    model.fit(scaled_features)
    
    return model, scaler

# 예시: 학습 데이터로 HMM 사전 학습
# hmm_model, hmm_scaler = train_market_hmm(data['train'])