# train_and_save_model_v5.py
> v3에서 변형한 버젼

---

## 주요 특징 및 개선 사항

### 1. K-Fold 교차 검증 도입 (가장 큰 변화)
   - 단일 Train/Test 분리 대신 StratifiedKFold를 사용하여 데이터셋 전체를
     훈련 및 검증에 활용함으로써 모델 성능 평가의 신뢰도를 대폭 향상
   - 각 Fold마다 모델을 새로 학습하고 평가하여 평균 정확도 및 표준편차 제공
   - 가장 높은 검증 정확도를 보인 모델을 최종 모델로 저장

### 2. L2 정규화 추가
   - LSTM 및 Dense 레이어에 L2 정규화(kernel_regularizer=l2(0.001))를 추가하여
     가중치가 너무 커지는 것을 방지하고 과적합 억제 강화

### 3. Dropout 비율 조정
   - Bidirectional LSTM 레이어의 Dropout 비율을 0.5에서 0.6으로 상향 조정하여
     과적합 방지 효과를 더욱 강화
   - Dense 레이어의 Dropout 비율도 0.4에서 0.5로 상향 조정

### 4. 학습 전략 유지
   - 학습률: 0.001
   - 배치 크기: 16
   - EarlyStopping Patience: 20
   - ReduceLROnPlateau Patience: 7

---

## v3와의 주요 차이점

### v3:
  데이터 로드 → Train/Test 분리 (단일) → Train만 증강 → 모델 학습 및 평가 (단일 Test셋)
  문제점: 적은 Test 데이터로 인한 평가 신뢰도 저하, 과적합 여부 판단 어려움

### v5 (개선):
  데이터 로드 → StratifiedKFold 분할 (N개 Fold)
  → 각 Fold마다: Train/Test 분리 → Train만 증강 → 모델 학습 및 평가
  → 모든 Fold 결과 집계 → 가장 좋은 모델 저장
  개선점: 모델 일반화 성능을 더 정확하게 측정, L2 정규화 및 Dropout 강화로 과적합 추가 억제

---

## 학습 로그 분석

### 핵심 요약
- 평균 테스트 정확도: 82.02%
- 표준 편차: ± 3.86%
- 가장 높은 정확도: 86.36% (Fold 2)
- 클래스 수: 5개

### 결과 분석
1. 높은 평균 정확도 (82.02%): 5개의 클래스 분류 문제에서 80%가 넘는 평균 정확도는 매우 훌륭한 결과

2. 낮은 표준 편차 (± 3.86%): 표준 편차가 낮다는 것은 각 Fold별 정확도 편차가 크지 않다는 것을 의미(과적합x, 일관된 성능, 일반화 성능 우수)

3. K-Fold 교차 검증의 효과: K-Fold 교차 검증을 통해 얻은 이 결과는 단일 Train/Test 분할보다 훨씬 신뢰 가능

4. 과적합 방지 전략의 성공: L2 정규화와 Dropout 비율 조정이 효과적으로 작동하여 과적합을 억제, 일반화 성능 향상

---

### 학습 로그

```text
python-project202508  python .\train_predict_model\train_and_save_model_v5.py

2025-10-06 16:00:28.867154: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
============================================================
train_and_save_model_v5.py
============================================================
K-Fold 교차 검증 및 과적합 방지 강화 적용
============================================================

============================================================
Step 1: 데이터 로딩
============================================================
원본 데이터: 111개
클래스 수: 5
클래스: ['무한', '미국', '수학', '월세', '일요일']

============================================================
Step 2: 원본 데이터 패딩
============================================================
Max sequence length: 149
Input dimension: 114
Padded shape: (111, 149, 114)

============================================================
Step 3: K-Fold 교차 검증 시작
============================================================
모델 일반화 성능을 더 신뢰성 있게 평가합니다.

--- Fold 1/5 시작 ---
Fold 1 Train 원본: 88개
Fold 1 Test 원본: 23개
Fold 1: 훈련 데이터만 증강 중...
Fold 1 증강 후 Train: (528, 149, 114)
Fold 1 증강 배수: 6.0배
Fold 1: 모델 생성 중...
2025-10-06 16:00:32.946063: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
C:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\keras\src\layers\rnn\bidirectional.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Fold 1: 모델 학습 시작...
Fold 1: 모델 학습 완료.
--- Fold 1 결과: 테스트 손실: 0.8393, 테스트 정확도: 0.7826 (78.26%) ---

--- Fold 2/5 시작 ---
Fold 2 Train 원본: 89개
Fold 2 Test 원본: 22개
Fold 2: 훈련 데이터만 증강 중...
Fold 2 증강 후 Train: (534, 149, 114)
Fold 2 증강 배수: 6.0배
Fold 2: 모델 생성 중...
Fold 2: 모델 학습 시작...
Fold 2: 모델 학습 완료.
--- Fold 2 결과: 테스트 손실: 0.6697, 테스트 정확도: 0.8636 (86.36%) ---

--- Fold 3/5 시작 ---
Fold 3 Train 원본: 89개
Fold 3 Test 원본: 22개
Fold 3: 훈련 데이터만 증강 중...
Fold 3 증강 후 Train: (534, 149, 114)
Fold 3 증강 배수: 6.0배
Fold 3: 모델 생성 중...
Fold 3: 모델 학습 시작...
Fold 3: 모델 학습 완료.
--- Fold 3 결과: 테스트 손실: 0.6642, 테스트 정확도: 0.7727 (77.27%) ---

--- Fold 4/5 시작 ---
Fold 4 Train 원본: 89개
Fold 4 Test 원본: 22개
Fold 4: 훈련 데이터만 증강 중...
Fold 4 증강 후 Train: (534, 149, 114)
Fold 4 증강 배수: 6.0배
Fold 4: 모델 생성 중...
Fold 4: 모델 학습 시작...
Fold 4: 모델 학습 완료.
--- Fold 4 결과: 테스트 손실: 0.5567, 테스트 정확도: 0.8636 (86.36%) ---

--- Fold 5/5 시작 ---
Fold 5 Train 원본: 89개
Fold 5 Test 원본: 22개
Fold 5: 훈련 데이터만 증강 중...
Fold 5 증강 후 Train: (534, 149, 114)
Fold 5 증강 배수: 6.0배
Fold 5: 모델 생성 중...
Fold 5: 모델 학습 시작...
Fold 5: 모델 학습 완료.
--- Fold 5 결과: 테스트 손실: 0.8413, 테스트 정확도: 0.8182 (81.82%) ---

============================================================
K-Fold 교차 검증 완료!
============================================================
모든 Fold의 테스트 정확도: ['78.26%', '86.36%', '77.27%', '86.36%', '81.82%']
평균 테스트 정확도: 82.02% ± 3.86%

============================================================
Step 9: 가장 좋은 모델 저장
============================================================
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
가장 높은 정확도 (86.36%)를 보인 Fold 2의 모델 저장 완료:
  - trained_model/trained_model_v5\sign_language_model.h5
  - trained_model/trained_model_v5\model_info.pkl

============================================================
학습 완료!
============================================================

예측 실행 방법:
   python .\train_predict_model\predict_sign_language.py --video "test.mp4" --model-dir "trained_model/trained_model_v5" --show-probs

============================================================

```

---

## 예측 로그
### 실제 정확도: 40% (2/5)
> 수학, 미국을 제외하고는 모델이 잘 작동하지 않음

### 수학 예측
- 수학 98.43%
- 미국 0.81%
- 무한 0.72%
- 일요일 0.04%
- 월세 0.00%

### 일요일 예측
- 미국 76.23%
- 무한 22.77%
- 일요일 0.87%
- 수학 0.11%
- 월세 0.01%

### 미국 예측
- 미국 82.73%
- 무한 10.56%
- 일요일 6.60%
- 수학 0.09%
- 월세 0.03%

### 무한 예측
- 미국 81.99%
- 무한 10.08%
- 일요일 7.81%
- 수학 0.09%
- 월세 0.03%

### 월세 예측
- 일요일 97.10%
- 미국 1.21%
- 수학 0.86%
- 월세 0.75%
- 무한 0.09%

---

## 예측 로그 분석

### 1. 데이터 불균형 또는 부족 (가장 큰 문제)
- 문제점
  - '수학'과 '미국'은 잘 맞추지만, '일요일', '무한', '월세'는 다른 단어로 오분류
  - 특히 '일요일'과 '월세'는 거의 예측 불가, '미국'이나 '무한'으로 오분류되는 경향이 강함
  - 해당 단어들의 훈련 데이터가 부족 혹은 특징이 충분히 학습되지 않았을 가능성 높음
- 해결책
  - 데이터 증강 강화:  '일요일', '무한', '월세'에 해당하는 수어 데이터의 수 증가 필요(다양한 조건의 원본 데이터 추가)
  - 데이터 품질 개선: 특정 단어의 키포인트 추출이 불안정하거나 노이즈가 많을 수 있음(비디오 확인, 키포인트 추출 여부 검토)

### 2. 클래스 간 유사성 (오분류의 원인)
- 문제점
  - '일요일', '무한'이 모두 '미국'으로 오분류되는 경향이 강함(모델이 이들의 미묘한 차이를 구분하기 어렵다는 것 의미)
  - '일요일' 예측 시 '미국' 76.23%
  - '무한' 예측 시 '미국' 81.99%
- 해결책
  - 특징 분석: '미국', '무한', '일요일' 수어 동작의 특징을 다시 한번 비교 분석하여, 모델이 어떤 부분에서 혼동하는지 파악해야할 필요성 있음
  - 데이터 라벨링 재확인: 혹시 훈련 데이터 내에서 유사한 동작의 라벨링이 잘못된 경우가 없는지 확인

### 결론
- 현재 가장 유력한 원인은 '일요일', '무한', '월세'에 대한 훈련 데이터의 양과 질 문제, 그리고 '미국'과의 시각적 유사성으로 인한 혼동

- 우선적으로는 '일요일', '무한', '월세'에 해당하는 수어 비디오 데이터를 더 많이 수집, 다양한 환경에서 촬영된 데이터를 포함하여 훈련 데이터셋을 확장할 필요가 있음
