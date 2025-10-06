# train_and_save_model_v2.py

## 🎯 주요 변경 사항
1. Bidirectional LSTM + Attention - 양방향 학습으로 패턴 파악 향상
2. 데이터 증강 15배 - 111개 → 약 1,776개 (충분한 학습 데이터)
3. 학습률 스케줄러 - 학습이 정체되면 자동으로 학습률 감소
4. 배치 사이즈 16 - 더 안정적인 학습
5. Patience 30 - 충분한 학습 시간 제공
6. 개선된 데이터 증강 전략 (4가지)
   - 시간적 스케일링(수어 동작의 속도 변화)
   - 가우시안 노이즈 추가(손 떨림이나 키포인트 추출 오차)
   - 위치 이동(카메라 앞에서 약간 왼쪽/오른쪽/위/아래로 이동)
   - 스케일 변화(카메라와의 거리 변화)
7. 개선된 증강의 장점:
  - 각 증강을 개별적으로 적용 (한 번에 하나씩)
  - 더 작고 현실적인 변화 범위
  - 순환 방식으로 모든 증강이 균등하게 적용
  - 원본 데이터의 본질적 특성 보존


---

## 🚨 심각한 과적합(Overfitting) 문제 발생
> 100% 정확도는 좋은 게 아니라 나쁜 신호

## 문제 분석

### 1️⃣ 명백한 과적합 증거
```text
Epoch 7: val_accuracy 100%, val_loss 0.0177
Epoch 8-50: 계속 100% 유지
테스트: 100%, loss 0.0001
```
모델이 데이터를 학습한 게 아니라 암기함.

### 2️⃣ 왜 이런 일이 발생했나?
핵심 문제: 데이터 증강이 잘못됨
```text
# 원본: 111개
# 증강: 1776개 (16배)
# 문제: 모두 같은 111개에서 파생된 데이터
```
- train/test를 나누기 전에 증강했기 때문에
- 같은 원본에서 나온 증강 데이터가 train/test에 모두 포함됨
- 모델이 "아, 이건 ID 1번 영상의 변형이네" 하고 암기

예시:
```text
원본 영상 A → 증강 16개 생성
  → 13개는 train에, 3개는 test에
  → 모델이 train에서 A를 봤으니 test의 A 변형도 쉽게 맞춤
```
> 실제 새로운 영상에는 전혀 작동하지 않을 가능성이 높습니다.

---

## ✅ 올바른 해결 방법

### 방법 1: Train/Test 분리 후 증강 ⭐⭐⭐ 필수

### 방법 2: 더 간단한 모델 사용
현재 모델이 너무 복잡합니다 (438K 파라미터 vs 1776개 샘플)

### 방법 3: 데이터 증강 줄이기
15배 → 5배로 감소

---

## 학습 로그

```text
python-project202508  python .\LSTM_model\train_and_save_model_v2.py

2025-10-06 00:14:16.371334: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
============================================================
데이터 로딩 중...
============================================================

✓ 총 111개의 원본 시퀀스 로드됨
✓ 클래스 수: 5
✓ 클래스: ['무한', '미국', '수학', '월세', '일요일']

============================================================
데이터 증강 중...
============================================================

✓ 증강 후: 1776개 시퀀스 생성
✓ 증강 배수: 16.0배

[클래스별 데이터 개수]
  무한: 336개
  미국: 368개
  수학: 368개
  월세: 352개
  일요일: 352개

============================================================
시퀀스 패딩 중...
============================================================

✓ 패딩 후 shape: (1776, 163, 114)
✓ 라벨 shape: (1776, 5)

✓ 훈련 데이터: (1420, 163, 114)
✓ 테스트 데이터: (356, 163, 114)

============================================================
Bidirectional LSTM + Attention 모델 생성 중...
============================================================
2025-10-06 00:14:20.803108: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)             │ (None, 163, 114)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bidirectional (Bidirectional)        │ (None, 163, 256)            │         248,832 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 163, 256)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bidirectional_1 (Bidirectional)      │ (None, 163, 128)            │         164,352 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 163, 128)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ attention_layer (AttentionLayer)     │ (None, 128)                 │          16,512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │           8,256 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 5)                   │             325 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 438,277 (1.67 MB)
 Trainable params: 438,277 (1.67 MB)
 Non-trainable params: 0 (0.00 B)

============================================================
모델 학습 시작...
============================================================
Epoch 1/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 11s 113ms/step - accuracy: 0.5176 - loss: 1.1717 - val_accuracy: 0.6761 - val_loss: 0.6958 - learning_rate: 5.0000e-04
Epoch 2/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 103ms/step - accuracy: 0.7139 - loss: 0.7052 - val_accuracy: 0.6549 - val_loss: 0.7401 - learning_rate: 5.0000e-04
Epoch 3/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 96ms/step - accuracy: 0.8099 - loss: 0.4640 - val_accuracy: 0.6972 - val_loss: 0.6522 - learning_rate: 5.0000e-04
Epoch 4/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 96ms/step - accuracy: 0.8618 - loss: 0.3605 - val_accuracy: 0.9225 - val_loss: 0.1890 - learning_rate: 5.0000e-04
Epoch 5/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 98ms/step - accuracy: 0.9243 - loss: 0.2195 - val_accuracy: 0.8310 - val_loss: 0.3268 - learning_rate: 5.0000e-04
Epoch 6/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 98ms/step - accuracy: 0.9349 - loss: 0.2003 - val_accuracy: 0.9401 - val_loss: 0.1420 - learning_rate: 5.0000e-04
Epoch 7/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 98ms/step - accuracy: 0.9665 - loss: 0.1006 - val_accuracy: 1.0000 - val_loss: 0.0177 - learning_rate: 5.0000e-04
Epoch 8/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 96ms/step - accuracy: 0.9991 - loss: 0.0181 - val_accuracy: 1.0000 - val_loss: 0.0033 - learning_rate: 5.0000e-04
Epoch 9/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 97ms/step - accuracy: 1.0000 - loss: 0.0078 - val_accuracy: 1.0000 - val_loss: 0.0018 - learning_rate: 5.0000e-04
Epoch 10/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 96ms/step - accuracy: 1.0000 - loss: 0.0063 - val_accuracy: 1.0000 - val_loss: 0.0011 - learning_rate: 5.0000e-04
Epoch 11/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 97ms/step - accuracy: 1.0000 - loss: 0.0036 - val_accuracy: 1.0000 - val_loss: 7.1113e-04 - learning_rate: 5.0000e-04
Epoch 12/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0044 - val_accuracy: 1.0000 - val_loss: 5.0869e-04 - learning_rate: 5.0000e-04
Epoch 13/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0030 - val_accuracy: 1.0000 - val_loss: 3.7429e-04 - learning_rate: 5.0000e-04
Epoch 14/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 101ms/step - accuracy: 1.0000 - loss: 0.0026 - val_accuracy: 1.0000 - val_loss: 2.5589e-04 - learning_rate: 5.0000e-04
Epoch 15/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0015 - val_accuracy: 1.0000 - val_loss: 1.9548e-04 - learning_rate: 5.0000e-04
Epoch 16/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0021 - val_accuracy: 1.0000 - val_loss: 1.5056e-04 - learning_rate: 5.0000e-04
Epoch 17/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 103ms/step - accuracy: 1.0000 - loss: 0.0013 - val_accuracy: 1.0000 - val_loss: 1.1978e-04 - learning_rate: 5.0000e-04
Epoch 18/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 102ms/step - accuracy: 1.0000 - loss: 0.0013 - val_accuracy: 1.0000 - val_loss: 9.9913e-05 - learning_rate: 5.0000e-04
Epoch 19/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 1.0000 - val_loss: 9.0160e-05 - learning_rate: 5.0000e-04
Epoch 20/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 8.8714e-04 - val_accuracy: 1.0000 - val_loss: 6.6545e-05 - learning_rate: 5.0000e-04
Epoch 21/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 0.9894 - loss: 0.0506 - val_accuracy: 0.8592 - val_loss: 0.4999 - learning_rate: 5.0000e-04
Epoch 22/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 0.8275 - loss: 0.4539 - val_accuracy: 0.8908 - val_loss: 0.4297 - learning_rate: 5.0000e-04
Epoch 23/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 0.9463 - loss: 0.1707 - val_accuracy: 0.9366 - val_loss: 0.1711 - learning_rate: 5.0000e-04
Epoch 24/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 103ms/step - accuracy: 0.9877 - loss: 0.0513 - val_accuracy: 1.0000 - val_loss: 0.0053 - learning_rate: 5.0000e-04
Epoch 25/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 0.9789 - loss: 0.0586 - val_accuracy: 1.0000 - val_loss: 0.0062 - learning_rate: 5.0000e-04
Epoch 26/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step - accuracy: 0.9791 - loss: 0.0761 
Epoch 26: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 0.9718 - loss: 0.0885 - val_accuracy: 1.0000 - val_loss: 0.0124 - learning_rate: 5.0000e-04
Epoch 27/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 0.9991 - loss: 0.0127 - val_accuracy: 1.0000 - val_loss: 0.0021 - learning_rate: 2.5000e-04
Epoch 28/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0055 - val_accuracy: 1.0000 - val_loss: 0.0014 - learning_rate: 2.5000e-04
Epoch 29/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0044 - val_accuracy: 1.0000 - val_loss: 0.0011 - learning_rate: 2.5000e-04
Epoch 30/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0037 - val_accuracy: 1.0000 - val_loss: 8.2293e-04 - learning_rate: 2.5000e-04
Epoch 31/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0030 - val_accuracy: 1.0000 - val_loss: 6.6292e-04 - learning_rate: 2.5000e-04
Epoch 32/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0024 - val_accuracy: 1.0000 - val_loss: 5.4263e-04 - learning_rate: 2.5000e-04
Epoch 33/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 98ms/step - accuracy: 1.0000 - loss: 0.0024 - val_accuracy: 1.0000 - val_loss: 4.4560e-04 - learning_rate: 2.5000e-04
Epoch 34/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0018 - val_accuracy: 1.0000 - val_loss: 3.7839e-04 - learning_rate: 2.5000e-04
Epoch 35/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 98ms/step - accuracy: 1.0000 - loss: 0.0022 - val_accuracy: 1.0000 - val_loss: 3.2703e-04 - learning_rate: 2.5000e-04
Epoch 36/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step - accuracy: 1.0000 - loss: 0.0019    
Epoch 36: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
71/71 ━━━━━━━━━━━━━━━━━━━━ 10s 99ms/step - accuracy: 1.0000 - loss: 0.0017 - val_accuracy: 1.0000 - val_loss: 2.7154e-04 - learning_rate: 2.5000e-04
Epoch 37/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0023 - val_accuracy: 1.0000 - val_loss: 2.4768e-04 - learning_rate: 1.2500e-04
Epoch 38/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 1.0000 - val_loss: 2.3253e-04 - learning_rate: 1.2500e-04
Epoch 39/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0021 - val_accuracy: 1.0000 - val_loss: 2.1401e-04 - learning_rate: 1.2500e-04
Epoch 40/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0013 - val_accuracy: 1.0000 - val_loss: 1.9889e-04 - learning_rate: 1.2500e-04
Epoch 41/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 1.0000 - val_loss: 1.8417e-04 - learning_rate: 1.2500e-04
Epoch 42/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 1.0000 - val_loss: 1.7208e-04 - learning_rate: 1.2500e-04
Epoch 43/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 1.0000 - val_loss: 1.6115e-04 - learning_rate: 1.2500e-04
Epoch 44/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 1.0000 - val_loss: 1.5218e-04 - learning_rate: 1.2500e-04
Epoch 45/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 1.0000 - val_loss: 1.3847e-04 - learning_rate: 1.2500e-04
Epoch 46/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 0s 90ms/step - accuracy: 1.0000 - loss: 0.0012
Epoch 46: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 9.6843e-04 - val_accuracy: 1.0000 - val_loss: 1.2849e-04 - learning_rate: 1.2500e-04
Epoch 47/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 99ms/step - accuracy: 1.0000 - loss: 0.0010 - val_accuracy: 1.0000 - val_loss: 1.2426e-04 - learning_rate: 6.2500e-05
Epoch 48/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 100ms/step - accuracy: 1.0000 - loss: 0.0011 - val_accuracy: 1.0000 - val_loss: 1.2307e-04 - learning_rate: 6.2500e-05
Epoch 49/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 102ms/step - accuracy: 1.0000 - loss: 0.0012 - val_accuracy: 1.0000 - val_loss: 1.1951e-04 - learning_rate: 6.2500e-05
Epoch 50/300
71/71 ━━━━━━━━━━━━━━━━━━━━ 7s 101ms/step - accuracy: 1.0000 - loss: 8.5435e-04 - val_accuracy: 1.0000 - val_loss: 1.1527e-04 - learning_rate: 6.2500e-05
Epoch 50: early stopping
Restoring model weights from the end of the best epoch: 20.

============================================================
모델 평가 중...

✓ 테스트 손실: 0.0001
✓ 테스트 정확도: 1.0000 (100.00%)

============================================================
모델 저장 중...
============================================================
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.

✓ 저장 완료!
  - 모델: trained_model/sign_language_model.h5
  - 설정: trained_model/model_info.pkl

============================================================
```

---

## 예측 로그
### 실제 정확도: 60% (3/5)
> 60%로 나름 준수한 정확도가 도출되었지만, 데이터 증강 실수로 교체 필요

---

### 수학 예측
- 수학 100.00%
- 일요일 0.00%
- 무한 0.00%
- 월세 0.00%
- 미국 0.00%

### 일요일 예측
- 일요일 46.82%
- 무한 35.81%
- 수학 16.99%
- 미국 0.37%
- 월세 0.01%

### 미국 예측
- 미국 96.51%
- 일요일 2.71%
- 무한 0.78%
- 수학 0.00%
- 월세 0.00%

### 무한 예측
- 일요일 99.29%
- 미국 0.58%
- 무한 0.12%
- 수학 0.00%
- 월세 0.00%

### 월세 예측
- 수학 84.53%
- 일요일 15.33%
- 무한 0.13%
- 월세 0.01%
- 미국 0.00%
