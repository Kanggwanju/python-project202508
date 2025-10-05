# train_and_save_model_v3.py

## 주요 특징 및 개선 사항

### 1. 올바른 데이터 증강 방식
   - v2의 치명적 문제 해결: Train/Test 분리 후 증강
   - Train 데이터만 증강하여 과적합 방지
   - Test는 원본 그대로 사용하여 실제 성능 측정

### 2. 단순화된 모델 구조
   - v2의 복잡한 Attention 제거 (438K → 약 100K 파라미터)
   - Bidirectional LSTM 2층 (64, 32 유닛)
   - Dropout 0.5로 증가하여 과적합 방지 강화

### 3. 데이터 증강 5배
   - v2의 15배 → 5배로 감소
   - 원본 111개 → 약 530개 (적절한 균형)
   - 4가지 증강 기법 순환 적용

### 4. 증강 기법 (v2와 동일)
   - 시간적 스케일링: 수어 속도 변화 (0.85~1.15배)
   - 가우시안 노이즈: 손 떨림 시뮬레이션 (0.2~0.8%)
   - 위치 이동: 카메라 각도 변화 (±3%)
   - 스케일 변화: 카메라 거리 변화 (0.95~1.05배)

### 5. 학습 전략
   - 학습률: 0.001 (v2의 0.0005보다 높게)
   - 배치 크기: 16
   - Patience: 20 (v2의 30에서 감소)
   - ReduceLROnPlateau: Patience 7

### 6. 예상 성능
   - 목표 정확도: 70-80% (100%는 과적합!)
   - Train: 85-95%, Validation: 70-80%
   - 실제 새로운 영상에서도 작동하는 모델

---

## 학습 로그와 수학 예측 로그 분석

### ✅ 긍정적인 부분
1. 올바른 방법 적용: Train/Test 분리 후 증강 ✓
2. 예측이 정확: test.mp4를 "수학"으로 99.98% 확신
3. v2보다 나음: 데이터 유출 문제는 해결됨

### ⚠️ 여전히 우려되는 부분
1. Test 데이터가 너무 적음
```text
Test: 23개 (클래스당 4-5개만)
```
- 통계적으로 신뢰하기 어려운 샘플 크기
- 우연히 잘 맞을 수도 있음

2. 100% 정확도는 여전히 의심스러움
```text
Epoch 49: val_accuracy 100%, val_loss 0.0193
Epoch 57-106: 계속 100% 유지
```
- 23개를 전부 맞춘 건 좋지만, 일반화 성능을 보장하지 않음

---

## 학습 로그

```text
python-project202508  python .\train_predict_model\train_and_save_model_v3.py

2025-10-06 00:40:26.700238: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
============================================================
train_and_save_model_v3.py
============================================================
과적합 방지를 위한 올바른 데이터 증강 방식 적용
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
Step 3: Train/Test 분리 (증강 전)
============================================================
중요: 분리 후 Train만 증강하여 과적합 방지!
Train 원본: 88개
Test 원본: 23개

============================================================
Step 4: 훈련 데이터만 증강
============================================================
Test는 원본 유지 -> 실제 성능 측정 가능
증강 후 Train: (528, 149, 114)
증강 배수: 6.0배
Test (증강 X): (23, 149, 114)

[Train 클래스별 데이터]
  무한: 102개
  미국: 108개
  수학: 108개
  월세: 102개
  일요일: 108개

[Test 클래스별 데이터]
  무한: 4개
  미국: 5개
  수학: 5개
  월세: 5개
  일요일: 4개

============================================================
Step 5: 모델 생성
============================================================
2025-10-06 00:40:30.918159: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
C:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\keras\src\layers\rnn\bidirectional.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ bidirectional (Bidirectional)        │ (None, 149, 128)            │          91,648 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 149, 128)            │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ bidirectional_1 (Bidirectional)      │ (None, 64)                  │          41,216 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 32)                  │           2,080 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_2 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 5)                   │             165 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 135,109 (527.77 KB)
 Trainable params: 135,109 (527.77 KB)
 Non-trainable params: 0 (0.00 B)

============================================================
Step 7: 모델 학습
============================================================
Validation에 원본 Test 사용 -> 진짜 성능 확인!
Epoch 1/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 5s 63ms/step - accuracy: 0.2917 - loss: 1.5163 - val_accuracy: 0.3478 - val_loss: 1.3988 - learning_rate: 0.0010
Epoch 2/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.4034 - loss: 1.3548 - val_accuracy: 0.4783 - val_loss: 1.2487 - learning_rate: 0.0010
Epoch 3/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.4583 - loss: 1.2193 - val_accuracy: 0.6087 - val_loss: 1.0920 - learning_rate: 0.0010
Epoch 4/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.5152 - loss: 1.0896 - val_accuracy: 0.5217 - val_loss: 1.0538 - learning_rate: 0.0010
Epoch 5/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.5284 - loss: 1.0554 - val_accuracy: 0.5217 - val_loss: 1.0272 - learning_rate: 0.0010
Epoch 6/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.5928 - loss: 0.9941 - val_accuracy: 0.5652 - val_loss: 0.9534 - learning_rate: 0.0010
Epoch 7/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.5985 - loss: 0.9109 - val_accuracy: 0.5217 - val_loss: 0.9385 - learning_rate: 0.0010
Epoch 8/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.6307 - loss: 0.8337 - val_accuracy: 0.4783 - val_loss: 0.9677 - learning_rate: 0.0010
Epoch 9/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.5814 - loss: 0.9107 - val_accuracy: 0.5217 - val_loss: 0.8528 - learning_rate: 0.0010
Epoch 10/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.6098 - loss: 0.9137 - val_accuracy: 0.4783 - val_loss: 1.3773 - learning_rate: 0.0010
Epoch 11/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.5777 - loss: 0.9469 - val_accuracy: 0.5217 - val_loss: 0.8946 - learning_rate: 0.0010
Epoch 12/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.6477 - loss: 0.8171 - val_accuracy: 0.6087 - val_loss: 0.8798 - learning_rate: 0.0010
Epoch 13/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.6231 - loss: 0.8246 - val_accuracy: 0.6087 - val_loss: 0.7938 - learning_rate: 0.0010
Epoch 14/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.6913 - loss: 0.6842 - val_accuracy: 0.6522 - val_loss: 0.6659 - learning_rate: 0.0010
Epoch 15/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.7670 - loss: 0.5794 - val_accuracy: 0.7391 - val_loss: 0.6783 - learning_rate: 0.0010
Epoch 16/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.7519 - loss: 0.5650 - val_accuracy: 0.5652 - val_loss: 0.7903 - learning_rate: 0.0010
Epoch 17/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.6875 - loss: 0.8236 - val_accuracy: 0.5652 - val_loss: 0.8517 - learning_rate: 0.0010
Epoch 18/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.6799 - loss: 0.6909 - val_accuracy: 0.7391 - val_loss: 0.5717 - learning_rate: 0.0010
Epoch 19/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.7595 - loss: 0.5473 - val_accuracy: 0.8261 - val_loss: 0.6379 - learning_rate: 0.0010
Epoch 20/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.8144 - loss: 0.5033 - val_accuracy: 0.8261 - val_loss: 0.3866 - learning_rate: 0.0010
Epoch 21/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.7822 - loss: 0.5370 - val_accuracy: 0.6957 - val_loss: 0.6641 - learning_rate: 0.0010
Epoch 22/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.7955 - loss: 0.4786 - val_accuracy: 0.8696 - val_loss: 0.3612 - learning_rate: 0.0010
Epoch 23/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.8561 - loss: 0.3685 - val_accuracy: 0.9130 - val_loss: 0.3032 - learning_rate: 0.0010
Epoch 24/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.8277 - loss: 0.4265 - val_accuracy: 0.7391 - val_loss: 0.5647 - learning_rate: 0.0010
Epoch 25/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.8542 - loss: 0.3702 - val_accuracy: 0.6087 - val_loss: 0.9488 - learning_rate: 0.0010
Epoch 26/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.7197 - loss: 0.6900 - val_accuracy: 0.6957 - val_loss: 0.7709 - learning_rate: 0.0010
Epoch 27/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.7803 - loss: 0.4802 - val_accuracy: 0.6957 - val_loss: 0.4320 - learning_rate: 0.0010
Epoch 28/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.7481 - loss: 0.6902 - val_accuracy: 0.8261 - val_loss: 0.4916 - learning_rate: 0.0010
Epoch 29/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.8447 - loss: 0.4451 - val_accuracy: 0.8696 - val_loss: 0.3036 - learning_rate: 0.0010
Epoch 30/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - accuracy: 0.8699 - loss: 0.3836
Epoch 30: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.8617 - loss: 0.3918 - val_accuracy: 0.8261 - val_loss: 0.5977 - learning_rate: 0.0010
Epoch 31/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.8883 - loss: 0.2863 - val_accuracy: 0.9130 - val_loss: 0.2390 - learning_rate: 5.0000e-04
Epoch 32/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9167 - loss: 0.2504 - val_accuracy: 0.9130 - val_loss: 0.2298 - learning_rate: 5.0000e-04
Epoch 33/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.9242 - loss: 0.2128 - val_accuracy: 0.9565 - val_loss: 0.1374 - learning_rate: 5.0000e-04
Epoch 34/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9205 - loss: 0.2095 - val_accuracy: 0.8696 - val_loss: 0.2673 - learning_rate: 5.0000e-04
Epoch 35/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9110 - loss: 0.2645 - val_accuracy: 0.7826 - val_loss: 0.6178 - learning_rate: 5.0000e-04
Epoch 36/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9129 - loss: 0.2238 - val_accuracy: 0.9130 - val_loss: 0.1765 - learning_rate: 5.0000e-04
Epoch 37/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9451 - loss: 0.1991 - val_accuracy: 0.8261 - val_loss: 0.2239 - learning_rate: 5.0000e-04
Epoch 38/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9489 - loss: 0.1713 - val_accuracy: 0.9130 - val_loss: 0.1333 - learning_rate: 5.0000e-04
Epoch 39/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9545 - loss: 0.1538 - val_accuracy: 0.9130 - val_loss: 0.1568 - learning_rate: 5.0000e-04
Epoch 40/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9508 - loss: 0.1359 - val_accuracy: 0.9130 - val_loss: 0.1539 - learning_rate: 5.0000e-04
Epoch 41/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9489 - loss: 0.1533 - val_accuracy: 0.9565 - val_loss: 0.1355 - learning_rate: 5.0000e-04
Epoch 42/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9527 - loss: 0.1495 - val_accuracy: 0.9130 - val_loss: 0.1340 - learning_rate: 5.0000e-04
Epoch 43/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9640 - loss: 0.1267 - val_accuracy: 0.8696 - val_loss: 0.2319 - learning_rate: 5.0000e-04
Epoch 44/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9697 - loss: 0.1267 - val_accuracy: 0.9130 - val_loss: 0.1000 - learning_rate: 5.0000e-04
Epoch 45/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9697 - loss: 0.1171 - val_accuracy: 0.9565 - val_loss: 0.0776 - learning_rate: 5.0000e-04
Epoch 46/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9811 - loss: 0.0960 - val_accuracy: 0.9130 - val_loss: 0.1512 - learning_rate: 5.0000e-04
Epoch 47/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9735 - loss: 0.0901 - val_accuracy: 0.9565 - val_loss: 0.1186 - learning_rate: 5.0000e-04
Epoch 48/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9640 - loss: 0.0979 - val_accuracy: 0.9565 - val_loss: 0.0882 - learning_rate: 5.0000e-04
Epoch 49/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9678 - loss: 0.0911 - val_accuracy: 1.0000 - val_loss: 0.0193 - learning_rate: 5.0000e-04
Epoch 50/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9716 - loss: 0.1264 - val_accuracy: 0.9565 - val_loss: 0.1162 - learning_rate: 5.0000e-04
Epoch 51/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9697 - loss: 0.1233 - val_accuracy: 0.9130 - val_loss: 0.1260 - learning_rate: 5.0000e-04
Epoch 52/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9564 - loss: 0.1577 - val_accuracy: 0.8696 - val_loss: 0.1669 - learning_rate: 5.0000e-04
Epoch 53/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9527 - loss: 0.1282 - val_accuracy: 0.9565 - val_loss: 0.0703 - learning_rate: 5.0000e-04
Epoch 54/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9640 - loss: 0.0935 - val_accuracy: 0.9130 - val_loss: 0.1719 - learning_rate: 5.0000e-04
Epoch 55/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9811 - loss: 0.0712 - val_accuracy: 0.9565 - val_loss: 0.1731 - learning_rate: 5.0000e-04
Epoch 56/150
32/33 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - accuracy: 0.9868 - loss: 0.0521
Epoch 56: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9867 - loss: 0.0592 - val_accuracy: 0.9130 - val_loss: 0.1147 - learning_rate: 5.0000e-04
Epoch 57/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9830 - loss: 0.0559 - val_accuracy: 1.0000 - val_loss: 0.0179 - learning_rate: 2.5000e-04
Epoch 58/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9905 - loss: 0.0433 - val_accuracy: 1.0000 - val_loss: 0.0228 - learning_rate: 2.5000e-04
Epoch 59/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9886 - loss: 0.0498 - val_accuracy: 1.0000 - val_loss: 0.0195 - learning_rate: 2.5000e-04
Epoch 60/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9962 - loss: 0.0372 - val_accuracy: 1.0000 - val_loss: 0.0147 - learning_rate: 2.5000e-04
Epoch 61/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9905 - loss: 0.0450 - val_accuracy: 1.0000 - val_loss: 0.0100 - learning_rate: 2.5000e-04
Epoch 62/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9830 - loss: 0.0657 - val_accuracy: 0.9565 - val_loss: 0.1515 - learning_rate: 2.5000e-04
Epoch 63/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9905 - loss: 0.0462 - val_accuracy: 1.0000 - val_loss: 0.0081 - learning_rate: 2.5000e-04
Epoch 64/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9886 - loss: 0.0585 - val_accuracy: 1.0000 - val_loss: 0.0082 - learning_rate: 2.5000e-04
Epoch 65/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9905 - loss: 0.0413 - val_accuracy: 1.0000 - val_loss: 0.0065 - learning_rate: 2.5000e-04
Epoch 66/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 3s 46ms/step - accuracy: 0.9924 - loss: 0.0385 - val_accuracy: 1.0000 - val_loss: 0.0056 - learning_rate: 2.5000e-04
Epoch 67/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9792 - loss: 0.0576 - val_accuracy: 1.0000 - val_loss: 0.0047 - learning_rate: 2.5000e-04
Epoch 68/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9943 - loss: 0.0382 - val_accuracy: 1.0000 - val_loss: 0.0048 - learning_rate: 2.5000e-04
Epoch 69/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9962 - loss: 0.0365 - val_accuracy: 1.0000 - val_loss: 0.0049 - learning_rate: 2.5000e-04
Epoch 70/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9943 - loss: 0.0366 - val_accuracy: 1.0000 - val_loss: 0.0063 - learning_rate: 2.5000e-04
Epoch 71/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9924 - loss: 0.0382 - val_accuracy: 1.0000 - val_loss: 0.0075 - learning_rate: 2.5000e-04
Epoch 72/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9981 - loss: 0.0277 - val_accuracy: 1.0000 - val_loss: 0.0071 - learning_rate: 2.5000e-04
Epoch 73/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9924 - loss: 0.0245 - val_accuracy: 1.0000 - val_loss: 0.0079 - learning_rate: 2.5000e-04
Epoch 74/150
32/33 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step - accuracy: 0.9935 - loss: 0.0395
Epoch 74: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9943 - loss: 0.0317 - val_accuracy: 1.0000 - val_loss: 0.0067 - learning_rate: 2.5000e-04
Epoch 75/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9943 - loss: 0.0321 - val_accuracy: 1.0000 - val_loss: 0.0072 - learning_rate: 1.2500e-04
Epoch 76/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9943 - loss: 0.0374 - val_accuracy: 1.0000 - val_loss: 0.0058 - learning_rate: 1.2500e-04
Epoch 77/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9905 - loss: 0.0359 - val_accuracy: 1.0000 - val_loss: 0.0038 - learning_rate: 1.2500e-04
Epoch 78/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9867 - loss: 0.0420 - val_accuracy: 1.0000 - val_loss: 0.0032 - learning_rate: 1.2500e-04
Epoch 79/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9962 - loss: 0.0322 - val_accuracy: 1.0000 - val_loss: 0.0034 - learning_rate: 1.2500e-04
Epoch 80/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9962 - loss: 0.0224 - val_accuracy: 1.0000 - val_loss: 0.0032 - learning_rate: 1.2500e-04
Epoch 81/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9943 - loss: 0.0298 - val_accuracy: 1.0000 - val_loss: 0.0029 - learning_rate: 1.2500e-04
Epoch 82/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9943 - loss: 0.0308 - val_accuracy: 1.0000 - val_loss: 0.0024 - learning_rate: 1.2500e-04
Epoch 83/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9981 - loss: 0.0192 - val_accuracy: 1.0000 - val_loss: 0.0023 - learning_rate: 1.2500e-04
Epoch 84/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9962 - loss: 0.0209 - val_accuracy: 1.0000 - val_loss: 0.0024 - learning_rate: 1.2500e-04
Epoch 85/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9905 - loss: 0.0340 - val_accuracy: 1.0000 - val_loss: 0.0026 - learning_rate: 1.2500e-04
Epoch 86/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9943 - loss: 0.0324 - val_accuracy: 1.0000 - val_loss: 0.0013 - learning_rate: 1.2500e-04
Epoch 87/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 50ms/step - accuracy: 0.9735 - loss: 0.1438 - val_accuracy: 1.0000 - val_loss: 0.0085 - learning_rate: 1.2500e-04
Epoch 88/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9905 - loss: 0.0357 - val_accuracy: 1.0000 - val_loss: 0.0083 - learning_rate: 1.2500e-04
Epoch 89/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.9924 - loss: 0.0302 - val_accuracy: 1.0000 - val_loss: 0.0108 - learning_rate: 1.2500e-04
Epoch 90/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9905 - loss: 0.0372 - val_accuracy: 1.0000 - val_loss: 0.0049 - learning_rate: 1.2500e-04
Epoch 91/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9886 - loss: 0.0296 - val_accuracy: 1.0000 - val_loss: 0.0051 - learning_rate: 1.2500e-04
Epoch 92/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9924 - loss: 0.0335 - val_accuracy: 1.0000 - val_loss: 0.0047 - learning_rate: 1.2500e-04
Epoch 93/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 0s 44ms/step - accuracy: 0.9994 - loss: 0.0212
Epoch 93: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9962 - loss: 0.0240 - val_accuracy: 1.0000 - val_loss: 0.0046 - learning_rate: 1.2500e-04
Epoch 94/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9905 - loss: 0.0333 - val_accuracy: 1.0000 - val_loss: 0.0038 - learning_rate: 6.2500e-05
Epoch 95/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9924 - loss: 0.0350 - val_accuracy: 1.0000 - val_loss: 0.0036 - learning_rate: 6.2500e-05
Epoch 96/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9962 - loss: 0.0247 - val_accuracy: 1.0000 - val_loss: 0.0036 - learning_rate: 6.2500e-05
Epoch 97/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9981 - loss: 0.0216 - val_accuracy: 1.0000 - val_loss: 0.0035 - learning_rate: 6.2500e-05
Epoch 98/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9943 - loss: 0.0272 - val_accuracy: 1.0000 - val_loss: 0.0034 - learning_rate: 6.2500e-05
Epoch 99/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9981 - loss: 0.0263 - val_accuracy: 1.0000 - val_loss: 0.0034 - learning_rate: 6.2500e-05
Epoch 100/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 0s 46ms/step - accuracy: 0.9978 - loss: 0.0213
Epoch 100: ReduceLROnPlateau reducing learning rate to 3.125000148429535e-05.
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.9962 - loss: 0.0214 - val_accuracy: 1.0000 - val_loss: 0.0034 - learning_rate: 6.2500e-05
Epoch 101/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 48ms/step - accuracy: 0.9962 - loss: 0.0267 - val_accuracy: 1.0000 - val_loss: 0.0033 - learning_rate: 3.1250e-05
Epoch 102/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9886 - loss: 0.0352 - val_accuracy: 1.0000 - val_loss: 0.0033 - learning_rate: 3.1250e-05
Epoch 103/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 1.0000 - loss: 0.0258 - val_accuracy: 1.0000 - val_loss: 0.0032 - learning_rate: 3.1250e-05
Epoch 104/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 46ms/step - accuracy: 0.9981 - loss: 0.0190 - val_accuracy: 1.0000 - val_loss: 0.0033 - learning_rate: 3.1250e-05
Epoch 105/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 47ms/step - accuracy: 0.9905 - loss: 0.0297 - val_accuracy: 1.0000 - val_loss: 0.0032 - learning_rate: 3.1250e-05
Epoch 106/150
33/33 ━━━━━━━━━━━━━━━━━━━━ 2s 49ms/step - accuracy: 0.9924 - loss: 0.0264 - val_accuracy: 1.0000 - val_loss: 0.0032 - learning_rate: 3.1250e-05
Epoch 106: early stopping
Restoring model weights from the end of the best epoch: 86.

============================================================
Step 8: 최종 평가
============================================================

테스트 손실: 0.0013
테스트 정확도: 1.0000 (100.00%)

경고: 정확도가 95% 이상입니다.
과적합 가능성이 있으니 실제 영상으로 테스트해보세요.

============================================================
Step 9: 모델 저장
============================================================
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.
저장 완료: trained_model/trained_model_v3/
  - sign_language_model.h5
  - model_info.pkl

============================================================
학습 완료!
============================================================

예측 실행 방법:
   python .\train_predict_model\predict_sign_language.py --video "test.mp4" --model-dir "trained_model/trained_model_v3" --show-probs

============================================================
```

---

## 수학 예측 로그
- 수학 예측 성공
- 나머지는 텍스트로 대체

```text
python-project202508  python .\train_predict_model\predict_sign_language.py --video ".\train_predict_model\test.mp4" --model-dir "trained_model/trained_model_v3" --show-probs

AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

2025-10-06 00:44:06.785152: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
모델 로딩 중...
2025-10-06 00:44:11.401436: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
모델 로드 완료!
학습된 수어 단어 (5개): ['무한', '미국', '수학', '월세', '일요일']

============================================================
예측 시작: .\train_predict_model\test.mp4
============================================================

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
영상 정보: 124프레임, 30.0fps
키포인트 추출 중...WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1759679051.734850   30668 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.763983   28664 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.769087   32916 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.769489   38300 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.771162   40304 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.783279   30668 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.800554   39032 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.802485   32916 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759679051.804510   30668 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.
...... 완료! (62프레임 추출)
데이터 전처리 중...
예측 중...

============================================================
📌 예측 결과
============================================================
예측된 수어: 수학
신뢰도: 99.98%

────────────────────────────────────────────────────────────
전체 확률 분포:
────────────────────────────────────────────────────────────
수학              │ █████████████████████████████████████████████████░ │  99.98%
무한              │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   0.02%
일요일             │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   0.00%
미국              │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   0.00%
월세              │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   0.00%
============================================================
```

---

## 🚨 심각한 문제 발견
### 실제 정확도: 40% (2/5)
> 수학 말고도 다른 수어 데이터에 대하여 모델이 잘 작동하지 않았음

---

## 문제 분석

### 1️⃣ "무한" 편향 문제
```text
일요일 → 무한 99.80% (틀림)
미국   → 무한 94.74% (틀림)
무한   → 무한 79.16% (맞음)
```
모델이 모든 걸 "무한"으로 예측하려는 경향

### 2️⃣ "월세"를 전혀 인식 못함
```text
월세 → 일요일 99.57% (완전히 틀림)
모든 예측에서 월세 확률 거의 0%
```
### 3️⃣ 클래스별 성능 차이
- ✅ 잘 맞춤: 수학, 무한
- ❌ 못 맞춤: 일요일, 미국, 월세

---

## 수학 제외, 나머지 수어 예측 기록
> 새롭게 영상을 촬영하여 예측함

### 일요일 예측
- 무한 99.80%
- 미국 0.14%
- 수학 0.05%
- 일요일 0.01%
- 월세 0.00%

### 미국 예측
- 무한 94.74%
- 미국 3.40%
- 일요일 1.54%
- 수학 0.32%
- 월세 0.00%

### 무한 예측
- 무한 79.16%
- 일요일 12.49%
- 미국 7.80%
- 수학 0.55%
- 월세 0.01%

### 월세 예측
- 일요일 99.57%
- 수학 0.40%
- 무한 0.03%
- 월세 0.00%
- 미국 0.00%