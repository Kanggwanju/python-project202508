# train_and_save_model.py

## 현재 모델의 문제점

### 1️⃣ 심각하게 낮은 성능
- 테스트 정확도 52% - 5개 클래스 중 랜덤 추측(20%)보다는 낫지만 실용성이 없는 수준
- 예측 신뢰도 30% - 모델이 확신을 갖지 못하고 있음
- 모든 클래스가 비슷한 확률(28-30%) → 의미있는 패턴을 학습하지 못함

### 2️⃣ 학습 불안정성
- Epoch 35-61: 정확도 50% 초반 (안정적)
- Epoch 62-74: 정확도 40%로 급락 후 회복 안됨

모델이 overfitting에서 collapse로 이어졌습니다.

### 3️⃣ 과도한 데이터 증강
- 40배 증강은 너무 과도함
- 원본 데이터의 고유한 특성을 희석시키고 노이즈를 학습

---

## 데이터 증강 10배로 변경
> 정확도가 52% → 40%로 떨어짐.

### 문제 진단
```
Epoch 24-26: 학습 시작 (37%)
Epoch 27: 완전 붕괴 (18%)
Epoch 28-41: 회복 불가능
```
데이터 증강을 줄였는데 성능이 악화된 이유:
1. 원본 데이터 111개가 너무 적음 - 증강 없이는 학습 불가능
2. 모델이 너무 단순 - 복잡한 수어 패턴을 학습할 수 없음
3. 학습이 매우 불안정 - gradient 폭발/소실 발생

---

## 데이터 증강 10배 학습 로그
```text
python-project202508  python .\LSTM_model\train_and_save_model.py
                                                    
2025-10-05 22:34:54.785085: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
데이터 로딩 중...
총 111개의 원본 시퀀스 로드됨. 클래스 수: 5
클래스 매핑: {'무한': 0, '미국': 1, '수학': 2, '월세': 3, '일요일': 4}

데이터 증강 중...
증강 후 총 1221개의 시퀀스 생성됨.

시퀀스 패딩 중...
패딩 후 시퀀스 shape: (1221, 208, 114)
라벨 shape: (1221, 5)

훈련 데이터 shape: (976, 208, 114)
테스트 데이터 shape: (245, 208, 114)

모델 생성 중...
2025-10-05 22:34:59.154406: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
C:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\keras\src\layers\rnn\rnn.py:199: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(**kwargs)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ lstm (LSTM)                          │ (None, 208, 64)             │          45,824 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 208, 64)             │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ lstm_1 (LSTM)                        │ (None, 32)                  │          12,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_1 (Dropout)                  │ (None, 32)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 5)                   │             165 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 58,405 (228.14 KB)
 Trainable params: 58,405 (228.14 KB)
 Non-trainable params: 0 (0.00 B)

모델 학습 시작...
Epoch 1/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 4s 87ms/step - accuracy: 0.2038 - loss: 1.6096 - val_accuracy: 0.1582 - val_loss: 1.6133
Epoch 2/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2205 - loss: 1.6073 - val_accuracy: 0.1582 - val_loss: 1.6181
Epoch 3/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2218 - loss: 1.6067 - val_accuracy: 0.1582 - val_loss: 1.6196
Epoch 4/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2154 - loss: 1.6074 - val_accuracy: 0.1582 - val_loss: 1.6153
Epoch 5/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2244 - loss: 1.6033 - val_accuracy: 0.1582 - val_loss: 1.6141
Epoch 6/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2038 - loss: 1.6058 - val_accuracy: 0.1582 - val_loss: 1.6173
Epoch 7/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2000 - loss: 1.6048 - val_accuracy: 0.2041 - val_loss: 1.6142
Epoch 8/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.1949 - loss: 1.6029 - val_accuracy: 0.1888 - val_loss: 1.6122
Epoch 9/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2103 - loss: 1.6042 - val_accuracy: 0.1582 - val_loss: 1.6148
Epoch 10/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2154 - loss: 1.6022 - val_accuracy: 0.1582 - val_loss: 1.6163
Epoch 11/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 74ms/step - accuracy: 0.2064 - loss: 1.6035 - val_accuracy: 0.1582 - val_loss: 1.6160
Epoch 12/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2000 - loss: 1.6003 - val_accuracy: 0.1582 - val_loss: 1.6169
Epoch 13/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2179 - loss: 1.6018 - val_accuracy: 0.2041 - val_loss: 1.6173
Epoch 14/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2154 - loss: 1.6022 - val_accuracy: 0.1582 - val_loss: 1.6156
Epoch 15/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2167 - loss: 1.5995 - val_accuracy: 0.1582 - val_loss: 1.6160
Epoch 16/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2026 - loss: 1.5994 - val_accuracy: 0.2041 - val_loss: 1.6187
Epoch 17/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2244 - loss: 1.6009 - val_accuracy: 0.1582 - val_loss: 1.6189
Epoch 18/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 77ms/step - accuracy: 0.2205 - loss: 1.5972 - val_accuracy: 0.1582 - val_loss: 1.6157
Epoch 19/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2192 - loss: 1.5962 - val_accuracy: 0.1633 - val_loss: 1.6105
Epoch 20/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2577 - loss: 1.5919 - val_accuracy: 0.2296 - val_loss: 1.5767
Epoch 21/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2474 - loss: 1.6025 - val_accuracy: 0.1990 - val_loss: 1.6081
Epoch 22/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 77ms/step - accuracy: 0.2359 - loss: 1.5869 - val_accuracy: 0.2194 - val_loss: 1.5965
Epoch 23/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 77ms/step - accuracy: 0.2372 - loss: 1.5707 - val_accuracy: 0.2143 - val_loss: 1.5938
Epoch 24/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.3244 - loss: 1.5038 - val_accuracy: 0.3367 - val_loss: 1.4709
Epoch 25/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.3628 - loss: 1.3612 - val_accuracy: 0.3469 - val_loss: 1.2521
Epoch 26/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.3641 - loss: 1.2933 - val_accuracy: 0.3724 - val_loss: 1.2227
Epoch 27/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2090 - loss: 1.5805 - val_accuracy: 0.1888 - val_loss: 1.6268
Epoch 28/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2077 - loss: 1.6108 - val_accuracy: 0.1582 - val_loss: 1.6158
Epoch 29/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 74ms/step - accuracy: 0.2179 - loss: 1.6103 - val_accuracy: 0.1582 - val_loss: 1.6199
Epoch 30/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 74ms/step - accuracy: 0.2154 - loss: 1.6078 - val_accuracy: 0.1582 - val_loss: 1.6180
Epoch 31/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2244 - loss: 1.6068 - val_accuracy: 0.1582 - val_loss: 1.6164
Epoch 32/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2090 - loss: 1.6076 - val_accuracy: 0.1582 - val_loss: 1.6220
Epoch 33/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 74ms/step - accuracy: 0.2282 - loss: 1.6077 - val_accuracy: 0.1582 - val_loss: 1.6172
Epoch 34/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2167 - loss: 1.6073 - val_accuracy: 0.1582 - val_loss: 1.6204
Epoch 35/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2154 - loss: 1.6054 - val_accuracy: 0.1582 - val_loss: 1.6175
Epoch 36/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2103 - loss: 1.6049 - val_accuracy: 0.1582 - val_loss: 1.6152
Epoch 37/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2179 - loss: 1.6051 - val_accuracy: 0.1582 - val_loss: 1.6165
Epoch 38/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2064 - loss: 1.6048 - val_accuracy: 0.1582 - val_loss: 1.6165
Epoch 39/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2064 - loss: 1.6048 - val_accuracy: 0.1582 - val_loss: 1.6176
Epoch 40/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 76ms/step - accuracy: 0.2141 - loss: 1.6034 - val_accuracy: 0.1582 - val_loss: 1.6201
Epoch 41/200
25/25 ━━━━━━━━━━━━━━━━━━━━ 2s 75ms/step - accuracy: 0.2346 - loss: 1.6044 - val_accuracy: 0.1582 - val_loss: 1.6150

모델 평가 중...
테스트 손실: 1.2139
테스트 정확도: 0.3959

모델 저장 중...
WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`.

저장 완료!
- 모델: trained_model/sign_language_model.h5
- 설정 정보: trained_model/model_info.pkl

학습된 수어 단어: ['무한', '미국', '수학', '월세', '일요일']
 python-project202508  python .\LSTM_model\predict_sign_language.py --video ".\LSTM_model\test.mp4" --show-probs                                                                                               
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
2025-10-06 00:00:26.849335: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
모델 로딩 중...
2025-10-06 00:00:31.654936: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
모델 로드 완료!
학습된 수어 단어 (5개): ['무한', '미국', '수학', '월세', '일요일']

============================================================
예측 시작: .\LSTM_model\test.mp4
============================================================

INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
영상 정보: 124프레임, 30.0fps
키포인트 추출 중...WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
W0000 00:00:1759676431.964131    3564 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676431.990421   38204 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676431.995290   24748 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676431.995490    2852 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676431.997264   20144 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676432.006089    2852 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676432.023850   22472 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676432.025083   38204 inference_feedback_manager.cc:114] Feedback manager requires a model with a single signature inference. Disabling support for feedback tensors.
W0000 00:00:1759676432.027176   20640 landmark_projection_calculator.cc:186] Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI. Provide IMAGE_DIMENSIONS or use PROJECTION_MATRIX.   
...... 완료! (62프레임 추출)
데이터 전처리 중...
예측 중...

============================================================
📌 예측 결과
============================================================
예측된 수어: 수학
신뢰도: 25.29%

────────────────────────────────────────────────────────────
전체 확률 분포:
────────────────────────────────────────────────────────────
수학              │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  25.29%
일요일             │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  24.98%
미국              │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  24.26%
무한              │ ██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │  21.77%
월세              │ █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │   3.71%
============================================================

```

