import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2 # L2 정규화를 위해 추가
import random

# ============================================================
# train_and_save_model_v5.py
# ============================================================
#
# 파일 위치: train_predict_model/train_and_save_model_v5.py
# 저장 위치: trained_model/trained_model_v5/
#
# ============================================================
# 실행 방법
# ============================================================
#
# 학습:
#    python .\train_predict_model\train_and_save_model_v5.py
#
# 예측:
#    python .\train_predict_model\predict_sign_language.py --video ".\train_predict_model\test.mp4" --model-dir "trained_model\trained_model_v5" --show-probs
#
# ============================================================
# 주요 특징 및 개선 사항
# ============================================================

# 1. K-Fold 교차 검증 도입 (가장 큰 변화)
#    - 단일 Train/Test 분리 대신 StratifiedKFold를 사용하여 데이터셋 전체를
#      훈련 및 검증에 활용함으로써 모델 성능 평가의 신뢰도를 대폭 향상
#    - 각 Fold마다 모델을 새로 학습하고 평가하여 평균 정확도 및 표준편차 제공
#    - 가장 높은 검증 정확도를 보인 모델을 최종 모델로 저장

# 2. L2 정규화 추가
#    - LSTM 및 Dense 레이어에 L2 정규화(kernel_regularizer=l2(0.001))를 추가하여
#      가중치가 너무 커지는 것을 방지하고 과적합 억제 강화

# 3. Dropout 비율 조정
#    - Bidirectional LSTM 레이어의 Dropout 비율을 0.5에서 0.6으로 상향 조정하여
#      과적합 방지 효과를 더욱 강화
#    - Dense 레이어의 Dropout 비율도 0.4에서 0.5로 상향 조정

# 4. 학습 전략 유지
#    - 학습률: 0.001
#    - 배치 크기: 16
#    - EarlyStopping Patience: 20
#    - ReduceLROnPlateau Patience: 7
#
# 5. 예상 성능
#    - K-Fold 교차 검증을 통해 얻는 평균 정확도가 실제 성능에 더 가까울 것으로 예상
#    - 목표 평균 정확도: 60-80% (데이터셋 크기 고려)
#    - 단일 Fold에서 100% 정확도가 나오더라도, 다른 Fold에서 낮은 정확도가 나올 수 있음
#
# 6. 저장 위치
#    - 모델: trained_model/trained_model_v5/sign_language_model.h5
#    - 설정: trained_model/trained_model_v5/model_info.pkl
#
# ============================================================
# v3와의 주요 차이점
# ============================================================
#
# v3:
#   데이터 로드 → Train/Test 분리 (단일) → Train만 증강 → 모델 학습 및 평가 (단일 Test셋)
#   문제점: 적은 Test 데이터로 인한 평가 신뢰도 저하, 과적합 여부 판단 어려움

# v5 (개선):
#   데이터 로드 → StratifiedKFold 분할 (N개 Fold)
#   → 각 Fold마다: Train/Test 분리 → Train만 증강 → 모델 학습 및 평가
#   → 모든 Fold 결과 집계 → 가장 좋은 모델 저장
#   개선점: 모델 일반화 성능을 더 정확하게 측정, L2 정규화 및 Dropout 강화로 과적합 추가 억제
#
# ============================================================


# ============================================================
# 1. 개선된 데이터 증강 함수 (v3와 동일)
# ============================================================
def augment_sequence_realistic(sequence: np.ndarray, num_augmentations: int = 8):
    """
    4가지 증강 기법을 순환하며 적용

    Args:
        sequence: 원본 시퀀스 (frames, features)
        num_augmentations: 생성할 증강 데이터 개수

    Returns:
        augmented_sequences: [원본 + 증강된 시퀀스들]
    """
    augmented_sequences = [sequence]  # 원본 포함

    for i in range(num_augmentations):
        aug_seq = sequence.copy()
        augmentation_type = i % 4

        if augmentation_type == 0:
            # 시간적 스케일링 (속도 변화)
            if aug_seq.shape[0] > 10:
                skip_factor = random.uniform(0.85, 1.15)
                new_len = int(aug_seq.shape[0] * skip_factor)
                if new_len > 5:
                    indices = np.linspace(0, aug_seq.shape[0] - 1, new_len).astype(int)
                    aug_seq = aug_seq[indices]

        elif augmentation_type == 1:
            # 가우시안 노이즈 추가
            noise_level = random.uniform(0.002, 0.008)
            aug_seq += np.random.normal(0, noise_level, aug_seq.shape)
            aug_seq = np.clip(aug_seq, 0.0, 1.0)

        elif augmentation_type == 2:
            # 위치 이동
            shift_x = random.uniform(-0.03, 0.03)
            shift_y = random.uniform(-0.03, 0.03)
            aug_seq[:, ::2] += shift_x
            aug_seq[:, 1::2] += shift_y
            aug_seq = np.clip(aug_seq, 0.0, 1.0)

        else:
            # 스케일 변화
            scale = random.uniform(0.95, 1.05)
            center_x, center_y = 0.5, 0.5
            aug_seq[:, ::2] = (aug_seq[:, ::2] - center_x) * scale + center_x
            aug_seq[:, 1::2] = (aug_seq[:, 1::2] - center_y) * scale + center_y
            aug_seq = np.clip(aug_seq, 0.0, 1.0)

        augmented_sequences.append(aug_seq)

    return augmented_sequences


# ============================================================
# 2. 단순화된 Bidirectional LSTM 모델 (L2 정규화 및 Dropout 강화)
# ============================================================
def build_simple_model(input_shape, num_classes):
    """
    과적합을 방지하는 단순한 Bidirectional LSTM (L2 정규화 및 Dropout 강화)

    Args:
        input_shape: (max_sequence_length, input_dim)
        num_classes: 분류할 클래스 개수

    Returns:
        compiled model
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=input_shape),
        Dropout(0.6), # v3: 0.5 -> v5: 0.6

        Bidirectional(LSTM(32, kernel_regularizer=l2(0.001))),
        Dropout(0.6), # v3: 0.5 -> v5: 0.6

        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.5), # v3: 0.4 -> v5: 0.5

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# 3. 데이터 로드 (v3와 동일)
# ============================================================
def load_and_preprocess_data(data_dir="coordinates_output"):
    """
    .npy 키포인트 파일과 metadata.csv 로드
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path, encoding='utf-8-sig')

    X_data = []
    y_labels = []

    unique_labels = sorted(metadata_df['label'].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}

    all_keypoints = []

    for _, row in metadata_df.iterrows():
        video_id = row['id']
        label = row['label']
        npy_path = os.path.join(data_dir, f"{video_id}_keypoints.npy")

        if os.path.exists(npy_path):
            keypoints = np.load(npy_path)
            flattened_keypoints = keypoints.reshape(keypoints.shape[0], -1)
            all_keypoints.append(flattened_keypoints)
            X_data.append(flattened_keypoints)
            y_labels.append(label_to_int[label])

    # 전역 스케일러
    all_keypoints_concat = np.vstack(all_keypoints)
    global_scaler = MinMaxScaler()
    global_scaler.fit(all_keypoints_concat)

    X_scaled = [global_scaler.transform(x) for x in X_data]

    return X_scaled, y_labels, label_to_int, int_to_label, unique_labels, global_scaler


# ============================================================
# 4. 시퀀스 패딩 (v3와 동일)
# ============================================================
def pad_sequences(sequences, max_len=None):
    """
    모든 시퀀스를 동일한 길이로 맞춤
    """
    if max_len is None:
        max_len = max(s.shape[0] for s in sequences)

    padded_sequences = []
    for seq in sequences:
        if seq.shape[0] >= max_len:
            padded_sequences.append(seq[:max_len])
        else:
            padding = np.zeros((max_len - seq.shape[0], seq.shape[1]))
            padded_sequences.append(np.vstack((seq, padding)))
    return np.array(padded_sequences)


# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    data_dir = "coordinates_output"
    model_dir = "trained_model/trained_model_v5"
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("train_and_save_model_v5.py")
    print("=" * 60)
    print("K-Fold 교차 검증 및 과적합 방지 강화 적용")
    print("=" * 60)

    # ========================================
    # Step 1: 데이터 로드
    # ========================================
    print("\n" + "=" * 60)
    print("Step 1: 데이터 로딩")
    print("=" * 60)
    X_raw, y_raw_int, label_to_int, int_to_label, unique_labels, global_scaler = load_and_preprocess_data(data_dir)

    if not X_raw:
        print("로드된 데이터가 없습니다.")
        exit()

    num_classes = len(unique_labels)
    print(f"원본 데이터: {len(X_raw)}개")
    print(f"클래스 수: {num_classes}")
    print(f"클래스: {list(label_to_int.keys())}")

    # ========================================
    # Step 2: 패딩 (원본만)
    # ========================================
    print("\n" + "=" * 60)
    print("Step 2: 원본 데이터 패딩")
    print("=" * 60)
    max_sequence_length = max(s.shape[0] for s in X_raw)
    input_dim = X_raw[0].shape[1]
    X_padded_raw = pad_sequences(X_raw, max_len=max_sequence_length)
    # y_raw_int는 KFold 분할에 사용하기 위해 to_categorical 전의 정수 레이블 유지

    print(f"Max sequence length: {max_sequence_length}")
    print(f"Input dimension: {input_dim}")
    print(f"Padded shape: {X_padded_raw.shape}")

    # input_shape 정의를 K-Fold 루프 밖으로 이동
    input_shape = (max_sequence_length, input_dim) # <-- 이 줄을 추가/이동했습니다.

    # ========================================
    # Step 3: K-Fold 교차 검증 도입
    # ========================================
    print("\n" + "=" * 60)
    print("Step 3: K-Fold 교차 검증 시작")
    print("=" * 60)
    print("모델 일반화 성능을 더 신뢰성 있게 평가합니다.")

    # 원본 데이터와 레이블을 numpy 배열로 변환
    X_padded_raw_np = np.array(X_padded_raw)
    y_raw_np = np.array(y_raw_int) # StratifiedKFold에 사용할 정수 레이블

    n_splits = 5 # 5-Fold 교차 검증
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    best_overall_accuracy = -1
    best_model = None
    best_model_info = None
    best_fold_num = -1

    for fold, (train_index, test_index) in enumerate(skf.split(X_padded_raw_np, y_raw_np)):
        print(f"\n--- Fold {fold+1}/{n_splits} 시작 ---")

        X_train_fold_raw, X_test_fold_raw = X_padded_raw_np[train_index], X_padded_raw_np[test_index]
        y_train_fold_raw_int, y_test_fold_raw_int = y_raw_np[train_index], y_raw_np[test_index]

        # y_categorical로 변환 (훈련 및 검증에 사용)
        y_train_fold_raw_cat = to_categorical(y_train_fold_raw_int, num_classes=num_classes)
        y_test_fold_raw_cat = to_categorical(y_test_fold_raw_int, num_classes=num_classes)

        print(f"Fold {fold+1} Train 원본: {X_train_fold_raw.shape[0]}개")
        print(f"Fold {fold+1} Test 원본: {X_test_fold_raw.shape[0]}개")

        # ========================================
        # Step 4: Train 데이터만 증강 (각 Fold마다)
        # ========================================
        print(f"Fold {fold+1}: 훈련 데이터만 증강 중...")
        X_train_list = []
        y_train_list = []

        for i in range(X_train_fold_raw.shape[0]):
            # 패딩 제거 (실제 프레임만)
            seq = X_train_fold_raw[i]
            non_zero_frames = np.any(seq != 0, axis=1)
            original_seq = seq[non_zero_frames]

            # 증강 (원본 + 5개)
            augmented = augment_sequence_realistic(original_seq, num_augmentations=8)

            # 다시 패딩
            for aug_seq in augmented:
                if aug_seq.shape[0] >= max_sequence_length:
                    padded = aug_seq[:max_sequence_length]
                else:
                    padding = np.zeros((max_sequence_length - aug_seq.shape[0], aug_seq.shape[1]))
                    padded = np.vstack((aug_seq, padding))

                X_train_list.append(padded)
                y_train_list.append(y_train_fold_raw_cat[i]) # 증강된 데이터에 원본 레이블 할당

        X_train_augmented = np.array(X_train_list)
        y_train_augmented = np.array(y_train_list)

        print(f"Fold {fold+1} 증강 후 Train: {X_train_augmented.shape}")
        print(f"Fold {fold+1} 증강 배수: {X_train_augmented.shape[0] / X_train_fold_raw.shape[0]:.1f}배")

        # ========================================
        # Step 5: 모델 생성 (각 Fold마다 초기화)
        # ========================================
        print(f"Fold {fold+1}: 모델 생성 중...")
        model = build_simple_model(input_shape, num_classes)
        # model.summary() # 각 Fold마다 출력하면 너무 길어지므로 주석 처리

        # ========================================
        # Step 6: 콜백 설정
        # ========================================
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0 # 각 Fold마다 출력 줄임
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=0 # 각 Fold마다 출력 줄임
        )

        # ========================================
        # Step 7: 학습 (각 Fold마다)
        # ========================================
        print(f"Fold {fold+1}: 모델 학습 시작...")
        history = model.fit(
            X_train_augmented, y_train_augmented,
            epochs=150,
            batch_size=16,
            validation_data=(X_test_fold_raw, y_test_fold_raw_cat),  # 원본 Test로 검증
            callbacks=[early_stopping, lr_scheduler],
            verbose=0 # 각 Fold마다 출력 줄임
        )
        print(f"Fold {fold+1}: 모델 학습 완료.")

        # ========================================
        # Step 8: 평가 (각 Fold마다)
        # ========================================
        loss, accuracy = model.evaluate(X_test_fold_raw, y_test_fold_raw_cat, verbose=0)
        print(f"--- Fold {fold+1} 결과: 테스트 손실: {loss:.4f}, 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%) ---")
        fold_accuracies.append(accuracy)

        # 가장 좋은 모델 저장 로직
        if accuracy > best_overall_accuracy:
            best_overall_accuracy = accuracy
            best_model = model
            best_model_info = {
                'label_to_int': label_to_int,
                'int_to_label': int_to_label,
                'max_sequence_length': max_sequence_length,
                'input_dim': input_dim,
                'num_classes': num_classes,
                'scaler': global_scaler
            }
            best_fold_num = fold + 1

    print("\n" + "=" * 60)
    print("K-Fold 교차 검증 완료!")
    print("=" * 60)
    print(f"모든 Fold의 테스트 정확도: {[f'{acc*100:.2f}%' for acc in fold_accuracies]}")
    print(f"평균 테스트 정확도: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")

    # ========================================
    # Step 9: 가장 좋은 모델 저장
    # ========================================
    if best_model:
        print("\n" + "=" * 60)
        print("Step 9: 가장 좋은 모델 저장")
        print("=" * 60)

        model_save_path = os.path.join(model_dir, "sign_language_model.h5")
        info_save_path = os.path.join(model_dir, "model_info.pkl")

        best_model.save(model_save_path)
        with open(info_save_path, 'wb') as f:
            pickle.dump(best_model_info, f)

        print(f"가장 높은 정확도 ({best_overall_accuracy*100:.2f}%)를 보인 Fold {best_fold_num}의 모델 저장 완료:")
        print(f"  - {model_save_path}")
        print(f"  - {info_save_path}")
    else:
        print("저장할 모델이 없습니다. 학습 과정에 문제가 있었을 수 있습니다.")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print("\n예측 실행 방법:")
    print(f"   python .\\train_predict_model\\predict_sign_language.py --video \".\\train_predict_model\\test.mp4\" --model-dir \"{model_dir}\" --show-probs")
    print("\n" + "=" * 60)