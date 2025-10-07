import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import random

# ============================================================
# train_and_save_model_v3.py
# ============================================================
# 
# 파일 위치: train_predict_model/train_and_save_model_v3.py
# 저장 위치: trained_model/trained_model_v3/
#
# ============================================================
# 실행 방법
# ============================================================
# 
# 학습:
#    python .\train_predict_model\train_and_save_model_v3.py
#
# 예측:
#    python .\train_predict_model\predict_sign_language.py --video "test.mp4" --model-dir "trained_model\trained_model_v3" --show-probs
#
# ============================================================
# 주요 특징 및 개선 사항
# ============================================================
# 
# 1. 올바른 데이터 증강 방식
#    - v2의 치명적 문제 해결: Train/Test 분리 후 증강
#    - Train 데이터만 증강하여 과적합 방지
#    - Test는 원본 그대로 사용하여 실제 성능 측정
#
# 2. 단순화된 모델 구조
#    - v2의 복잡한 Attention 제거 (438K → 약 100K 파라미터)
#    - Bidirectional LSTM 2층 (64, 32 유닛)
#    - Dropout 0.5로 증가하여 과적합 방지 강화
#
# 3. 데이터 증강 5배
#    - v2의 15배 → 5배로 감소
#    - 원본 111개 → 약 530개 (적절한 균형)
#    - 4가지 증강 기법 순환 적용
#
# 4. 증강 기법 (v2와 동일)
#    - 시간적 스케일링: 수어 속도 변화 (0.85~1.15배)
#    - 가우시안 노이즈: 손 떨림 시뮬레이션 (0.2~0.8%)
#    - 위치 이동: 카메라 각도 변화 (±3%)
#    - 스케일 변화: 카메라 거리 변화 (0.95~1.05배)
#
# 5. 학습 전략
#    - 학습률: 0.001 (v2의 0.0005보다 높게)
#    - 배치 크기: 16
#    - Patience: 20 (v2의 30에서 감소)
#    - ReduceLROnPlateau: Patience 7
#
# 6. 예상 성능
#    - 목표 정확도: 70-80% (100%는 과적합!)
#    - Train: 85-95%, Validation: 70-80%
#    - 실제 새로운 영상에서도 작동하는 모델
#
# 7. 저장 위치
#    - 모델: trained_model/trained_model_v3/sign_language_model.h5
#    - 설정: trained_model/trained_model_v3/model_info.pkl
#
# ============================================================
# v2와의 주요 차이점
# ============================================================
# 
# v2 (잘못된 방법):
#   데이터 로드 → 증강 → Train/Test 분리
#   결과: 100% 정확도 (과적합, 실제로는 사용 불가)
#
# v3 (올바른 방법):
#   데이터 로드 → Train/Test 분리 → Train만 증강
#   결과: 70-80% 정확도 (일반화 성능 우수)
#
# ============================================================


# ============================================================
# 1. 개선된 데이터 증강 함수
# ============================================================
def augment_sequence_realistic(sequence: np.ndarray, num_augmentations: int = 5):
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
# 2. 단순화된 Bidirectional LSTM 모델
# ============================================================
def build_simple_model(input_shape, num_classes):
    """
    과적합을 방지하는 단순한 Bidirectional LSTM
    
    Args:
        input_shape: (max_sequence_length, input_dim)
        num_classes: 분류할 클래스 개수
    
    Returns:
        compiled model
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.5),
        
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        
        Dense(32, activation='relu'),
        Dropout(0.4),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================
# 3. 데이터 로드
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
# 4. 시퀀스 패딩
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
    model_dir = "trained_model/trained_model_v3"
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("train_and_save_model_v3.py")
    print("=" * 60)
    print("과적합 방지를 위한 올바른 데이터 증강 방식 적용")
    print("=" * 60)
    
    # ========================================
    # Step 1: 데이터 로드 (증강 X)
    # ========================================
    print("\n" + "=" * 60)
    print("Step 1: 데이터 로딩")
    print("=" * 60)
    X_raw, y_raw, label_to_int, int_to_label, unique_labels, global_scaler = load_and_preprocess_data(data_dir)

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
    y_categorical_raw = to_categorical(y_raw, num_classes=num_classes)
    
    print(f"Max sequence length: {max_sequence_length}")
    print(f"Input dimension: {input_dim}")
    print(f"Padded shape: {X_padded_raw.shape}")
    
    # ========================================
    # Step 3: Train/Test 분리 (원본만)
    # ========================================
    print("\n" + "=" * 60)
    print("Step 3: Train/Test 분리 (증강 전)")
    print("=" * 60)
    print("중요: 분리 후 Train만 증강하여 과적합 방지!")
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, idx_train, idx_test = train_test_split(
        X_padded_raw, y_categorical_raw, range(len(X_raw)),
        test_size=0.2,
        random_state=42,
        stratify=y_raw
    )
    
    print(f"Train 원본: {X_train_raw.shape[0]}개")
    print(f"Test 원본: {X_test_raw.shape[0]}개")
    
    # ========================================
    # Step 4: Train 데이터만 증강
    # ========================================
    print("\n" + "=" * 60)
    print("Step 4: 훈련 데이터만 증강")
    print("=" * 60)
    print("Test는 원본 유지 -> 실제 성능 측정 가능")
    
    X_train_list = []
    y_train_list = []
    
    for i in range(X_train_raw.shape[0]):
        # 패딩 제거 (실제 프레임만)
        seq = X_train_raw[i]
        non_zero_frames = np.any(seq != 0, axis=1)
        original_seq = seq[non_zero_frames]
        
        # 증강 (원본 + 5개)
        augmented = augment_sequence_realistic(original_seq, num_augmentations=5)
        
        # 다시 패딩
        for aug_seq in augmented:
            if aug_seq.shape[0] >= max_sequence_length:
                padded = aug_seq[:max_sequence_length]
            else:
                padding = np.zeros((max_sequence_length - aug_seq.shape[0], aug_seq.shape[1]))
                padded = np.vstack((aug_seq, padding))
            
            X_train_list.append(padded)
            y_train_list.append(y_train_raw[i])
    
    X_train_augmented = np.array(X_train_list)
    y_train_augmented = np.array(y_train_list)
    
    print(f"증강 후 Train: {X_train_augmented.shape}")
    print(f"증강 배수: {X_train_augmented.shape[0] / X_train_raw.shape[0]:.1f}배")
    print(f"Test (증강 X): {X_test_raw.shape}")
    
    # 클래스별 분포 확인
    from collections import Counter
    train_class_counts = Counter(np.argmax(y_train_augmented, axis=1))
    test_class_counts = Counter(np.argmax(y_test_raw, axis=1))
    
    print("\n[Train 클래스별 데이터]")
    for class_id, count in sorted(train_class_counts.items()):
        print(f"  {int_to_label[class_id]}: {count}개")
    
    print("\n[Test 클래스별 데이터]")
    for class_id, count in sorted(test_class_counts.items()):
        print(f"  {int_to_label[class_id]}: {count}개")
    
    # ========================================
    # Step 5: 모델 생성
    # ========================================
    print("\n" + "=" * 60)
    print("Step 5: 모델 생성")
    print("=" * 60)
    input_shape = (max_sequence_length, input_dim)
    model = build_simple_model(input_shape, num_classes)
    model.summary()
    
    # ========================================
    # Step 6: 콜백 설정
    # ========================================
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
    
    # ========================================
    # Step 7: 학습
    # ========================================
    print("\n" + "=" * 60)
    print("Step 7: 모델 학습")
    print("=" * 60)
    print("Validation에 원본 Test 사용 -> 진짜 성능 확인!")
    
    history = model.fit(
        X_train_augmented, y_train_augmented,
        epochs=150,
        batch_size=16,
        validation_data=(X_test_raw, y_test_raw),  # 원본 Test로 검증
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    # ========================================
    # Step 8: 평가
    # ========================================
    print("\n" + "=" * 60)
    print("Step 8: 최종 평가")
    print("=" * 60)
    loss, accuracy = model.evaluate(X_test_raw, y_test_raw, verbose=0)
    
    print(f"\n테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy > 0.95:
        print("\n경고: 정확도가 95% 이상입니다.")
        print("과적합 가능성이 있으니 실제 영상으로 테스트해보세요.")
    elif accuracy < 0.5:
        print("\n경고: 정확도가 50% 미만입니다.")
        print("데이터 증강을 늘리거나 모델을 조정해보세요.")
    else:
        print("\n적절한 정확도입니다!")
        print("실제 영상에서도 잘 작동할 가능성이 높습니다.")
    
    # ========================================
    # Step 9: 저장
    # ========================================
    print("\n" + "=" * 60)
    print("Step 9: 모델 저장")
    print("=" * 60)
    
    model.save(os.path.join(model_dir, "sign_language_model.h5"))
    
    model_info = {
        'label_to_int': label_to_int,
        'int_to_label': int_to_label,
        'max_sequence_length': max_sequence_length,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'scaler': global_scaler
    }
    
    with open(os.path.join(model_dir, "model_info.pkl"), 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"저장 완료: {model_dir}/")
    print(f"  - sign_language_model.h5")
    print(f"  - model_info.pkl")
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)
    print("\n예측 실행 방법:")
    print(f"   python .\\train_predict_model\\predict_sign_language.py --video \"test.mp4\" --model-dir \"{model_dir}\" --show-probs")
    print("\n" + "=" * 60)