import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.layers import Dense, Dropout, Concatenate, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import random

# ============================================================
# train_and_save_model_v4.py - CNN + BiLSTM 하이브리드
# ============================================================
# 
# 파일 위치: train_predict_model/train_and_save_model_v4.py
# 저장 위치: trained_model/trained_model_v4/
#
# ============================================================
# 실행 방법
# ============================================================
# 
# 학습:
#    python .\train_predict_model\train_and_save_model_v4.py
#
# 예측:
#    python .\train_predict_model\predict_sign_language.py --video "test.mp4" --model-dir "trained_model\trained_model_v4" --show-probs
#
# ============================================================
# CNN + BiLSTM 하이브리드란?
# ============================================================
#
# 🔍 CNN (Convolutional Neural Network)
#    - 한글: 합성곱 신경망
#    - 역할: "사진에서 패턴 찾기"
#    - 예시: 손 모양, 손가락 각도, 팔 위치 등의 "공간적 패턴" 인식
#    - 비유: 그림을 보고 "아, 이건 주먹 모양이구나" 하고 알아차리는 것
#
# 🔍 BiLSTM (Bidirectional Long Short-Term Memory)
#    - 한글: 양방향 장단기 메모리
#    - 역할: "동작의 순서 이해하기"
#    - 예시: 손이 위→아래→옆으로 움직이는 "시간적 순서" 학습
#    - 비유: 영화를 앞뒤로 보면서 "이 동작은 이렇게 진행되는구나" 이해하는 것
#
# 💡 왜 둘을 결합하나?
#    수어 = 손 모양(CNN) + 손 동작 순서(BiLSTM)
#    
#    예: "수학" 수어
#    - CNN이 인식: "두 손이 이런 모양으로 벌어져 있네"
#    - BiLSTM이 인식: "그 손이 이렇게 움직이는구나"
#    - 결합: "아하! 이건 '수학' 이구나!"
#
# ============================================================
# v3 대비 주요 개선 사항
# ============================================================
#
# 1. CNN 브랜치 추가
#    - Conv1D 레이어로 공간적 특징 추출
#    - 손 모양, 위치 관계를 더 잘 파악
#
# 2. BiLSTM 브랜치 유지
#    - 시간적 패턴 학습
#    - 동작 순서를 이해
#
# 3. 두 브랜치 결합 (Concatenate)
#    - CNN의 공간 정보 + LSTM의 시간 정보
#    - 더 풍부한 특징으로 판단
#
# 4. 데이터 증강 20배로 증가
#    - v3: 5배 → v4: 20배
#    - 데이터 부족 문제 완화
#
# 5. 공격적 증강 기법 추가
#    - 회전, 좌우 반전 등
#    - 다양한 환경 시뮬레이션
#
# 6. Class Weight 적용
#    - 클래스 불균형 자동 보정
#    - 잘 못 맞추는 클래스에 가중치
#
# 7. BatchNormalization 추가
#    - 학습 안정화
#    - 수렴 속도 향상
#
# ============================================================
# 예상 성능
# ============================================================
#
# v3 결과 (실제): 40% (직접 촬영 영상)
# v4 예상: 60-70% (개선 목표)
#
# ============================================================


# ============================================================
# 1. 공격적 데이터 증강 함수
# ============================================================
def augment_sequence_aggressive(sequence: np.ndarray, num_augmentations: int = 20):
    """
    더 다양하고 공격적인 증강으로 데이터 부족 문제 해결
    
    여러 증강 기법을 동시에 랜덤하게 적용하여
    실제 촬영 환경의 다양성을 시뮬레이션
    
    Args:
        sequence: 원본 시퀀스 (frames, features)
        num_augmentations: 생성할 증강 데이터 개수
    
    Returns:
        augmented_sequences: [원본 + 증강된 시퀀스들]
    """
    augmented_sequences = [sequence]  # 원본 포함
    
    for i in range(num_augmentations):
        aug_seq = sequence.copy()
        
        # 1. 속도 변화 (50% 확률)
        if np.random.random() < 0.5:
            skip_factor = np.random.uniform(0.7, 1.3)
            if aug_seq.shape[0] > 10:
                new_len = int(aug_seq.shape[0] * skip_factor)
                if new_len > 5:
                    indices = np.linspace(0, aug_seq.shape[0] - 1, new_len).astype(int)
                    aug_seq = aug_seq[indices]
        
        # 2. 노이즈 추가 (50% 확률)
        if np.random.random() < 0.5:
            noise_level = np.random.uniform(0.002, 0.015)
            aug_seq += np.random.normal(0, noise_level, aug_seq.shape)
            aug_seq = np.clip(aug_seq, 0.0, 1.0)
        
        # 3. 위치 이동 (50% 확률)
        if np.random.random() < 0.5:
            shift_x = np.random.uniform(-0.05, 0.05)
            shift_y = np.random.uniform(-0.05, 0.05)
            aug_seq[:, ::2] += shift_x
            aug_seq[:, 1::2] += shift_y
            aug_seq = np.clip(aug_seq, 0.0, 1.0)
        
        # 4. 스케일 변화 (50% 확률)
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.85, 1.15)
            center_x, center_y = 0.5, 0.5
            aug_seq[:, ::2] = (aug_seq[:, ::2] - center_x) * scale + center_x
            aug_seq[:, 1::2] = (aug_seq[:, 1::2] - center_y) * scale + center_y
            aug_seq = np.clip(aug_seq, 0.0, 1.0)
        
        # 5. 회전 (30% 확률) - 새로 추가
        if np.random.random() < 0.3:
            angle = np.random.uniform(-10, 10)
            aug_seq = rotate_keypoints(aug_seq, angle)
        
        # 6. 좌우 반전 (10% 확률) - 조심스럽게
        if np.random.random() < 0.1:
            aug_seq[:, ::2] = 1.0 - aug_seq[:, ::2]
        
        augmented_sequences.append(aug_seq)
    
    return augmented_sequences


def rotate_keypoints(keypoints, angle_degrees):
    """
    키포인트를 중심점 기준으로 회전
    
    카메라 각도가 약간 틀어진 상황 시뮬레이션
    """
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated = keypoints.copy()
    center_x, center_y = 0.5, 0.5
    
    for i in range(0, keypoints.shape[1], 2):
        x = keypoints[:, i] - center_x
        y = keypoints[:, i+1] - center_y
        
        rotated[:, i] = x * cos_a - y * sin_a + center_x
        rotated[:, i+1] = x * sin_a + y * cos_a + center_y
    
    return np.clip(rotated, 0.0, 1.0)


# ============================================================
# 2. CNN + BiLSTM 하이브리드 모델
# ============================================================
def build_cnn_bilstm_model(input_shape, num_classes):
    """
    CNN과 BiLSTM을 결합한 하이브리드 모델
    
    구조:
    ┌─────────────┐
    │   Input     │ (프레임별 키포인트)
    └──────┬──────┘
           ├─────────────────┬─────────────────┐
           ↓                 ↓                 ↓
    ┌─────────────┐   ┌─────────────┐   ┌──────────┐
    │  CNN 브랜치  │   │ LSTM 브랜치  │   │ 원본     │
    │  (손 모양)   │   │  (동작순서)  │   │          │
    └──────┬──────┘   └──────┬──────┘   └────┬─────┘
           └─────────────────┴──────────────────┘
                          ↓
                    ┌──────────┐
                    │  결합     │
                    └─────┬────┘
                          ↓
                    ┌──────────┐
                    │  분류     │
                    └──────────┘
    
    Args:
        input_shape: (max_sequence_length, input_dim)
        num_classes: 분류할 클래스 개수
    
    Returns:
        compiled model
    """
    inputs = Input(shape=input_shape)
    
    # ===== CNN 브랜치: 공간적 패턴 =====
    # "각 프레임에서 손 모양 특징 추출"
    
    # 첫 번째 Conv 레이어
    x1 = Conv1D(64, kernel_size=5, padding='same', activation='relu')(inputs)
    x1 = BatchNormalization()(x1)  # 학습 안정화
    x1 = Dropout(0.3)(x1)
    
    # 두 번째 Conv 레이어
    x1 = Conv1D(128, kernel_size=3, padding='same', activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.4)(x1)
    
    # ===== BiLSTM 브랜치: 시간적 패턴 =====
    # "프레임 순서대로 동작 흐름 이해"
    
    x2 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x2 = Dropout(0.5)(x2)
    
    x2 = Bidirectional(LSTM(32, return_sequences=True))(x2)
    x2 = Dropout(0.5)(x2)
    
    # ===== 두 브랜치 결합 =====
    # CNN의 공간 정보 + LSTM의 시간 정보
    
    merged = Concatenate()([x1, x2]) # (None, 149, 128+64)
    
    # 추가 LSTM으로 결합된 정보 처리
    x = Bidirectional(LSTM(32, return_sequences=True))(merged)
    x = Dropout(0.5)(x)
    
    # Global Average Pooling: 시퀀스 → 벡터
    x = GlobalAveragePooling1D()(x)
    
    # ===== 분류 레이어 =====
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # 모델 생성
    model = Model(inputs=inputs, outputs=outputs)
    
    # 컴파일
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
    model_dir = "trained_model/trained_model_v4"
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("train_and_save_model_v4.py - CNN + BiLSTM")
    print("=" * 60)
    print("공간 정보(CNN) + 시간 정보(BiLSTM) 하이브리드")
    print("=" * 60)
    
    # ========================================
    # Step 1: 데이터 로드
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
    # Step 2: 패딩
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
    # Step 3: Train/Test 분리
    # ========================================
    print("\n" + "=" * 60)
    print("Step 3: Train/Test 분리")
    print("=" * 60)
    
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, idx_train, idx_test = train_test_split(
        X_padded_raw, y_categorical_raw, range(len(X_raw)),
        test_size=0.2,
        random_state=42,
        stratify=y_raw
    )
    
    print(f"Train 원본: {X_train_raw.shape[0]}개")
    print(f"Test 원본: {X_test_raw.shape[0]}개")
    
    # ========================================
    # Step 4: Train 데이터 증강 (20배)
    # ========================================
    print("\n" + "=" * 60)
    print("Step 4: 공격적 데이터 증강 (20배)")
    print("=" * 60)
    
    X_train_list = []
    y_train_list = []
    
    for i in range(X_train_raw.shape[0]):
        seq = X_train_raw[i]
        non_zero_frames = np.any(seq != 0, axis=1)
        original_seq = seq[non_zero_frames]
        
        # 20배 증강
        augmented = augment_sequence_aggressive(original_seq, num_augmentations=20)
        
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
    
    # 클래스별 분포
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
    # Step 5: Class Weight 계산
    # ========================================
    print("\n" + "=" * 60)
    print("Step 5: Class Weight 계산")
    print("=" * 60)
    
    y_integers = np.argmax(y_train_augmented, axis=1)
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_integers),
        y=y_integers
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    print("클래스별 가중치:")
    for class_id, weight in class_weight_dict.items():
        print(f"  {int_to_label[class_id]}: {weight:.4f}")
    
    # ========================================
    # Step 6: 모델 생성
    # ========================================
    print("\n" + "=" * 60)
    print("Step 6: CNN + BiLSTM 모델 생성")
    print("=" * 60)
    input_shape = (max_sequence_length, input_dim)
    model = build_cnn_bilstm_model(input_shape, num_classes)
    model.summary()
    
    # ========================================
    # Step 7: 콜백 설정
    # ========================================
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=1
    )
    
    # ========================================
    # Step 8: 학습
    # ========================================
    print("\n" + "=" * 60)
    print("Step 8: 모델 학습")
    print("=" * 60)
    
    history = model.fit(
        X_train_augmented, y_train_augmented,
        epochs=200,
        batch_size=16,
        validation_data=(X_test_raw, y_test_raw),
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight_dict,  # Class Weight 적용
        verbose=1
    )
    
    # ========================================
    # Step 9: 평가
    # ========================================
    print("\n" + "=" * 60)
    print("Step 9: 최종 평가")
    print("=" * 60)
    loss, accuracy = model.evaluate(X_test_raw, y_test_raw, verbose=0)
    
    print(f"\n테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    if accuracy > 0.95:
        print("\n경고: 정확도가 95% 이상입니다.")
        print("과적합 가능성이 있으니 직접 촬영한 영상으로 테스트해보세요.")
    elif accuracy >= 0.60:
        print("\n좋습니다! 적절한 정확도입니다.")
        print("직접 촬영한 영상으로 실제 성능을 확인해보세요.")
    else:
        print("\n개선 필요: 정확도가 낮습니다.")
        print("더 많은 데이터를 수집하거나 증강을 늘려보세요.")
    
    # ========================================
    # Step 10: 저장
    # ========================================
    print("\n" + "=" * 60)
    print("Step 10: 모델 저장")
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