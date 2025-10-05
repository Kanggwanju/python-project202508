import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import random

# 1. Bidirectional LSTM + Attention - 양방향 학습으로 패턴 파악 향상
# 2. 데이터 증강 15배 - 111개 → 약 1,776개 (충분한 학습 데이터)
# 3. 학습률 스케줄러 - 학습이 정체되면 자동으로 학습률 감소
# 4. 배치 사이즈 16 - 더 안정적인 학습
# 5. Patience 30 - 충분한 학습 시간 제공
# 6. 개선된 데이터 증강 전략 (4가지)
#    - 시간적 스케일링(수어 동작의 속도 변화)
#    - 가우시안 노이즈 추가(손 떨림이나 키포인트 추출 오차)
#    - 위치 이동(카메라 앞에서 약간 왼쪽/오른쪽/위/아래로 이동)
#    - 스케일 변화(카메라와의 거리 변화)
# 7. 문제점: 심각한 과적합
#           - Epoch 8-50: 계속 예측 100% 유지 (데이터를 학습한 게 아니라 암기)
# 8. 문제 발생 이유: 데이터 증강이 잘못됨
#                   - train/test를 나누기 전에 증강 (같은 원본에서 나온 증강 데이터가 train/test에 모두 포함됨)
# 9. 문제 해결 방법: Train/Test 분리 후 증강

# ============================================================
# 1. Attention Layer 정의
# ============================================================
class AttentionLayer(Layer):
    """
    중요한 프레임에 가중치를 주는 Attention 메커니즘
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='att_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# ============================================================
# 2. 개선된 데이터 증강 함수
# ============================================================
def augment_sequence_realistic(sequence: np.ndarray, num_augmentations: int = 7):
    """
    더 현실적이고 절제된 데이터 증강
    """
    augmented_sequences = [sequence]  # 원본 포함
    
    for i in range(num_augmentations):
        aug_seq = sequence.copy()
        
        # 각 증강마다 다른 조합 적용
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
            # 작은 노이즈 추가
            noise_level = random.uniform(0.002, 0.008)
            aug_seq += np.random.normal(0, noise_level, aug_seq.shape)
            aug_seq = np.clip(aug_seq, 0.0, 1.0)
        
        elif augmentation_type == 2:
            # 위치 이동 (카메라 각도 변화)
            shift_x = random.uniform(-0.03, 0.03)
            shift_y = random.uniform(-0.03, 0.03)
            aug_seq[:, ::2] += shift_x
            aug_seq[:, 1::2] += shift_y
            aug_seq = np.clip(aug_seq, 0.0, 1.0)
        
        else:
            # 스케일 변화 (손 크기 변화)
            scale = random.uniform(0.95, 1.05)
            center_x, center_y = 0.5, 0.5
            aug_seq[:, ::2] = (aug_seq[:, ::2] - center_x) * scale + center_x
            aug_seq[:, 1::2] = (aug_seq[:, 1::2] - center_y) * scale + center_y
            aug_seq = np.clip(aug_seq, 0.0, 1.0)
        
        augmented_sequences.append(aug_seq)
    
    return augmented_sequences


# ============================================================
# 3. Bidirectional LSTM + Attention 모델
# ============================================================
def build_attention_model(input_shape, num_classes):
    """
    Bidirectional LSTM + Attention 메커니즘
    양방향으로 시퀀스를 학습하고 중요한 프레임에 집중
    """
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM 레이어
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.4)(x)
    
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    # Attention 메커니즘
    x = AttentionLayer()(x)
    
    # 분류 레이어
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


# ============================================================
# 4. 데이터 로드 (기존과 동일)
# ============================================================
def load_and_preprocess_data(data_dir="coordinates_output"):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path, encoding='utf-8-sig')

    X_data = []
    y_labels = []
    video_ids = []

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
            video_ids.append(video_id)

    all_keypoints_concat = np.vstack(all_keypoints)
    global_scaler = MinMaxScaler()
    global_scaler.fit(all_keypoints_concat)
    
    X_scaled = [global_scaler.transform(x) for x in X_data]

    return X_scaled, y_labels, video_ids, label_to_int, int_to_label, unique_labels, global_scaler


def pad_sequences(sequences, max_len=None):
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
    model_dir = "trained_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. 데이터 로드
    print("=" * 60)
    print("데이터 로딩 중...")
    print("=" * 60)
    X_raw, y_raw, video_ids_raw, label_to_int, int_to_label, unique_labels, global_scaler = load_and_preprocess_data(data_dir)

    if not X_raw:
        print("로드된 데이터가 없습니다. 스크립트를 종료합니다.")
        exit()

    num_classes = len(unique_labels)
    print(f"\n✓ 총 {len(X_raw)}개의 원본 시퀀스 로드됨")
    print(f"✓ 클래스 수: {num_classes}")
    print(f"✓ 클래스: {list(label_to_int.keys())}")

    # 2. 데이터 증강 - 15배로 증가
    print("\n" + "=" * 60)
    print("데이터 증강 중...")
    print("=" * 60)
    X_augmented = []
    y_augmented = []
    
    for i, seq in enumerate(X_raw):
        # 클래스별 증강 횟수 확인
        augmented_seqs = augment_sequence_realistic(seq, num_augmentations=15)
        X_augmented.extend(augmented_seqs)
        y_augmented.extend([y_raw[i]] * len(augmented_seqs))
    
    print(f"\n✓ 증강 후: {len(X_augmented)}개 시퀀스 생성")
    print(f"✓ 증강 배수: {len(X_augmented) / len(X_raw):.1f}배")

    # 클래스별 데이터 개수 확인
    from collections import Counter
    class_counts = Counter(y_augmented)
    print("\n[클래스별 데이터 개수]")
    for class_id, count in sorted(class_counts.items()):
        print(f"  {int_to_label[class_id]}: {count}개")

    # 3. 패딩
    max_sequence_length = max(s.shape[0] for s in X_augmented)
    input_dim = X_augmented[0].shape[1]

    print("\n" + "=" * 60)
    print("시퀀스 패딩 중...")
    print("=" * 60)
    X_padded = pad_sequences(X_augmented, max_len=max_sequence_length)
    y_categorical = to_categorical(y_augmented, num_classes=num_classes)

    print(f"\n✓ 패딩 후 shape: {X_padded.shape}")
    print(f"✓ 라벨 shape: {y_categorical.shape}")

    # 4. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_categorical, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_augmented
    )
    
    print(f"\n✓ 훈련 데이터: {X_train.shape}")
    print(f"✓ 테스트 데이터: {X_test.shape}")

    # 5. 모델 생성
    print("\n" + "=" * 60)
    print("Bidirectional LSTM + Attention 모델 생성 중...")
    print("=" * 60)
    input_shape = (max_sequence_length, input_dim)
    model = build_attention_model(input_shape, num_classes)
    
    # 학습률 조정
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # 6. 콜백 설정
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    # 7. 학습
    print("\n" + "=" * 60)
    print("모델 학습 시작...")
    print("=" * 60)
    history = model.fit(
        X_train, y_train,
        epochs=300,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    # 8. 평가
    print("\n" + "=" * 60)
    print("모델 평가 중...")
    print("=" * 60)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✓ 테스트 손실: {loss:.4f}")
    print(f"✓ 테스트 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 9. 저장
    print("\n" + "=" * 60)
    print("모델 저장 중...")
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
    
    print(f"\n✓ 저장 완료!")
    print(f"  - 모델: {model_dir}/sign_language_model.h5")
    print(f"  - 설정: {model_dir}/model_info.pkl")
    print("\n" + "=" * 60)