#--- START OF FILE simple_sign_language_model_no_flip.py ---

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import random

# 1. 데이터 로드 및 전처리 함수
def load_and_preprocess_data(data_dir="coordinates_output"):
    """
    저장된 .npy 좌표 파일과 metadata.csv를 로드하고 전처리합니다.
    """
    metadata_path = os.path.join(data_dir, "metadata.csv")
    metadata_df = pd.read_csv(metadata_path, encoding='utf-8-sig')

    X_data = [] # 시퀀스 데이터
    y_labels = [] # 라벨 (수어 단어)
    video_ids = [] # 원본 비디오 ID

    # 라벨 인코딩을 위한 맵 생성
    unique_labels = metadata_df['label'].unique()
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}

    for _, row in metadata_df.iterrows():
        video_id = row['id']
        label = row['label']
        npy_path = os.path.join(data_dir, f"{video_id}_keypoints.npy")

        if os.path.exists(npy_path):
            keypoints = np.load(npy_path)
            
            # 각 프레임의 모든 좌표를 하나의 벡터로 평탄화 (N_frames, N_keypoints * 2)
            flattened_keypoints = keypoints.reshape(keypoints.shape[0], -1)
            
            # MinMaxScaler 적용 (각 특징별로 0~1 스케일링)
            scaler = MinMaxScaler()
            scaled_keypoints = scaler.fit_transform(flattened_keypoints)
            
            X_data.append(scaled_keypoints)
            y_labels.append(label_to_int[label])
            video_ids.append(video_id)
        else:
            print(f"경고: {npy_path} 파일을 찾을 수 없습니다. 건너뜀.")

    return X_data, y_labels, video_ids, label_to_int, int_to_label, unique_labels

# 2. 데이터 증강 함수 (좌우 반전 제외)
def augment_sequence(sequence: np.ndarray, num_augmentations: int = 5):
    """
    단일 시퀀스 데이터에 대해 간단한 증강을 수행합니다.
    - 시간적 스케일링 (프레임 건너뛰기/복제)
    - 노이즈 추가
    """
    augmented_sequences = [sequence] # 원본 포함

    for _ in range(num_augmentations):
        aug_seq = sequence.copy()

        # 1. 시간적 스케일링 (랜덤하게 프레임 건너뛰기 또는 복제)
        if random.random() < 0.7 and aug_seq.shape[0] > 10: # 70% 확률로 적용, 최소 프레임 수 제한
            skip_factor = random.uniform(0.7, 1.4) # 0.7배 ~ 1.3배 길이
            new_len = int(aug_seq.shape[0] * skip_factor)
            if new_len < 5: new_len = 5 # 최소 길이 보장
            
            indices = np.linspace(0, aug_seq.shape[0] - 1, new_len).astype(int)
            aug_seq = aug_seq[indices]

        # 2. 가우시안 노이즈 추가
        noise_level = random.uniform(0.001, 0.020) # 작은 노이즈 범위 조정
        aug_seq += np.random.normal(0, noise_level, aug_seq.shape)
        
        # 좌표 범위 0~1 유지
        aug_seq = np.clip(aug_seq, 0.0, 1.0)

        # 3. (선택 사항) 미세한 위치 이동 (x, y 축)
        if random.random() < 0.5: # 50% 확률로 적용
            shift_x = random.uniform(-0.05, 0.05) # -0.02 ~ 0.02 범위로 이동
            shift_y = random.uniform(-0.05, 0.05)
            aug_seq[:, ::2] += shift_x # x 좌표에 이동 적용
            aug_seq[:, 1::2] += shift_y # y 좌표에 이동 적용
            aug_seq = np.clip(aug_seq, 0.0, 1.0) # 범위 유지

        augmented_sequences.append(aug_seq)
    
    return augmented_sequences

# 3. 모델 정의 함수
def build_lstm_model(input_shape, num_classes):
    """
    간단한 LSTM 모델을 정의합니다.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. 시퀀스 패딩 (길이 맞추기)
def pad_sequences(sequences, max_len=None):
    """
    시퀀스 길이를 맞추기 위해 패딩을 적용합니다.
    """
    if max_len is None:
        max_len = max(s.shape[0] for s in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if seq.shape[0] >= max_len:
            padded_sequences.append(seq[:max_len]) # 너무 길면 자름
        else:
            # 패딩 값은 0으로 설정 (MediaPipe 좌표는 0~1 사이)
            padding = np.zeros((max_len - seq.shape[0], seq.shape[1]))
            padded_sequences.append(np.vstack((seq, padding)))
    return np.array(padded_sequences)


# 메인 실행
if __name__ == "__main__":
    data_dir = "coordinates_output"
    
    # 1. 데이터 로드 및 전처리
    X_raw, y_raw, video_ids_raw, label_to_int, int_to_label, unique_labels = load_and_preprocess_data(data_dir)

    if not X_raw:
        print("로드된 데이터가 없습니다. 스크립트를 종료합니다.")
        exit()

    num_classes = len(unique_labels)
    print(f"총 {len(X_raw)}개의 원본 시퀀스 로드됨. 클래스 수: {num_classes}")
    print(f"클래스 매핑: {label_to_int}")

    # 데이터 증강 적용
    X_augmented = []
    y_augmented = []
    for i, seq in enumerate(X_raw):
        # 각 원본 시퀀스당 15개 증강 (원본 포함 총 16개)
        # 증강 횟수를 늘려 데이터 부족을 보완 시도
        augmented_seqs = augment_sequence(seq, num_augmentations=40) 
        X_augmented.extend(augmented_seqs)
        y_augmented.extend([y_raw[i]] * len(augmented_seqs))
    
    print(f"증강 후 총 {len(X_augmented)}개의 시퀀스 생성됨.")

    # 모든 시퀀스의 최대 길이 찾기 (패딩을 위해)
    max_sequence_length = max(s.shape[0] for s in X_augmented)
    input_dim = X_augmented[0].shape[1] # 각 프레임의 특징 차원 (키포인트 수 * 2)

    # 시퀀스 패딩
    X_padded = pad_sequences(X_augmented, max_len=max_sequence_length)
    y_categorical = to_categorical(y_augmented, num_classes=num_classes)

    print(f"패딩 후 시퀀스 shape: {X_padded.shape}")
    print(f"라벨 shape: {y_categorical.shape}")

    # 데이터 분할 (훈련/테스트)
    # 각 클래스당 1개의 원본 데이터이므로, 일반적인 train_test_split은 어렵습니다.
    # 여기서는 증강된 데이터를 섞어서 분할하지만, 실제 성능 평가는 LOOCV에 가깝게 해야 합니다.
    # 간단한 예시를 위해 증강된 데이터 전체를 섞어 분할합니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_categorical, test_size=0.2, random_state=42, stratify=y_augmented
    )
    
    print(f"훈련 데이터 shape: {X_train.shape}, 라벨 shape: {y_train.shape}")
    print(f"테스트 데이터 shape: {X_test.shape}, 라벨 shape: {y_test.shape}")

    # 5. 모델 빌드 및 학습
    input_shape = (max_sequence_length, input_dim)
    model = build_lstm_model(input_shape, num_classes)
    model.summary()

    # EarlyStopping 콜백 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True) # patience 증가

    print("\n모델 학습 시작...")
    history = model.fit(
        X_train, y_train,
        epochs=200, # 충분히 많은 에폭 설정, EarlyStopping이 조절
        batch_size=32,
        validation_split=0.2, # 훈련 데이터 내에서 검증 데이터 분할
        callbacks=[early_stopping],
        verbose=1
    )

    # 6. 모델 평가
    print("\n모델 평가 시작...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.4f}")

    # 7. 예측 예시 (선택 사항)
    print("\n예측 예시:")
    sample_index = random.randint(0, len(X_test) - 1)
    sample_X = X_test[sample_index:sample_index+1]
    sample_y_true_int = np.argmax(y_test[sample_index])
    sample_y_true_label = int_to_label[sample_y_true_int]

    prediction = model.predict(sample_X)
    predicted_class_int = np.argmax(prediction)
    predicted_class_label = int_to_label[predicted_class_int]

    print(f"실제 라벨: {sample_y_true_label}")
    print(f"예측 라벨: {predicted_class_label}")
    print(f"예측 확률: {prediction[0]}")

    # 모델 저장 (선택 사항)
    # model.save("simple_sign_language_model_no_flip.h5")
    # print("모델이 'simple_sign_language_model_no_flip.h5'로 저장되었습니다.")

#--- END OF FILE simple_sign_language_model_no_flip.py ---