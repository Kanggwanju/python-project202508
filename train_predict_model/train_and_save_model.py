import numpy as np
import pandas as pd
import os
import pickle
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

    X_data = []
    y_labels = []
    video_ids = []

    # 라벨 인코딩을 위한 맵 생성
    unique_labels = sorted(metadata_df['label'].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for i, label in enumerate(unique_labels)}

    # 모든 데이터를 수집 (전역 스케일러를 위해)
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
        else:
            print(f"경고: {npy_path} 파일을 찾을 수 없습니다. 건너뜀.")

    # 전역 스케일러 생성 및 학습
    all_keypoints_concat = np.vstack(all_keypoints)
    global_scaler = MinMaxScaler()
    global_scaler.fit(all_keypoints_concat)
    
    # 스케일링 적용
    X_scaled = [global_scaler.transform(x) for x in X_data]

    return X_scaled, y_labels, video_ids, label_to_int, int_to_label, unique_labels, global_scaler

# 2. 데이터 증강 함수
def augment_sequence(sequence: np.ndarray, num_augmentations: int = 5):
    """
    단일 시퀀스 데이터에 대해 간단한 증강을 수행합니다.
    """
    augmented_sequences = [sequence]

    for _ in range(num_augmentations):
        aug_seq = sequence.copy()

        # 1. 시간적 스케일링
        if random.random() < 0.7 and aug_seq.shape[0] > 10:
            skip_factor = random.uniform(0.7, 1.4)
            new_len = int(aug_seq.shape[0] * skip_factor)
            if new_len < 5: new_len = 5
            
            indices = np.linspace(0, aug_seq.shape[0] - 1, new_len).astype(int)
            aug_seq = aug_seq[indices]

        # 2. 가우시안 노이즈 추가
        noise_level = random.uniform(0.001, 0.020)
        aug_seq += np.random.normal(0, noise_level, aug_seq.shape)
        aug_seq = np.clip(aug_seq, 0.0, 1.0)

        # 3. 미세한 위치 이동
        if random.random() < 0.5:
            shift_x = random.uniform(-0.05, 0.05)
            shift_y = random.uniform(-0.05, 0.05)
            aug_seq[:, ::2] += shift_x
            aug_seq[:, 1::2] += shift_y
            aug_seq = np.clip(aug_seq, 0.0, 1.0)

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

# 4. 시퀀스 패딩
def pad_sequences(sequences, max_len=None):
    """
    시퀀스 길이를 맞추기 위해 패딩을 적용합니다.
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


# 메인 실행
if __name__ == "__main__":
    data_dir = "coordinates_output"
    model_dir = "trained_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # 1. 데이터 로드 및 전처리
    print("데이터 로딩 중...")
    X_raw, y_raw, video_ids_raw, label_to_int, int_to_label, unique_labels, global_scaler = load_and_preprocess_data(data_dir)

    if not X_raw:
        print("로드된 데이터가 없습니다. 스크립트를 종료합니다.")
        exit()

    num_classes = len(unique_labels)
    print(f"총 {len(X_raw)}개의 원본 시퀀스 로드됨. 클래스 수: {num_classes}")
    print(f"클래스 매핑: {label_to_int}")

    # 데이터 증강 적용
    print("\n데이터 증강 중...")
    X_augmented = []
    y_augmented = []
    for i, seq in enumerate(X_raw):
        augmented_seqs = augment_sequence(seq, num_augmentations=10) 
        X_augmented.extend(augmented_seqs)
        y_augmented.extend([y_raw[i]] * len(augmented_seqs))
    
    print(f"증강 후 총 {len(X_augmented)}개의 시퀀스 생성됨.")

    # 모든 시퀀스의 최대 길이 찾기
    max_sequence_length = max(s.shape[0] for s in X_augmented)
    input_dim = X_augmented[0].shape[1]

    # 시퀀스 패딩
    print("\n시퀀스 패딩 중...")
    X_padded = pad_sequences(X_augmented, max_len=max_sequence_length)
    y_categorical = to_categorical(y_augmented, num_classes=num_classes)

    print(f"패딩 후 시퀀스 shape: {X_padded.shape}")
    print(f"라벨 shape: {y_categorical.shape}")

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, y_categorical, test_size=0.2, random_state=42, stratify=y_augmented
    )
    
    print(f"\n훈련 데이터 shape: {X_train.shape}")
    print(f"테스트 데이터 shape: {X_test.shape}")

    # 모델 빌드 및 학습
    print("\n모델 생성 중...")
    input_shape = (max_sequence_length, input_dim)
    model = build_lstm_model(input_shape, num_classes)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    print("\n모델 학습 시작...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # 모델 평가
    print("\n모델 평가 중...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"테스트 손실: {loss:.4f}")
    print(f"테스트 정확도: {accuracy:.4f}")

    # 모델 및 관련 정보 저장
    print("\n모델 저장 중...")
    model.save(os.path.join(model_dir, "sign_language_model.h5"))
    
    # 라벨 매핑 및 스케일러, 설정 정보 저장
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
    
    print(f"\n저장 완료!")
    print(f"- 모델: {model_dir}/sign_language_model.h5")
    print(f"- 설정 정보: {model_dir}/model_info.pkl")
    print(f"\n학습된 수어 단어: {list(label_to_int.keys())}")