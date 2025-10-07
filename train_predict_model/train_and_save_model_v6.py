"""
3D 좌표 기반 수어 인식 모델 학습 스크립트 (npy 파일 사용)
3d_coordinate_extractor로 추출된 데이터를 사용합니다.

데이터 구조:
coordinates_output/
  ├── metadata.csv  (id, label 정보 포함)
  ├── L_001_keypoints.npy
  ├── L_002_keypoints.npy
  └── ...

사용법:
# 기본 학습
python train_and_save_model_v6.py --data-dir coordinates_output

# 데이터 증강 적용
python train_and_save_model_v6.py --data-dir coordinates_output --augment

# 에포크 및 배치 크기 조정
python train_and_save_model_v6.py --data-dir coordinates_output --epochs 150 --batch-size 8

# 저장 경로 지정
python train_and_save_model_v6.py --data-dir coordinates_output --output-dir trained_model/my_model
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    Masking, GRU
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd

# seaborn은 선택적 의존성
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not found. Using matplotlib only for visualizations.")


class DataAugmentor:
    """데이터 증강 클래스"""
    
    @staticmethod
    def augment_sequence(sequence: np.ndarray, aug_type: str = 'noise'):
        """
        데이터 증강
        - noise: 가우시안 노이즈 추가
        - speed: 속도 변화 (시간 워핑)
        - scale: 크기 변화
        """
        if aug_type == 'noise':
            # 약한 가우시안 노이즈 추가
            noise = np.random.normal(0, 0.02, sequence.shape)
            return sequence + noise
        
        elif aug_type == 'speed':
            # 속도 변화 (0.8x ~ 1.2x)
            factor = np.random.uniform(0.8, 1.2)
            indices = np.linspace(0, len(sequence) - 1, int(len(sequence) * factor))
            indices = np.clip(indices, 0, len(sequence) - 1).astype(int)
            return sequence[indices]
        
        elif aug_type == 'scale':
            # 크기 변화 (0.9x ~ 1.1x)
            factor = np.random.uniform(0.9, 1.1)
            return sequence * factor
        
        return sequence


class SignLanguageTrainer:
    """수어 인식 모델 학습 클래스 (npy 파일 + metadata.csv 기반)"""
    
    def __init__(self, data_dir: str, output_dir: str = "trained_model/trained_model_v6"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.augmentor = DataAugmentor()
        
        self.label_to_int = {}
        self.int_to_label = {}
        self.max_sequence_length = 0
        self.input_dim = 147  # 49개 키포인트 × 3 (x, y, z)
        
        # metadata.csv 로드
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """metadata.csv 파일 로드"""
        metadata_path = self.data_dir / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.csv 파일을 찾을 수 없습니다: {metadata_path}\n"
                "3d_coordinate_extractor.py로 생성된 metadata.csv가 필요합니다."
            )
        
        df = pd.read_csv(metadata_path)
        
        # 필수 컬럼 확인
        required_cols = ['id', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"metadata.csv에 필수 컬럼이 없습니다: {missing_cols}")
        
        print(f"metadata.csv 로드 완료: {len(df)}개 항목")
        
        return df
        
    def load_data(self, augment: bool = False):
        """metadata.csv를 참조하여 npy 파일 로드"""
        print("\n" + "="*70)
        print("데이터 로딩 시작 (metadata.csv 기반)")
        print("="*70)
        
        sequences = []
        labels = []
        file_info = []
        
        # 고유한 라벨 추출 및 매핑 생성
        unique_labels = sorted(self.metadata['label'].unique())
        for idx, label in enumerate(unique_labels):
            self.label_to_int[label] = idx
            self.int_to_label[idx] = label
        
        print(f"\n발견된 수어 단어 ({len(self.label_to_int)}개):")
        for label, idx in self.label_to_int.items():
            count = len(self.metadata[self.metadata['label'] == label])
            print(f"  {idx}: {label} ({count}개 파일)")
        
        # 각 파일 로드
        print(f"\n총 {len(self.metadata)}개 파일 처리 중...")
        
        success_count = 0
        failed_files = []
        
        for _, row in self.metadata.iterrows():
            file_id = row['id']
            label = row['label']
            
            # npy 파일 경로 생성 (id_keypoints.npy 형식)
            npy_filename = f"{file_id}_keypoints.npy"
            npy_path = self.data_dir / npy_filename
            
            if not npy_path.exists():
                failed_files.append(f"{npy_filename} (파일 없음)")
                continue
            
            try:
                # npy 파일 로드
                data = np.load(npy_path)
                
                if len(data) == 0:
                    failed_files.append(f"{npy_filename} (빈 데이터)")
                    continue
                
                label_int = self.label_to_int[label]
                
                sequences.append(data)
                labels.append(label_int)
                file_info.append({
                    'file': npy_filename,
                    'id': file_id,
                    'label': label,
                    'frames': len(data)
                })
                success_count += 1
                
                # 데이터 증강
                if augment:
                    # 노이즈 버전
                    aug_seq = self.augmentor.augment_sequence(data, 'noise')
                    sequences.append(aug_seq)
                    labels.append(label_int)
                    
                    # 속도 변화 버전
                    aug_seq = self.augmentor.augment_sequence(data, 'speed')
                    sequences.append(aug_seq)
                    labels.append(label_int)
                    
            except Exception as e:
                failed_files.append(f"{npy_filename} ({str(e)})")
        
        print(f"\n로딩 결과:")
        print(f"  성공: {success_count}개")
        print(f"  실패: {len(failed_files)}개")
        
        if failed_files and len(failed_files) <= 10:
            print(f"\n실패한 파일:")
            for failed in failed_files:
                print(f"  - {failed}")
        elif len(failed_files) > 10:
            print(f"\n실패한 파일 (처음 10개):")
            for failed in failed_files[:10]:
                print(f"  - {failed}")
            print(f"  ... 외 {len(failed_files) - 10}개")
        
        if augment:
            print(f"\n데이터 증강 후: {len(sequences)}개 시퀀스 (원본 × 3)")
        else:
            print(f"\n총 시퀀스: {len(sequences)}개")
        
        if len(sequences) == 0:
            raise ValueError("로드 가능한 데이터가 없습니다.")
        
        # 라벨별 통계
        print(f"\n라벨별 데이터 수:")
        label_array = np.array(labels)
        for label, idx in sorted(self.label_to_int.items(), key=lambda x: x[1]):
            count = np.sum(label_array == idx)
            print(f"  {label}: {count}개")
        
        # 최대 시퀀스 길이 계산
        sequence_lengths = [len(seq) for seq in sequences]
        self.max_sequence_length = int(np.percentile(sequence_lengths, 95))
        print(f"\n시퀀스 길이 통계:")
        print(f"  95th percentile: {self.max_sequence_length} 프레임 (모델 입력 길이)")
        print(f"  평균: {np.mean(sequence_lengths):.1f} 프레임")
        print(f"  중앙값: {np.median(sequence_lengths):.1f} 프레임")
        print(f"  최소/최대: {min(sequence_lengths)}/{max(sequence_lengths)} 프레임")
        
        # 파일 정보 저장
        if file_info:
            df_info = pd.DataFrame(file_info)
            info_path = self.output_dir / "training_data_info.csv"
            df_info.to_csv(info_path, index=False, encoding='utf-8-sig')
            print(f"\n학습 데이터 정보 저장: {info_path}")
        
        return sequences, np.array(labels)
    
    def preprocess_data(self, sequences, labels):
        """데이터 전처리 및 분할"""
        print("\n" + "="*70)
        print("데이터 전처리")
        print("="*70)
        
        # 시퀀스 평탄화 및 스케일링
        all_frames = []
        for seq in sequences:
            flattened = seq.reshape(seq.shape[0], -1)
            all_frames.append(flattened)
        
        all_frames = np.vstack(all_frames)
        
        # 스케일러 학습
        print("스케일러 학습 중...")
        self.scaler.fit(all_frames)
        print(f"  특징 평균: {self.scaler.mean_[0]:.3f}, 표준편차: {self.scaler.scale_[0]:.3f}")
        
        # 각 시퀀스에 스케일링 및 패딩 적용
        processed_sequences = []
        for seq in sequences:
            flattened = seq.reshape(seq.shape[0], -1)
            scaled = self.scaler.transform(flattened)
            
            # 패딩 또는 자르기
            if scaled.shape[0] >= self.max_sequence_length:
                padded = scaled[:self.max_sequence_length]
            else:
                padding = np.zeros((self.max_sequence_length - scaled.shape[0], scaled.shape[1]))
                padded = np.vstack((scaled, padding))
            
            processed_sequences.append(padded)
        
        X = np.array(processed_sequences)
        y = to_categorical(labels, num_classes=len(self.label_to_int))
        
        print(f"\n전처리 완료:")
        print(f"  입력 shape: {X.shape}")
        print(f"  출력 shape: {y.shape}")
        
        # Train/Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\n데이터 분할:")
        print(f"  학습 데이터: {X_train.shape[0]}개 ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  테스트 데이터: {X_test.shape[0]}개 ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # 분할 후 라벨 분포 확인
        y_train_labels = np.argmax(y_train, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)
        
        print(f"\n학습 데이터 라벨 분포:")
        for label, idx in sorted(self.label_to_int.items(), key=lambda x: x[1]):
            train_count = np.sum(y_train_labels == idx)
            test_count = np.sum(y_test_labels == idx)
            print(f"  {label}: 학습 {train_count}개, 테스트 {test_count}개")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """모델 구축 (강력한 정규화 적용)"""
        print("\n" + "="*70)
        print("모델 구축 (강력한 정규화 버전)")
        print("="*70)
        
        model = Sequential([
            # Masking layer (패딩된 부분 무시)
            Masking(mask_value=0.0, input_shape=(self.max_sequence_length, self.input_dim)),
            
            # 첫 번째 Bi-LSTM (크기 감소 + dropout 증가)
            Bidirectional(LSTM(64, return_sequences=True, dropout=0.5, recurrent_dropout=0.3)),
            BatchNormalization(),
            
            # 두 번째 Bi-LSTM (크기 감소 + dropout 증가)
            Bidirectional(LSTM(32, return_sequences=False, dropout=0.5, recurrent_dropout=0.3)),
            BatchNormalization(),
            
            # Dense layers (L2 정규화 + dropout 증가)
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            
            # 출력층
            Dense(len(self.label_to_int), activation='softmax')
        ])
        
        # 컴파일 (학습률 감소)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n모델 구조:")
        model.summary()
        
        total_params = model.count_params()
        print(f"\n총 파라미터 수: {total_params:,}")
        print(f"\n정규화 설정:")
        print(f"  - LSTM 크기: 128→64, 64→32")
        print(f"  - Dropout: 0.5 (LSTM), 0.5 (Dense)")
        print(f"  - Recurrent Dropout: 0.3")
        print(f"  - L2 정규화: 0.01")
        print(f"  - 학습률: 0.001→0.0005")
        
        return model
    
    def train(self, model, X_train, X_test, y_train, y_test, epochs: int = 100, batch_size: int = 16):
        """모델 학습"""
        print("\n" + "="*70)
        print("모델 학습 시작")
        print("="*70)
        print(f"에포크: {epochs}, 배치 크기: {batch_size}")
        
        # 콜백 설정 (더 빠른 조기 종료)
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # 15→10으로 감소
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # 5→3으로 감소
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=str(self.output_dir / "best_model.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 학습
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n학습 완료!")
        print(f"  최종 학습 정확도: {history.history['accuracy'][-1]*100:.2f}%")
        print(f"  최종 검증 정확도: {history.history['val_accuracy'][-1]*100:.2f}%")
        print(f"  최종 학습 손실: {history.history['loss'][-1]:.4f}")
        print(f"  최종 검증 손실: {history.history['val_loss'][-1]:.4f}")
        
        # 과적합 경고
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        gap = train_acc - val_acc
        
        if gap > 0.15:
            print(f"\n⚠️  과적합 경고!")
            print(f"  학습/검증 정확도 차이: {gap*100:.2f}%")
            print(f"  추가 정규화 또는 더 많은 데이터가 필요합니다.")
        elif gap > 0.10:
            print(f"\n⚠️  약간의 과적합 감지")
            print(f"  학습/검증 정확도 차이: {gap*100:.2f}%")
        else:
            print(f"\n✅ 적절한 일반화 (차이: {gap*100:.2f}%)")
        
        return history
    
    def evaluate(self, model, X_test, y_test):
        """모델 평가"""
        print("\n" + "="*70)
        print("모델 평가")
        print("="*70)
        
        # 예측
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        
        # 분류 리포트
        print("\n분류 리포트:")
        print(classification_report(
            y_true_classes, 
            y_pred_classes,
            target_names=[self.int_to_label[i] for i in range(len(self.int_to_label))],
            digits=3
        ))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        # 정규화된 confusion matrix도 계산
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 두 개의 confusion matrix 그리기
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        labels = [self.int_to_label[i] for i in range(len(self.int_to_label))]
        
        if HAS_SEABORN:
            # seaborn 사용
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[0]
            )
            axes[0].set_title('Confusion Matrix (Counts)')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
            
            sns.heatmap(
                cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=axes[1],
                vmin=0,
                vmax=1
            )
            axes[1].set_title('Confusion Matrix (Normalized)')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')
        else:
            # matplotlib만 사용
            im0 = axes[0].imshow(cm, cmap='Blues', aspect='auto')
            axes[0].set_title('Confusion Matrix (Counts)')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
            axes[0].set_xticks(range(len(labels)))
            axes[0].set_yticks(range(len(labels)))
            axes[0].set_xticklabels(labels, rotation=45, ha='right')
            axes[0].set_yticklabels(labels)
            
            # 숫자 표시
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = axes[0].text(j, i, int(cm[i, j]),
                                       ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white")
            plt.colorbar(im0, ax=axes[0])
            
            im1 = axes[1].imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            axes[1].set_title('Confusion Matrix (Normalized)')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')
            axes[1].set_xticks(range(len(labels)))
            axes[1].set_yticks(range(len(labels)))
            axes[1].set_xticklabels(labels, rotation=45, ha='right')
            axes[1].set_yticklabels(labels)
            
            # 숫자 표시
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = axes[1].text(j, i, f'{cm_normalized[i, j]:.2f}',
                                       ha="center", va="center", color="black" if cm_normalized[i, j] < 0.5 else "white")
            plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix 저장: {self.output_dir / 'confusion_matrix.png'}")
        plt.close()
        
        return y_pred, y_pred_classes, y_true_classes
    
    def plot_history(self, history):
        """학습 곡선 그리기"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 최소값 표시
        min_val_loss_epoch = np.argmin(history.history['val_loss'])
        min_val_loss = history.history['val_loss'][min_val_loss_epoch]
        axes[0].scatter([min_val_loss_epoch], [min_val_loss], color='red', s=100, zorder=5)
        axes[0].annotate(f'Best: {min_val_loss:.4f}', 
                        xy=(min_val_loss_epoch, min_val_loss),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, color='red')
        
        # Accuracy
        axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # 최대값 표시
        max_val_acc_epoch = np.argmax(history.history['val_accuracy'])
        max_val_acc = history.history['val_accuracy'][max_val_acc_epoch]
        axes[1].scatter([max_val_acc_epoch], [max_val_acc], color='green', s=100, zorder=5)
        axes[1].annotate(f'Best: {max_val_acc:.4f}', 
                        xy=(max_val_acc_epoch, max_val_acc),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=10, color='green')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"학습 곡선 저장: {self.output_dir / 'training_history.png'}")
        plt.close()
    
    def save_model(self, model):
        """모델 및 설정 저장"""
        print("\n" + "="*70)
        print("모델 저장")
        print("="*70)
        
        # 모델 저장
        model_path = self.output_dir / "sign_language_model.h5"
        model.save(model_path)
        print(f"모델 저장: {model_path}")
        
        # 설정 정보 저장
        model_info = {
            'label_to_int': self.label_to_int,
            'int_to_label': self.int_to_label,
            'max_sequence_length': self.max_sequence_length,
            'input_dim': self.input_dim,
            'num_classes': len(self.label_to_int),
            'scaler': self.scaler
        }
        
        info_path = self.output_dir / "model_info.pkl"
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"설정 저장: {info_path}")
        
        # JSON 형식으로도 저장 (scaler 제외)
        json_info = {
            'label_to_int': self.label_to_int,
            'int_to_label': self.int_to_label,
            'max_sequence_length': int(self.max_sequence_length),
            'input_dim': int(self.input_dim),
            'num_classes': len(self.label_to_int),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'labels': list(self.label_to_int.keys())
        }
        
        json_path = self.output_dir / "model_info.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_info, f, indent=2, ensure_ascii=False)
        print(f"JSON 정보 저장: {json_path}")
        
        print("\n✅ 모델 저장 완료!")


def main():
    parser = argparse.ArgumentParser(
        description="3D 좌표 기반 수어 인식 모델 학습 (npy 파일 + metadata.csv 사용)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 학습
  python train_and_save_model_v6.py --data-dir coordinates_output
  
  # 데이터 증강 적용 (권장)
  python train_and_save_model_v6.py --data-dir coordinates_output --augment
  
  # 에포크 및 배치 크기 조정
  python train_and_save_model_v6.py --data-dir coordinates_output --epochs 150 --batch-size 8
  
  # 저장 경로 지정
  python train_and_save_model_v6.py --data-dir coordinates_output --output-dir my_model

데이터 구조:
  coordinates_output/
    ├── metadata.csv
    ├── L_001_keypoints.npy
    ├── L_002_keypoints.npy
    └── ...
        """
    )
    
    parser.add_argument("--data-dir", "-d", required=True, type=str,
                       help="npy 파일 및 metadata.csv가 있는 폴더 경로")
    parser.add_argument("--output-dir", "-o", default="trained_model/trained_model_v6",
                       help="모델 저장 경로 (기본: trained_model/trained_model_v6)")
    parser.add_argument("--epochs", "-e", type=int, default=100,
                       help="학습 에포크 수 (기본: 100)")
    parser.add_argument("--batch-size", "-b", type=int, default=16,
                       help="배치 크기 (기본: 16)")
    parser.add_argument("--augment", "-a", action="store_true",
                       help="데이터 증강 활성화 (3배 증가)")
    
    args = parser.parse_args()
    
    print("="*70)
    print("수어 인식 모델 학습 시작")
    print("="*70)
    print(f"데이터 경로: {args.data_dir}")
    print(f"저장 경로: {args.output_dir}")
    print(f"에포크: {args.epochs}")
    print(f"배치 크기: {args.batch_size}")
    print(f"데이터 증강: {'활성화' if args.augment else '비활성화'}")
    
    try:
        # 트레이너 초기화
        trainer = SignLanguageTrainer(args.data_dir, args.output_dir)
        
        # 데이터 로드
        sequences, labels = trainer.load_data(augment=args.augment)
        
        # 전처리
        X_train, X_test, y_train, y_test = trainer.preprocess_data(sequences, labels)
        
        # 모델 구축
        model = trainer.build_model()
        
        # 학습
        history = trainer.train(model, X_train, X_test, y_train, y_test, 
                               epochs=args.epochs, batch_size=args.batch_size)
        
        # 평가
        trainer.evaluate(model, X_test, y_test)
        
        # 결과 시각화
        trainer.plot_history(history)
        
        # 모델 저장
        trainer.save_model(model)
        
        print("\n" + "="*70)
        print("🎉 학습 완료!")
        print("="*70)
        print(f"\n저장된 파일:")
        print(f"  📁 {args.output_dir}/")
        print(f"    ├── sign_language_model.h5")
        print(f"    ├── best_model.h5")
        print(f"    ├── model_info.pkl")
        print(f"    ├── model_info.json")
        print(f"    ├── training_data_info.csv")
        print(f"    ├── confusion_matrix.png")
        print(f"    └── training_history.png")
        print(f"\n다음 명령어로 예측을 실행하세요:")
        print(f"  python 3d_predict_sign_language.py --video test.mp4 --model-dir {args.output_dir}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()