"""
수어 인식 - 데이터 증강 + Random Forest 모델
적은 데이터로 빠르게 프로토타입 만들기
"""

import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


class SignLanguageAugmenter:
    """수어 좌표 데이터 증강"""
    
    def __init__(self, n_augmentations=100):
        self.n_augmentations = n_augmentations
    
    def augment_sequence(self, keypoints):
        """하나의 시퀀스를 여러 개로 증강 (강도 높임)"""
        augmented = []
        
        for _ in range(self.n_augmentations):
            aug_kp = keypoints.copy()
            
            # 1. 노이즈 추가 (더 크게)
            noise = np.random.normal(0, 0.03, aug_kp.shape)
            aug_kp += noise
            
            # 2. 크기 조정 (더 다양하게)
            scale = np.random.uniform(0.85, 1.15)
            center = np.mean(aug_kp, axis=(0, 1))
            aug_kp = (aug_kp - center) * scale + center
            
            # 3. 이동 (더 크게)
            shift = np.random.uniform(-0.15, 0.15, size=(1, 1, 2))
            aug_kp += shift
            
            # 4. 회전 (더 크게)
            angle = np.random.uniform(-25, 25) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # 중심점 기준으로 회전
            aug_kp_centered = aug_kp - center
            aug_kp_rotated = np.dot(aug_kp_centered, rotation_matrix.T)
            aug_kp = aug_kp_rotated + center
            
            # 5. 시간 왜곡 (더 크게)
            if len(aug_kp) > 5:
                indices = np.linspace(0, len(aug_kp) - 1, len(aug_kp))
                time_warp = np.random.uniform(0.7, 1.3)
                new_indices = np.clip(indices * time_warp, 0, len(aug_kp) - 1)
                aug_kp = np.array([aug_kp[int(i)] for i in new_indices])
            
            # 6. 프레임 샘플링 (일부 프레임 건너뛰기)
            if len(aug_kp) > 10:
                skip = np.random.choice([1, 2, 3])
                aug_kp = aug_kp[::skip]
            
            augmented.append(aug_kp)
        
        return augmented


class SignLanguageClassifier:
    """수어 인식 모델"""
    
    def __init__(self, model_type='random_forest'):
        """
        model_type: 'random_forest', 'svm', 'knn'
        """
        self.model_type = model_type
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        elif model_type == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='euclidean'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.label_encoder = {}
        self.augmenter = SignLanguageAugmenter(n_augmentations=100)
    
    def load_data(self, coordinates_dir="coordinates_output"):
        """좌표 데이터 로드 및 증강"""
        coord_path = Path(coordinates_dir)
        metadata_path = coord_path / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError("metadata.csv 파일이 없습니다")
        
        # 메타데이터 읽기
        df = pd.read_csv(metadata_path, encoding='utf-8-sig')
        
        X_all = []
        y_all = []
        
        print("데이터 로딩 및 증강 중...")
        print("="*50)
        
        for idx, row in df.iterrows():
            video_id = row['id']
            label = row['label']
            
            # npy 파일 로드
            npy_path = coord_path / f"{video_id}_keypoints.npy"
            if not npy_path.exists():
                print(f"  ⚠ ID {video_id}: npy 파일 없음")
                continue
            
            keypoints = np.load(npy_path)
            
            # 데이터 증강
            augmented = self.augmenter.augment_sequence(keypoints)
            
            print(f"  ✓ '{label}': 원본 1개 → 증강 {len(augmented)}개")
            
            # 각 증강 데이터를 평탄화하여 추가
            for aug_kp in augmented:
                # (frames, keypoints, 2) → 1D 벡터로 평탄화
                flattened = aug_kp.flatten()
                X_all.append(flattened)
                y_all.append(label)
        
        print("="*50)
        
        # 길이를 맞추기 위해 최대 길이 찾기
        max_length = max(len(x) for x in X_all)
        
        # 패딩 (짧은 시퀀스는 0으로 채움)
        X_padded = []
        for x in X_all:
            if len(x) < max_length:
                padded = np.pad(x, (0, max_length - len(x)), mode='constant')
            else:
                padded = x[:max_length]  # 혹시 더 길면 자르기
            X_padded.append(padded)
        
        return np.array(X_padded), np.array(y_all)
    
    def train(self, X, y, test_size=0.2):
        """모델 학습"""
        print("\n모델 학습 중...")
        
        # 학습/테스트 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"  학습 데이터: {len(X_train)}개")
        print(f"  테스트 데이터: {len(X_test)}개")
        
        # 학습
        self.model.fit(X_train, y_train)
        
        # 평가
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n  학습 정확도: {train_acc:.2%}")
        print(f"  테스트 정확도: {test_acc:.2%}")
        
        print("\n상세 분류 리포트:")
        print(classification_report(y_test, test_pred))
        
        return train_acc, test_acc
    
    def save_model(self, path="sign_language_model.pkl"):
        """모델 저장"""
        joblib.dump(self.model, path)
        print(f"\n모델 저장 완료: {path}")
    
    def load_model(self, path="sign_language_model.pkl"):
        """모델 로드"""
        self.model = joblib.load(path)
        print(f"모델 로드 완료: {path}")
    
    def predict(self, keypoints):
        """새로운 데이터 예측"""
        # 평탄화
        flattened = keypoints.flatten()
        
        # 학습 데이터와 같은 길이로 맞추기 (패딩)
        if hasattr(self.model, 'n_features_in_'):
            expected_length = self.model.n_features_in_
            if len(flattened) < expected_length:
                flattened = np.pad(flattened, (0, expected_length - len(flattened)))
            else:
                flattened = flattened[:expected_length]
        
        # 예측
        prediction = self.model.predict([flattened])[0]
        probabilities = self.model.predict_proba([flattened])[0]
        
        return prediction, probabilities


def main():
    """메인 실행"""
    print("="*50)
    print("수어 인식 모델 학습")
    print("="*50)
    
    # 1. 모델 초기화 (Random Forest 사용)
    model = SignLanguageClassifier(model_type='random_forest')
    print(f"모델 타입: Random Forest")
    print("  - 더 현실적인 확률 분포 제공")
    print("  - 과적합 방지")
    print()
    
    # 2. 데이터 로드 및 증강
    try:
        X, y = model.load_data("coordinates_output")
        print(f"\n총 데이터: {len(X)}개 (증강 후)")
        print(f"클래스: {np.unique(y)}")
    except FileNotFoundError as e:
        print(f"\n❌ 에러: {e}")
        print("먼저 coordinate_extraction.py를 실행하여 좌표를 추출하세요")
        return
    
    # 3. 학습
    train_acc, test_acc = model.train(X, y, test_size=0.2)
    
    # 4. 모델 저장
    model.save_model("sign_language_model.pkl")
    
    print("\n" + "="*50)
    print("학습 완료!")
    print("="*50)
    print(f"✓ 모델이 저장되었습니다: sign_language_model.pkl")
    print(f"✓ 테스트 정확도: {test_acc:.2%}")


if __name__ == "__main__":
    main()