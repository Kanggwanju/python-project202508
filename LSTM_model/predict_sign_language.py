"""
새로운 수어 영상을 입력받아 학습된 모델로 예측하는 스크립트

사용법:
python predict_sign_language.py --video "test_video.mp4"
python predict_sign_language.py --video "test_video.mp4" --show-probs
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model

class SignLanguagePredictor:
    def __init__(self, model_dir="trained_model"):
        """
        학습된 모델과 설정 정보를 로드합니다.
        """
        print("모델 로딩 중...")
        
        # 모델 로드
        model_path = Path(model_dir) / "sign_language_model.h5"
        info_path = Path(model_dir) / "model_info.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        if not info_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {info_path}")
        
        self.model = load_model(model_path)
        
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        self.label_to_int = model_info['label_to_int']
        self.int_to_label = model_info['int_to_label']
        self.max_sequence_length = model_info['max_sequence_length']
        self.input_dim = model_info['input_dim']
        self.num_classes = model_info['num_classes']
        self.scaler = model_info['scaler']
        
        # MediaPipe Holistic 초기화
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        
        print(f"모델 로드 완료!")
        print(f"학습된 수어 단어 ({self.num_classes}개): {list(self.label_to_int.keys())}")
    
    def extract_keypoints_from_video(self, video_path: str, frame_skip: int = 2):
        """
        비디오 파일에서 키포인트를 추출합니다.
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 30fps 목표로 frame_skip 조정
        if original_fps > 30:
            auto_skip = int(original_fps / 30)
            frame_skip = max(frame_skip, auto_skip)
        
        frames_data = []
        frame_count = 0
        extracted = 0
        
        print(f"영상 정보: {total_frames}프레임, {original_fps:.1f}fps")
        print(f"키포인트 추출 중...", end="", flush=True)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            if frame_count % frame_skip == 0:
                coords = self._process_frame(frame)
                if coords:
                    frames_data.append(coords)
                    extracted += 1
                    
                    # 진행 상황 표시
                    if extracted % 10 == 0:
                        print(".", end="", flush=True)
            
            frame_count += 1
        
        cap.release()
        print(f" 완료! ({extracted}프레임 추출)")
        
        if len(frames_data) == 0:
            raise ValueError("키포인트를 추출할 수 없습니다. 영상에 사람이 감지되지 않았을 수 있습니다.")
        
        return np.array(frames_data)
    
    def _process_frame(self, frame: np.ndarray) -> list:
        """
        단일 프레임에서 키포인트를 추출합니다.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        
        if not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None:
            return []
        
        keypoints = []
        
        # Pose (15개)
        for idx in self.pose_indices:
            lm = results.pose_landmarks.landmark[idx]
            keypoints.append([float(lm.x), float(lm.y)])
        
        # Left Hand (21개)
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                keypoints.append([float(lm.x), float(lm.y)])
        else:
            keypoints.extend([[0.0, 0.0]] * 21)
        
        # Right Hand (21개)
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                keypoints.append([float(lm.x), float(lm.y)])
        else:
            keypoints.extend([[0.0, 0.0]] * 21)
        
        return keypoints
    
    def preprocess_sequence(self, keypoints: np.ndarray):
        """
        추출된 키포인트를 모델 입력 형식으로 전처리합니다.
        """
        # 평탄화
        flattened = keypoints.reshape(keypoints.shape[0], -1)
        
        # 스케일링 (학습 시 사용한 스케일러 사용)
        scaled = self.scaler.transform(flattened)
        
        # 패딩
        if scaled.shape[0] >= self.max_sequence_length:
            padded = scaled[:self.max_sequence_length]
        else:
            padding = np.zeros((self.max_sequence_length - scaled.shape[0], scaled.shape[1]))
            padded = np.vstack((scaled, padding))
        
        # 배치 차원 추가
        return np.expand_dims(padded, axis=0)
    
    def predict(self, video_path: str, show_probabilities: bool = False):
        """
        비디오 파일에서 수어를 예측합니다.
        """
        print(f"\n{'='*60}")
        print(f"예측 시작: {video_path}")
        print(f"{'='*60}\n")
        
        # 1. 키포인트 추출
        keypoints = self.extract_keypoints_from_video(video_path)
        
        # 2. 전처리
        print("데이터 전처리 중...")
        preprocessed = self.preprocess_sequence(keypoints)
        
        # 3. 예측
        print("예측 중...")
        prediction = self.model.predict(preprocessed, verbose=0)
        
        # 4. 결과 분석
        predicted_class_int = np.argmax(prediction[0])
        predicted_label = self.int_to_label[predicted_class_int]
        confidence = prediction[0][predicted_class_int] * 100
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"📌 예측 결과")
        print(f"{'='*60}")
        print(f"예측된 수어: {predicted_label}")
        print(f"신뢰도: {confidence:.2f}%")
        
        if show_probabilities:
            print(f"\n{'─'*60}")
            print(f"전체 확률 분포:")
            print(f"{'─'*60}")
            
            # 확률 순으로 정렬
            sorted_indices = np.argsort(prediction[0])[::-1]
            for idx in sorted_indices:
                label = self.int_to_label[idx]
                prob = prediction[0][idx] * 100
                bar_length = int(prob / 2)  # 50% = 25칸
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"{label:15s} │ {bar} │ {prob:6.2f}%")
        
        print(f"{'='*60}\n")
        
        return {
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': {self.int_to_label[i]: float(prediction[0][i] * 100) 
                                 for i in range(len(prediction[0]))}
        }


def main():
    parser = argparse.ArgumentParser(
        description="학습된 모델로 새로운 수어 영상을 예측합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 예측
  python predict_sign_language.py --video "test_video.mp4"
  
  # 모든 클래스 확률 표시
  python predict_sign_language.py --video "test_video.mp4" --show-probs
  
  # 다른 모델 디렉토리 사용
  python predict_sign_language.py --video "test.mp4" --model-dir "my_model"
        """
    )
    
    parser.add_argument("--video", "-v", required=True,
                       help="예측할 mp4 비디오 파일 경로")
    parser.add_argument("--model-dir", "-m", default="trained_model",
                       help="학습된 모델이 저장된 디렉토리 (기본: trained_model)")
    parser.add_argument("--show-probs", "-p", action="store_true",
                       help="모든 클래스에 대한 확률 분포 표시")
    parser.add_argument("--frame-skip", "-s", type=int, default=2,
                       help="프레임 건너뛰기 (기본: 2)")
    
    args = parser.parse_args()
    
    # 비디오 파일 존재 확인
    if not Path(args.video).exists():
        print(f"❌ 오류: 비디오 파일을 찾을 수 없습니다: {args.video}")
        return
    
    try:
        # 예측기 초기화
        predictor = SignLanguagePredictor(model_dir=args.model_dir)
        
        # 예측 수행
        result = predictor.predict(args.video, show_probabilities=args.show_probs)
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("\n먼저 train_and_save_model.py를 실행하여 모델을 학습시켜주세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()