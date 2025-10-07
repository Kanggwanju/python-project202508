"""
수어 인식 예측
학습된 모델로 새로운 영상 예측하기

사용법:
python predict.py --video test_video.mp4
python predict.py --url https://example.com/video.mp4
python predict.py --video_id 123  (DB에서 가져오기)
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import joblib
from pathlib import Path
import pymysql
from dotenv import load_dotenv
import os

load_dotenv()


class CoordinateExtractor:
    """좌표 추출기 (coordinate_extraction.py와 동일)"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def extract_from_source(self, source: str, frame_skip: int = 2) -> np.ndarray:
        """파일 경로 또는 URL에서 좌표 추출"""
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {source}")

        frames_data = []
        frame_count = 0

        print("좌표 추출 중...", end="", flush=True)

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_skip == 0:
                coords = self._process_frame(frame)
                if coords:
                    frames_data.append(coords)
                
                # 진행 상황 표시
                if len(frames_data) % 10 == 0:
                    print(".", end="", flush=True)

            frame_count += 1

        cap.release()
        print(f" 완료! ({len(frames_data)} 프레임)")

        if len(frames_data) == 0:
            raise ValueError("좌표를 추출할 수 없습니다")

        return np.array(frames_data)

    def _process_frame(self, frame: np.ndarray) -> list:
        """단일 프레임 처리"""
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


class SignLanguagePredictor:
    """수어 예측기"""
    
    def __init__(self, model_path="sign_language_model.pkl"):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"모델 파일이 없습니다: {model_path}\n"
                f"먼저 train_model.py를 실행하여 모델을 학습하세요"
            )
        
        self.model = joblib.load(model_path)
        self.extractor = CoordinateExtractor()
        print(f"✓ 모델 로드 완료: {model_path}")
    
    def predict(self, source: str) -> dict:
        """영상 소스로부터 예측"""
        # 1. 좌표 추출
        keypoints = self.extractor.extract_from_source(source)
        
        # 2. 전처리 (평탄화 및 패딩)
        flattened = keypoints.flatten()
        
        # 학습 데이터와 같은 길이로 맞추기
        expected_length = self.model.n_features_in_
        if len(flattened) < expected_length:
            flattened = np.pad(flattened, (0, expected_length - len(flattened)))
        else:
            flattened = flattened[:expected_length]
        
        # 3. 예측
        prediction = self.model.predict([flattened])[0]
        probabilities = self.model.predict_proba([flattened])[0]
        
        # 4. 결과 정리
        classes = self.model.classes_
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        # 확률 높은 순으로 정렬
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "prediction": prediction,
            "confidence": float(max(probabilities)),
            "all_probabilities": sorted_probs,
            "keypoints_shape": keypoints.shape
        }


def get_video_url_from_db(video_id: int) -> str:
    """DB에서 영상 URL 가져오기"""
    connection = pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "test"),
        charset='utf8mb4'
    )

    try:
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            query = "SELECT video_url FROM sign_language_data WHERE id = %s"
            cursor.execute(query, (video_id,))
            result = cursor.fetchone()
            
            if not result:
                raise ValueError(f"ID {video_id}에 해당하는 영상이 없습니다")
            
            return result['video_url']
    finally:
        connection.close()


def print_result(result: dict):
    """결과 출력"""
    print("\n" + "="*50)
    print("🎯 예측 결과")
    print("="*50)
    print(f"\n예측: {result['prediction']}")
    print(f"신뢰도: {result['confidence']:.2%}")
    print(f"추출된 프레임: {result['keypoints_shape'][0]}개")
    
    print("\n📊 전체 확률 분포:")
    for label, prob in result['all_probabilities']:
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {label:15s} {bar} {prob:.2%}")
    
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="수어 예측")
    parser.add_argument("--video", type=str, help="비디오 파일 경로")
    parser.add_argument("--url", type=str, help="비디오 URL")
    parser.add_argument("--video_id", type=int, help="DB 비디오 ID")
    parser.add_argument("--model", type=str, default="sign_language_knn_model.pkl", 
                        help="모델 파일 경로")
    
    args = parser.parse_args()
    
    # 입력 소스 결정
    if args.video:
        source = args.video
        print(f"📹 비디오 파일: {source}")
    elif args.url:
        source = args.url
        print(f"🌐 URL: {source}")
    elif args.video_id:
        print(f"🔍 DB에서 ID {args.video_id} 조회 중...")
        source = get_video_url_from_db(args.video_id)
        print(f"✓ URL: {source}")
    else:
        print("❌ 에러: 입력 소스를 지정해주세요")
        print("\n사용법:")
        print("  python predict.py --video test.mp4")
        print("  python predict.py --url https://example.com/video.mp4")
        print("  python predict.py --video_id 123")
        return
    
    try:
        # 예측 실행
        predictor = SignLanguagePredictor(args.model)
        result = predictor.predict(source)
        
        # 결과 출력
        print_result(result)
        
    except FileNotFoundError as e:
        print(f"\n❌ 에러: {e}")
    except Exception as e:
        print(f"\n❌ 예측 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()