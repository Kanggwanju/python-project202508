"""
3D 좌표 기반 수어 예측 스크립트 (3d_coordinate_extractor.py와 완벽 호환)

사용법:
# 단일 영상 예측
python 3d_predict_sign_language.py --video test.mp4

# 폴더 내 모든 영상 일괄 예측
python 3d_predict_sign_language.py --folder test_videos

# 확률 분포 표시
python 3d_predict_sign_language.py --video test.mp4 --show-probs

# 모델 경로 지정
python 3d_predict_sign_language.py --folder test_videos --model-dir trained_model/trained_model_v5
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse
from pathlib import Path
from tensorflow.keras.models import load_model
import pandas as pd

class SignLanguagePredictor3D:
    def __init__(self, model_dir="trained_model/trained_model_v5"):
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
        
        # MediaPipe Holistic 초기화 (3d_coordinate_extractor.py와 동일)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_indices = [11, 12, 13, 14, 15, 16]  # 어깨, 팔꿈치, 손목
        
        print(f"모델 로드 완료!")
        print(f"학습된 수어 단어 ({self.num_classes}개): {list(self.label_to_int.keys())}")
        print(f"입력 차원: {self.input_dim}, 최대 시퀀스 길이: {self.max_sequence_length}")
    
    def extract_keypoints_from_video(self, video_path: str):
        """
        비디오 파일에서 3D 키포인트를 추출합니다.
        (3d_coordinate_extractor.py의 extract_from_file과 동일한 로직)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_data = []
        frame_count = 0
        extracted = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # 모든 프레임 처리
            coords = self._process_frame(frame)
            if coords:
                frames_data.append(coords)
                extracted += 1
            
            frame_count += 1
        
        cap.release()
        
        if len(frames_data) == 0:
            raise ValueError("키포인트를 추출할 수 없습니다.")
        
        return np.array(frames_data)
    
    def _process_frame(self, frame: np.ndarray) -> list:
        """
        단일 프레임에서 3D 키포인트를 추출합니다.
        (3d_coordinate_extractor.py의 _process_frame과 완전히 동일)
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        
        if not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None:
            return []
        
        pose_landmarks = results.pose_landmarks.landmark
        
        # 기준점: 어깨 중심 + 코 평균
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        nose = pose_landmarks[0]
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
        shoulder_center_z = (left_shoulder.z + right_shoulder.z) / 2
        stable_center_x = (shoulder_center_x + nose.x) / 2
        stable_center_y = (shoulder_center_y + nose.y) / 2
        stable_center_z = (shoulder_center_z + nose.z) / 2
        normalization_scale = 0.3
        
        keypoints = []
        
        # Pose 3D 상대좌표 (Z축 감쇠)
        for idx in self.pose_indices:
            lm = pose_landmarks[idx]
            rel_x = (lm.x - stable_center_x) / normalization_scale
            rel_y = (lm.y - stable_center_y) / normalization_scale
            rel_z = ((lm.z - stable_center_z) / normalization_scale) * 0.7
            keypoints.append([rel_x, rel_y, rel_z])
        
        # 왼손
        left_wrist_pos = None
        if hasattr(results, 'left_hand_landmarks') and results.left_hand_landmarks:
            hand_landmarks = results.left_hand_landmarks.landmark
            wrist = hand_landmarks[0]
            wrist_rel_x = (wrist.x - stable_center_x) / normalization_scale
            wrist_rel_y = (wrist.y - stable_center_y) / normalization_scale
            wrist_rel_z = ((wrist.z - stable_center_z) / normalization_scale) * 0.7
            left_wrist_pos = [wrist_rel_x, wrist_rel_y, wrist_rel_z]
            keypoints.append(left_wrist_pos)
            hand_scale = normalization_scale
            for i in range(1, 21):
                finger_lm = hand_landmarks[i]
                finger_rel_x = (finger_lm.x - wrist.x) / hand_scale
                finger_rel_y = (finger_lm.y - wrist.y) / hand_scale
                finger_rel_z = ((finger_lm.z - wrist.z) / hand_scale) * 0.6
                keypoints.append([finger_rel_x, finger_rel_y, finger_rel_z])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)
        
        # 오른손
        right_wrist_pos = None
        if hasattr(results, 'right_hand_landmarks') and results.right_hand_landmarks:
            hand_landmarks = results.right_hand_landmarks.landmark
            wrist = hand_landmarks[0]
            wrist_rel_x = (wrist.x - stable_center_x) / normalization_scale
            wrist_rel_y = (wrist.y - stable_center_y) / normalization_scale
            wrist_rel_z = ((wrist.z - stable_center_z) / normalization_scale) * 0.7
            right_wrist_pos = [wrist_rel_x, wrist_rel_y, wrist_rel_z]
            keypoints.append(right_wrist_pos)
            hand_scale = normalization_scale
            for i in range(1, 21):
                finger_lm = hand_landmarks[i]
                finger_rel_x = (finger_lm.x - wrist.x) / hand_scale
                finger_rel_y = (finger_lm.y - wrist.y) / hand_scale
                finger_rel_z = ((finger_lm.z - wrist.z) / hand_scale) * 0.6
                keypoints.append([finger_rel_x, finger_rel_y, finger_rel_z])
        else:
            keypoints.extend([[0.0, 0.0, 0.0]] * 21)
        
        # 양손 손목 3D 상대거리 추가
        if left_wrist_pos and right_wrist_pos:
            hand_distance = [
                right_wrist_pos[0] - left_wrist_pos[0],
                right_wrist_pos[1] - left_wrist_pos[1],
                right_wrist_pos[2] - left_wrist_pos[2]
            ]
            keypoints.append(hand_distance)
        else:
            keypoints.append([0.0, 0.0, 0.0])
        
        return keypoints
    
    def preprocess_sequence(self, keypoints: np.ndarray):
        """
        추출된 키포인트를 모델 입력 형식으로 전처리합니다.
        """
        # 평탄화
        flattened = keypoints.reshape(keypoints.shape[0], -1)
        
        # 스케일링
        scaled = self.scaler.transform(flattened)
        
        # 패딩
        if scaled.shape[0] >= self.max_sequence_length:
            padded = scaled[:self.max_sequence_length]
        else:
            padding = np.zeros((self.max_sequence_length - scaled.shape[0], scaled.shape[1]))
            padded = np.vstack((scaled, padding))
        
        # 배치 차원 추가
        return np.expand_dims(padded, axis=0)
    
    def predict_single(self, video_path: str, show_probabilities: bool = False):
        """
        단일 비디오 파일에서 수어를 예측합니다.
        """
        print(f"\n{'='*70}")
        print(f"예측 시작: {Path(video_path).name}")
        print(f"{'='*70}")
        
        # 1. 키포인트 추출
        print("키포인트 추출 중...", end=" ")
        keypoints = self.extract_keypoints_from_video(video_path)
        print(f"완료 ({keypoints.shape[0]}프레임)")
        
        # 2. 전처리
        print("데이터 전처리 중...", end=" ")
        preprocessed = self.preprocess_sequence(keypoints)
        print("완료")
        
        # 3. 예측
        print("예측 중...", end=" ")
        prediction = self.model.predict(preprocessed, verbose=0)
        print("완료")
        
        # 4. 결과 분석
        predicted_class_int = np.argmax(prediction[0])
        predicted_label = self.int_to_label[predicted_class_int]
        confidence = prediction[0][predicted_class_int] * 100
        
        # 결과 출력
        print(f"\n{'='*70}")
        print(f"예측 결과")
        print(f"{'='*70}")
        print(f"예측된 수어: {predicted_label}")
        print(f"신뢰도: {confidence:.2f}%")
        
        if show_probabilities:
            print(f"\n{'-'*70}")
            print(f"전체 확률 분포:")
            print(f"{'-'*70}")
            
            # 확률 순으로 정렬
            sorted_indices = np.argsort(prediction[0])[::-1]
            for idx in sorted_indices:
                label = self.int_to_label[idx]
                prob = prediction[0][idx] * 100
                bar_length = int(prob / 2)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"{label:20s} │ {bar} │ {prob:6.2f}%")
        
        print(f"{'='*70}\n")
        
        return {
            'file': Path(video_path).name,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'frames': keypoints.shape[0],
            'all_probabilities': {self.int_to_label[i]: float(prediction[0][i] * 100) 
                                 for i in range(len(prediction[0]))}
        }
    
    def predict_folder(self, folder_path: str, show_probabilities: bool = False):
        """
        폴더 내 모든 영상을 일괄 예측합니다.
        """
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder_path}")
        
        # 비디오 파일 찾기
        video_extensions = ['.mp4', '.mov', '.MP4', '.MOV', '.avi', '.AVI']
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder.glob(f"*{ext}"))

        # 중복 제거
        video_files = list(set(video_files))
        
        if not video_files:
            print(f"폴더에서 비디오 파일을 찾을 수 없습니다: {folder_path}")
            return []
        
        video_files.sort()
        
        print(f"\n{'='*70}")
        print(f"일괄 예측 시작: {folder_path}")
        print(f"총 {len(video_files)}개 영상 발견")
        print(f"{'='*70}")
        
        results = []
        
        for idx, video_file in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}] {video_file.name}")
            print("-" * 70)
            
            try:
                # 키포인트 추출
                print("키포인트 추출 중...", end=" ")
                keypoints = self.extract_keypoints_from_video(video_file)
                print(f"완료 ({keypoints.shape[0]}프레임)")
                
                # 전처리 및 예측
                print("예측 중...", end=" ")
                preprocessed = self.preprocess_sequence(keypoints)
                prediction = self.model.predict(preprocessed, verbose=0)
                print("완료")
                
                # 결과 분석
                predicted_class_int = np.argmax(prediction[0])
                predicted_label = self.int_to_label[predicted_class_int]
                confidence = prediction[0][predicted_class_int] * 100
                
                print(f"결과: {predicted_label} ({confidence:.2f}%)")
                
                results.append({
                    'file': video_file.name,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'frames': keypoints.shape[0]
                })
                
            except Exception as e:
                print(f"실패: {e}")
                results.append({
                    'file': video_file.name,
                    'predicted_label': 'ERROR',
                    'confidence': 0.0,
                    'frames': 0
                })
        
        # 전체 결과 요약
        self._print_summary(results, show_probabilities)
        
        return results
    
    def _print_summary(self, results, show_probabilities):
        """
        전체 결과를 표로 정리하여 출력합니다.
        """
        print(f"\n\n{'='*70}")
        print("전체 예측 결과 요약")
        print(f"{'='*70}\n")
        
        # DataFrame으로 변환
        df = pd.DataFrame(results)
        
        # 표 출력
        print(f"{'파일명':<30} {'예측 결과':<15} {'신뢰도':<10} {'프레임'}")
        print("-" * 70)
        
        for _, row in df.iterrows():
            file_name = row['file'][:28] + '..' if len(row['file']) > 30 else row['file']
            print(f"{file_name:<30} {row['predicted_label']:<15} {row['confidence']:>6.2f}%  {row['frames']:>6}")
        
        print("-" * 70)
        
        # 통계
        success_count = len(df[df['predicted_label'] != 'ERROR'])
        error_count = len(df[df['predicted_label'] == 'ERROR'])
        avg_confidence = df[df['predicted_label'] != 'ERROR']['confidence'].mean()
        
        print(f"\n통계:")
        print(f"  성공: {success_count}개")
        print(f"  실패: {error_count}개")
        if success_count > 0:
            print(f"  평균 신뢰도: {avg_confidence:.2f}%")
        
        # 예측 분포
        if success_count > 0:
            print(f"\n예측 분포:")
            label_counts = df[df['predicted_label'] != 'ERROR']['predicted_label'].value_counts()
            for label, count in label_counts.items():
                print(f"  {label}: {count}개")
        
        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="3D 좌표 기반 수어 예측 (3d_coordinate_extractor.py 호환)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 영상 예측
  python 3d_predict_sign_language.py --video test.mp4
  
  # 폴더 내 모든 영상 일괄 예측
  python 3d_predict_sign_language.py --folder test_videos
  
  # 확률 분포 표시
  python 3d_predict_sign_language.py --video test.mp4 --show-probs
  
  # 다른 모델 사용
  python 3d_predict_sign_language.py --folder test_videos --model-dir trained_model/trained_model_v5
        """
    )
    
    # 입력 모드 (둘 중 하나만 선택)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", "-v", type=str,
                            help="예측할 단일 비디오 파일 경로")
    input_group.add_argument("--folder", "-f", type=str,
                            help="예측할 비디오들이 있는 폴더 경로")
    
    parser.add_argument("--model-dir", "-m", default="trained_model/trained_model_v5",
                       help="학습된 모델이 저장된 디렉토리 (기본: trained_model/trained_model_v5)")
    parser.add_argument("--show-probs", "-p", action="store_true",
                       help="모든 클래스에 대한 확률 분포 표시")
    
    args = parser.parse_args()
    
    try:
        # 예측기 초기화
        predictor = SignLanguagePredictor3D(model_dir=args.model_dir)
        
        # 예측 수행
        if args.video:
            # 단일 파일 예측
            if not Path(args.video).exists():
                print(f"오류: 비디오 파일을 찾을 수 없습니다: {args.video}")
                return
            
            result = predictor.predict_single(args.video, show_probabilities=args.show_probs)
            
        else:
            # 폴더 일괄 예측
            results = predictor.predict_folder(args.folder, show_probabilities=args.show_probs)
        
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("\n먼저 train_and_save_model_v5.py를 실행하여 모델을 학습시켜주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()