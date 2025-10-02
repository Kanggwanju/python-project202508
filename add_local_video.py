"""
로컬 영상 파일을 학습 데이터에 추가하는 스크립트

사용법:
python add_local_video.py --video test_video.mp4 --label "수학" --id 999
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


class CoordinateExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def extract_from_video(self, video_path: str, frame_skip: int = 2) -> dict:
        """로컬 비디오 파일에서 좌표 추출"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        frames_data = []
        frame_count = 0
        extracted = 0

        print(f"영상 처리 중: {video_path}")
        print(f"총 프레임: {total_frames}, FPS: {original_fps:.2f}")

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_skip == 0:
                coords = self._process_frame(frame)
                if coords:
                    frames_data.append(coords)
                    extracted += 1
                
                if extracted % 10 == 0:
                    print(f"  진행중... {extracted} 프레임 추출")

            frame_count += 1

        cap.release()

        actual_fps = (extracted / (total_frames / original_fps)) if total_frames > 0 else 0

        print(f"✓ 완료: {extracted} 프레임 추출됨")

        return {
            "keypoints": np.array(frames_data),
            "metadata": {
                "total_frames": total_frames,
                "original_fps": original_fps,
                "actual_fps": actual_fps,
                "frame_skip": frame_skip,
                "extracted_frames": extracted
            }
        }

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


def save_data(video_id, keypoints, metadata, label, output_dir):
    """데이터 저장"""
    output_dir = Path(output_dir)
    
    # numpy 파일로 좌표 저장
    numpy_path = output_dir / f"{video_id}_keypoints.npy"
    np.save(numpy_path, keypoints)
    print(f"✓ 저장됨: {numpy_path}")

    # 통합 메타데이터 CSV에 추가
    csv_path = output_dir / "metadata.csv"

    new_row = {
        "id": video_id,
        "label": label,
        "shape": str(keypoints.shape),
        **metadata
    }

    # 기존 파일이 있으면 읽어서 추가, 없으면 새로 생성
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # 중복 ID 제거
        df = df[df['id'] != video_id]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df = df.sort_values('id')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ 메타데이터 업데이트: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="로컬 영상을 학습 데이터에 추가")
    parser.add_argument("--video", type=str, required=True, help="비디오 파일 경로")
    parser.add_argument("--label", type=str, required=True, help="수어 레이블 (예: 수학)")
    parser.add_argument("--id", type=int, required=True, help="고유 ID (예: 999)")
    parser.add_argument("--output", type=str, default="coordinates_output", help="출력 디렉토리")
    parser.add_argument("--skip", type=int, default=2, help="프레임 건너뛰기")

    args = parser.parse_args()

    # 비디오 파일 존재 확인
    if not Path(args.video).exists():
        print(f"❌ 에러: 파일이 없습니다 - {args.video}")
        return

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    print("="*50)
    print(f"로컬 영상 추가")
    print("="*50)
    print(f"영상: {args.video}")
    print(f"레이블: {args.label}")
    print(f"ID: {args.id}")
    print("="*50)

    try:
        # 좌표 추출
        extractor = CoordinateExtractor()
        result = extractor.extract_from_video(args.video, args.skip)

        if len(result["keypoints"]) == 0:
            print("❌ 실패: 좌표를 추출할 수 없습니다")
            return

        # 데이터 저장
        save_data(args.id, result["keypoints"], result["metadata"], args.label, output_dir)

        print("="*50)
        print("✅ 완료!")
        print(f"Shape: {result['keypoints'].shape}")
        print("="*50)
        print("\n다음 단계:")
        print("  python train_model.py  # 모델 재학습")

    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()