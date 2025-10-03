"""
로컬 mp4 파일에서 좌표를 추출하여 기존 데이터에 추가하는 스크립트

사용법:
python local_video_extractor.py --video "내영상.mp4" --label "안녕하세요" --id 101
python local_video_extractor.py --video "영상들/*.mp4" --label "감사합니다" --id 102 --auto-increment
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import glob

class CoordinateExtractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # 상체 Pose 15개 포인트 인덱스
        self.pose_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def extract_from_file(self, file_path: str, frame_skip: int = 2) -> dict:
        """로컬 파일에서 좌표 추출 (30fps로 고정)"""
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            raise ValueError(f"영상을 열 수 없습니다: {file_path}")

        # 메타데이터
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        target_fps = 30.0

        # 원본 FPS에 따라 frame_skip 자동 조정
        if original_fps > 30:
            auto_skip = int(original_fps / target_fps)
            frame_skip = max(frame_skip, auto_skip)

        frames_data = []
        frame_count = 0
        extracted = 0

        print(f"  영상 정보: {total_frames}프레임, {original_fps:.1f}fps")
        
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % frame_skip == 0:
                coords = self._process_frame(frame)
                if coords:
                    frames_data.append(coords)
                    extracted += 1

            frame_count += 1

        cap.release()

        actual_fps = (extracted / (total_frames / original_fps)) if total_frames > 0 else 0

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


def get_next_id(output_dir):
    """기존 데이터에서 다음 사용 가능한 ID 찾기"""
    csv_path = output_dir / "metadata.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return int(df['id'].max()) + 1
    else:
        return 1


def save_data(video_id, keypoints, metadata, label, output_dir):
    """데이터 저장 (기존 coordinate_extraction.py와 동일)"""
    # numpy 파일로 좌표 저장
    numpy_path = output_dir / f"{video_id}_keypoints.npy"
    np.save(numpy_path, keypoints)
    
    print(f"  저장됨: {numpy_path.name}")

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


def main(video_path, label, video_id=None, auto_increment=False, frame_skip=2):
    """메인 실행 함수"""
    # 출력 디렉토리 (기존과 동일)
    output_dir = Path("coordinates_output")
    output_dir.mkdir(exist_ok=True)

    # 비디오 파일 목록 가져오기
    if '*' in video_path or '?' in video_path:
        video_files = glob.glob(video_path)
    else:
        video_files = [video_path]

    if not video_files:
        print(f"영상 파일을 찾을 수 없습니다: {video_path}")
        return

    print(f"총 {len(video_files)}개 영상 처리 시작")
    print("="*50)

    extractor = CoordinateExtractor()
    current_id = video_id if video_id else get_next_id(output_dir)

    for idx, file_path in enumerate(video_files, 1):
        file_name = Path(file_path).name
        print(f"[{idx}/{len(video_files)}] 파일: {file_name}")

        try:
            # 좌표 추출
            result = extractor.extract_from_file(file_path, frame_skip)

            if len(result["keypoints"]) == 0:
                print(f"  ✗ 실패: 좌표를 추출할 수 없음")
                continue

            # 데이터 저장
            save_data(current_id, result["keypoints"], result["metadata"], label, output_dir)

            shape = result["keypoints"].shape
            print(f"  ✓ 성공: ID={current_id}, {shape[0]}프레임, shape={shape}")

            # 다음 ID로 증가 (auto_increment 모드일 때)
            if auto_increment:
                current_id += 1

        except Exception as e:
            print(f"  ✗ 실패: {e}")
            import traceback
            traceback.print_exc()

    print("="*50)
    print(f"완료! 저장 위치: {output_dir}")
    print(f"생성된 파일:")
    print(f"  - *_keypoints.npy (좌표 데이터)")
    print(f"  - metadata.csv (메타데이터)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="로컬 mp4 파일에서 좌표 추출하여 기존 데이터에 추가",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 파일 추가 (ID 자동 할당)
  python local_video_extractor.py --video "내영상.mp4" --label "안녕하세요"
  
  # ID 직접 지정
  python local_video_extractor.py --video "내영상.mp4" --label "감사합니다" --id 101
  
  # 여러 파일 일괄 처리 (ID 자동 증가)
  python local_video_extractor.py --video "영상들/*.mp4" --label "사랑해요" --auto-increment
  
  # 프레임 건너뛰기 조정
  python local_video_extractor.py --video "내영상.mp4" --label "안녕" --skip 3
        """
    )
    
    parser.add_argument("--video", "-v", required=True, 
                       help="mp4 파일 경로 (와일드카드 * 사용 가능)")
    parser.add_argument("--label", "-l", required=True, 
                       help="수어 단어 레이블 (예: '안녕하세요', '감사합니다')")
    parser.add_argument("--id", type=int, 
                       help="비디오 ID (미지정시 자동 할당)")
    parser.add_argument("--auto-increment", action="store_true",
                       help="여러 파일 처리시 ID 자동 증가")
    parser.add_argument("--skip", type=int, default=2, 
                       help="프레임 건너뛰기 (기본: 2)")

    args = parser.parse_args()

    main(
        video_path=args.video,
        label=args.label,
        video_id=args.id,
        auto_increment=args.auto_increment,
        frame_skip=args.skip
    )