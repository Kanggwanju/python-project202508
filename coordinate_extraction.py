"""
DB에서 영상 범위를 지정해서 좌표 추출하는 간단한 스크립트

사용법:
python coordinate_extractor.py --start 1 --end 100
python coordinate_extractor.py --ids 1,2,3,5,8
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv
import argparse
import os
from pathlib import Path

load_dotenv()

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

    def extract_from_url(self, url: str, frame_skip: int = 2) -> dict:
        """URL에서 직접 스트리밍하여 좌표 추출 (30fps로 고정)"""
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다")

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


def get_videos_from_db(start_id=None, end_id=None, video_ids=None):
    """DB에서 영상 정보 조회"""
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
            if video_ids:
                placeholders = ','.join(['%s'] * len(video_ids))
                query = f"SELECT id, title, video_url FROM sign_language_data WHERE id IN ({placeholders})"
                cursor.execute(query, video_ids)
            elif start_id and end_id:
                query = "SELECT id, title, video_url FROM sign_language_data WHERE id BETWEEN %s AND %s ORDER BY id"
                cursor.execute(query, (start_id, end_id))
            else:
                raise ValueError("start_id/end_id 또는 video_ids를 지정해야 합니다")

            return cursor.fetchall()
    finally:
        connection.close()


def save_data(video_id, keypoints, metadata, label, output_dir):
    """데이터 저장"""
    # numpy 파일로 좌표 저장
    numpy_path = output_dir / f"{video_id}_keypoints.npy"
    np.save(numpy_path, keypoints)

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


def main(start_id=None, end_id=None, video_ids=None, frame_skip=2):
    """메인 실행 함수"""
    # 출력 디렉토리 생성
    output_dir = Path("coordinates_output")
    output_dir.mkdir(exist_ok=True)

    # DB에서 영상 목록 가져오기
    videos = get_videos_from_db(start_id, end_id, video_ids)

    if not videos:
        print("조회된 영상이 없습니다.")
        return

    print(f"총 {len(videos)}개 영상 처리 시작")
    print("="*50)

    extractor = CoordinateExtractor()

    for idx, video in enumerate(videos, 1):
        video_id = video['id']
        title = video['title']
        url = video['video_url']

        print(f"[{idx}/{len(videos)}] ID: {video_id}, 제목: {title}")

        try:
            # 좌표 추출
            result = extractor.extract_from_url(url, frame_skip)

            if len(result["keypoints"]) == 0:
                print(f"  ✗ 실패: 좌표를 추출할 수 없음")
                continue

            # 데이터 저장
            save_data(video_id, result["keypoints"], result["metadata"], title, output_dir)

            shape = result["keypoints"].shape
            print(f"  ✓ 성공: {shape[0]}프레임, shape={shape}")

        except Exception as e:
            print(f"  ✗ 실패: {e}")

    print("="*50)
    print(f"완료! 저장 위치: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DB에서 영상 좌표 추출")
    parser.add_argument("--start", type=int, help="시작 ID")
    parser.add_argument("--end", type=int, help="끝 ID")
    parser.add_argument("--ids", type=str, help="특정 ID들 (쉼표로 구분: 1,2,3)")
    parser.add_argument("--skip", type=int, default=2, help="프레임 건너뛰기 (기본: 2)")

    args = parser.parse_args()

    if args.ids:
        video_ids = [int(x.strip()) for x in args.ids.split(',')]
        main(video_ids=video_ids, frame_skip=args.skip)
    elif args.start and args.end:
        main(start_id=args.start, end_id=args.end, frame_skip=args.skip)
    else:
        print("사용법:")
        print("  범위 지정: python coordinate_extractor.py --start 1 --end 100")
        print("  특정 ID: python coordinate_extractor.py --ids 1,2,3,5,8")