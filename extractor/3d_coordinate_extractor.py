# =============================================
# 좌표 추출 스크립트 실행/사용 예시
# =============================================
# DB 범위 지정:
#   python data_preparation/coordinate_extractor.py --start 1 --end 100
# DB 특정 ID:
#   python data_preparation/coordinate_extractor.py --ids 3013,753,390,484,3015
# 로컬 영상 처리 (기본 경로):
#   python data_preparation/coordinate_extractor.py --local
# 로컬 영상 처리 (커스텀 경로):
#   python data_preparation/coordinate_extractor.py --local --local-path data/america
# 주요 옵션:
#   --skip 2   # 프레임 건너뛰기(기본 2, 실제는 모든 프레임 사용)
# =============================================
# 코드 설명
# =============================================
# - DB 또는 로컬 영상에서 프레임별로 MediaPipe Holistic 좌표를 추출합니다.
# - 기준점: 어깨 중심+코 평균으로 정규화(흔들림 최소화)
# - Z축 감쇠 및 이동평균, 양손 손목 3D 상대거리 등 수어 특화 특징 반영
# - 추출 좌표는 numpy(.npy)와 메타데이터(.csv)로 저장
# - 학습용 데이터셋 구축에 활용
# =============================================

"""
DB에서 영상 범위를 지정해서 좌표 추출하는 간단한 스크립트
로컬 영상도 처리 가능

사용법:
python data_preparation\coordinate_extractor.py --start 1 --end 100
python data_preparation\coordinate_extractor.py --ids 3013,753,390,484,3015
python data_preparation\coordinate_extractor.py --local  # 기본 경로: localvideo
python data_preparation\coordinate_extractor.py --local --local-path data/america
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv
import argparse
import os
import random
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
        # ✅ 개선: 수어에 핵심적인 상체 Pose만 선택 (어깨, 팔꿈치, 손목)
        # 기존 15개 → 6개로 축소하여 잡음 제거
        self.pose_indices = [11, 12, 13, 14, 15, 16]  # 어깨(11,12), 팔꿈치(13,14), 손목(15,16)

    def extract_from_url(self, url: str) -> dict:
        """URL에서 직접 스트리밍하여 좌표 추출 (모든 프레임 처리)"""
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다")

        # 메타데이터
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        frames_data = []
        frame_count = 0
        extracted = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            # 모든 프레임 처리 (frame_skip 로직 제거)
            coords = self._process_frame(frame)
            if coords:
                frames_data.append(coords)
                extracted += 1

            frame_count += 1

        cap.release()

        return {
            "keypoints": np.array(frames_data),
            "metadata": {
                "total_frames": total_frames,
                "original_fps": original_fps,
                "actual_fps": original_fps,  # 모든 프레임을 처리하므로 원본 FPS와 동일
                "frame_skip": 1,  # 프레임 스킵 없음을 의미
                "extracted_frames": extracted
            }
        }

    def extract_from_file(self, file_path: str) -> dict:
        """로컬 파일에서 좌표 추출 (모든 프레임 처리)"""
        cap = cv2.VideoCapture(str(file_path))

        if not cap.isOpened():
            raise ValueError(f"영상 파일을 열 수 없습니다: {file_path}")

        # 메타데이터
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

        return {
            "keypoints": np.array(frames_data),
            "metadata": {
                "total_frames": total_frames,
                "original_fps": original_fps,
                "actual_fps": original_fps,
                "frame_skip": 1,
                "extracted_frames": extracted
            }
        }

    def _smooth_z_coordinates(self, z_sequence):
        """Z축 좌표 이동평균(3프레임) 필터"""
        if len(z_sequence) < 3:
            return z_sequence
        smoothed = []
        for i in range(len(z_sequence)):
            if i == 0:
                smoothed.append(z_sequence[i])
            elif i == len(z_sequence) - 1:
                smoothed.append(z_sequence[i])
            else:
                avg = (z_sequence[i-1] + z_sequence[i] + z_sequence[i+1]) / 3
                smoothed.append(avg)
        return smoothed

    def _process_frame(self, frame: np.ndarray) -> list:
        """개선된 기준점, Z축 감쇠/평활화, 양손 상대거리 추가"""
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
            rel_z = ((lm.z - stable_center_z) / normalization_scale) * 0.7  # Z축 감쇠
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
                finger_rel_z = ((finger_lm.z - wrist.z) / hand_scale) * 0.6  # 손가락 Z축 더 강한 감쇠
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
                query = f"SELECT id, gloss, url FROM sign WHERE id IN ({placeholders})"
                cursor.execute(query, video_ids)
            elif start_id and end_id:
                query = "SELECT id, gloss, url FROM sign WHERE id BETWEEN %s AND %s ORDER BY id"
                cursor.execute(query, (start_id, end_id))
            else:
                raise ValueError("start_id/end_id 또는 video_ids를 지정해야 합니다")

            return cursor.fetchall()
    finally:
        connection.close()


def get_local_videos(local_dir_path="localvideo"):
    """로컬 영상 파일 목록 조회"""
    local_dir = Path(local_dir_path)

    if not local_dir.exists():
        print(f"로컬 영상 디렉토리가 없습니다: {local_dir}")
        return []

    # 기존 로컬 ID의 최대 번호 확인
    csv_path = Path("coordinates_output") / "metadata.csv"
    max_local_num = 0

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            local_ids = df[df['id'].astype(str).str.startswith('L_')]['id'].astype(str)
            for local_id in local_ids:
                try:
                    num = int(local_id.split('_')[1])
                    max_local_num = max(max_local_num, num)
                except (IndexError, ValueError):
                    continue
        except Exception as e:
            print(f"기존 CSV 파일 읽기 실패: {e}")

    video_extensions = ['.mp4', '.mov', '.MP4', '.MOV']
    videos = []
    current_num = max_local_num + 1

    # 파일명으로 정렬하여 일관된 순서 보장
    video_files = []
    for file_path in local_dir.glob("*"):
        if file_path.suffix in video_extensions:
            video_files.append(file_path)

    video_files.sort(key=lambda x: x.name)

    for file_path in video_files:
        # 파일명에서 단어 추출 (예: "무한_01.MP4" -> "무한")
        word = file_path.stem.split('_')[0]

        # 로컬 ID 생성: L_ + 3자리 순차번호 (L_001, L_002, ...)
        local_id = f"L_{current_num:03d}"

        videos.append({
            'id': local_id,
            'gloss': word,
            'file_path': file_path
        })

        current_num += 1

    return videos


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

    # ID를 문자열로 변환하여 정렬 (숫자/문자 혼합 정렬 문제 해결)
    df['id'] = df['id'].astype(str)
    df = df.sort_values('id')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')


def main(start_id=None, end_id=None, video_ids=None, frame_skip=2, process_local=False, local_path="localvideo"):
    """메인 실행 함수"""
    # 출력 디렉토리 생성
    output_dir = Path("coordinates_output")
    output_dir.mkdir(exist_ok=True)

    videos = []

    if process_local:
        # 로컬 영상 처리
        videos = get_local_videos(local_path)
        print(f"로컬 영상 디렉토리: {local_path}")
        print(f"로컬 영상 {len(videos)}개 발견")
    else:
        # DB에서 영상 목록 가져오기
        videos = get_videos_from_db(start_id, end_id, video_ids)
        print(f"DB에서 {len(videos)}개 영상 조회")

    if not videos:
        print("처리할 영상이 없습니다.")
        return

    print(f"총 {len(videos)}개 영상 처리 시작")
    print("="*50)

    extractor = CoordinateExtractor()

    for idx, video in enumerate(videos, 1):
        video_id = video['id']
        title = video['gloss']

        print(f"[{idx}/{len(videos)}] ID: {video_id}, 제목: {title}")

        try:
            # 좌표 추출
            if process_local:
                # 로컬 파일에서 추출
                result = extractor.extract_from_file(video['file_path'])
            else:
                # URL에서 추출
                url = video['url']
                result = extractor.extract_from_url(url)

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
    parser = argparse.ArgumentParser(description="DB 및 로컬 영상에서 좌표 추출")
    parser.add_argument("--start", type=int, help="시작 ID (DB 모드)")
    parser.add_argument("--end", type=int, help="끝 ID (DB 모드)")
    parser.add_argument("--ids", type=str, help="특정 ID들 (쉼표로 구분: 1,2,3)")
    parser.add_argument("--local", action="store_true", help="로컬 영상 처리 모드")
    parser.add_argument("--local-path", type=str, default="localvideo", help="로컬 영상 경로 (기본: localvideo)")
    parser.add_argument("--skip", type=int, default=2, help="프레임 건너뛰기 (기본: 2)")

    args = parser.parse_args()

    if args.local:
        main(frame_skip=args.skip, process_local=True, local_path=args.local_path)
    elif args.ids:
        video_ids = [int(x.strip()) for x in args.ids.split(',')]
        main(video_ids=video_ids, frame_skip=args.skip)
    elif args.start and args.end:
        main(start_id=args.start, end_id=args.end, frame_skip=args.skip)
    else:
        print("사용법:")
        print("  DB 범위 지정: python coordinate_extractor.py --start 1 --end 100")
        print("  DB 특정 ID: python coordinate_extractor.py --ids 1,2,3,5,8")
        print("  로컬 영상 (기본 경로): python coordinate_extractor.py --local")
        print("  로컬 영상 (커스텀 경로): python 3d_coordinate_extractor.py --local --local-path data/america")