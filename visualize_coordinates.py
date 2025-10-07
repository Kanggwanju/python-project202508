"""
추출된 좌표를 원본 영상 위에 시각화하는 스크립트

사용법:
python visualize_coordinates.py --video_url "http://..." --npy_path "coordinates_output/123_keypoints.npy"
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

class CoordinateVisualizer:
    def __init__(self):
        # MediaPipe Holistic 연결 정보
        self.pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # 얼굴
            (0, 4), (4, 5), (5, 6), (6, 8),  # 얼굴
            (9, 10),  # 입
            (11, 12), (11, 13), (13, 15),  # 왼쪽 팔
            (12, 14), (14, 16),  # 오른쪽 팔 (원래는 16이지만 우리는 14까지만)
        ]
        
        # 손 연결 (21개 포인트)
        self.hand_connections = [
            # 엄지
            (0, 1), (1, 2), (2, 3), (3, 4),
            # 검지
            (0, 5), (5, 6), (6, 7), (7, 8),
            # 중지
            (0, 9), (9, 10), (10, 11), (11, 12),
            # 약지
            (0, 13), (13, 14), (14, 15), (15, 16),
            # 새끼
            (0, 17), (17, 18), (18, 19), (19, 20),
        ]
        
        # 색상 정의
        self.color_pose = (0, 255, 0)  # 초록색
        self.color_left_hand = (255, 0, 0)  # 파란색
        self.color_right_hand = (0, 0, 255)  # 빨간색

    def visualize(self, video_url: str, keypoints_path: str, output_path: str = None, 
                  frame_skip: int = 2, max_frames: int = None):
        """
        영상과 좌표를 오버레이하여 시각화
        
        Args:
            video_url: 원본 영상 URL
            keypoints_path: .npy 파일 경로
            output_path: 저장할 영상 경로 (None이면 화면에만 표시)
            frame_skip: 프레임 건너뛰기 (원본 추출 시와 동일하게)
            max_frames: 최대 처리 프레임 수 (None이면 전체)
        """
        # 좌표 데이터 로드
        keypoints = np.load(keypoints_path)
        print(f"좌표 데이터 로드: {keypoints.shape}")
        print(f"총 {keypoints.shape[0]}개 프레임")
        
        # 영상 열기
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            raise ValueError("영상을 열 수 없습니다")
        
        # 영상 정보
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"영상 정보: {width}x{height}, {fps}fps")
        
        # 출력 설정
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps/frame_skip, (width, height))
        
        frame_count = 0
        coord_idx = 0
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # 프레임 건너뛰기 로직 (추출 시와 동일)
            if frame_count % frame_skip == 0:
                if coord_idx < len(keypoints):
                    # 좌표 그리기
                    frame = self._draw_keypoints(frame, keypoints[coord_idx], width, height)
                    coord_idx += 1
                    
                    # 프레임 번호 표시
                    cv2.putText(frame, f"Frame: {coord_idx}/{len(keypoints)}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 저장 또는 화면 표시
            if writer:
                writer.write(frame)
            else:
                cv2.imshow('Coordinate Visualization', frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("사용자에 의해 중단됨")
                    break
                elif key == ord(' '):  # 스페이스바로 일시정지
                    cv2.waitKey(0)
            
            frame_count += 1
            
            # 최대 프레임 제한
            if max_frames and coord_idx >= max_frames:
                break
        
        # 정리
        cap.release()
        if writer:
            writer.release()
            print(f"영상 저장 완료: {output_path}")
        else:
            cv2.destroyAllWindows()
        
        print(f"처리 완료: {coord_idx}개 프레임 시각화")

    def _draw_keypoints(self, frame, keypoints_2d, width, height):
        """프레임에 키포인트 그리기"""
        # 정규화된 좌표를 픽셀 좌표로 변환
        keypoints_px = []
        for x, y in keypoints_2d:
            px = int(x * width)
            py = int(y * height)
            keypoints_px.append((px, py))
        
        # Pose (0~14)
        pose_points = keypoints_px[:15]
        self._draw_landmarks(frame, pose_points, self.pose_connections, self.color_pose)
        
        # Left Hand (15~35)
        left_hand_points = keypoints_px[15:36]
        if not all(x == 0 and y == 0 for x, y in left_hand_points):
            left_hand_offset = [(x, y) for x, y in left_hand_points]
            self._draw_landmarks(frame, left_hand_offset, self.hand_connections, self.color_left_hand)
        
        # Right Hand (36~56)
        right_hand_points = keypoints_px[36:57]
        if not all(x == 0 and y == 0 for x, y in right_hand_points):
            right_hand_offset = [(x, y) for x, y in right_hand_points]
            self._draw_landmarks(frame, right_hand_offset, self.hand_connections, self.color_right_hand)
        
        return frame

    def _draw_landmarks(self, frame, points, connections, color):
        """랜드마크와 연결선 그리기"""
        # 연결선 그리기
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                start_point = points[start_idx]
                end_point = points[end_idx]
                
                # 유효한 좌표인지 확인
                if start_point != (0, 0) and end_point != (0, 0):
                    cv2.line(frame, start_point, end_point, color, 2)
        
        # 포인트 그리기
        for point in points:
            if point != (0, 0):
                cv2.circle(frame, point, 4, color, -1)
                cv2.circle(frame, point, 4, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(description="좌표 시각화")
    parser.add_argument("--video_url", type=str, required=True, help="원본 영상 URL")
    parser.add_argument("--npy_path", type=str, required=True, help=".npy 파일 경로")
    parser.add_argument("--output", type=str, default=None, help="출력 영상 경로 (선택)")
    parser.add_argument("--skip", type=int, default=2, help="프레임 건너뛰기 (기본: 2)")
    parser.add_argument("--max_frames", type=int, default=None, help="최대 프레임 수")
    
    args = parser.parse_args()
    
    visualizer = CoordinateVisualizer()
    
    print("시각화 시작...")
    print("- 'q' 키: 종료")
    print("- 스페이스바: 일시정지/재생")
    print()
    
    visualizer.visualize(
        video_url=args.video_url,
        keypoints_path=args.npy_path,
        output_path=args.output,
        frame_skip=args.skip
    )


if __name__ == "__main__":
    main()