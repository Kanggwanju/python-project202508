import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def visualize_single_frame_2d(landmarks, frame_index=0):
    """
    특정 프레임의 손 좌표를 2D로 시각화
    
    Args:
        landmarks: numpy 배열 (프레임수, 126)
        frame_index: 확인할 프레임 번호
    """
    if frame_index >= len(landmarks):
        print(f"❌ 프레임 {frame_index}는 존재하지 않습니다.")
        return
    
    frame = landmarks[frame_index]
    
    # 첫 번째 손
    hand1_x = frame[0::3][:21]
    hand1_y = frame[1::3][:21]
    
    # 두 번째 손
    hand2_x = frame[63::3][:21]
    hand2_y = frame[64::3][:21]
    
    # 손 연결선 정의 (MediaPipe 기준)
    connections = [
        (0,1),(1,2),(2,3),(3,4),        # 엄지
        (0,5),(5,6),(6,7),(7,8),        # 검지
        (0,9),(9,10),(10,11),(11,12),   # 중지
        (0,13),(13,14),(14,15),(15,16), # 약지
        (0,17),(17,18),(18,19),(19,20), # 새끼
        (5,9),(9,13),(13,17)            # 손바닥
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # 첫 번째 손
    ax1 = axes[0]
    if np.any(hand1_x != 0):
        # 연결선 그리기
        for start, end in connections:
            ax1.plot([hand1_x[start], hand1_x[end]], 
                    [hand1_y[start], hand1_y[end]], 
                    'b-', linewidth=2)
        # 관절점 그리기
        ax1.scatter(hand1_x, hand1_y, c='red', s=100, zorder=5)
        # 관절 번호 표시
        for i, (x, y) in enumerate(zip(hand1_x, hand1_y)):
            ax1.text(x, y, str(i), fontsize=8, ha='center', va='center')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0)  # Y축 뒤집기
    ax1.set_aspect('equal')
    ax1.set_title(f'Hand 1 - Frame {frame_index}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    
    # 두 번째 손
    ax2 = axes[1]
    if np.any(hand2_x != 0):
        for start, end in connections:
            ax2.plot([hand2_x[start], hand2_x[end]], 
                    [hand2_y[start], hand2_y[end]], 
                    'g-', linewidth=2)
        ax2.scatter(hand2_x, hand2_y, c='blue', s=100, zorder=5)
        for i, (x, y) in enumerate(zip(hand2_x, hand2_y)):
            ax2.text(x, y, str(i), fontsize=8, ha='center', va='center')
    else:
        ax2.text(0.5, 0.5, 'No Hand Detected', 
                ha='center', va='center', fontsize=20, color='gray')
    
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1, 0)
    ax2.set_aspect('equal')
    ax2.set_title(f'Hand 2 - Frame {frame_index}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_single_frame_3d(landmarks, frame_index=0):
    """
    특정 프레임의 손 좌표를 3D로 시각화
    """
    if frame_index >= len(landmarks):
        print(f"❌ 프레임 {frame_index}는 존재하지 않습니다.")
        return
    
    frame = landmarks[frame_index]
    
    # 첫 번째 손
    hand1_x = frame[0::3][:21]
    hand1_y = frame[1::3][:21]
    hand1_z = frame[2::3][:21]
    
    # 두 번째 손
    hand2_x = frame[63::3][:21]
    hand2_y = frame[64::3][:21]
    hand2_z = frame[65::3][:21]
    
    fig = plt.figure(figsize=(15, 7))
    
    # 첫 번째 손 (3D)
    ax1 = fig.add_subplot(121, projection='3d')
    if np.any(hand1_x != 0):
        ax1.scatter(hand1_x, hand1_y, hand1_z, c='red', marker='o', s=100)
        for i, (x, y, z) in enumerate(zip(hand1_x, hand1_y, hand1_z)):
            ax1.text(x, y, z, str(i), fontsize=8)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Hand 1 (3D) - Frame {frame_index}')
    
    # 두 번째 손 (3D)
    ax2 = fig.add_subplot(122, projection='3d')
    if np.any(hand2_x != 0):
        ax2.scatter(hand2_x, hand2_y, hand2_z, c='blue', marker='o', s=100)
        for i, (x, y, z) in enumerate(zip(hand2_x, hand2_y, hand2_z)):
            ax2.text(x, y, z, str(i), fontsize=8)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Hand 2 (3D) - Frame {frame_index}')
    
    plt.tight_layout()
    plt.show()


def visualize_trajectory(landmarks, landmark_index=8):
    """
    특정 관절(예: 검지 끝)의 궤적 시각화
    
    Args:
        landmarks: numpy 배열 (프레임수, 126)
        landmark_index: 추적할 관절 번호 (0~20)
    """
    num_frames = len(landmarks)
    
    # 첫 번째 손의 특정 관절 좌표
    x_coords = landmarks[:, landmark_index * 3]
    y_coords = landmarks[:, landmark_index * 3 + 1]
    z_coords = landmarks[:, landmark_index * 3 + 2]
    
    # 손이 감지된 프레임만 필터링
    valid_frames = x_coords != 0
    
    fig = plt.figure(figsize=(15, 5))
    
    # 2D 궤적 (X-Y)
    ax1 = fig.add_subplot(131)
    ax1.plot(x_coords[valid_frames], y_coords[valid_frames], 
            'b-', linewidth=2, alpha=0.7)
    ax1.scatter(x_coords[valid_frames], y_coords[valid_frames], 
               c=range(np.sum(valid_frames)), cmap='viridis', s=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'관절 {landmark_index} 궤적 (X-Y)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(1, 0)
    
    # 시간에 따른 X 좌표
    ax2 = fig.add_subplot(132)
    ax2.plot(x_coords, 'r-', linewidth=2, label='X')
    ax2.plot(y_coords, 'g-', linewidth=2, label='Y')
    ax2.plot(z_coords, 'b-', linewidth=2, label='Z')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Coordinate')
    ax2.set_title(f'관절 {landmark_index} 좌표 변화')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3D 궤적
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(x_coords[valid_frames], 
            y_coords[valid_frames], 
            z_coords[valid_frames], 
            'b-', linewidth=2)
    ax3.scatter(x_coords[valid_frames], 
               y_coords[valid_frames], 
               z_coords[valid_frames],
               c=range(np.sum(valid_frames)), cmap='viridis', s=20)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'관절 {landmark_index} 궤적 (3D)')
    
    plt.tight_layout()
    plt.show()


def create_video_visualization(landmarks, original_video_path, output_path='output_visualization.mp4'):
    """
    원본 비디오와 손 좌표를 함께 보여주는 영상 생성
    
    Args:
        landmarks: numpy 배열 (프레임수, 126)
        original_video_path: 원본 비디오 경로
        output_path: 출력 영상 경로
    """
    cap = cv2.VideoCapture(original_video_path)
    
    if not cap.isOpened():
        print(f"❌ 비디오를 열 수 없습니다: {original_video_path}")
        return
    
    # 비디오 정보
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 출력 비디오 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"📹 영상 생성 중... ({len(landmarks)}개 프레임)")
    
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20)
    ]
    
    for frame_idx in range(len(landmarks)):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 손 좌표 가져오기
        frame_landmarks = landmarks[frame_idx]
        
        # 첫 번째 손 그리기
        hand1_x = frame_landmarks[0::3][:21]
        hand1_y = frame_landmarks[1::3][:21]
        
        if np.any(hand1_x != 0):
            # 연결선
            for start, end in connections:
                pt1 = (int(hand1_x[start] * width), int(hand1_y[start] * height))
                pt2 = (int(hand1_x[end] * width), int(hand1_y[end] * height))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # 관절점
            for i in range(21):
                x = int(hand1_x[i] * width)
                y = int(hand1_y[i] * height)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # 두 번째 손 그리기
        hand2_x = frame_landmarks[63::3][:21]
        hand2_y = frame_landmarks[64::3][:21]
        
        if np.any(hand2_x != 0):
            for start, end in connections:
                pt1 = (int(hand2_x[start] * width), int(hand2_y[start] * height))
                pt2 = (int(hand2_x[end] * width), int(hand2_y[end] * height))
                cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
            
            for i in range(21):
                x = int(hand2_x[i] * width)
                y = int(hand2_y[i] * height)
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
        
        # 프레임 번호 표시
        cv2.putText(frame, f'Frame: {frame_idx}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if (frame_idx + 1) % 50 == 0:
            print(f"  진행: {frame_idx + 1}/{len(landmarks)}")
    
    cap.release()
    out.release()
    
    print(f"✅ 완료: {output_path}")


def analyze_landmarks(landmarks):
    """
    랜드마크 데이터 통계 분석
    """
    print("=" * 50)
    print("📊 랜드마크 데이터 분석")
    print("=" * 50)
    
    print(f"\n기본 정보:")
    print(f"  Shape: {landmarks.shape}")
    print(f"  총 프레임: {len(landmarks)}")
    print(f"  각 프레임 좌표 수: {landmarks.shape[1]}")
    
    # 손 감지율
    hand1_detected = np.any(landmarks[:, :63] != 0, axis=1)
    hand2_detected = np.any(landmarks[:, 63:126] != 0, axis=1)
    
    print(f"\n손 감지 통계:")
    print(f"  첫 번째 손 감지: {np.sum(hand1_detected)}/{len(landmarks)} 프레임 ({np.sum(hand1_detected)/len(landmarks)*100:.1f}%)")
    print(f"  두 번째 손 감지: {np.sum(hand2_detected)}/{len(landmarks)} 프레임 ({np.sum(hand2_detected)/len(landmarks)*100:.1f}%)")
    print(f"  양손 모두 감지: {np.sum(hand1_detected & hand2_detected)} 프레임")
    
    # 좌표 범위
    non_zero = landmarks[landmarks != 0]
    if len(non_zero) > 0:
        print(f"\n좌표 범위:")
        print(f"  최소값: {non_zero.min():.3f}")
        print(f"  최대값: {non_zero.max():.3f}")
        print(f"  평균값: {non_zero.mean():.3f}")


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("    NPY 파일 시각화 도구")
    print("=" * 50)
    
    # 파일 경로 입력
    npy_path = input("\nNPY 파일 경로를 입력하세요: ").strip()
    
    try:
        # 데이터 로드
        print(f"\n📂 로딩 중: {npy_path}")
        landmarks = np.load(npy_path)
        print(f"✅ 로드 완료: Shape {landmarks.shape}")
        
        # 통계 분석
        analyze_landmarks(landmarks)
        
        # 시각화 옵션
        print("\n" + "=" * 50)
        print("시각화 옵션:")
        print("1. 단일 프레임 2D 시각화")
        print("2. 단일 프레임 3D 시각화")
        print("3. 관절 궤적 시각화")
        print("4. 비디오로 출력 (원본 비디오 필요)")
        print("5. 모두 보기")
        
        choice = input("\n선택 (1-5): ").strip()
        
        if choice == '1':
            frame_idx = int(input("프레임 번호 (0부터 시작): "))
            visualize_single_frame_2d(landmarks, frame_idx)
        
        elif choice == '2':
            frame_idx = int(input("프레임 번호 (0부터 시작): "))
            visualize_single_frame_3d(landmarks, frame_idx)
        
        elif choice == '3':
            print("\n주요 관절:")
            print("  0: 손목, 4: 엄지 끝, 8: 검지 끝")
            print("  12: 중지 끝, 16: 약지 끝, 20: 새끼 끝")
            landmark_idx = int(input("관절 번호 (0-20): "))
            visualize_trajectory(landmarks, landmark_idx)
        
        elif choice == '4':
            video_path = input("원본 비디오 경로: ").strip()
            output_path = input("출력 파일명 (예: output.mp4): ").strip()
            create_video_visualization(landmarks, video_path, output_path)
        
        elif choice == '5':
            # 모두 보기
            visualize_single_frame_2d(landmarks, 0)
            visualize_single_frame_3d(landmarks, 0)
            visualize_trajectory(landmarks, 8)  # 검지 끝
        
        else:
            print("❌ 잘못된 선택입니다.")
    
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {npy_path}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")