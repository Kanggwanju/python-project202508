import cv2
import mediapipe as mp
import numpy as np
import json
import os

class VideoToLandmarks:
    def __init__(self):
        """MediaPipe 손 추적 초기화"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks_from_video(self, video_path):
        """
        비디오 파일에서 손 좌표 추출
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"❌ 비디오를 열 수 없습니다: {video_path}")
            return None
        
        landmarks_list = []
        frame_count = 0
        
        print(f"📹 처리 중: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGR을 RGB로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # 손 감지
            results = self.hands.process(image)
            
            # 항상 126개 크기로 초기화
            frame_landmarks = [0.0] * 126
            
            if results.multi_hand_landmarks:
                # 첫 번째 손 (최대 63개)
                if len(results.multi_hand_landmarks) > 0:
                    hand_0 = results.multi_hand_landmarks[0]
                    for i, landmark in enumerate(hand_0.landmark):
                        if i < 21:  # 21개 관절만
                            idx = i * 3
                            frame_landmarks[idx] = landmark.x
                            frame_landmarks[idx + 1] = landmark.y
                            frame_landmarks[idx + 2] = landmark.z
                
                # 두 번째 손 (있다면, 63~125번 인덱스)
                if len(results.multi_hand_landmarks) > 1:
                    hand_1 = results.multi_hand_landmarks[1]
                    for i, landmark in enumerate(hand_1.landmark):
                        if i < 21:
                            idx = 63 + (i * 3)
                            frame_landmarks[idx] = landmark.x
                            frame_landmarks[idx + 1] = landmark.y
                            frame_landmarks[idx + 2] = landmark.z
            
            landmarks_list.append(frame_landmarks)
            frame_count += 1
            
            # 진행 상황 표시 (선택)
            if frame_count % 50 == 0:
                print(f"  처리 중... {frame_count} 프레임")
        
        cap.release()
        
        print(f"✅ 완료: {frame_count}개 프레임 처리됨")
        
        return np.array(landmarks_list)
    
    def save_landmarks(self, landmarks, output_path):
        """
        추출한 좌표를 파일로 저장
        
        Args:
            landmarks: numpy 배열 (프레임수, 126)
            output_path: 저장할 파일 경로
        """
        # numpy 배열로 저장 (.npy)
        if output_path.endswith('.npy'):
            np.save(output_path, landmarks)
            print(f"💾 저장 완료: {output_path}")
        
        # JSON으로 저장
        elif output_path.endswith('.json'):
            data = {
                'landmarks': landmarks.tolist(),
                'shape': landmarks.shape,
                'num_frames': len(landmarks)
            }
            with open(output_path, 'w') as f:
                json.dump(data, f)
            print(f"💾 저장 완료: {output_path}")
        
        # CSV로 저장
        elif output_path.endswith('.csv'):
            np.savetxt(output_path, landmarks, delimiter=',')
            print(f"💾 저장 완료: {output_path}")
        
        else:
            print("❌ 지원하지 않는 형식입니다. (.npy, .json, .csv만 가능)")


def process_single_video(video_path, output_path):
    """
    단일 비디오 처리
    
    Args:
        video_path: 입력 비디오 파일
        output_path: 출력 파일 경로
    """
    extractor = VideoToLandmarks()
    
    # 좌표 추출
    landmarks = extractor.extract_landmarks_from_video(video_path)
    
    if landmarks is not None:
        print(f"📊 추출된 데이터 shape: {landmarks.shape}")
        
        # 저장
        extractor.save_landmarks(landmarks, output_path)
        
        return landmarks
    
    return None


def process_multiple_videos(video_folder, output_folder):
    """
    폴더 안의 모든 비디오 처리
    
    Args:
        video_folder: 비디오 파일들이 있는 폴더
        output_folder: 결과를 저장할 폴더
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    extractor = VideoToLandmarks()
    
    # 지원하는 비디오 확장자
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # 폴더 내 모든 파일 확인
    video_files = [
        f for f in os.listdir(video_folder) 
        if any(f.lower().endswith(ext) for ext in video_extensions)
    ]
    
    if not video_files:
        print(f"❌ {video_folder}에 비디오 파일이 없습니다.")
        return
    
    print(f"\n📂 총 {len(video_files)}개 비디오 발견")
    print("=" * 50)
    
    results = []
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 처리 중...")
        
        video_path = os.path.join(video_folder, video_file)
        
        # 출력 파일명 생성 (확장자를 .npy로 변경)
        output_filename = os.path.splitext(video_file)[0] + '.npy'
        output_path = os.path.join(output_folder, output_filename)
        
        # 좌표 추출
        landmarks = extractor.extract_landmarks_from_video(video_path)
        
        if landmarks is not None:
            # 저장
            extractor.save_landmarks(landmarks, output_path)
            
            results.append({
                'video': video_file,
                'output': output_filename,
                'num_frames': len(landmarks),
                'shape': landmarks.shape
            })
        else:
            print(f"❌ 실패: {video_file}")
    
    # 결과 요약
    print("\n" + "=" * 50)
    print("📊 처리 결과 요약")
    print("=" * 50)
    for result in results:
        print(f"✅ {result['video']}")
        print(f"   → {result['output']}")
        print(f"   프레임: {result['num_frames']}, Shape: {result['shape']}")
    
    print(f"\n총 {len(results)}/{len(video_files)}개 성공")


def visualize_landmarks(landmarks, frame_index=0):
    """
    추출한 좌표를 시각화 (선택 사항)
    
    Args:
        landmarks: numpy 배열 (프레임수, 126)
        frame_index: 확인할 프레임 번호
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    if frame_index >= len(landmarks):
        print(f"❌ 프레임 {frame_index}는 존재하지 않습니다.")
        return
    
    frame = landmarks[frame_index]
    
    # 첫 번째 손 (63개 좌표)
    hand1_x = frame[0::3][:21]
    hand1_y = frame[1::3][:21]
    hand1_z = frame[2::3][:21]
    
    # 두 번째 손 (있다면)
    hand2_x = frame[63::3][:21] if len(frame) >= 126 else None
    hand2_y = frame[64::3][:21] if len(frame) >= 126 else None
    hand2_z = frame[65::3][:21] if len(frame) >= 126 else None
    
    # 3D 플롯
    fig = plt.figure(figsize=(10, 5))
    
    # 첫 번째 손
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(hand1_x, hand1_y, hand1_z, c='red', marker='o')
    ax1.set_title('Hand 1')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # 두 번째 손
    if hand2_x is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(hand2_x, hand2_y, hand2_z, c='blue', marker='o')
        ax2.set_title('Hand 2')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()


# ============================================================
# 사용 예시
# ============================================================

if __name__ == "__main__":
    print("=" * 50)
    print("    비디오 → 손 좌표 변환기")
    print("=" * 50)
    print("\n사용 방법을 선택하세요:")
    print("1. 단일 비디오 처리")
    print("2. 폴더 내 모든 비디오 처리")
    
    choice = input("\n선택 (1 또는 2): ").strip()
    
    if choice == '1':
        # 단일 비디오 처리
        video_path = input("비디오 파일 경로를 입력하세요: ").strip()
        output_path = input("저장할 파일명 (.npy, .json, .csv): ").strip()
        
        if not output_path:
            # 기본 출력 파일명
            output_path = os.path.splitext(video_path)[0] + '_landmarks.npy'
        
        landmarks = process_single_video(video_path, output_path)
        
        # 시각화 여부
        if landmarks is not None:
            viz = input("\n첫 프레임 시각화할까요? (y/n): ").strip().lower()
            if viz == 'y':
                visualize_landmarks(landmarks, frame_index=0)
    
    elif choice == '2':
        # 폴더 처리
        video_folder = input("비디오 폴더 경로를 입력하세요: ").strip()
        output_folder = input("결과 저장 폴더 경로: ").strip()
        
        if not output_folder:
            output_folder = video_folder + "_landmarks"
        
        process_multiple_videos(video_folder, output_folder)
    
    else:
        print("❌ 잘못된 선택입니다.")