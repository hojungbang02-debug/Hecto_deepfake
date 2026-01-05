import cv2
import os
import glob
import torch
import numpy as np
import random
from tqdm import tqdm
from facenet_pytorch import MTCNN

# ====================================================
# [설정] 경로 및 파라미터
# ====================================================
SOURCE_VIDEO_DIR = r"C:\Users\hojun\.cache\kagglehub\datasets\xdxd003\ff-c23\versions\1"

# 내부 최상위 폴더 이름
DATASET_FOLDER_NAME = "FaceForensics++_C23"

# 저장할 경로
SAVE_ROOT = "./ff_dataset" 

# 모델 입력 크기
TARGET_SIZE = 380      
FRAMES_PER_VIDEO = 30  
# ======================================
# ==============

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, select_largest=True, device=device, margin=0)

# 얼굴 중심 기준으로 cropping 좌표 계산
def get_crop_coords(center, size):
    cx, cy = center
    half = size // 2
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size
    return x1, y1, x2, y2

def process_video(video_path, save_dir, prefix):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 데이터가 없으면 중단
    if total_frames <= 0: 
        cap.release()
        return

    # 비디오의 총 프레임이 30보다 작으면
    if total_frames < FRAMES_PER_VIDEO:
        # 모든 프레임 사용
        indices = range(total_frames)
    else:
        # 고르게 30프레임 샘플링
        indices = np.linspace(0, total_frames-1, FRAMES_PER_VIDEO, dtype=int)
    
    file_base = os.path.basename(video_path).split('.')[0]
    

    for idx in indices:
        # idx번째 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            boxes, _ = mtcnn.detect(frame_rgb)
        except:
            continue
            
        if boxes is not None:
            # 얼굴 좌표 추출 및 380x380 크롭 계산
            x1, y1, x2, y2 = map(int, boxes[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # 380*380 영역에 맞춰 좌표 계산
            startX, startY, endX, endY = get_crop_coords((cx, cy), TARGET_SIZE)
            
            canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
            
            # 화면 밖으로 벗어나지 않게 처리
            src_sx, src_sy = max(0, startX), max(0, startY)
            src_ex, src_ey = min(frame.shape[1], endX), min(frame.shape[0], endY)
            
            # 새로 그릴 캔버스 좌표 계산
            dst_sx = src_sx - startX
            dst_sy = src_sy - startY
            dst_ex = dst_sx + (src_ex - src_sx)
            dst_ey = dst_sy + (src_ey - src_sy)
            
            if src_ex > src_sx and src_ey > src_sy:
                crop = frame[src_sy:src_ey, src_sx:src_ex]
                canvas[dst_sy:dst_ey, dst_sx:dst_ex] = crop
                
                save_name = f"{prefix}_{file_base}_{idx}.jpg"
                cv2.imwrite(os.path.join(save_dir, save_name), canvas)

    cap.release()

def main():
    # 저장 폴더 생성
    os.makedirs(os.path.join(SAVE_ROOT, "0_real"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "1_fake"), exist_ok=True)
    
    # 실제 데이터셋 루트 경로 조합
    dataset_root = os.path.join(SOURCE_VIDEO_DIR, DATASET_FOLDER_NAME)
    
    print(f"FF++ 전처리 시작 (Target: {TARGET_SIZE}px)")
    print(f"데이터셋 루트: {dataset_root}")

    # ==========================================
    # Real 비디오 검색
    # ==========================================
    real_path = os.path.join(dataset_root, "original", "*.mp4")
    real_videos = glob.glob(real_path) 
    
    # 혹시 확장자가 대문자일 경우 대비
    if len(real_videos) == 0:
        real_videos = glob.glob(os.path.join(dataset_root, "original", "*.MP4"))

    if len(real_videos) == 0:
        print(f"경로 확인 필요: {real_path}")
        raise ValueError("Real 비디오('original')를 찾지 못했습니다.")

    print(f"Real 비디오 발견: {len(real_videos)}개")

    # ==========================================
    # Fake 비디오 검색 
    # ==========================================
    # 사용할 Fake 폴더 목록
    fake_folders = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
    
    fake_videos = []
    for folder in fake_folders:
        f_path = os.path.join(dataset_root, folder, "*.mp4")
        found = glob.glob(f_path)
        fake_videos.extend(found)
        print(f"   └─ {folder}: {len(found)}개")

    if len(fake_videos) == 0:
         raise ValueError("Fake 비디오를 하나도 못 찾았습니다!")

    # ==========================================
    # 데이터 밸런싱
    # ==========================================
    # 중복 제거
    real_videos = list(set(real_videos))
    fake_videos = list(set(fake_videos))

    if len(fake_videos) > len(real_videos):
        print(f"불균형 감지 (Real {len(real_videos)} vs Fake {len(fake_videos)}) -> 1:1로 조정합니다.")
        random.seed(42)
        random.shuffle(fake_videos)
        fake_videos = fake_videos[:len(real_videos)]
    
    print(f"최종 처리 대상: Real {len(real_videos)}개 / Fake {len(fake_videos)}개")

    # ==========================================
    # 실행
    # ==========================================
    for v_path in tqdm(real_videos, desc="Processing Real"):
        process_video(v_path, os.path.join(SAVE_ROOT, "0_real"), prefix="real")
        
    for v_path in tqdm(fake_videos, desc="Processing Fake"):
        process_video(v_path, os.path.join(SAVE_ROOT, "1_fake"), prefix="fake")

    print("\n✨ 전처리 완료! 생성된 폴더를 확인하세요.")

if __name__ == "__main__":
    main()