import os
import glob
import shutil
import random
import math
from tqdm import tqdm


SOURCE_DIR = "./ff_dataset"  
TRAIN_DIR = "./train_data"
VAL_DIR = "./val_data"
SPLIT_RATIO = 0.8  # 8:2 분할

def makedirs(path):
    if os.path.exists(path):
        # 기존 폴더가 있으면 삭제 후 재생성
        shutil.rmtree(path)
    os.makedirs(path)

def get_video_id(filename):
    """
    파일명에서 비디오 ID를 추출하는 핵심 로직
    예: real_001_0.jpg -> real_001
    예: fake_001_002_15.jpg -> fake_001_002
    """
    return "_".join(filename.split('_')[:-1])

def split_and_move_by_video(label_name):
    src_folder = os.path.join(SOURCE_DIR, label_name)
    
    if not os.path.exists(src_folder):
        print(f"경고: '{src_folder}' 폴더가 없습니다.")
        return

    # 모든 이미지 가져오기
    images = glob.glob(os.path.join(src_folder, "*.jpg")) + \
             glob.glob(os.path.join(src_folder, "*.png"))
    
    if not images:
        print(f"{label_name} 폴더에 이미지가 없습니다.")
        return

    print(f"\n[{label_name}] 이미지 분석 중... (비디오 ID 추출)")

    # 비디오 ID별로 그룹화
    video_groups = {}
    for img_path in tqdm(images, desc="Grouping"):
        filename = os.path.basename(img_path)
        vid_id = get_video_id(filename)
        
        if vid_id not in video_groups:
            video_groups[vid_id] = []
        video_groups[vid_id].append(img_path)
    
    video_ids = list(video_groups.keys())
    total_videos = len(video_ids)
    
    print(f"   총 {total_videos}개의 고유 비디오 발견 (이미지 {len(images)}장)")

    # 비디오 섞기
    random.seed(42)
    random.shuffle(video_ids)
    
    # train, val 분할
    train_count = math.ceil(total_videos * SPLIT_RATIO)
    train_vids = video_ids[:train_count]
    val_vids = video_ids[train_count:]
    
    print(f"   Train 할당: 비디오 {len(train_vids)}개")
    print(f"   Val   할당: 비디오 {len(val_vids)}개")
  
    # Train 이동
    train_dest = os.path.join(TRAIN_DIR, label_name)
    os.makedirs(train_dest, exist_ok=True)
    
    count_train_imgs = 0
    for vid in tqdm(train_vids, desc="Moving to Train"):
        for img_path in video_groups[vid]:
            shutil.move(img_path, os.path.join(train_dest, os.path.basename(img_path)))
            count_train_imgs += 1
            
    # Val 이동
    val_dest = os.path.join(VAL_DIR, label_name)
    os.makedirs(val_dest, exist_ok=True)
    
    count_val_imgs = 0
    for vid in tqdm(val_vids, desc="Moving to Val"):
        for img_path in video_groups[vid]:
            shutil.move(img_path, os.path.join(val_dest, os.path.basename(img_path)))
            count_val_imgs += 1
            
    print(f"   이동 완료: Train {count_train_imgs}장 / Val {count_val_imgs}장")

def main():
    print("[비디오 단위] 데이터셋 재분할을 시작합니다...")
    
    # Train/Val 폴더 초기화
    makedirs(TRAIN_DIR)
    makedirs(VAL_DIR)
    
    # 0_real 처리
    split_and_move_by_video("0_real")
    
    # 1_fake 처리
    split_and_move_by_video("1_fake")

if __name__ == "__main__":
    main()