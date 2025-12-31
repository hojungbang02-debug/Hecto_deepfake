import os
import shutil
import random
import glob
from collections import defaultdict
from tqdm import tqdm

# ==========================================
# ⚙️ 설정
# ==========================================
SOURCE_DIR = './train_data'  # 원본 데이터 (합쳐진 상태)
TARGET_DIR = './val_data'    # 검증 데이터 보낼 곳 (비어있어야 함)
SPLIT_RATIO = 0.2            # 20% 분할
# ==========================================

def get_video_id(filename):
    """
    [최종 수정] 복합 데이터셋 대응 로직
    - Case A: ff_251_0.png -> ID: 'ff_251' (앞에 두 덩어리)
    - Case B: ceymbecxnj_20_0.png -> ID: 'ceymbecxnj' (앞에 한 덩어리)
    """
    base_name = os.path.basename(filename)
    name_no_ext = os.path.splitext(base_name)[0]
    parts = name_no_ext.split('_')
    
    # 1. FF++ 데이터인 경우 ('ff_'로 시작)
    if base_name.startswith('ff_') and len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"  # 예: ff_251
        
    # 2. 그 외 (DFDC 등 일반적인 경우)
    else:
        return parts[0]  # 예: ceymbecxnj

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ {SOURCE_DIR} 폴더가 없습니다!")
        return

    for class_name in ['0_real', '1_fake']:
        src_path = os.path.join(SOURCE_DIR, class_name)
        dst_path = os.path.join(TARGET_DIR, class_name)
        
        os.makedirs(dst_path, exist_ok=True)
        
        files = glob.glob(os.path.join(src_path, '*.*'))
        if not files:
            print(f"⚠️ {class_name} 폴더가 비어있습니다.")
            continue
            
        print(f"\n📂 {class_name} 분석 및 그룹핑 중... (총 {len(files)}개)")

        # 그룹핑
        video_groups = defaultdict(list)
        for f in files:
            vid_id = get_video_id(f)
            video_groups[vid_id].append(f)
            
        video_ids = list(video_groups.keys())
        print(f"   🎬 고유 비디오 ID 개수: {len(video_ids)}개")
        
        # ID 추출 샘플 확인 (사용자가 안심하도록 출력)
        print(f"   [샘플 ID 확인] {video_ids[:5]} ...")
        
        # 섞고 나누기
        random.shuffle(video_ids)
        num_val = int(len(video_ids) * SPLIT_RATIO)
        val_vids = video_ids[:num_val]
        
        print(f"   🚚 검증용 이동 대상: 비디오 {len(val_vids)}개")
        
        # 이동
        move_cnt = 0
        for vid in tqdm(val_vids, desc=f"Moving {class_name}"):
            for file_path in video_groups[vid]:
                try:
                    shutil.move(file_path, dst_path)
                    move_cnt += 1
                except Exception as e:
                    print(f"Error: {e}")
                    
        print(f"✅ {class_name} 완료: {move_cnt}장 이동됨.")

if __name__ == "__main__":
    main()