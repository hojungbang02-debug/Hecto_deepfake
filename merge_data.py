import os
import shutil
import glob
from tqdm import tqdm

# ==========================================
# ⚙️ 설정
# ==========================================
SRC_ROOT = './val_data'    # 여기 있는 걸
DST_ROOT = './train_data'  # 여기로 보냄
# ==========================================

def merge_data():
    print("🔄 데이터 합치기(원상복구)를 시작합니다...")

    if not os.path.exists(SRC_ROOT):
        print("❓ val_data 폴더가 없어서 합칠 게 없습니다.")
        return

    # 0_real, 1_fake 각각 수행
    for class_name in ['0_real', '1_fake']:
        src_path = os.path.join(SRC_ROOT, class_name)
        dst_path = os.path.join(DST_ROOT, class_name)
        
        # 목적지 폴더가 없으면 생성 (혹시 모르니)
        os.makedirs(dst_path, exist_ok=True)
        
        # 파일 찾기
        files = glob.glob(os.path.join(src_path, '*.*'))
        
        if not files:
            print(f"    {class_name}: 옮길 파일이 없습니다.")
            continue
            
        print(f"📦 {class_name}: {len(files)}개 파일을 train_data로 이동 중...")
        
        move_cnt = 0
        
        for f in tqdm(files):
            try:
                # 파일 이동 (shutil.move)
                shutil.move(f, dst_path)
                move_cnt += 1
            except Exception as e:
                print(f"   ❌ 이동 실패 ({os.path.basename(f)}): {e}")
                
        print(f"   ✅ {class_name} 완료! ({move_cnt}장 이동됨)")

    # (옵션) 빈 val_data 폴더 삭제
    try:
        if len(os.listdir(SRC_ROOT)) == 0: # 내용물이 비었으면
            os.rmdir(SRC_ROOT) # 삭제
            print("🗑️ 빈 val_data 폴더를 삭제했습니다.")
    except:
        pass

    print("\n🎉 모든 데이터가 train_data로 합쳐졌습니다!")

if __name__ == "__main__":
    merge_data()