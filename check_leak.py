import os

def check_overlap():
    # 1. 파일명 수집 함수
    def get_filenames(root):
        fnames = set()
        for path, subdirs, files in os.walk(root):
            for name in files:
                fnames.add(name)
        return fnames

    print("🔍 데이터 중복 검사를 시작합니다...")
    
    # 2. 파일명 싹 긁어오기
    train_files = get_filenames('./train_data')
    val_files = get_filenames('./val_data')
    
    print(f"📄 Train 파일 개수: {len(train_files)}장")
    print(f"📄 Valid 파일 개수: {len(val_files)}장")
    
    # 3. 교집합(중복) 확인
    overlap = train_files.intersection(val_files)
    
    print("-" * 30)
    if len(overlap) > 0:
        print(f"🚨 [경고] 중복된 파일이 {len(overlap)}장 발견되었습니다!")
        print("   => 학습 데이터가 검증 데이터에 섞여 있습니다. (점수 뻥튀기 원인)")
        print("   => 해결책: split_data.py를 다시 돌리거나 중복을 제거해야 합니다.")
    else:
        print("✅ [통과] 중복된 파일이 없습니다! (Clean)")

if __name__ == "__main__":
    check_overlap()