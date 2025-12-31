import torch
import cv2
import os
import glob
import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 모델 불러오기
from src.model import DeepFakeModel

# ==========================================
# 설정
# ==========================================
CONFIG = {
    'model_path': './model/best_model.pth',
    'test_dir': './test_data',               # 테스트 데이터 폴더
    'model_name': 'efficientnet_b4',
    'save_name': 'submission.csv',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 예시 코드의 구조를 맞추기 위한 결과 객체 클래스
class ProcessOutput:
    def __init__(self, filename, imgs=None, error=None):
        self.filename = filename  # 파일명 (예: video.mp4)
        self.imgs = imgs          # 전처리된 텐서 (없으면 None)
        self.error = error        # 에러 메시지 (없으면 None)

# ==========================================
# 함수 정의
# ==========================================

def get_transforms():
    """이미지 전처리 (리사이징 + 정규화)"""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def preprocess_one(file_path, transform):
    """
    파일 하나를 읽어서 텐서로 변환하는 함수
    - 이미지: 그대로 읽음
    - 비디오: 랜덤 5프레임 추출
    """
    filename = file_path.name
    str_path = str(file_path)
    
    try:
        # 비디오인 경우
        if str_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            cap = cv2.VideoCapture(str_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_count == 0:
                return ProcessOutput(filename, error="Empty Video")
            
            # 랜덤 5프레임 추출 (너무 짧으면 전체)
            if frame_count > 5:
                indices = sorted(np.random.choice(frame_count, 5, replace=False))
            else:
                indices = range(frame_count)
                
            frames = []
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Albumentations 적용
                    frame = transform(image=frame)['image']
                    frames.append(frame)
            cap.release()
            
            if not frames:
                return ProcessOutput(filename, error="Read Fail")
            
            # [5, 3, 224, 224] 형태로 스택
            imgs = torch.stack(frames)
            return ProcessOutput(filename, imgs=imgs)

        # 이미지인 경우
        else:
            image = cv2.imread(str_path)
            if image is None:
                return ProcessOutput(filename, error="Image Read Fail")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image=image)['image']
            # [1, 3, 224, 224] (배치 차원 추가)
            imgs = image.unsqueeze(0)
            return ProcessOutput(filename, imgs=imgs)

    except Exception as e:
        return ProcessOutput(filename, error=str(e))

def infer_fake_probs(model, imgs, device):
    """
    모델에 넣어서 가짜일 확률(0~1)을 뱉어주는 함수
    """
    with torch.no_grad():
        imgs = imgs.to(device)
        outputs = model(imgs)      # Logits
        probs = torch.sigmoid(outputs) # 0~1 확률 변환
        
        # CPU로 가져와서 리스트로 변환
        return probs.cpu().numpy().flatten().tolist()

# ==========================================
# 메인 실행
# ==========================================
def main():
    print(f"🔥 추론 시작! (Device: {CONFIG['device']})")
    
    # 1. 모델 준비
    model = DeepFakeModel(model_name=CONFIG['model_name'], pretrained=False).to(CONFIG['device'])
    
    if os.path.exists(CONFIG['model_path']):
        model.load_state_dict(torch.load(CONFIG['model_path']))
        print("✅ 모델 로드 완료")
    else:
        print("❌ 학습된 모델 파일이 없습니다! 0.5로 찍습니다.")
    
    model.eval()
    transform = get_transforms()

    # 2. 파일 목록 가져오기 (pathlib 사용 - 예시 코드 스타일)
    TEST_DIR = Path(CONFIG['test_dir'])
    # 모든 파일 가져오기 (숨김 파일 제외)
    files = sorted([p for p in TEST_DIR.iterdir() if p.is_file() and p.name[0] != '.'])
    
    print(f"📂 테스트 데이터 개수: {len(files)}개")

    # 3. 루프 돌면서 추론 (예시 코드 로직 그대로 적용)
    results = {}
    
    for file_path in tqdm(files, desc="Processing"):
        # 전처리
        out = preprocess_one(file_path, transform)
        
        # Case 1: 에러 발생 (파일 깨짐 등) -> 0.5 (모름) 또는 0.0 (Real) 처리
        if out.error:
            results[out.filename] = 0.5 # 에러나면 그냥 반반 확률로 던짐 (전략)
        
        # Case 2: 정상 (이미지/비디오 프레임 있음)
        elif out.imgs is not None:
            probs = infer_fake_probs(model, out.imgs, CONFIG['device'])
            # 확률들의 평균을 사용 (비디오 프레임이 5개면 5개 평균)
            avg_prob = float(np.mean(probs))
            results[out.filename] = avg_prob
            
        # Case 3: 이상한 경우
        else:
            results[out.filename] = 0.5

    # 4. CSV 저장
    print(f"💾 '{CONFIG['save_name']}' 저장 중...")
    
    with open(CONFIG['save_name'], 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prediction']) # 헤더 (대회 규격 확인 필수!)
        
        for filename, prob in results.items():
            writer.writerow([filename, prob])
            
    print("🎉 Submission 파일 생성 완료!")

if __name__ == "__main__":
    main()