import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import DeepFakeModel, DeepFakeModelDinoV2

# ==========================================
# 설정
# ==========================================
MODEL_PATH = './model/best_model.pth'
TEST_DATA_DIR = './test_data'
SUBMISSION_PATH = './submission.csv'

IMG_SIZE = 378
MARGIN_RATIO = 1.3
FRAME_INTERVAL = 10 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 모델 & MTCNN
# ==========================================
# 민감도 조정하여 얼굴 최대한 찾기
mtcnn = MTCNN(keep_all=True, select_largest=True, 
              thresholds=[0.5, 0.6, 0.6], 
              device=device, post_process=False)

# model = DeepFakeModel(model_name='efficientnet_b4', pretrained=False)
model = DeepFakeModelDinoV2(model_name='dinov2_vitb14', pretrained=False)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
else:
    print(f"모델 파일 없음: {MODEL_PATH}")
    exit()

# ==========================================
# 학습 코드와 동일하게 Albumentations 사용
# ==========================================
test_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def get_face_crop(frame, mtcnn):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        
        if boxes is None or len(boxes) == 0: return None

        box = boxes[0] 
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        
        new_w, new_h = w * MARGIN_RATIO, h * MARGIN_RATIO
        side = max(new_w, new_h)
        
        img_h, img_w = frame.shape[:2]
        x1 = int(max(0, cx - side // 2))
        y1 = int(max(0, cy - side // 2))
        x2 = int(min(img_w, cx + side // 2))
        y2 = int(min(img_h, cy + side // 2))
        
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0 or cropped.shape[0] < 10: return None
        return cropped
    except: return None

def get_center_crop(frame):
    h, w, _ = frame.shape
    cy, cx = h // 2, w // 2
    crop_size = min(h, w) // 2
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x2 = min(w, cx + crop_size // 2)
    y2 = min(h, cy + crop_size // 2)
    return frame[y1:y2, x1:x2]

def predict_frame(frame, model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 학습 때와 같은 Transform 적용
    augmented = test_transform(image=img_rgb)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
    return prob

# ==========================================
# 메인 추론
# ==========================================
def main():
    # 파일 목록
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.m4v'}
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.jfif'}
    valid_extensions = video_exts.union(image_exts)
    
    # test_data 폴더의 파일들을 이름순으로 가져옴
    test_files = []
    for root, dirs, files in os.walk(TEST_DATA_DIR):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                test_files.append(os.path.join(root, file))
    
    test_files.sort() # 순서 보장
    
    print(f"총 {len(test_files)}개의 파일을 처리합니다 (Albumentations 적용).")
    
    results = []
    
    for file_path in tqdm(test_files):
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        frames_to_process = []

        # 프레임 로드
        if ext in video_exts:
            cap = cv2.VideoCapture(file_path)
            frame_cnt = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_cnt += 1
                if frame_cnt % FRAME_INTERVAL == 0:
                    frames_to_process.append(frame)
            cap.release()
        elif ext in image_exts:
            frame = cv2.imread(file_path)
            if frame is not None:
                frames_to_process.append(frame)

        # 예측
        valid_probs = []
        fallback_frames = []
        
        for frame in frames_to_process:
            cropped = get_face_crop(frame, mtcnn)
            if cropped is not None:
                prob = predict_frame(cropped, model)
                valid_probs.append(prob)
            else:
                fallback_frames.append(frame)

        # 점수 집계
        if len(valid_probs) > 0:
            final_prob = np.mean(valid_probs)
        else:
            fallback_probs = []
            for frame in fallback_frames:
                center_crop = get_center_crop(frame)
                prob = predict_frame(center_crop, model)
                fallback_probs.append(prob)
            
            if len(fallback_probs) > 0:
                final_prob = np.mean(fallback_probs)
            else:
                final_prob = 0.5

        results.append({'path': file_name, 'label': final_prob})

    # 저장
    df = pd.DataFrame(results)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"{SUBMISSION_PATH} 저장 완료")

if __name__ == '__main__':
    main()