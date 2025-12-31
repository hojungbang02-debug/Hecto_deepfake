import os
import cv2
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, mode='train', image_size=224, sample_ratio=1.0):
        """
        root_dir: 데이터셋 루트 경로
        mode: 'train' / 'val'
        image_size: 이미지 크기
        sample_ratio: 전체 데이터 중 몇 %를 사용할지 (0.1 = 10%만 사용)
        """
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.image_paths = []
        self.labels = []

        # 1. 일단 데이터 전부 로드
        self._load_data()
        
        # 2. 데이터 샘플링 (Subset Training)
        if sample_ratio < 1.0:
            print(f"✂️ 전체 데이터의 {sample_ratio*100}%만 랜덤하게 추출합니다...")
            # 이미지와 라벨을 묶어서 같이 섞어야 짝이 안 맞지 않음
            combined = list(zip(self.image_paths, self.labels))
            random.shuffle(combined) # 무작위 섞기
            
            # 비율만큼 자르기
            cutoff = int(len(combined) * sample_ratio)
            combined = combined[:cutoff]
            
            # 다시 분리
            self.image_paths, self.labels = zip(*combined)
            # 튜플을 리스트로 변환
            self.image_paths = list(self.image_paths)
            self.labels = list(self.labels)
            
            print(f"   -> 추출 후 남은 데이터: {len(self.image_paths)}장")

        # 3. 변환(Transform) 정의
        self.transform = self._get_transforms()

    def _load_data(self):
        """폴더를 뒤져서 jpg, png, jpeg 파일을 싹 긁어모으는 함수"""
        classes = {'0_real': 0, '1_fake': 1}
        
        for class_name, label in classes.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                files = glob.glob(os.path.join(class_dir, ext))
                self.image_paths.extend(files)
                self.labels.extend([label] * len(files))
                
        print(f"[{self.mode.upper()}] 원본 데이터 로드: 총 {len(self.image_paths)}장")

    def _get_transforms(self):
        if self.mode == 'train':
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.3),
                A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
                A.GaussNoise(p=0.2),
                A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            print(f"⚠️ 에러: 이미지 로드 실패 - {img_path}")
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image_tensor = augmented['image']

        return image_tensor, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 10%만 뽑아서 테스트해보는 코드
    dataset = DeepFakeDataset(root_dir='./train_data', mode='train', sample_ratio=0.1)
    print(f"최종 데이터 개수: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(loader))
    print(f"배치 이미지 크기: {images.shape}")