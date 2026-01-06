import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, mode='train', image_size=380):
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.image_paths = []
        self.labels = []

        # 데이터 로드
        self._load_data()
        
        # 변환
        self.transform = self._get_transforms()

    def _load_data(self):
        # 0: Real, 1: Fake
        class_map = {
            0: ['0_real', 'real', 'Real', 'REAL', '0'], 
            1: ['1_fake', 'fake', 'Fake', 'FAKE', '1']
        }

        print(f"[{self.mode.upper()}] 경로 탐색 시작: {os.path.abspath(self.root_dir)}")
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"폴더가 없습니다: {self.root_dir}")

        for label, folder_names in class_map.items():
            for folder_name in folder_names:
                target_path = os.path.join(self.root_dir, folder_name)
                
                if os.path.exists(target_path):
                    files = []
                    extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
                    for ext in extensions:
                        found = glob.glob(os.path.join(target_path, "**", ext), recursive=True)
                        files.extend(found)
                    
                    if len(files) > 0:
                        self.image_paths.extend(files)
                        self.labels.extend([label] * len(files))
                        print(f"   '{folder_name}' 폴더에서 {len(files)}장 찾음! (Label: {label})")
                    else:
                        print(f"   '{folder_name}' 폴더는 있지만 비어있습니다.")

        if len(self.image_paths) == 0:
            print(f"'{self.root_dir}' 안에 이미지가 하나도 없습니다.")
            raise ValueError(f"No images found in {self.root_dir}")

        print(f"🎉 [{self.mode.upper()}] 로드 완료: 총 {len(self.image_paths)}장 준비됨.")

    def _get_transforms(self):
        if self.mode == 'train':
            return A.Compose([
                # 혹시 몰라서 input과 같은 크기로 리사이즈
                # TODO: 이후에 하드코딩 제거
                A.CenterCrop(self.image_size, self.image_size),
                
                # flip
                A.HorizontalFlip(p=0.5),
                
                # 픽셀값 변환
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.2),
                
                # 두 개 중 랜덤 noise 추가 (blur 대신)
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                ], p=0.2),
                
                # 정규화
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            # 검증
            return A.Compose([
                A.CenterCrop(self.image_size, self.image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # 경로에 한글이 섞이면 안 읽혀서 imread 대신 사용
            img_array = np.fromfile(img_path, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if image is None: raise Exception("Decode failed")
        except Exception:
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 정의한 transform 적용
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, torch.tensor(label, dtype=torch.long)