import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transform(image_size: int, p_horizontal_flip: float = 0.5, p_random_rotate90: float = 0.5, p_transpose: float = 0.2):
    return A.Compose([
        A.HorizontalFlip(p=p_horizontal_flip),
        A.RandomRotate90(p=p_random_rotate90),     # 0/90/180/270
        A.Transpose(p=p_transpose),          # swap H/W
        A.Resize(image_size, image_size, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  
        ToTensorV2(),
    ])
