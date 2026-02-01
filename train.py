import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np

# 모듈들 불러오기
from src.dataset import DeepFakeDataset
from src.model import DeepFakeModel

# ====================================================
# 하이퍼파라미터 설정
# ====================================================
CONFIG = {
    'model_name': 'efficientnet_b4',  
    'image_size': 380,        
    'batch_size': 8,          
    'epochs': 20,             
    'lr': 1e-4,
    'seed': 42,
    'save_path': './model/best_model.pth' 
}

# ====================================================
#  유틸리티 함수
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Mixed Precision (메모리 절약 & 속도 향상)
        with torch.amp.autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
        
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total * 100
    return epoch_loss, epoch_acc

# ====================================================
# 메인 실행 함수
# ====================================================
def main():
    seed_everything(CONFIG['seed'])
    
    save_dir = os.path.dirname(CONFIG['save_path']) 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"'{save_dir}' 폴더를 생성")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    # ----------------------------------------------------
    # 1. 데이터셋 & 로더 준비
    # ----------------------------------------------------
    print("데이터 로딩 중...")

    train_dataset = DeepFakeDataset(
        root_dir='./train_data', 
        mode='train', 
        image_size=CONFIG['image_size'] 
    )
    
    if os.path.exists('./val_data'):
        val_dataset = DeepFakeDataset(
            root_dir='./val_data', 
            mode='val', 
            image_size=CONFIG['image_size']
        )
    else:
        print("val_data 폴더가 없습니다. train_data를 검증용으로 사용합니다.")
        val_dataset = DeepFakeDataset(root_dir='./train_data', mode='val', image_size=CONFIG['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    
    # ----------------------------------------------------
    # 모델 불러오기
    # ----------------------------------------------------
    print(f"모델 로드 중: {CONFIG['model_name']} (Size: {CONFIG['image_size']})...")
    model = DeepFakeModel(model_name=CONFIG['model_name'], pretrained=True)
    
    # model.py에 구현해둔 함수 호출
    if hasattr(model, 'set_gradient_checkpointing'):
        model.set_gradient_checkpointing(True)
        
    model = model.to(device)
    
    # ----------------------------------------------------
    # 설정 (Loss, Optimizer, Scheduler)
    # ----------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    
    # ----------------------------------------------------
    # 학습 시작
    # ----------------------------------------------------
    best_acc = 0.0
    print(f"\n학습 시작! (Epochs: {CONFIG['epochs']})")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(f"   [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"   [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # 스케줄러 업데이트
        scheduler.step()
        
        if val_acc > best_acc:
            print(f"   최고 성능 갱신! ({best_acc:.2f}% -> {val_acc:.2f}%) './model' 폴더에 저장 중...")
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['save_path'])
            
    print(f"\n학습 완료! 최고 정확도: {best_acc:.2f}%")
    print(f"모델 저장 위치: {CONFIG['save_path']}")

if __name__ == "__main__":
    main()