import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
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
    'batch_size': 8,                  
    'epochs': 10,
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
        
        with torch.cuda.amp.autocast():
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
    
    save_dir = os.path.dirname(CONFIG['save_path']) # './model' 추출
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 '{save_dir}' 폴더를 생성했습니다.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 사용 장치: {device}")
    
    # 1. 데이터셋 & 로더 준비
    print("💿 데이터 로딩 중...")

    sample_ratio = 1.0
    
    train_dataset = DeepFakeDataset(root_dir='./train_data', mode='train', sample_ratio=sample_ratio)
    
    if os.path.exists('./val_data'):
        val_dataset = DeepFakeDataset(root_dir='./val_data', mode='val', sample_ratio=sample_ratio)
    else:
        print("⚠️ 주의: val_data 폴더가 없습니다. train_data를 검증용으로 사용합니다.")
        val_dataset = DeepFakeDataset(root_dir='./train_data', mode='val', sample_ratio=sample_ratio)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. 모델 불러오기
    print(f"🤖 모델 로드 중: {CONFIG['model_name']}...")
    model = DeepFakeModel(model_name=CONFIG['model_name'], pretrained=True).to(device)
    
    # 3. 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()
    
    # 4. 학습 시작
    best_acc = 0.0
    print("\n🚀 학습 시작!")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n📢 Epoch {epoch+1}/{CONFIG['epochs']}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        print(f"   [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"   [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            print(f"   🎉 최고 성능 갱신! ({best_acc:.2f}% -> {val_acc:.2f}%) './model' 폴더에 저장 중...")
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['save_path'])
            
    print(f"\n🏁 학습 완료! 최고 정확도: {best_acc:.2f}%")
    print(f"💾 모델 저장 위치: {CONFIG['save_path']}")

if __name__ == "__main__":
    main()