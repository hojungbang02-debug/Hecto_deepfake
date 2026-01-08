import gc
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import wandb

# 모듈들 불러오기
from src.dataset import DeepFakeDataset
from src.model import DeepFakeModel, DeepFakeModelDinoV2
from src.scheduler import cosine_with_min_lr

# ====================================================
# 하이퍼파라미터 설정
# ====================================================
OPTIM_CONFIG = {
    "backbone": {
        "lr": 1e-5,
        "weight_decay": 1e-2,
    },
    "head": {
        "lr": 1e-4,
        "weight_decay": 1e-2,
    },
}
CONFIG = {
    'model_name': 'dinov2_vitb14',  
    'image_size': 378,        
    'batch_size': 32,          
    'epochs': 20,             
    'optim_config': OPTIM_CONFIG,
    'backbone_lr': 5e-5,
    'seed': 42,
    'save_path': './model/best_model.pth',
    'val_image_size': 378,
    'run_name': 'dinov2_vitb14_378_SRM12',
    'warmup_epochs': 2,
    'hidden_dim': [],
    'min_lr': 1e-7,
    'dropout_rate': 0.3,
    'filter_type': 'srm12',  # 'none', 'srm', 'srm6', 'srm12'   
}

SRM_FILTER = {'none': 3, 'srm': 3, 'srm6': 6, 'srm12': 12}
CONFIG['in_chs'] = SRM_FILTER[CONFIG['filter_type']]

USE_WANDB = True

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

def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler: optim.lr_scheduler, scaler, device, epoch, use_wandb=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Mixed Precision (메모리 절약 & 속도 향상)
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
        global_step = (epoch-1) * len(loader) + step

        # Gradient Norm Cliipping
        grad_head = torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=float('inf'))
        grad_backbone = torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=float('inf'))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        

        if not isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # Log metrics to wandb/step
        if use_wandb:
            
            with torch.no_grad():
                SRM_rate = model.srm.logit.sigmoid().item() if hasattr(model, 'srm') else 0.0

            wandb.log({
                "global_step": global_step,
                "train_loss/step": loss.item(),
                "lr/step": optimizer.param_groups[0]['lr'],
                "grad_norm/step": grad_norm.item(),
                "grad_norm/head": grad_head.item(),
                "grad_norm/backbone": grad_backbone.item(),
                "pred/prob_mean": probs.mean().item(),
                "pred/prob_std": probs.std().item(),
                "filter/srm_rate": SRM_rate,
            })


        
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
    
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="kdch6686-",
        # Set the wandb project where this run will be logged.
        project="HAI",
        # Track hyperparameters and run metadata.
        config=CONFIG,
        name=CONFIG['run_name']
    )
    # 기본 x축은 global_step
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    wandb.define_metric("train_loss/*", step_metric="global_step")
    wandb.define_metric("lr/*", step_metric="global_step")
    wandb.define_metric("grad_norm/*", step_metric="global_step")
    wandb.define_metric("pred/*", step_metric="global_step")
    wandb.define_metric("filter/*", step_metric="global_step")

    # val 쪽은 epoch을 x축으로
    wandb.define_metric("epoch")
    wandb.define_metric("val/*", step_metric="epoch")

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

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=6  , pin_memory=True)
    
    # ----------------------------------------------------
    # 모델 불러오기
    # ----------------------------------------------------
    print(f"모델 로드 중: {CONFIG['model_name']} (Size: {CONFIG['image_size']})...")
    model = DeepFakeModelDinoV2(model_name=CONFIG['model_name'], in_chs=CONFIG['in_chs'], pretrained=True, hidden_dim=CONFIG['hidden_dim'], drop_rate=CONFIG['dropout_rate'])
    print(model)
    
    # model.py에 구현해둔 함수 호출
    if hasattr(model, 'set_gradient_checkpointing'):
        model.set_gradient_checkpointing(True)
        
    model = model.to(device)

    # ----------------------------------------------------
    # Backbone의 일부 레이어 동결 (Fine-tuning 시)
    # ----------------------------------------------------

    # 일단 BackBone전체동결
    for param in model.model.parameters():
        param.requires_grad = False
    
    # ----------------------------------------------------
    # 설정 (Loss, Optimizer, Scheduler)
    # ----------------------------------------------------
    criterion = nn.BCEWithLogitsLoss()

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("model"):   # backbone 기준
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {
            "params": backbone_params,
            **OPTIM_CONFIG["backbone"]
        },
        {
            "params": head_params,
            **OPTIM_CONFIG["head"]
        },
    ]
    
    optimizer = torch.optim.AdamW(param_groups)
    total_steps = CONFIG["epochs"] * len(train_loader)
    scheduler = cosine_with_min_lr(
        optimizer,
        warmup_steps=CONFIG['warmup_epochs']*len(train_loader),
        total_steps=total_steps,
        min_lr=CONFIG['min_lr']
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=1e-6)
    
    scaler = torch.amp.GradScaler(device=device.type)
    
    # ----------------------------------------------------
    # 학습 시작
    # ----------------------------------------------------
    best_acc = 0.0
    print(f"\n학습 시작! (Epochs: {CONFIG['epochs']})")
    
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | LR: {optimizer.param_groups[0]['lr']:.9f}")
        
    
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch+1, use_wandb=USE_WANDB)
        print(f"   [Train] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"   [Valid] Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # 스케줄러 업데이트
            scheduler.step()
        
        if val_acc > best_acc:
            print(f"   최고 성능 갱신! ({best_acc:.2f}% -> {val_acc:.2f}%) './model' 폴더에 저장 중...")
            best_acc = val_acc
            torch.save(model.state_dict(), CONFIG['save_path'])

        if USE_WANDB is not None:
            # Wandb에 epoch별 메트릭 기록

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
            })
        # 1. GPU: 캐시 비우기
        torch.cuda.empty_cache()
        # 2. Python GC 강제 실행
        gc.collect()
            
    print(f"\n학습 완료! 최고 정확도: {best_acc:.2f}%")
    print(f"모델 저장 위치: {CONFIG['save_path']}")

    

if __name__ == "__main__":
    main()