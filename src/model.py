import torch
import torch.nn as nn
import timm

class DeepFakeModelDinoV2(nn.Module):
    def __init__(self, model_name='dinov2_vitb14', in_chs=3, pretrained=True):
        """
        Args:
            model_name (str): 사용할 모델 이름
            pretrained (bool): ImageNet 사전 학습 가중치 사용 여부
        """
        super(DeepFakeModelDinoV2, self).__init__()

        self.proj = nn.Conv2d(in_chs, 3, kernel_size=1) if in_chs != 3 else nn.Identity()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.head = nn.Linear(self.model.num_features, 1)

    def forward(self, x):
        x = self.proj(x)
        x = self.model(x)
        x = self.head(x)
        return x
        
    
    def set_gradient_checkpointing(self, enable=True):
        """
        VRAM 부족 시 이 함수를 호출하면 메모리를 아낄 수 있음.
        (대신 학습 속도가 약 20~30% 느려짐)
        """
        if hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(enable)
            print(f"Gradient Checkpointing {'Enabled' if enable else 'Disabled'} (Memory Saving)")
        else:
            print("이 모델은 Gradient Checkpointing을 지원하지 않습니다.")

class DeepFakeModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', pretrained=True):
        """
        Args:
            model_name (str): 사용할 모델 이름
            pretrained (bool): ImageNet 사전 학습 가중치 사용 여부
        """
        super(DeepFakeModel, self).__init__()

        # drop_rate: 마지막 분류기(Classifier) 앞의 Dropout 비율 (0.3~0.5 추천)
        # drop_path_rate: 모델 중간중간 레이어를 랜덤하게 건너뛰는 비율 (Stochastic Depth). 
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=1,
            drop_rate=0.4,       # 40% 확률로 노드 끄기 (Overfitting 방지)
            drop_path_rate=0.2   # 깊은 모델 학습 안정화
        )
        
    def forward(self, x):
        return self.model(x)
    
    def set_gradient_checkpointing(self, enable=True):
        """
        VRAM 부족 시 이 함수를 호출하면 메모리를 아낄 수 있음.
        (대신 학습 속도가 약 20~30% 느려짐)
        """
        if hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(enable)
            print(f"Gradient Checkpointing {'Enabled' if enable else 'Disabled'} (Memory Saving)")
        else:
            print("이 모델은 Gradient Checkpointing을 지원하지 않습니다.")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 생성 
    try:
        model = DeepFakeModel(model_name='efficientnet_b4', pretrained=True)
        model.to(device)
        print("모델 로드 성공!")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        exit()

    # 메모리 확인용
    dummy_input = torch.randn(2, 3, 380, 380).to(device)
    
    # 순전파 (Forward Pass)
    output = model(dummy_input)
    
    print(f"입력 크기: {dummy_input.shape}")    
    print(f"출력 크기: {output.shape}")          
    print(f"출력 값(Logits):\n{output.detach().cpu().numpy()}")
    
    print("\n모델 구조 이상 무! 학습 준비 완료.")