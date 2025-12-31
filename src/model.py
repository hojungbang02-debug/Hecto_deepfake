import torch
import torch.nn as nn
import timm

class DeepFakeModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', pretrained=True):
        """
        Args:
            model_name (str): 사용할 모델 이름 (예: 'efficientnet_b0', 'efficientnet_b4', 'convnext_tiny')
            pretrained (bool): ImageNet 사전 학습 가중치 사용 여부 (무조건 True 권장)
        """
        super(DeepFakeModel, self).__init__()
        
        # 1. TIMM 라이브러리로 SOTA 모델 불러오기
        # num_classes=1: 우리는 가짜(1)일 확률 하나만 뽑으면 됨 (Binary Classification)
        self.model = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=1
        )
        
        # 참고: EfficientNet은 기본적으로 다음과 같은 구조를 가짐
        # Input -> Features(CNN) -> GlobalAvgPool -> Dropout -> Classifier(Linear)
        # timm이 num_classes=1에 맞춰서 마지막 Linear 층을 자동으로 교체해 줌.

    def forward(self, x):
        # x: [Batch_Size, 3, 224, 224]
        
        # 모델 통과
        output = self.model(x) 
        
        # output: [Batch_Size, 1] -> Logits 값 (0~1 사이 확률이 아니라 -무한대 ~ +무한대 값)
        return output

# ==========================================
# 🧪 모델 테스트 코드 (이 파일을 직접 실행할 때만 작동)
# ==========================================
if __name__ == "__main__":
    # 1. 모델 생성 테스트 (가벼운 b0 버전으로 테스트)
    # 실제 학습 땐 'efficientnet_b4'나 'convnext_base' 등을 추천
    try:
        model = DeepFakeModel(model_name='efficientnet_b4', pretrained=True)
        print("✅ 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        exit()

    # 2. 더미 데이터(가짜 이미지)를 넣어서 잘 뱉어내는지 확인
    # 배치크기 4, 채널 3(RGB), 높이 224, 너비 224
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # 순전파 (Forward Pass)
    output = model(dummy_input)
    
    print(f"입력 크기: {dummy_input.shape}")     # torch.Size([4, 3, 224, 224])
    print(f"출력 크기: {output.shape}")          # torch.Size([4, 1]) 이어야 함
    print(f"출력 값(Logits):\n{output.detach().numpy()}")
    
    print("\n🎉 모델 구조 이상 무! 학습 준비 완료.")