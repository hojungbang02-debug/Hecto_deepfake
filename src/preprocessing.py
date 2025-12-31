import cv2
import os
import glob
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch

# 처리할 작업 목록: (비디오가 있는 소스 폴더, 이미지를 저장할 타겟 폴더)
TASKS = [
    # 1. Real Data (FF++ Original) -> 0_real 폴더로
    {
        "source": "./external_data/original_sequences/youtube/c23/videos",
        "target": "./train_data/0_real",
        "max_frames": 20  # 비디오 1개당 뽑을 이미지 장수 (Real은 데이터가 많으니 적당히)
    },
    # 2. Fake Data (FF++ Deepfakes) -> 1_fake 폴더로
    {
        "source": "./external_data/manipulated_sequences/Deepfakes/c23/videos",
        "target": "./train_data/1_fake",
        "max_frames": 20  # Fake도 균형을 맞춰줍니다.
    }
]
# ======================================================================

def process_videos():
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # MTCNN 얼굴 감지기 로드
    mtcnn = MTCNN(keep_all=False, select_largest=False, device=device, post_process=False, margin=50)

    for task in TASKS:
        source_dir = task["source"]
        save_dir = task["target"]
        max_frames = task["max_frames"]
        
        # 저장 폴더가 없으면 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 비디오 파일 목록 가져오기
        video_files = glob.glob(os.path.join(source_dir, "*.mp4"))
        print(f"\n🚀 시작: {source_dir} -> {save_dir}")
        print(f"총 {len(video_files)}개의 비디오를 처리합니다.")

        for video_path in tqdm(video_files):
            filename = os.path.basename(video_path).split('.')[0]
            cap = cv2.VideoCapture(video_path)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                continue

            # 비디오 길이에 맞춰 일정한 간격 계산 (예: 100프레임인데 10장 뽑으려면 10프레임마다)
            interval = max(1, total_frames // max_frames)
            
            frame_idx = 0
            saved_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 인터벌에 맞고, 목표 장수를 아직 못 채웠다면 처리
                if frame_idx % interval == 0 and saved_count < max_frames:
                    try:
                        # BGR(OpenCV) -> RGB(Pytorch/PIL) 변환
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 파일명: 원본비디오명_프레임번호.jpg (예: 001_frame0.jpg)
                        # 이렇게 하면 나중에 어떤 영상 출신인지 알 수 있음
                        save_path = os.path.join(save_dir, f"ff_{filename}_{frame_idx}.jpg")
                        
                        # 얼굴 감지 및 저장 (MTCNN이 알아서 크롭해서 저장해줌)
                        mtcnn(frame_rgb, save_path=save_path)
                        
                        saved_count += 1
                    except Exception as e:
                        # 가끔 얼굴 인식이 실패하거나 에러나면 그냥 패스
                        pass
                    
                frame_idx += 1

            cap.release()

if __name__ == "__main__":
    process_videos()