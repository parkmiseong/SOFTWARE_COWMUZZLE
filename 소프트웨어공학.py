import cv2
import os
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# 1. 딥러닝 기반 YOLOv8 모델 로드 (cascade.xml 대체)
model_path = './CowMuzzle_Project/v1_train/weights/best.pt' # 학습된 모델 가중치
yolo_model = YOLO(model_path)

# 2. 이미지 불러오기 (흑백 변환 불필요)
image_path = 'test_images/cow_face.jpg'
img = cv2.imread(image_path)

# 3. 비문(코) 영역 탐지 (AI 예측 수행)
# conf=0.5: 50% 이상 비문이라고 확신하는 경우에만 결과 반환
results = yolo_model.predict(source=img, conf=0.5, verbose=False)

# 4. 탐지된 영역 크롭 및 저장
output_dir = 'noses_yolo/'
os.makedirs(output_dir, exist_ok=True)

# 탐지된 바운딩 박스(네모 상자)들을 순회
boxes = results[0].boxes
for i, box in enumerate(boxes):
    # YOLO는 좌표를 [시작X, 시작Y, 끝X, 끝Y] 형태로 반환합니다.
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # 원본 컬러 이미지에서 코 부분만 정확히 자르기 (Crop)
    cropped_nose = img[y1:y2, x1:x2]
    
    # 잘라낸 이미지 저장
    output_path = os.path.join(output_dir, f'extracted_nose_{i}.jpg')
    cv2.imwrite(output_path, cropped_nose)

print(f"총 {len(boxes)}개의 비문을 성공적으로 추출하여 저장했습니다.")