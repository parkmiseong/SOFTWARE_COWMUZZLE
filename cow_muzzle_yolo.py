'''
YOLOv8을 활용한 소 얼굴 비문 탐지 모델 학습 스크립트
이 스크립트는 Ultralytics의 YOLOv8 모델을 사용하여 소 얼굴 비문 탐지 모델을 학습하는 과정을 자동화합니다.
- 1단계: YOLOv8의 가장 작은 모델인 yolov8n.pt을 불러옵니다.
- 2단계: data.yaml 파일의 경로를 설정합니다. (실제 경로에 맞게 수정해야 합니다)
- 3단계: 모델을 학습합니다. 학습이 완료되면 결과물이 'CowMuzzle_Project/YOLO_Detector' 폴더에 저장됩니다.
'''
#라이브러리
import os
import torch
from ultralytics import YOLO

#YOLOv8 모델 불러오기
#yolov8n.pt 모델을 불러오며, 해당 모델은 미리 학습된 가장 가벼운 모델
model = YOLO('yolov8n.pt')

#data.yaml 경로
data_yaml_path = 'cow-muzzle-dataset/data.yaml'


print("YOLOv8 비문 탐지 모델 학습 시작")
#모델 학습
results = model.train(
    data=data_yaml_path,        #data.yaml 파일 경로
    epochs=50,                  #데이터셋 반복 학습 횟수
    imgsz=640,                  #학습에 사용할 이미지 크기
    batch=16,                   #배치 크기
    name='cow_muzzle_yolo'      #학습 결과가 저장될 폴더명
)
print("\n모델 학습 완료")