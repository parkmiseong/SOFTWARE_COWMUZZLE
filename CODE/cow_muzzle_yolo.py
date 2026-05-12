'''
사용 코드 : https://wikidocs.net/295761
YOLOv8을 활용한 소 얼굴 비문 탐지 모델 학습
- yolov8n.pt 불러오기 : YOLOv8의 가장 작은 모델
- data.yaml 파일의 경로 설정
- 모델 학습 -> 결과물은 'cow_muzzle_yolo' 폴더에 저장
'''
#라이브러리
import os
import torch
from ultralytics import YOLO

#YOLOv8 모델 불러오기
#yolov8n.pt 모델을 불러오며, 해당 모델은 미리 학습된 가장 가벼운 모델
model = YOLO('yolov8n.pt')

#GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("사용 가능 장치 :", device)
#모델을 GPU로 이동
model.to(device)

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