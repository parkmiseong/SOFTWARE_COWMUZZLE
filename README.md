소의 코지문 인식 시스템

cow-muzzle-dataset : 소의 코가 찍혀있는 원 데이터셋 -> 캐글에서 다운받은 데이터 셋
- 다운 링크 : https://www.kaggle.com/datasets/sharifashik/cow-muzzle-dataset

noses_yolo : 소의 코지문이 되는 부분만 있는 이미지들

runs : 데이터셋에 있는 YOLOv8을 활용한 소의 코 지문 탐지 모델 -> best.pt를 이용할 용도

yolov8n.pt : YOLOv8을 활용한 모델을 생성하기 위해 사용할 미리 학습된 가장 가벼운 모델

[코드]
- cow_muzzle_yolo.py : YOLOv8을 활용한 소의 코 지문 탐지 모델 생성 코드
- cow_muzzle_cutting.py : 소의 코 지문 탐지 모델을 활용하여 소의 코 지문 부분만을 자르는 코드
- 
