# cow_muzzle_cutting.py
## 라이브러리 임포트
import os                       # 파일 생성 등을 위한 라이브러리
import cv2                      # 이미지 처리 라이브러리
from ultralytics import YOLO    # YOLO 모델을 사용하기 위한 라이브러리

## cow_muzzle_yolo를 통해 얻은 YOLOv8 모델 로드
model_path = './DATA/runs/detect/cow_muzzle_yolo-3/weights/best.pt'
yolo_model = YOLO(model_path)

## 디렉토리 경로 설정
# 전처리할 이미지 불러오기
input_image_directory = './DATA/cow-muzzle-dataset/train/images'
# 전처리 완료 후 저장할 디렉토리 생성
output_image_directory = './DATA/noses_yolo/'
# 출력 폴더가 존재하지 않을 경우 생성
os.makedirs(output_image_directory, exist_ok=True)

# 전체 추출 개수를 누적할 변수 생성
total_extracted = 0

## 이미지를 디렉토리에서 불러오기
# 이미지의 확장자가 .jpg, .png인 파일만 리스트에 포함
image_files = [f for f in os.listdir(input_image_directory) if f.endswith('.jpg') or f.endswith('.png')]

# 이미지 읽기 실패 횟수 카운트
fail = 0

## 비문 추출
print("[비문 추출 시작]")
for image_file in image_files:
    # 이미지 경로 불러오기
    image_path = os.path.join(input_image_directory, image_file)
    # 이미지 읽기
    img = cv2.imread(image_path)

    # 이미지 읽기 실패 시, 오류 메시지 출력 및 실패 횟수 증가
    if img is None:
        print("이미지 읽기 실패 : " + image_path)
        fail += 1
    else:
        ## 이미지 읽기 성공
        # 비문(코) 영역 탐지
        results = yolo_model.predict(source=img, conf=0.5, verbose=False)
        boxes = results[0].boxes

        # 탐지된 바운딩 박스 순회
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped_nose = img[y1:y2, x1:x2]

            # 덮어쓰기 방지: 원본 파일명(image_file)을 활용하여 새로운 이름 부여
            # 예: cow_01.jpg -> cow_01_nose_0.jpg
            base_name = os.path.splitext(image_file)[0]
            output_filename = f'{base_name}_nose_{i}.jpg'
            output_path = os.path.join(output_image_directory, output_filename)
        
            cv2.imwrite(output_path, cropped_nose)
            total_extracted += 1 # 성공할 때마다 카운트 1 증가
    

print("[비문 추출 완료]")
print("총 " + str(total_extracted) + "개의 비문을 '" + output_image_directory + "' 폴더에 성공적으로 저장했습니다.")