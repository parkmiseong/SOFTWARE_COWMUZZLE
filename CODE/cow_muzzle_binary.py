## 라이브러리 임포트 ##
import os
import cv2
import numpy as np
from ultralytics import YOLO

## 경로 설정 ##
input_directory = './DATA/muzzle_dataset'

# 최종 저장 폴더
output_directory = './DATA/processed_dataset'

# 출력 폴더 생성
os.makedirs(output_directory, exist_ok=True)

## YOLO 모델 로드 ##
model_path = './DATA/best.pt'
yolo_model = YOLO(model_path)


## 소 비문 추출 + 이진화 + 저장 ##
def process_cow_muzzle_dataset(input_dir, output_dir):

    # 통계 변수
    total_images = 0
    total_saved = 0
    detect_fail = 0
    read_fail = 0

    print("\n[소 비문 데이터셋 처리 시작]")

    # 모든 이미지 탐색
    image_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image_files.append(image_path)

    print(f"총 이미지 수 : {len(image_files)}")

    ## 이미지 처리 ##
    for image_path in image_files:
        total_images += 1       # 전체 이미지 수 증가
        # 개체 폴더 이름 추출
        cattle_id = os.path.basename(os.path.dirname(image_path))
        # 저장 폴더 생성
        cattle_output_dir = os.path.join(output_dir, cattle_id)
        os.makedirs(cattle_output_dir, exist_ok=True)

        # 이미지 읽기
        img = cv2.imread(image_path)

        # 읽기 실패
        if img is None:
            print(f"[읽기 실패] {image_path}")
            read_fail += 1
            continue

        ## YOLO 비문 탐지 ##
        results = yolo_model.predict(source=img, conf=0.25, verbose=False)
        boxes = results[0].boxes

        # 탐지 실패
        if len(boxes) == 0:
            detect_fail += 1
            continue

        ## 가장 큰 바운딩 박스 선택 ##
        largest_box = None
        largest_area = 0

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = (x1, y1, x2, y2)

        # 자른 이미지 추출
        x1, y1, x2, y2 = largest_box
        cropped_nose = img[y1:y2, x1:x2]

        ## 이진화 처리 ##
        # 그레이스케일 변환
        gray_image = cv2.cvtColor(cropped_nose,cv2.COLOR_BGR2GRAY)
        # 이미지 이진화
        binary_image = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,2)
        # 이미지 반전
        reverse_image = cv2.bitwise_not(binary_image)
        # 모플 연산으로 노이즈 제거
        process_image = cv2.morphologyEx(reverse_image,cv2.MORPH_CLOSE,np.ones((3, 3), np.uint8))
        # 이미지 크기 조정(128x128)
        resized_image = cv2.resize(process_image,(128, 128))
        
        ## 저장 이름 ##
        original_name = os.path.splitext(os.path.basename(image_path))[0]
        save_name = f"{original_name}.png"
        save_path = os.path.join(cattle_output_dir, save_name)

        ## 저장 ##
        cv2.imwrite(save_path, resized_image)
        total_saved += 1

    ## 결과 출력 ##
    print("\n==============================")
    print("[데이터셋 처리 완료]")
    print("==============================")

    print(f"총 이미지 수      : {total_images}")
    print(f"저장 완료 수      : {total_saved}")
    print(f"이미지 읽기 실패  : {read_fail}")
    print(f"비문 탐지 실패    : {detect_fail}")


## 메인 ##
process_cow_muzzle_dataset(input_directory,output_directory)