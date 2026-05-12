import cv2
import os
from ultralytics import YOLO

# 1. 딥러닝 기반 YOLOv8 모델 로드
model_path = 'C:\\Users\\emaet\\OneDrive\\바탕 화면\\대학\\Git\\runs(epochs=2)\\detect\\cow_muzzle_yolo\\weights\\best.pt'
yolo_model = YOLO(model_path)

# 2. 입출력 경로 설정
image_dir = 'cow-muzzle-dataset/train/images'
output_dir = 'noses_yolo/'
os.makedirs(output_dir, exist_ok=True)

# 3. 전체 추출 개수를 누적할 변수 생성
total_extracted = 0

print("비문 추출을 시작합니다...")

# 4. 이미지 폴더 순회
for img_file in os.listdir(image_dir):
    # 이미지 파일(.jpg, .png 등)만 처리하도록 필터링
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = os.path.join(image_dir, img_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"⚠️ 이미지 읽기 실패: {img_path}. 건너뜁니다.")
        continue # 이미지 읽기 실패 시 건너뜀

    # 비문(코) 영역 탐지 (신뢰도 50% 이상)
    results = yolo_model.predict(source=img, conf=0.1, verbose=False)
    boxes = results[0].boxes

    # 탐지된 바운딩 박스 순회
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_nose = img[y1:y2, x1:x2]

        # ⭐️ 덮어쓰기 방지: 원본 파일명(img_file)을 활용하여 새로운 이름 부여
        # 예: cow_01.jpg -> cow_01_nose_0.jpg
        base_name = os.path.splitext(img_file)[0]
        output_filename = f'{base_name}_nose_{i}.jpg'
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, cropped_nose)
        total_extracted += 1 # 성공할 때마다 카운트 1 증가

print(f"✅ 추출 완료! 총 {total_extracted}개의 비문을 '{output_dir}' 폴더에 성공적으로 저장했습니다.")