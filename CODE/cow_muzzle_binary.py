# cow_muzzle_binary.py
## 라이브러리 임포트
import os                               # 파일 생성 등을 위한 라이브러리
import cv2                              # 이미지 처리 라이브러리
import numpy as np                      # 수치 계산 라이브러리
from matplotlib import pyplot as plt    # 이미지 시각화 라이브러리

## 디렉토리 경로 설정
# 전처리할 이미지 불러오기
input_image_directory = './DATA/noses_yolo'
# 전처리 완료 후 저장할 디렉토리 생성
output_image_directory = './DATA/noses_binary'
# 출력 폴더가 존재하지 않을 경우 생성
os.makedirs(output_image_directory, exist_ok=True)

## 이미지를 디렉토리에서 불러오기
# 이미지의 확장자가 .jpg, .png인 파일만 리스트에 포함
image_files = [f for f in os.listdir(input_image_directory) if f.endswith('.jpg') or f.endswith('.png')]

# 이미지 읽기 실패 횟수 카운트
fail = 0

## 이미지 처리(흑백 변환 및 이진화)
print("[이미지 처리 시작]")
for image_file in image_files:
    # 이미지 경로 불러오기
    image_path = os.path.join(input_image_directory, image_file)    # input_image_directory + '/' + image_file
    # 이미지 읽기 (그레이스케일로 읽기 -> 이진화를 위해서 그레이스케일로 읽어오기)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 이미지 읽기 실패 시, 오류 메시지 출력 및 실패 횟수 증가
    if image is None:
        print("이미지 읽기 실패 :", image_path)
        fail += 1
    else:
        #ret, binary_image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
        # 적응형 이진화를 이용하여 이미지를 검정과 흰색으로만 이루어진 이진 이미지로 변환 -> 비문 패턴 강조
        binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        # 이미지 반전
        reverse_image = cv2.bitwise_not(binary_image)
        # 모폴로지 연산을 사용하여 노이즈 제거 및 비문 패턴 강조
        process_image = cv2.morphologyEx(reverse_image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # 전처리된 이미지 저장(256*256 크기로 재조정)
        cv2.imwrite(os.path.join(output_image_directory, image_file), cv2.resize(process_image,(256,256)))

print("[이미지 처리 완료]")
print("이미지 읽기 실패 횟수 :", fail)
print("이미지 처리 완료 횟수 : ", len(image_files) - fail)
print("총 이미지 수 : ", len(image_files))