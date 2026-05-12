# 라이브러리 임포트
# OpenCV(cv2)와 NumPy, Matplotlib 라이브러리를 사용하여 이미지 처리 및 시각화를 수행
# os : 운영체제와 상호작용하기 위한 라이브러리로, 파일 및 디렉토리 작업에 사용
# cv2 : OpenCV 라이브러리로 이미지 처리에 사용
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 경로 설정
input_directory = './cow-muzzle-dataset/test/images'
output_directory = './Processed_Cow_Dataset'

# 이미지 처리 함수
def convert_image(image_path, output_path):
    # 이미지 읽기
    image = cv2.imread(image_path)

    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화
    # ADAPTIVE_THRESH_GAUSSIAN_C : 픽셀 주변의 작은 영역에서 가우시안 가중치를 적용하여 임계값 계산
    # THRESH_BINARY : 픽셀 값이 임계값보다 크면 최대값(255)으로 설정, 그렇지 않으면 0으로 설정
    # 검정과 흰색으로만 이루어진 이미지로 변환하여 비문 패턴 강조
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 이미지 반전
    inverted_image = cv2.bitwise_not(adaptive_thresh)

    # 모폴로지 연산을 사용하여 노이즈 제거 및 비문 패턴 강조
    # cv2.morphologyEx(원본 배열, 연산 방법, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)
    # MORPH_CLOSE : 닫힘 연산, 객체 내부의 구멍이 사라지게 됨
    # np.ones((3, 3), np.uint8) : 3x3 크기의 구조 요소를 생성하여 노이즈 제거에 사용
    kernel = np.ones((3, 3), np.uint8)
    processed_image = cv2.morphologyEx(inverted_image, cv2.MORPH_CLOSE, kernel)

    # 이미지 저장(256*256 크기로 재조정)
    cv2.imwrite(output_path, cv2.resize(processed_image, (256, 256)))

    # 처리된 이미지 저장 경로 출력
    print("Processed and saved : " + output_path)

# 입력 디렉토리에서 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(input_directory) if f.endswith('.jpg') or f.endswith('.png')]

# 이미지 파일을 하나씩 처리
for image_file in image_files:
    input_path = os.path.join(input_directory, image_file)
    output_path = os.path.join(output_directory, image_file)
    convert_image(input_path, output_path)

# 모든 이미지 처리 완료 메세지
print("Image processing completed.")