"""이미지에서 텍스트 제거 스크립트 (인페인팅 방식)"""
import cv2
import numpy as np
import os

input_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\split_slides'
output_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\split_slides_no_text'
os.makedirs(output_dir, exist_ok=True)

def remove_text(img_path, output_path):
    """텍스트 영역을 인페인팅으로 제거"""
    # 이미지 로드
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading: {img_path}")
        return False

    h, w = img.shape[:2]

    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 밝은 텍스트 감지 (흰색/밝은 색 텍스트)
    # 상단 20%와 하단 15% 영역만 처리 (텍스트가 주로 있는 곳)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 상단 영역 (제목)
    top_region = gray[:int(h*0.22), :]
    _, top_mask = cv2.threshold(top_region, 200, 255, cv2.THRESH_BINARY)
    # 마스크 확장 (텍스트 주변도 포함)
    kernel = np.ones((5, 5), np.uint8)
    top_mask = cv2.dilate(top_mask, kernel, iterations=3)
    mask[:int(h*0.22), :] = top_mask

    # 하단 영역 (파일명)
    bottom_region = gray[int(h*0.88):, :]
    _, bottom_mask = cv2.threshold(bottom_region, 180, 255, cv2.THRESH_BINARY)
    bottom_mask = cv2.dilate(bottom_mask, kernel, iterations=3)
    mask[int(h*0.88):, :] = bottom_mask

    # 인페인팅 적용
    result = cv2.inpaint(img, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # 저장
    cv2.imwrite(output_path, result)
    return True

# 모든 분할 이미지 처리
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if remove_text(input_path, output_path):
            print(f"처리 완료: {filename}")
        else:
            print(f"처리 실패: {filename}")

print(f"\n텍스트 제거 완료!")
print(f"저장 위치: {output_dir}")
