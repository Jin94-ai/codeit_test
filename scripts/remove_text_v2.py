"""텍스트 제거 v2 - 더 강력한 인페인팅"""
import cv2
import numpy as np
import os

input_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\split_slides'
output_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\split_slides_clean'
os.makedirs(output_dir, exist_ok=True)

def remove_text_v2(img_path, output_path):
    """밝은 텍스트를 감지하고 인페인팅으로 제거"""
    img = cv2.imread(img_path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # HSV 변환으로 밝은 영역 감지
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 밝고 채도 낮은 영역 (흰색/밝은 텍스트) 마스크
    # V(명도) 높고, S(채도) 낮은 픽셀
    lower = np.array([0, 0, 200])
    upper = np.array([180, 60, 255])
    mask1 = cv2.inRange(hsv, lower, upper)

    # 청록색 계열 밝은 텍스트도 감지
    lower2 = np.array([80, 50, 180])
    upper2 = np.array([130, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # 마스크 합치기
    mask = cv2.bitwise_or(mask1, mask2)

    # 상단 35%와 하단 15%만 처리 (텍스트 영역 한정)
    region_mask = np.zeros_like(mask)
    region_mask[:int(h*0.35), :] = 255
    region_mask[int(h*0.85):, :] = 255

    # 텍스트 영역만 남기기
    mask = cv2.bitwise_and(mask, region_mask)

    # 마스크 확장 (텍스트 주변도 포함)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # 노이즈 제거
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 인페인팅
    result = cv2.inpaint(img, mask, inpaintRadius=10, flags=cv2.INPAINT_NS)

    cv2.imwrite(output_path, result)
    return True

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if remove_text_v2(input_path, output_path):
            print(f"처리 완료: {filename}")

print(f"\n저장 위치: {output_dir}")
