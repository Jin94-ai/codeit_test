"""텍스트 영역을 크롭하여 제거하는 스크립트"""
from PIL import Image
import os

input_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\split_slides'
output_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\split_slides_cropped'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = Image.open(input_path)
        w, h = img.size

        # 상단 22%, 하단 12% 크롭 (텍스트 영역 제거)
        top_crop = int(h * 0.22)
        bottom_crop = int(h * 0.12)

        # 크롭 영역: (left, upper, right, lower)
        cropped = img.crop((0, top_crop, w, h - bottom_crop))

        # 저장
        cropped.save(output_path)
        print(f"크롭 완료: {filename} -> {cropped.size[0]}x{cropped.size[1]}")

print(f"\n모든 이미지 크롭 완료!")
print(f"저장 위치: {output_dir}")
