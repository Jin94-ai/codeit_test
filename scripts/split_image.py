"""이미지 8분할 스크립트"""
from PIL import Image
import os

# 입력 이미지
input_path = r'C:\Users\USER\codeit_team8_project1\docs\images\Gemini_Generated_Image_hc342rhc342rhc34.png'
output_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\split_slides'
os.makedirs(output_dir, exist_ok=True)

# 이미지 로드
img = Image.open(input_path)
width, height = img.size
print(f"원본 이미지 크기: {width} x {height}")

# 2행 4열 그리드
cols, rows = 4, 2
cell_width = width // cols
cell_height = height // rows

print(f"각 셀 크기: {cell_width} x {cell_height}")

# 슬라이드 이름
slide_names = [
    "01_medical_ai",
    "02_project_overview",
    "03_eda_analysis",
    "04_trials_failure",
    "05_2stage_pipeline",
    "06_best_model",
    "07_team_partnership",
    "08_conclusion"
]

# 8분할
for row in range(rows):
    for col in range(cols):
        idx = row * cols + col
        left = col * cell_width
        upper = row * cell_height
        right = left + cell_width
        lower = upper + cell_height

        # 크롭
        cropped = img.crop((left, upper, right, lower))

        # 저장
        output_path = os.path.join(output_dir, f"{slide_names[idx]}.png")
        cropped.save(output_path)
        print(f"저장: {slide_names[idx]}.png ({cell_width}x{cell_height})")

print(f"\n총 {rows * cols}개 이미지 분할 완료!")
print(f"저장 위치: {output_dir}")
