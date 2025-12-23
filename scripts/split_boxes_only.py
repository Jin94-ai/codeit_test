"""이미지에서 네모 박스 부분만 추출"""
from PIL import Image
import os

input_path = r'C:\Users\USER\codeit_team8_project1\docs\images\Gemini_Generated_Image_hc342rhc342rhc34.png'
output_dir = r'C:\Users\USER\codeit_team8_project1\docs\images\slide_boxes'
os.makedirs(output_dir, exist_ok=True)

img = Image.open(input_path)
width, height = img.size
print(f"원본: {width} x {height}")

# 2행 4열 그리드
cols, rows = 4, 2
cell_width = width // cols
cell_height = height // rows

# 각 셀 내에서 박스 영역 (텍스트 제외)
# 상단 여백: 약 2%, 하단 여백(파일명): 약 13%
top_margin_ratio = 0.02
bottom_margin_ratio = 0.14

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

for row in range(rows):
    for col in range(cols):
        idx = row * cols + col

        # 셀 경계
        cell_left = col * cell_width
        cell_top = row * cell_height

        # 박스 영역 (셀 내부에서 여백 제외)
        box_left = cell_left + 2
        box_top = cell_top + int(cell_height * top_margin_ratio)
        box_right = cell_left + cell_width - 2
        box_bottom = cell_top + cell_height - int(cell_height * bottom_margin_ratio)

        # 크롭
        cropped = img.crop((box_left, box_top, box_right, box_bottom))

        # 저장
        output_path = os.path.join(output_dir, f"{slide_names[idx]}.png")
        cropped.save(output_path)
        print(f"저장: {slide_names[idx]}.png ({cropped.size[0]}x{cropped.size[1]})")

print(f"\n완료! 저장 위치: {output_dir}")
