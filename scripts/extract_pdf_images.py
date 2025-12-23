"""PDF에서 이미지 추출 스크립트"""
import fitz
import os

pdf_path = r'c:\Users\USER\codeit_team8_project1\docs\앱개발 정리.pdf'
output_dir = r'c:\Users\USER\codeit_team8_project1\docs\images\app_dev'
os.makedirs(output_dir, exist_ok=True)

doc = fitz.open(pdf_path)
img_count = 0

for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images()

    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        img_count += 1
        img_filename = os.path.join(output_dir, f"app_dev_p{page_num+1}_img{img_count}.{image_ext}")

        with open(img_filename, "wb") as f:
            f.write(image_bytes)
        print(f"Saved: {img_filename}")

print(f"\nTotal images extracted: {img_count}")
doc.close()
