"""
COCO 포맷 → YOLO 포맷 변환 스크립트

역할:
1. 651개 이미지 + 1001개 JSON → 232개 유효 데이터 필터링
2. COCO bbox → YOLO bbox 변환
3. Train/Val split (80:20, Stratified)
4. YOLOv8 학습용 구조 생성

실행:
    python src/data/coco_to_yolo.py

결과:
    data/processed/
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    ├── labels/val/
    └── pill_config.yaml
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import yaml


class COCOtoYOLOConverter:
    def __init__(self, data_root='data/raw', output_root='data/processed'):
        self.data_root = Path(data_root)
        self.output_root = Path(output_root)

        self.train_img_dir = self.data_root / 'train_images'
        self.train_anno_dir = self.data_root / 'train_annotations'

        # 출력 디렉토리
        self.output_images_train = self.output_root / 'images' / 'train'
        self.output_images_val = self.output_root / 'images' / 'val'
        self.output_labels_train = self.output_root / 'labels' / 'train'
        self.output_labels_val = self.output_root / 'labels' / 'val'

        # 데이터 저장
        self.all_images = []
        self.all_annotations = []
        self.all_categories = []

        # 통계
        self.stats = {
            'total_images': 0,
            'total_json': 0,
            'valid_images': 0,
            'total_annotations': 0,
            'train_count': 0,
            'val_count': 0,
            'num_classes': 0
        }

    def find_json_files(self):
        """JSON 파일 재귀 탐색"""
        json_files = []
        for root, dirs, files in os.walk(self.train_anno_dir):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(Path(root) / file)
        return json_files

    def parse_annotations(self):
        """
        모든 JSON 파일 파싱 및 데이터 통합
        - 나연님 로직: file_name 기준 중복 제거
        - 민우님 로직: bbox 유효성 검사
        """
        print("=" * 60)
        print("JSON 파일 파싱 중...")
        print("=" * 60)

        json_files = self.find_json_files()
        self.stats['total_json'] = len(json_files)

        # 중복 방지
        parsed_filenames = set()
        parsed_anno_ids = set()
        parsed_category_pairs = set()

        for json_path in tqdm(json_files, desc="Parsing JSON"):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️ JSON 로드 실패: {json_path} - {e}")
                continue

            # Images: file_name 기준 중복 제거 (나연님 방식)
            if 'images' in data:
                for img_info in data['images']:
                    if img_info['file_name'] not in parsed_filenames:
                        img_info['source_json'] = str(json_path)
                        self.all_images.append(img_info)
                        parsed_filenames.add(img_info['file_name'])

            # Annotations: bbox 유효성 + id 중복 제거 (민우님 방식)
            if 'annotations' in data:
                for anno in data['annotations']:
                    # bbox 유효성 검사 (데이터셋 설명서 기준)
                    is_valid = (
                        'bbox' in anno and
                        anno['bbox'] and
                        isinstance(anno['bbox'], list) and
                        len(anno['bbox']) == 4 and
                        anno['id'] not in parsed_anno_ids
                    )
                    if is_valid:
                        anno['source_json'] = str(json_path)
                        self.all_annotations.append(anno)
                        parsed_anno_ids.add(anno['id'])

            # Categories: (id, name) 중복 제거
            if 'categories' in data:
                for cat in data['categories']:
                    cat_pair = (cat['id'], cat['name'])
                    if cat_pair not in parsed_category_pairs:
                        self.all_categories.append(cat)
                        parsed_category_pairs.add(cat_pair)

        print(f"✅ 파싱 완료: {len(self.all_images)}개 이미지, {len(self.all_annotations)}개 어노테이션")

    def filter_valid_data(self):
        """
        이미지-JSON 양방향 매칭 검증 (나연님 로직)
        651개 → 232개로 필터링
        """
        print("\n" + "=" * 60)
        print("이미지-JSON 매칭 검증 중...")
        print("=" * 60)

        # 실제 이미지 파일
        actual_images = set([f.name for f in self.train_img_dir.glob('*.png')])
        self.stats['total_images'] = len(actual_images)

        # JSON에 언급된 이미지
        json_images = set([img['file_name'] for img in self.all_images])

        # 양방향 매칭
        valid_images = actual_images.intersection(json_images)
        missing_json = actual_images - valid_images
        missing_image = json_images - valid_images

        print(f"정상 매칭:     {len(valid_images):,}개")
        print(f"JSON 없음:    {len(missing_json):,}개 (이미지만 존재)")
        print(f"이미지 없음:  {len(missing_image):,}개 (JSON만 존재)")

        if len(missing_json) > 0:
            print(f"   예시: {list(missing_json)[:3]}")
        if len(missing_image) > 0:
            print(f"   예시: {list(missing_image)[:3]}")

        # 유효한 데이터만 필터링
        images_df = pd.DataFrame(self.all_images)
        images_df = images_df[images_df['file_name'].isin(valid_images)].reset_index(drop=True)

        annotations_df = pd.DataFrame(self.all_annotations)
        valid_image_ids = set(images_df['id'].tolist())
        annotations_df = annotations_df[annotations_df['image_id'].isin(valid_image_ids)].reset_index(drop=True)

        self.stats['valid_images'] = len(images_df)
        self.stats['total_annotations'] = len(annotations_df)

        print(f"\n최종 사용 가능 데이터:")
        print(f"   이미지:       {len(images_df):,}개")
        print(f"   어노테이션:   {len(annotations_df):,}개")

        return images_df, annotations_df

    def create_category_mapping(self):
        """카테고리 ID → 클래스 인덱스 매핑 생성"""
        categories_df = pd.DataFrame(self.all_categories)
        categories_df = categories_df.sort_values('id').reset_index(drop=True)

        # YOLO는 0부터 시작하는 연속된 인덱스 필요
        category_to_idx = {
            cat['id']: idx
            for idx, cat in enumerate(categories_df.to_dict('records'))
        }

        self.stats['num_classes'] = len(category_to_idx)

        print(f"\n클래스 정보:")
        print(f"   총 클래스 수: {self.stats['num_classes']}개")
        print(f"   클래스 예시:")
        for idx, (cat_id, name) in enumerate(zip(categories_df['id'][:5], categories_df['name'][:5])):
            print(f"      {category_to_idx[cat_id]:2d}. {name}")

        return category_to_idx, categories_df

    def coco_to_yolo_bbox(self, coco_bbox, img_width, img_height):
        """
        COCO bbox → YOLO bbox 변환

        COCO: [x_min, y_min, width, height] (절대좌표)
        YOLO: [x_center, y_center, width, height] (정규화)
        """
        x_min, y_min, width, height = coco_bbox

        # 중심점 계산
        x_center = x_min + width / 2
        y_center = y_min + height / 2

        # 정규화 (0~1)
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        width_norm = width / img_width
        height_norm = height / img_height

        return [x_center_norm, y_center_norm, width_norm, height_norm]

    def train_val_split(self, images_df, annotations_df):
        """
        Train/Val split (Stratified)
        클래스 분포 유지하면서 80:20 분할
        """
        print("\n" + "=" * 60)
        print("Train/Val Split (Stratified, 80:20)")
        print("=" * 60)

        # 이미지별 대표 클래스 결정 (첫 번째 어노테이션 클래스)
        image_to_class = annotations_df.groupby('image_id')['category_id'].first().to_dict()
        images_df['class_id'] = images_df['id'].map(image_to_class)

        # Stratified split
        train_df, val_df = train_test_split(
            images_df,
            test_size=0.2,
            stratify=images_df['class_id'],
            random_state=42
        )

        self.stats['train_count'] = len(train_df)
        self.stats['val_count'] = len(val_df)

        print(f"Train: {len(train_df):,}개 ({len(train_df)/len(images_df)*100:.1f}%)")
        print(f"Val:   {len(val_df):,}개 ({len(val_df)/len(images_df)*100:.1f}%)")

        return train_df, val_df

    def create_directories(self):
        """출력 디렉토리 생성"""
        print("\n" + "=" * 60)
        print("출력 디렉토리 생성 중...")
        print("=" * 60)

        for directory in [
            self.output_images_train,
            self.output_images_val,
            self.output_labels_train,
            self.output_labels_val
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"{directory}")

    def process_dataset(self, df, annotations_df, category_to_idx, split='train'):
        """
        데이터셋 처리: 이미지 복사 + 라벨 생성
        """
        output_img_dir = self.output_images_train if split == 'train' else self.output_images_val
        output_lbl_dir = self.output_labels_train if split == 'train' else self.output_labels_val

        print(f"\n{split.upper()} 데이터 처리 중...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
            file_name = row['file_name']
            image_id = row['id']
            img_width = row['width']
            img_height = row['height']

            # 이미지 복사
            src_img = self.train_img_dir / file_name
            dst_img = output_img_dir / file_name

            try:
                shutil.copy2(src_img, dst_img)
            except Exception as e:
                print(f"이미지 복사 실패: {file_name} - {e}")
                continue

            # 해당 이미지의 모든 어노테이션 가져오기
            img_annotations = annotations_df[annotations_df['image_id'] == image_id]

            # YOLO 라벨 파일 생성
            label_file = output_lbl_dir / file_name.replace('.png', '.txt')

            with open(label_file, 'w') as f:
                for _, anno in img_annotations.iterrows():
                    class_id = category_to_idx[anno['category_id']]
                    coco_bbox = anno['bbox']

                    # YOLO bbox 변환
                    yolo_bbox = self.coco_to_yolo_bbox(coco_bbox, img_width, img_height)

                    # YOLO 형식: class_id x_center y_center width height
                    line = f"{class_id} {' '.join([f'{v:.6f}' for v in yolo_bbox])}\n"
                    f.write(line)

        print(f"{split.upper()} 완료: {len(df)}개 이미지")

    def create_yaml_config(self, categories_df):
        """
        YOLOv8 학습용 YAML config 생성
        """
        print("\n" + "=" * 60)
        print("YAML Config 생성 중...")
        print("=" * 60)

        # 절대 경로로 변환
        project_root = Path.cwd()

        config = {
            'path': str(project_root / self.output_root),
            'train': 'images/train',
            'val': 'images/val',
            'test': '',  # 테스트셋 없음

            'nc': len(categories_df),
            'names': categories_df['name'].tolist(),

            # Hyperparameters (EDA 기반)
            'imgsz': 640,

            # Augmentation (배경 과적합 방지)
            'hsv_h': 0.03,      # Hue
            'hsv_s': 0.9,       # Saturation (강화)
            'hsv_v': 0.6,       # Value (강화)
            'degrees': 15.0,    # Rotation
            'translate': 0.1,   # Translation
            'scale': 0.5,       # Scale
            'fliplr': 0.5,      # Horizontal flip
            'mosaic': 1.0,      # Mosaic augmentation
            'mixup': 0.0,       # MixUp (선택)
        }

        yaml_path = self.output_root / 'pill_config.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"Config 저장: {yaml_path}")
        print(f"\n설정 내용:")
        print(f"   클래스 수: {config['nc']}")
        print(f"   이미지 크기: {config['imgsz']}")
        print(f"   ColorJitter: hsv_s={config['hsv_s']}, hsv_v={config['hsv_v']}")

        return yaml_path

    def print_summary(self):
        """최종 통계 출력"""
        print("\n" + "=" * 60)
        print("변환 완료 - 최종 통계")
        print("=" * 60)
        print(f"원본 데이터:")
        print(f"  - 총 이미지:           {self.stats['total_images']:,}개")
        print(f"  - 총 JSON 파일:        {self.stats['total_json']:,}개")
        print(f"\n필터링 후:")
        print(f"  - 유효 이미지:         {self.stats['valid_images']:,}개")
        print(f"  - 총 어노테이션:       {self.stats['total_annotations']:,}개")
        print(f"  - 클래스 수:           {self.stats['num_classes']}개")
        print(f"\nTrain/Val Split:")
        print(f"  - Train:               {self.stats['train_count']:,}개 (80%)")
        print(f"  - Val:                 {self.stats['val_count']:,}개 (20%)")
        print("=" * 60)
        print(f"\n✨ 데이터 준비 완료!")
        print(f"   위치: {self.output_root}")
        print(f"\n다음 단계:")
        print(f"   yolo train data={self.output_root / 'pill_config.yaml'} model=yolov8n.pt epochs=50")
        print("=" * 60)

    def convert(self):
        """전체 변환 프로세스 실행"""
        print("\n" + "=" * 60)
        print("COCO → YOLO 포맷 변환 시작")
        print("=" * 60)

        # 1. JSON 파싱
        self.parse_annotations()

        # 2. 유효 데이터 필터링 (232개)
        images_df, annotations_df = self.filter_valid_data()

        # 3. 카테고리 매핑
        category_to_idx, categories_df = self.create_category_mapping()

        # 4. Train/Val split
        train_df, val_df = self.train_val_split(images_df, annotations_df)

        # 5. 출력 디렉토리 생성
        self.create_directories()

        # 6. Train 데이터 처리
        self.process_dataset(train_df, annotations_df, category_to_idx, split='train')

        # 7. Val 데이터 처리
        self.process_dataset(val_df, annotations_df, category_to_idx, split='val')

        # 8. YAML config 생성
        self.create_yaml_config(categories_df)

        # 9. 최종 통계
        self.print_summary()


def main():
    """메인 함수"""
    converter = COCOtoYOLOConverter(
        data_root='data/raw',
        output_root='data/processed'
    )
    converter.convert()


if __name__ == '__main__':
    main()
