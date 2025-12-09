# src/data/yolo_dataset/yolo_dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from .yolo_augmentation import get_train_transform, set_global_seed


class YoloPillDataset(Dataset):
    """
    YOLO 포맷(.txt) 라벨을 사용하는 커스텀 Dataset.
    - 이미지 경로: img_dir/*.jpg|*.jpeg|*.png
    - 라벨 경로: label_dir/<same_name>.txt
    - txt 형식: class x_center y_center w h  (모두 0~1)
    """

    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms

        self.image_files = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _load_labels(self, label_path):
        boxes = []
        labels = []
        if not os.path.exists(label_path):
            return boxes, labels

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x_c, y_c, w, h = map(float, parts)
                labels.append(int(cls))
                boxes.append([x_c, y_c, w, h])
        return boxes, labels

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, file_name)
        label_path = os.path.join(
            self.label_dir, os.path.splitext(file_name)[0] + ".txt"
        )

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = self._load_labels(label_path)

        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                labels=labels,
            )
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([idx]),
        }

        return image, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def get_train_loader(
    img_dir="datasets/pills/images/train",
    label_dir="datasets/pills/labels/train",
    batch_size=4,
    num_workers=0,
):
    """
    on-the-fly Albumentations 증강이 적용된 Train DataLoader를 반환한다.
    """
    set_global_seed()
    train_transform = get_train_transform()

    dataset = YoloPillDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transforms=train_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader, dataset