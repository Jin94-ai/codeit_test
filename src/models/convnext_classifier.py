"""
Stage 2: ConvNeXt Pill Classifier
- timm의 ConvNeXt 사용
- cropped 이미지에서 K-code 분류
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "cropped"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

CONFIG = {
    "model_name": "convnext_tiny",
    "pretrained": True,
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "img_size": 224,
    "val_split": 0.1,
    "patience": 10,
    "num_workers": 2,
}


class PillDataset(Dataset):
    """크롭된 알약 이미지 데이터셋"""

    def __init__(self, image_paths: list, labels: list, label2idx: dict, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_idx = self.label2idx[label]
        return image, label_idx


def load_data():
    """데이터 로드 및 라벨 매핑 생성"""
    print("\n[데이터 로드]")

    image_paths = []
    labels = []

    # 어노테이션에서 K-code 추출
    for ann_path in ANNOTATIONS_DIR.glob("*.json"):
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # K-code 추출 (파일명에서)
            k_code = ann_path.stem.split("_")[0]  # K-001900_0001 -> K-001900

            # 대응 이미지 경로
            img_path = IMAGES_DIR / f"{ann_path.stem}.png"
            if img_path.exists():
                image_paths.append(img_path)
                labels.append(k_code)
        except Exception:
            continue

    # 라벨 매핑
    unique_labels = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    print(f"  총 이미지: {len(image_paths)}개")
    print(f"  클래스 수: {len(unique_labels)}개")

    return image_paths, labels, label2idx, idx2label


def create_dataloaders(image_paths, labels, label2idx, config):
    """Train/Val DataLoader 생성"""

    # Train/Val 분할
    from sklearn.model_selection import train_test_split

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=config["val_split"],
        stratify=labels,
        random_state=42
    )

    print(f"  Train: {len(train_paths)}개, Val: {len(val_paths)}개")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = PillDataset(train_paths, train_labels, label2idx, train_transform)
    val_dataset = PillDataset(val_paths, val_labels, label2idx, val_transform)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 에폭 학습"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Train")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"})

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    """검증"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main():
    print("=" * 60)
    print("Stage 2: ConvNeXt Pill Classifier")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # 데이터 로드
    image_paths, labels, label2idx, idx2label = load_data()
    num_classes = len(label2idx)

    # DataLoaders
    train_loader, val_loader = create_dataloaders(image_paths, labels, label2idx, CONFIG)

    # 모델
    print(f"\n[모델: {CONFIG['model_name']}]")
    model = timm.create_model(
        CONFIG["model_name"],
        pretrained=CONFIG["pretrained"],
        num_classes=num_classes
    )
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # W&B
    if WANDB_AVAILABLE:
        try:
            wandb.init(
                project="codeit_team8",
                entity="codeit_team8",
                name="convnext_classifier",
                config=CONFIG,
                settings=wandb.Settings(init_timeout=120)
            )
        except Exception as e:
            print(f"W&B 연결 실패: {e}")

    # 저장 경로
    save_dir = PROJECT_ROOT / "runs" / "classify" / "convnext"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 학습
    print("\n[학습 시작]")
    best_acc = 0
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        # W&B 로깅
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Best 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "label2idx": label2idx,
                "idx2label": idx2label,
            }, save_dir / "best.pt")
            print(f"  ✓ Best 모델 저장 (Acc: {best_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # 최종 저장
    torch.save({
        "model_state_dict": model.state_dict(),
        "label2idx": label2idx,
        "idx2label": idx2label,
    }, save_dir / "last.pt")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"모델 저장: {save_dir}")
    print("=" * 60)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
