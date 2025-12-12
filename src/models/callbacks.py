import wandb
import torch
import glob
import os
from PIL import Image   # resize 위해 추가

_last_train_log = {}

def wandb_train_logging(trainer):
    """train loss를 모아두는 역할만 수행"""
    global _last_train_log
    log_dict = {}

    # ---- train loss ----
    loss = getattr(trainer, "loss", None)
    if isinstance(loss, torch.Tensor):
        if loss.numel() == 1:
            log_dict["train/loss_total"] = loss.item()
        else:
            for i, v in enumerate(loss):
                log_dict[f"train/loss_part{i}"] = v.item()

    # ---- LR ----
    if hasattr(trainer, "optimizer"):
        for i, g in enumerate(trainer.optimizer.param_groups):
            log_dict[f"lr/group{i}"] = g["lr"]

    _last_train_log = log_dict


def wandb_val_logging(validator):
    """validation metrics + train loss + 이미지 로깅"""
    global _last_train_log
    log_dict = dict(_last_train_log)

    # --------------------------
    # Validation metrics
    # --------------------------
    metrics = getattr(validator, "metrics", None)
    if metrics:
        results = getattr(metrics, "results_dict", {})
        for k, v in results.items():
            try:
                log_dict[f"val/{k}"] = float(v)
            except:
                pass

    # ============================================================
    # 이미지 업로드 (이미지 파일만 필터링 + resize 적용)
    # ============================================================
    image_dir = getattr(validator, "save_dir", None)

    # 이미지 확장자 리스트
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    if image_dir and os.path.isdir(image_dir):
        all_files = glob.glob(os.path.join(image_dir, "*.*"))

        # ---- 이미지 파일만 선택 ----
        image_files = [f for f in all_files if f.lower().endswith(valid_exts)]

        wandb_images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((640, 640))   # (방법1) resize
                wandb_images.append(
                    wandb.Image(img, caption=os.path.basename(img_path))
                )
            except Exception as e:
                print(f"Image upload error: {e}")

        if wandb_images:
            log_dict["val/images"] = wandb_images

    # --------------------------
    # wandb 업로드
    # --------------------------
    if log_dict:
        wandb.log(log_dict)
