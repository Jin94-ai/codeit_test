import wandb
import torch
import numpy as np

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
    """validation metrics와 함께 train loss를 한 번에 wandb로 로깅"""
    global _last_train_log
    log_dict = dict(_last_train_log)

    # --------------------------
    #   Ultralytics Validator metrics
    #   metrics = DetMetrics object
    #   실제 값은 metrics.results_dict에 존재
    # --------------------------
    metrics = getattr(validator, "metrics", None)

    if metrics:
        results = {}
        try:
            results = metrics.results_dict  # ⭐ 올바른 방식
        except:
            pass

        for k, v in results.items():
            try:
                log_dict[f"val/{k}"] = float(v)
            except:
                pass

        # --------------------------
        # mAP@0.75:0.95 계산 (캐글 평가 기준)
        # IoU thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        # index 5~9가 0.75~0.95에 해당
        # --------------------------
        try:
            if hasattr(metrics, 'box') and hasattr(metrics.box, 'ap'):
                ap = metrics.box.ap  # shape: (num_classes, 10)
                if ap is not None and len(ap) > 0:
                    # IoU 0.75~0.95 (index 5~9)
                    ap_75_95 = ap[:, 5:].mean()
                    log_dict["val/mAP75-95"] = float(ap_75_95)
        except Exception as e:
            pass

    # --------------------------
    # wandb 업로드 (한 epoch에 한 번)
    # --------------------------
    if log_dict:
        wandb.log(log_dict)
