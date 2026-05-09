"""Small supervised train/evaluate loops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from segcraft.config import SegCraftConfig
from segcraft.data import SegmentationDataset
from segcraft.models import create_model


def run_training(config: SegCraftConfig) -> dict[str, Any]:
    if not _split_ready(config, "train"):
        return _missing_data_summary(config, "train")

    torch = _torch()
    _set_seed(config.runtime.seed, torch)
    device = _resolve_device(config.runtime.device, torch)

    train_loader = _loader(config, "train", shuffle=True)
    val_loader = _loader(config, "val", shuffle=False) if _split_ready(config, "val") else None
    model = create_model(config.model, config.task).to(device)
    optimizer = _optimizer(model, config, torch)
    criterion = _loss(config, torch)

    history = []
    best_score = -1.0
    checkpoint_dir = Path(config.runtime.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _model_output(model(images))
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            batch_size = images.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            samples += batch_size

        epoch_summary = {
            "epoch": epoch,
            "train_loss": total_loss / max(samples, 1),
        }
        if val_loader is not None:
            epoch_summary.update(_evaluate_loader(model, val_loader, config, device, torch))
            score = float(epoch_summary.get("miou", epoch_summary.get("iou", 0.0)))
        else:
            score = -epoch_summary["train_loss"]

        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.to_dict(),
                    "epoch": epoch,
                    "score": score,
                },
                best_checkpoint,
            )
        history.append(epoch_summary)

    return {
        "status": "completed",
        "device": str(device),
        "epochs_completed": config.train.epochs,
        "train_batches": len(train_loader),
        "val_batches": len(val_loader) if val_loader is not None else 0,
        "best_checkpoint": str(best_checkpoint),
        "history": history,
    }


def run_evaluation(config: SegCraftConfig) -> dict[str, Any]:
    if not _split_ready(config, "val"):
        return _missing_data_summary(config, "val")

    torch = _torch()
    device = _resolve_device(config.runtime.device, torch)
    loader = _loader(config, "val", shuffle=False)
    model = create_model(config.model, config.task).to(device)
    model.eval()

    checkpoint = Path(config.runtime.output_dir) / "checkpoints" / "best.pt"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model_state_dict"])

    metrics = _evaluate_loader(model, loader, config, device, torch)
    return {
        "status": "completed",
        "device": str(device),
        "batches": len(loader),
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "metrics": metrics,
    }


def _loader(config: SegCraftConfig, split: str, *, shuffle: bool) -> Any:
    torch = _torch()
    dataset = SegmentationDataset.from_config(config.data, split=split, task_type=config.task.type)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
    )


def _evaluate_loader(
    model: Any,
    loader: Any,
    config: SegCraftConfig,
    device: Any,
    torch: Any,
) -> dict[str, float]:
    model.eval()
    metric_classes = _metric_classes(config)
    confusion = torch.zeros(
        (metric_classes, metric_classes),
        dtype=torch.int64,
        device=device,
    )
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = _model_output(model(images))
            predictions = _predictions(logits, config, torch)
            targets = _targets(masks, config)
            confusion += _confusion_matrix(predictions, targets, config, torch)

    return _metrics_from_confusion(confusion.cpu(), config, torch)


def _loss(config: SegCraftConfig, torch: Any) -> Any:
    requested = config.train.loss.lower()
    if config.task.type == "binary":
        bce = torch.nn.BCEWithLogitsLoss()
        if requested in {"auto", "bce_dice", "dice_bce"}:
            return lambda logits, masks: bce(logits, masks.float()) + _binary_dice_loss(
                logits, masks, torch
            )
        if requested == "bce":
            return bce
        raise ValueError("Binary train.loss must be one of: auto, bce_dice, bce")

    kwargs = {}
    if config.task.ignore_index is not None:
        kwargs["ignore_index"] = config.task.ignore_index
    cross_entropy = torch.nn.CrossEntropyLoss(**kwargs)
    if requested in {"auto", "cross_entropy_dice", "dice_ce", "ce_dice"}:
        return lambda logits, masks: cross_entropy(logits, masks.long()) + _multiclass_dice_loss(
            logits, masks, config, torch
        )
    if requested in {"cross_entropy", "ce"}:
        return cross_entropy
    raise ValueError(
        "Multiclass train.loss must be one of: auto, cross_entropy_dice, dice_ce, cross_entropy"
    )


def _binary_dice_loss(logits: Any, masks: Any, torch: Any) -> Any:
    probs = torch.sigmoid(logits)
    targets = masks.float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _multiclass_dice_loss(logits: Any, masks: Any, config: SegCraftConfig, torch: Any) -> Any:
    import torch.nn.functional as functional

    targets = masks.long()
    valid = torch.ones_like(targets, dtype=torch.bool)
    if config.task.ignore_index is not None:
        valid &= targets != config.task.ignore_index

    safe_targets = targets.clamp(min=0, max=config.task.num_classes - 1)
    one_hot = functional.one_hot(safe_targets, num_classes=config.task.num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)

    probs = torch.softmax(logits, dim=1) * valid
    one_hot = one_hot * valid
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _optimizer(model: Any, config: SegCraftConfig, torch: Any) -> Any:
    name = config.train.optimizer.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.train.learning_rate, momentum=0.9)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    raise ValueError("train.optimizer must be one of: adam, adamw, sgd")


def _predictions(logits: Any, config: SegCraftConfig, torch: Any) -> Any:
    if config.task.type == "binary":
        return (torch.sigmoid(logits[:, 0]) > 0.5).long()
    return logits.argmax(dim=1)


def _targets(masks: Any, config: SegCraftConfig) -> Any:
    if config.task.type == "binary":
        return masks[:, 0].long()
    return masks.long()


def _confusion_matrix(predictions: Any, targets: Any, config: SegCraftConfig, torch: Any) -> Any:
    valid = torch.ones_like(targets, dtype=torch.bool)
    if config.task.ignore_index is not None:
        valid &= targets != config.task.ignore_index
    valid &= targets >= 0
    metric_classes = _metric_classes(config)
    valid &= targets < metric_classes

    encoded = targets[valid] * metric_classes + predictions[valid]
    return torch.bincount(
        encoded,
        minlength=metric_classes * metric_classes,
    ).reshape(metric_classes, metric_classes)


def _metrics_from_confusion(confusion: Any, config: SegCraftConfig, torch: Any) -> dict[str, float]:
    confusion = confusion.float()
    true_positive = confusion.diag()
    false_positive = confusion.sum(dim=0) - true_positive
    false_negative = confusion.sum(dim=1) - true_positive
    union = true_positive + false_positive + false_negative
    iou = true_positive / union.clamp_min(1)
    dice = (2 * true_positive) / (2 * true_positive + false_positive + false_negative).clamp_min(1)
    pixel_accuracy = true_positive.sum() / confusion.sum().clamp_min(1)

    if config.task.type == "binary":
        return {
            "iou": float(iou[1]),
            "dice": float(dice[1]),
            "pixel_accuracy": float(pixel_accuracy),
        }
    return {
        "miou": float(iou.mean()),
        "dice_macro": float(dice.mean()),
        "pixel_accuracy": float(pixel_accuracy),
    }


def _model_output(output: Any) -> Any:
    return output["out"] if isinstance(output, dict) else output


def _metric_classes(config: SegCraftConfig) -> int:
    return 2 if config.task.type == "binary" else config.task.num_classes


def _split_ready(config: SegCraftConfig, split: str) -> bool:
    if split == "train":
        paths = [config.data.train_images, config.data.train_masks]
    else:
        paths = [config.data.val_images, config.data.val_masks]
    return all(Path(path).exists() for path in paths)


def _missing_data_summary(config: SegCraftConfig, split: str) -> dict[str, Any]:
    if split == "train":
        paths = [config.data.train_images, config.data.train_masks]
    else:
        paths = [config.data.val_images, config.data.val_masks]
    missing = [path for path in paths if not Path(path).exists()]
    return {
        "status": "data_missing",
        "split": split,
        "missing_paths": missing,
        "message": "Create these paths or set them in configs/local.yaml to run the real loop.",
    }


def _resolve_device(requested: str, torch: Any) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is 'cuda', but CUDA is not available")
    return torch.device(requested)


def _set_seed(seed: int, torch: Any) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Training and evaluation require PyTorch. Install with `pip install -e .[torch]`."
        ) from exc
    return torch
