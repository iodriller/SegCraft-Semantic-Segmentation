"""Torch dataset utilities for paired image/mask segmentation data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from segcraft.config import DataConfig
from segcraft.runtime import INSTALL_HINTS

from .discovery import ImageMaskPair, pair_image_masks


try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed
    Dataset = object  # type: ignore[misc,assignment]


class SegmentationDataset(Dataset):
    """A small paired image/mask dataset.

    It keeps assumptions simple: RGB images, single-channel masks, and matching
    filenames by stem. More specialized datasets can still plug in later.
    """

    def __init__(
        self,
        pairs: Sequence[ImageMaskPair],
        *,
        image_size: tuple[int, int] | None = None,
        task_type: str = "multiclass",
        normalize: bool = True,
    ) -> None:
        _require_image_stack()
        self.pairs = list(pairs)
        self.image_size = image_size
        self.task_type = task_type
        self.normalize = normalize

    @classmethod
    def from_config(
        cls,
        data_config: DataConfig | dict[str, Any],
        *,
        split: str,
        task_type: str,
    ) -> "SegmentationDataset":
        cfg = data_config if isinstance(data_config, DataConfig) else DataConfig.from_mapping(data_config)
        if split == "train":
            image_dir, mask_dir = cfg.train_images, cfg.train_masks
        elif split in {"val", "validation"}:
            image_dir, mask_dir = cfg.val_images, cfg.val_masks
        else:
            raise ValueError("split must be 'train' or 'val'")

        pairs = pair_image_masks(image_dir, mask_dir, mask_suffix=cfg.mask_suffix)
        return cls(pairs, image_size=cfg.image_size, task_type=task_type)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        image = _read_image(pair.image_path)
        mask = _read_mask(pair.mask_path)

        if self.image_size:
            height, width = self.image_size
            image = image.resize((width, height), resample=_pil().BILINEAR)
            mask = mask.resize((width, height), resample=_pil().NEAREST)

        image_tensor = _image_to_tensor(image, normalize=self.normalize)
        mask_tensor = _mask_to_tensor(mask, task_type=self.task_type)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
            "stem": pair.stem,
        }


def _require_image_stack() -> None:
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"SegmentationDataset requires the image stack. {INSTALL_HINTS['torch']}"
        ) from exc


def _pil():
    from PIL import Image

    return Image


def _read_image(path: str | Path):
    return _pil().open(path).convert("RGB")


def _read_mask(path: str | Path):
    return _pil().open(path).convert("L")


def _image_to_tensor(image: Any, *, normalize: bool):
    import numpy as np
    import torch

    array = np.asarray(image).astype("float32") / 255.0
    if normalize:
        mean = np.asarray([0.485, 0.456, 0.406], dtype="float32")
        std = np.asarray([0.229, 0.224, 0.225], dtype="float32")
        array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1))


def _mask_to_tensor(mask: Any, *, task_type: str):
    import numpy as np
    import torch

    array = np.asarray(mask)
    if task_type == "binary":
        return torch.from_numpy((array > 0).astype("float32")).unsqueeze(0)
    return torch.from_numpy(array.astype("int64"))
