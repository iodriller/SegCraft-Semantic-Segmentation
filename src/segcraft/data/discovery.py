"""File discovery helpers for segmentation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageMaskPair:
    image_path: Path
    mask_path: Path
    stem: str

    def to_dict(self) -> dict[str, str]:
        return {
            "image_path": str(self.image_path),
            "mask_path": str(self.mask_path),
            "stem": self.stem,
        }


def list_image_files(root: str | Path, extensions: Iterable[str] = IMAGE_EXTENSIONS) -> list[Path]:
    root = Path(root)
    suffixes = {item.lower() for item in extensions}

    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    if root.is_file():
        if root.suffix.lower() not in suffixes:
            raise ValueError(f"Unsupported image extension: {root}")
        return [root]

    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def pair_image_masks(
    image_dir: str | Path,
    mask_dir: str | Path,
    *,
    image_extensions: Iterable[str] = IMAGE_EXTENSIONS,
    mask_extensions: Iterable[str] = MASK_EXTENSIONS,
    mask_suffix: str = "",
) -> list[ImageMaskPair]:
    images = _index_by_stem(list_image_files(image_dir, image_extensions))
    masks = _index_by_stem(list_image_files(mask_dir, mask_extensions), suffix=mask_suffix)

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        details = []
        if missing_masks:
            details.append(f"missing masks for: {', '.join(missing_masks[:5])}")
        if missing_images:
            details.append(f"missing images for: {', '.join(missing_images[:5])}")
        raise ValueError("Image/mask pairing failed; " + "; ".join(details))

    return [ImageMaskPair(images[stem], masks[stem], stem) for stem in sorted(images)]


def _index_by_stem(paths: list[Path], suffix: str = "") -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in paths:
        stem = path.stem
        if suffix and stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        if stem in indexed:
            raise ValueError(f"Duplicate file stem found while pairing dataset: {stem}")
        indexed[stem] = path
    return indexed
