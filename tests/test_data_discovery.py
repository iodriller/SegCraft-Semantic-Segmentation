from segcraft.data import pair_image_masks


def test_pair_image_masks_matches_by_stem(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "frame_001.jpg").write_text("image")
    (masks / "frame_001.png").write_text("mask")

    pairs = pair_image_masks(images, masks)

    assert len(pairs) == 1
    assert pairs[0].stem == "frame_001"


def test_pair_image_masks_supports_mask_suffix(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "frame_001.jpg").write_text("image")
    (masks / "frame_001_mask.png").write_text("mask")

    pairs = pair_image_masks(images, masks, mask_suffix="_mask")

    assert len(pairs) == 1
    assert pairs[0].stem == "frame_001"


def test_pair_image_masks_reports_missing_mask(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    (images / "frame_001.jpg").write_text("image")

    try:
        pair_image_masks(images, masks)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "missing masks" in str(exc)
