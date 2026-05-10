from segcraft.config import ConfigValidationError, SegCraftConfig, parse_config


def base_config():
    return {
        "task": {"type": "binary", "num_classes": 1, "class_names": ["background", "foreground"]},
        "model": {"name": "deeplabv3", "backend": "auto", "pretrained": True},
        "data": {"image_size": [128, 96], "batch_size": 2, "num_workers": 1},
        "train": {"epochs": 1, "optimizer": "adam", "learning_rate": 1e-3},
        "eval": {"metrics": []},
        "predict": {"input_path": "in", "output_path": "out", "overlay_alpha": 0.25},
        "runtime": {},
    }


def test_parse_config_returns_typed_sections():
    config = parse_config(base_config())
    assert isinstance(config, SegCraftConfig)
    assert config.task.type == "binary"
    assert config.data.image_size == (128, 96)
    assert config.model.backend == "auto"
    assert config.task.background_class_id == 0
    assert config.train.loss == "auto"
    assert config.train.scheduler == "none"
    assert config.train.amp is False
    assert config.predict.preserve_audio is True
    assert config.predict.display.palette == "vivid"


def test_parse_config_rejects_invalid_overlay_alpha():
    cfg = base_config()
    cfg["predict"]["overlay_alpha"] = 1.5
    try:
        parse_config(cfg)
        assert False, "expected ConfigValidationError"
    except ConfigValidationError:
        assert True


def test_parse_config_rejects_invalid_scheduler():
    cfg = base_config()
    cfg["train"]["scheduler"] = "sometimes"
    try:
        parse_config(cfg)
        assert False, "expected ConfigValidationError"
    except ConfigValidationError:
        assert True


def test_parse_config_accepts_cityscapes_without_background_class():
    cfg = base_config()
    cfg["task"] = {
        "type": "multiclass",
        "num_classes": 2,
        "class_names": ["road", "car"],
        "background_class_id": None,
    }
    cfg["model"] = {"name": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", "backend": "transformers"}
    parsed = parse_config(cfg)
    assert parsed.task.background_class_id is None
    assert parsed.model.backend == "transformers"
