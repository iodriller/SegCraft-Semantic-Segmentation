from segcraft.config.schema import ConfigValidationError, validate_config


def valid_config(task_type: str = "multiclass", num_classes: int = 3):
    return {
        "task": {"type": task_type, "num_classes": num_classes},
        "model": {"name": "unet"},
        "data": {},
        "train": {"epochs": 1, "optimizer": "adam", "learning_rate": 1e-3},
        "eval": {},
        "predict": {"input_path": "in", "output_path": "out"},
        "runtime": {},
    }


def test_schema_accepts_multiclass():
    validate_config(valid_config("multiclass", 3))


def test_schema_rejects_binary_with_wrong_num_classes():
    try:
        validate_config(valid_config("binary", 2))
        assert False, "expected ConfigValidationError"
    except ConfigValidationError:
        assert True


def test_schema_allows_display_name_count_mismatch():
    cfg = valid_config("multiclass", 4)
    cfg["task"]["class_names"] = ["background", "foreground"]

    validate_config(cfg)
