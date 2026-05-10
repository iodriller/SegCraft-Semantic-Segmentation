from segcraft.models import build_model


def task_config():
    return {"type": "multiclass", "num_classes": 3}


def test_build_model_resolves_torchvision_alias():
    spec = build_model({"name": "deeplabv3", "backend": "auto"}, task_config())

    assert spec["backend"] == "torchvision"
    assert spec["factory"] == "deeplabv3_resnet50"
    assert spec["num_classes"] == 3


def test_build_model_resolves_smp_alias():
    spec = build_model({"name": "unet", "backend": "auto"}, task_config())

    assert spec["backend"] == "smp"
    assert spec["factory"] == "Unet"


def test_build_model_resolves_transformers_model_id():
    spec = build_model(
        {"name": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", "backend": "auto"},
        {"type": "multiclass", "num_classes": 19},
    )

    assert spec["backend"] == "transformers"
    assert spec["factory"] == "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    assert spec["num_classes"] == 19


def test_build_model_rejects_wrong_backend():
    try:
        build_model({"name": "unet", "backend": "torchvision"}, task_config())
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported model" in str(exc)
