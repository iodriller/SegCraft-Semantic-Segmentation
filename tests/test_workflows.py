from segcraft.engine.workflows import evaluate, predict, train


def base_config(task_type="multiclass", num_classes=3):
    return {
        "task": {"type": task_type, "num_classes": num_classes},
        "model": {"name": "unet", "encoder": "resnet34", "pretrained": True},
        "data": {},
        "train": {"epochs": 3, "optimizer": "adam", "learning_rate": 1e-3, "loss": "auto"},
        "eval": {"metrics": []},
        "predict": {"input_path": "in", "output_path": "out", "overlay_alpha": 0.5},
        "runtime": {},
    }


def test_train_auto_loss_multiclass():
    summary = train(base_config())
    assert summary["train"]["loss"] == "cross_entropy_dice"


def test_train_auto_loss_binary():
    summary = train(base_config(task_type="binary", num_classes=1))
    assert summary["train"]["loss"] == "bce_dice"


def test_predict_and_evaluate_modes():
    cfg = base_config()
    assert evaluate(cfg)["mode"] == "evaluate"
    assert predict(cfg)["mode"] == "predict"
