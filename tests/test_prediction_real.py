import pytest


torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


def test_image_prediction_writes_outputs_with_dummy_model(tmp_path, monkeypatch):
    from segcraft.prediction.predictor import run_prediction

    input_dir = tmp_path / "images"
    output_dir = tmp_path / "predictions"
    input_dir.mkdir()
    image_path = input_dir / "frame_001.png"
    Image.new("RGB", (16, 16), color=(30, 80, 120)).save(image_path)

    class DummyModel(torch.nn.Module):
        def forward(self, images):
            batch_size, _, height, width = images.shape
            logits = torch.zeros((batch_size, 2, height, width), dtype=images.dtype)
            logits[:, 1] = 8.0
            return {"out": logits}

    monkeypatch.setattr(
        "segcraft.prediction.predictor.create_model",
        lambda *_args, **_kwargs: DummyModel(),
    )
    progress_events = []
    summary = run_prediction(
        {
            "task": {
                "type": "multiclass",
                "num_classes": 2,
                "class_names": ["background", "foreground"],
                "background_class_id": 0,
            },
            "model": {"name": "dummy", "backend": "torchvision", "pretrained": False},
            "data": {"image_size": [8, 8]},
            "train": {"epochs": 1, "optimizer": "adam", "learning_rate": 1e-3},
            "eval": {"metrics": []},
            "predict": {
                "input_path": str(input_dir),
                "output_path": str(output_dir),
                "annotate": False,
                "save_video": False,
            },
            "runtime": {"device": "cpu"},
        },
        progress_callback=progress_events.append,
    )

    assert summary["status"] == "completed"
    assert summary["images_processed"] == 1
    assert (output_dir / "masks" / "frame_001.png").exists()
    assert (output_dir / "overlays" / "frame_001.jpg").exists()
    assert (output_dir / "summary.json").exists()
    assert progress_events[-1]["stage"] == "completed"
    assert progress_events[-1]["percent"] == 100.0


def test_image_prediction_honors_cancellation(tmp_path, monkeypatch):
    from segcraft.prediction import PredictionCancelled
    from segcraft.prediction.predictor import run_prediction

    input_dir = tmp_path / "images"
    output_dir = tmp_path / "predictions"
    input_dir.mkdir()
    Image.new("RGB", (16, 16), color=(30, 80, 120)).save(input_dir / "frame_001.png")

    class DummyModel(torch.nn.Module):
        def forward(self, images):
            batch_size, _, height, width = images.shape
            return {"out": torch.zeros((batch_size, 2, height, width), dtype=images.dtype)}

    monkeypatch.setattr(
        "segcraft.prediction.predictor.create_model",
        lambda *_args, **_kwargs: DummyModel(),
    )

    with pytest.raises(PredictionCancelled):
        run_prediction(
            {
                "task": {
                    "type": "multiclass",
                    "num_classes": 2,
                    "class_names": ["background", "foreground"],
                    "background_class_id": 0,
                },
                "model": {"name": "dummy", "backend": "torchvision", "pretrained": False},
                "data": {"image_size": [8, 8]},
                "train": {"epochs": 1, "optimizer": "adam", "learning_rate": 1e-3},
                "eval": {"metrics": []},
                "predict": {
                    "input_path": str(input_dir),
                    "output_path": str(output_dir),
                    "annotate": False,
                    "save_video": False,
                },
                "runtime": {"device": "cpu"},
            },
            should_stop=lambda: True,
        )
