from segcraft.config.loader import _deep_merge


def test_deep_merge_nested_dicts():
    base = {"task": {"type": "multiclass", "num_classes": 3}, "train": {"epochs": 20}}
    override = {"task": {"num_classes": 5}, "train": {"epochs": 2}}
    merged = _deep_merge(base, override)
    assert merged["task"]["type"] == "multiclass"
    assert merged["task"]["num_classes"] == 5
    assert merged["train"]["epochs"] == 2
