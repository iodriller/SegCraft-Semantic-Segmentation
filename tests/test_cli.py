from segcraft.cli.main import resolve_config_path
from segcraft.config.loader import list_available_presets, load_and_validate_config


def test_cli_default_config_works_outside_source_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with resolve_config_path() as config_path:
        assert config_path.name == "base.yaml"
        assert config_path.exists()


def test_packaged_preset_name_loads_outside_source_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with resolve_config_path() as config_path:
        config = load_and_validate_config(config_path, preset_path="pascal_video")

    assert config["data"]["image_size"] == [360, 640]
    assert "pascal_video" in list_available_presets()
