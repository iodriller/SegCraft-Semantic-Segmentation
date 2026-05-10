from segcraft.cli.main import resolve_config_path


def test_cli_default_config_works_outside_source_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with resolve_config_path() as config_path:
        assert config_path.name == "base.yaml"
        assert config_path.exists()
