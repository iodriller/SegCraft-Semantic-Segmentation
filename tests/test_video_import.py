import json

import segcraft.video as video


def test_video_module_imports_without_loading_cv2():
    assert callable(video.extract_frames)
    assert callable(video.copy_video_file)
    assert callable(video.is_video_file)
    assert callable(video.probe_video)
    assert callable(video.mux_audio_from_source)
    assert callable(video.write_side_by_side_video)
    assert callable(video.write_video_from_images)


def test_is_video_file_uses_known_suffixes():
    assert video.is_video_file("clip.mp4")
    assert video.is_video_file("clip.MOV")
    assert not video.is_video_file("frame.jpg")


def test_download_youtube_reuses_matching_output(tmp_path, monkeypatch):
    output = tmp_path / "youtube.mp4"
    output.write_bytes(b"video")
    output.with_name("youtube.mp4.segcraft-download.json").write_text(
        json.dumps({"url": "https://example.com/video", "format_selector": "mp4"}),
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("download should not run")

    monkeypatch.setattr(video.subprocess, "run", fail_run)

    result = video.download_youtube(
        "https://example.com/video",
        output,
        format_selector="mp4",
    )

    assert result == output


def test_download_youtube_reuses_cache_file(tmp_path, monkeypatch):
    output = tmp_path / "job" / "youtube.mp4"
    cache_dir = tmp_path / "cache"
    cached = video._cached_download_path(cache_dir, "https://example.com/video", "mp4", ".mp4")
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached video")
    cached.with_name(f"{cached.name}.segcraft-download.json").write_text(
        json.dumps({"url": "https://example.com/video", "format_selector": "mp4"}),
        encoding="utf-8",
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("download should not run")

    monkeypatch.setattr(video.subprocess, "run", fail_run)

    result = video.download_youtube(
        "https://example.com/video",
        output,
        format_selector="mp4",
        cache_dir=cache_dir,
    )

    assert result == output
    assert output.read_bytes() == b"cached video"


def test_download_youtube_writes_metadata_and_populates_cache(tmp_path, monkeypatch):
    output = tmp_path / "job" / "youtube.mp4"
    cache_dir = tmp_path / "cache"

    def fake_run(command, *, should_stop=None):
        output_path = command[command.index("-o") + 1]
        assert should_stop is None
        with open(output_path, "wb") as handle:
            handle.write(b"fresh video")

    monkeypatch.setattr(video, "_run_download", fake_run)

    result = video.download_youtube(
        "https://example.com/fresh",
        output,
        format_selector="mp4",
        cache_dir=cache_dir,
    )
    cached = video._cached_download_path(cache_dir, "https://example.com/fresh", "mp4", ".mp4")

    assert result == output
    assert output.read_bytes() == b"fresh video"
    assert cached.read_bytes() == b"fresh video"
    assert json.loads(video._download_metadata_path(output).read_text(encoding="utf-8")) == {
        "url": "https://example.com/fresh",
        "format_selector": "mp4",
    }
    assert json.loads(video._download_metadata_path(cached).read_text(encoding="utf-8")) == {
        "url": "https://example.com/fresh",
        "format_selector": "mp4",
    }


def test_run_download_terminates_when_cancelled(monkeypatch):
    class SlowProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    process = SlowProcess()
    monkeypatch.setattr(video.subprocess, "Popen", lambda _command: process)

    try:
        video._run_download(["yt-dlp"], should_stop=lambda: True)
        assert False, "expected DownloadCancelled"
    except video.DownloadCancelled:
        assert process.terminated is True
