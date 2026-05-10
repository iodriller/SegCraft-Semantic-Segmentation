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
