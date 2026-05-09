import segcraft.video as video


def test_video_module_imports_without_loading_cv2():
    assert callable(video.extract_frames)
    assert callable(video.write_video_from_images)
