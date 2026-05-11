import types

import pytest

from segcraft.runtime import (
    collect_runtime_diagnostics,
    cuda_unavailable_message,
    format_runtime_diagnostics,
    resolve_torch_device,
)


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0


class FakeTorch:
    __version__ = "2.0.0+cpu"
    version = types.SimpleNamespace(cuda=None)
    cuda = FakeCuda()

    @staticmethod
    def device(name):
        return name


def test_resolve_torch_device_auto_falls_back_to_cpu():
    assert resolve_torch_device("auto", FakeTorch) == "cpu"


def test_resolve_torch_device_cuda_error_has_install_context():
    with pytest.raises(RuntimeError) as exc_info:
        resolve_torch_device("cuda", FakeTorch)

    message = str(exc_info.value)
    assert "CPU-only PyTorch build" in message
    assert "python -c" in message
    assert "pytorch.org/get-started/locally" in message


def test_cuda_unavailable_message_includes_requested_device():
    message = cuda_unavailable_message("cuda:0", FakeTorch)
    assert "runtime.device is 'cuda:0'" in message
    assert "visible CUDA devices: 0" in message


def test_collect_runtime_diagnostics_checks_numpy_and_opencv_imports():
    report = collect_runtime_diagnostics()

    assert "numpy" in report["packages"]
    assert "numpy" in report["imports"]
    assert "cv2" in report["imports"]


def test_format_runtime_diagnostics_reports_import_failures():
    text = format_runtime_diagnostics(
        {
            "python_executable": "python",
            "python_version": "3.12.0",
            "packages": {"numpy": "1.26.4", "opencv-python": "4.11.0"},
            "imports": {
                "cv2": {
                    "ok": False,
                    "error": "ImportError: numpy.core.multiarray failed to import",
                },
                "numpy": {"ok": True, "version": "1.26.4"},
            },
            "torch": {"installed": False, "message": "torch missing"},
        }
    )

    assert "cv2: failed (ImportError: numpy.core.multiarray failed to import)" in text
    assert "numpy: ok (1.26.4)" in text
    assert "--force-reinstall" in text
