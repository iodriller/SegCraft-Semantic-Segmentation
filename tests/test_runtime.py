import types

import pytest

from segcraft.runtime import cuda_unavailable_message, resolve_torch_device


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
