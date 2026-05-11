"""Runtime environment diagnostics for SegCraft."""

from __future__ import annotations

from importlib import metadata
import platform
import sys
from typing import Any


INSTALL_HINTS = {
    "torch": 'Install with `python -m pip install "segcraft[torch]"`.',
    "video": 'Install with `python -m pip install "segcraft[video]"`.',
    "app": 'Install with `python -m pip install "segcraft[app]"`.',
    "web": 'Install the web stack with `python -m pip install "segcraft[web]"`.',
    "smp": 'Install with `python -m pip install "segcraft[torch,smp]"`.',
    "transformers": 'Install with `python -m pip install "segcraft[torch,transformers]"`.',
}


def resolve_torch_device(requested: str, torch: Any) -> Any:
    """Resolve a torch device and raise a diagnostic CUDA error when needed."""
    requested = (requested or "auto").strip().lower()
    if requested == "gpu":
        requested = "cuda"
    if requested == "auto":
        requested = "cuda" if _cuda_available(torch) else "cpu"
    if requested.startswith("cuda") and not _cuda_available(torch):
        raise RuntimeError(cuda_unavailable_message(requested, torch))
    return torch.device(requested)


def cuda_unavailable_message(requested: str, torch: Any) -> str:
    report = torch_cuda_report(torch)
    cuda_version = report.get("cuda_version") or "none (CPU-only PyTorch build)"
    device_count = report.get("device_count")
    python_executable = report.get("python_executable")

    if report.get("cuda_version") is None:
        cause = "This PyTorch build does not include CUDA support."
    elif device_count == 0:
        cause = "This PyTorch build has CUDA support, but it cannot see an NVIDIA CUDA device."
    else:
        cause = "PyTorch can see CUDA metadata, but torch.cuda.is_available() returned False."

    return (
        f"runtime.device is '{requested}', but CUDA is not available in this Python environment. "
        f"{cause} Python: {python_executable}; torch: {report.get('torch_version')}; "
        f"torch CUDA build: {cuda_version}; visible CUDA devices: {device_count}. "
        "Launch SegCraft from the same environment where "
        "`python -c \"import torch; print(torch.cuda.is_available())\"` prints True, or set "
        "`runtime.device: auto`/choose `auto` in the UI to fall back to CPU. For NVIDIA GPUs, "
        "install a CUDA-enabled PyTorch wheel from https://pytorch.org/get-started/locally/."
    )


def torch_cuda_report(torch: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
    }
    try:
        report["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on local driver state
        report["cuda_error"] = str(exc)

    try:
        report["device_count"] = int(torch.cuda.device_count())
    except Exception as exc:  # pragma: no cover - depends on local driver state
        report["device_count_error"] = str(exc)
        return report

    names = []
    for index in range(report["device_count"]):
        try:
            names.append(torch.cuda.get_device_name(index))
        except Exception as exc:  # pragma: no cover - depends on local driver state
            names.append(f"unavailable ({exc})")
    report["device_names"] = names
    return report


def collect_runtime_diagnostics() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "packages": {
            "segcraft": _version("segcraft"),
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "opencv-python": _version("opencv-python"),
            "fastapi": _version("fastapi"),
            "uvicorn": _version("uvicorn"),
            "transformers": _version("transformers"),
        },
    }
    try:
        import torch
    except ModuleNotFoundError:
        report["torch"] = {
            "installed": False,
            "message": INSTALL_HINTS["torch"],
        }
    else:
        report["torch"] = {"installed": True, **torch_cuda_report(torch)}
    return report


def format_runtime_diagnostics(report: dict[str, Any]) -> str:
    lines = [
        f"Python: {report['python_version']} ({report['python_executable']})",
        "Packages:",
    ]
    packages = report.get("packages", {})
    for name in sorted(packages):
        lines.append(f"  {name}: {packages[name] or 'not installed'}")

    torch_report = report.get("torch", {})
    if not torch_report.get("installed"):
        lines.append(f"Torch: not installed. {torch_report.get('message', INSTALL_HINTS['torch'])}")
        return "\n".join(lines)

    lines.extend(
        [
            f"Torch CUDA build: {torch_report.get('cuda_version') or 'none (CPU-only build)'}",
            f"CUDA available: {torch_report.get('cuda_available')}",
            f"CUDA device count: {torch_report.get('device_count')}",
        ]
    )
    device_names = torch_report.get("device_names") or []
    if device_names:
        lines.append("CUDA devices:")
        lines.extend(f"  {name}" for name in device_names)
    if not torch_report.get("cuda_available"):
        lines.append("CUDA note: choose runtime.device=auto for CPU fallback, or install a CUDA-enabled PyTorch wheel.")
    return "\n".join(lines)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _cuda_available(torch: Any) -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False
