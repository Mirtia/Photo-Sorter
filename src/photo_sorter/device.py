"""Pick the compute device for torch and ONNX Runtime, once per process.

Everything ML-heavy funnels through here so the pipeline uses the GPU when one
is present and silently falls back to CPU otherwise. Set
``PHOTO_SORTER_FORCE_CPU=1`` to pin CPU even when CUDA is available (useful for
debugging or when VRAM is needed elsewhere).
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import sysconfig
from functools import lru_cache

logger = logging.getLogger(__name__)

# onnxruntime-gpu dlopens its CUDA provider by SONAME, but the CUDA 13 / cuDNN 9
# libraries ship inside torch's ``nvidia-*`` wheels and are not on the linker
# path. Preloading them (dependency order) with RTLD_GLOBAL lets the provider
# resolve them without a system-wide CUDA install or LD_LIBRARY_PATH juggling.
_CUDA_PRELOAD_ORDER = (
    "libcudart.so.13",
    "libnvJitLink.so.13",
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libcufft.so.12",
    "libcurand.so.10",
    "libcusparse.so.12",
    "libnvrtc.so.13",
    "libcudnn_graph.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
    "libcudnn.so.9",
)


_CUDA_REQUIRED = (
    "libcudart.so.13",
    "libcublasLt.so.13",
    "libcudnn.so.9",
)


def _nvidia_lib_dirs() -> list[str]:
    seen: list[str] = []
    purelib = sysconfig.get_paths().get("purelib", "")
    roots = {purelib, purelib.replace("/lib/", "/lib64/")}
    for root in roots:
        if not root:
            continue
        for lib_dir in sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib"))):
            if lib_dir not in seen and os.path.isdir(lib_dir):
                seen.append(lib_dir)
    return seen


@lru_cache(maxsize=1)
def _preload_cuda_libs() -> bool:
    """Preload the venv-bundled CUDA 13 / cuDNN 9 libs.

    Returns True when the libraries ONNX Runtime actually needs (cudart,
    cublasLt, cudnn) resolved. Extra cuDNN plugin libs are best-effort.
    """
    dirs = _nvidia_lib_dirs()
    if not dirs:
        return False
    loaded: set[str] = set()
    for name in _CUDA_PRELOAD_ORDER:
        for lib_dir in dirs:
            candidate = os.path.join(lib_dir, name)
            if os.path.exists(candidate):
                try:
                    ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
                    loaded.add(name)
                except OSError as exc:
                    logger.debug("Could not preload %s: %s", name, exc)
                break
    return all(name in loaded for name in _CUDA_REQUIRED)


def force_cpu() -> bool:
    """True when the user pinned CPU via ``PHOTO_SORTER_FORCE_CPU``."""
    flag = os.environ.get("PHOTO_SORTER_FORCE_CPU", "").strip().lower()
    return flag in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True when torch can see a usable CUDA device and CPU is not forced."""
    if force_cpu():
        return False
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def torch_device() -> str:
    """Return ``"cuda"`` or ``"cpu"`` for torch models."""
    return "cuda" if cuda_available() else "cpu"


@lru_cache(maxsize=1)
def use_fp16() -> bool:
    """Use half precision only on CUDA (fp16 on CPU is slower, not faster)."""
    return cuda_available()


@lru_cache(maxsize=1)
def onnx_providers() -> tuple[str, ...]:
    """ONNX Runtime providers, preferring CUDA when the EP is installed.

    Returns a CPU-only tuple when ``onnxruntime-gpu`` is missing or CPU is
    forced, so callers never crash on a machine without the GPU runtime.
    """
    if not cuda_available():
        return ("CPUExecutionProvider",)
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        return ("CPUExecutionProvider",)
    if "CUDAExecutionProvider" in available and _preload_cuda_libs():
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    logger.info("onnxruntime CUDA provider unavailable; running ONNX models on CPU")
    return ("CPUExecutionProvider",)


@lru_cache(maxsize=1)
def insightface_ctx_id() -> int:
    """InsightFace ctx_id: GPU index 0 when CUDA is usable, else -1 (CPU)."""
    return 0 if onnx_providers()[0] == "CUDAExecutionProvider" else -1


def describe() -> str:
    """One-line human summary of the selected device (for logs/CLI)."""
    if not cuda_available():
        return "CPU"
    try:
        import torch

        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "CUDA device"
    provider = "CUDA" if onnx_providers()[0] == "CUDAExecutionProvider" else "CPU (ONNX on CPU)"
    return f"{name} · torch=cuda · onnx={provider}"
