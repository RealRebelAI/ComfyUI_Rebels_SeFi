# device_compat.py — cross-platform device helpers.
#
# Added 2026-07-18 (Apple Silicon support). The original code only ever checked
# torch.cuda.is_available() before picking a device, which silently falls back to
# CPU on Mac instead of using the Metal (MPS) GPU — confirmed ~24 min/step for the
# 5B transformer on Apple Silicon CPU vs the seconds/step this pack is designed for
# on GPU. This file is the single place that knows about non-CUDA GPUs so the rest
# of the pack doesn't need scattered platform checks.
import torch


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_cache(device: str | None = None) -> None:
    """Best-effort cache clear for whichever GPU backend is active. No-op on CPU."""
    dev = device or best_device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
