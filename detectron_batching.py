"""Utilities for sizing Detectron2 batches based on available VRAM."""

from __future__ import annotations

from typing import Iterable, List


def estimate_images_per_gpu(memory_bytes: int) -> int:
    """Estimate how many images a single GPU can hold based on VRAM.

    The heuristic favors safer lower batch sizes on smaller cards while allowing
    larger cards to process more images concurrently.
    """

    if memory_bytes >= 24 * 2**30:
        return 4
    if memory_bytes >= 16 * 2**30:
        return 3
    if memory_bytes >= 12 * 2**30:
        return 2
    return 1


def compute_vram_aware_batch_size(memories: Iterable[int], fallback_batch: int) -> int:
    """Aggregate a global batch size from per-GPU VRAM estimates.

    Args:
        memories: VRAM sizes in bytes for each visible GPU.
        fallback_batch: Value to return when no GPU information is available.

    Returns:
        Total images per batch across all devices. Always at least 1.
    """

    memories = list(memories)
    if not memories:
        return max(1, int(fallback_batch))

    total = sum(estimate_images_per_gpu(mem) for mem in memories)
    return max(1, total)


def collect_gpu_vram() -> List[int]:
    """Return the VRAM of each visible GPU in bytes."""

    try:
        import torch
    except Exception:  # pragma: no cover - torch may be unavailable in tests
        return []

    if not torch.cuda.is_available():
        return []

    return [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
