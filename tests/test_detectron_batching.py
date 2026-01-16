import unittest

from detectron_batching import compute_vram_aware_batch_size, estimate_images_per_gpu


class DetectronBatchingTests(unittest.TestCase):
    def test_batch_size_scales_with_vram(self):
        gpu_memories = [24 * 2**30, 16 * 2**30, 8 * 2**30, 12 * 2**30]
        expected = [
            estimate_images_per_gpu(mem) for mem in gpu_memories
        ]  # [4, 3, 1, 2]

        batch_size = compute_vram_aware_batch_size(gpu_memories, fallback_batch=4)
        self.assertEqual(batch_size, sum(expected))

    def test_fallback_batch_size_used_without_gpus(self):
        self.assertEqual(compute_vram_aware_batch_size([], fallback_batch=3), 3)


if __name__ == "__main__":
    unittest.main()
