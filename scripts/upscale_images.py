#!/usr/bin/env python3
# Usage: python scripts/upscale_images.py --input data/inference_examples --output data/inference_examples_upscaled
# Ownership: Copyright (c) 2026 Adrian Meyer
# License: AGPL-3.0 only (code); model weights and dependencies may be under AGPL-3.0.
import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

# Reduce CPU oversubscription when running multiple GPU processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def list_images(input_dir: Path, recursive: bool = True):
    exts = {".jpg", ".jpeg"}
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files

def ensure_weights(weights_path: Path, url: str):
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        return
    print(f"[weights] Missing: {weights_path}")
    print(f"[weights] Attempting download from: {url}")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(weights_path))
        print(f"[weights] Downloaded to: {weights_path}")
    except Exception as e:
        raise RuntimeError(
            "Failed to download weights (common on locked-down HPC networks). "
            "Manually download the RealESRGAN_x2plus.pth file and pass --weights /path/to/RealESRGAN_x2plus.pth"
        ) from e

def worker(rank: int, gpu_id: int, files, args_dict):
    # Imports inside worker: avoids CUDA context issues on some systems
    import cv2
    import torch
    import numpy as np
    from tqdm import tqdm
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    cv2.setNumThreads(0)
    torch.set_num_threads(1)

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    scale = int(args_dict["scale"])
    weights = args_dict["weights"]
    tile = int(args_dict["tile"])
    tile_pad = int(args_dict["tile_pad"])
    pre_pad = int(args_dict["pre_pad"])
    fp16 = bool(args_dict["fp16"])
    jpg_quality = int(args_dict["jpg_quality"])
    overwrite = bool(args_dict["overwrite"])
    input_dir = Path(args_dict["input_dir"])
    output_dir = Path(args_dict["output_dir"])

    # Model for Real-ESRGAN x2
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32,
        scale=scale
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=weights,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=fp16,
        device=device,
    )

    # Optional warmup (helps stabilize timings)
    if args_dict["warmup"]:
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        with torch.inference_mode():
            _ = upsampler.enhance(dummy, outscale=scale)

    pbar = tqdm(files, desc=f"GPU {gpu_id}", position=rank, dynamic_ncols=True)
    for in_path in pbar:
        rel = in_path.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if (not overwrite) and out_path.exists():
            continue

        img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[GPU {gpu_id}] Skipping unreadable image: {in_path}", file=sys.stderr)
            continue

        try:
            with torch.inference_mode():
                out, _ = upsampler.enhance(img, outscale=scale)
        except RuntimeError as e:
            # Helpful hint for OOM
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(
                    f"[GPU {gpu_id}] CUDA error on {in_path}. "
                    f"Try setting --tile 512 or --tile 1024 to reduce VRAM use.",
                    file=sys.stderr
                )
            raise

        # Write JPG with quality
        ok = cv2.imwrite(
            str(out_path),
            out,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
        )
        if not ok:
            print(f"[GPU {gpu_id}] Failed to write: {out_path}", file=sys.stderr)

def parse_args():
    ap = argparse.ArgumentParser(description="Multi-GPU 2x AI upscaler (Real-ESRGAN x2) for JPG directories.")
    ap.add_argument("--input_dir", required=True, help="Directory containing .jpg/.jpeg images (recursively).")
    ap.add_argument("--output_dir", required=True, help="Output directory (same relative structure as input).")

    ap.add_argument("--scale", type=int, default=2, choices=[2], help="Upscale factor (fixed to 2x here).")

    ap.add_argument("--weights", default="", help="Path to RealESRGAN_x2plus.pth (optional; auto-download if empty).")
    ap.add_argument("--gpus", default="", help="Comma-separated GPU ids to use (e.g. 0,1,2,3). Default: all visible.")

    ap.add_argument("--tile", type=int, default=0, help="Tile size. 0 = no tiling (fastest, more VRAM). Try 512/1024 if OOM.")
    ap.add_argument("--tile_pad", type=int, default=10, help="Tile padding (avoid seams).")
    ap.add_argument("--pre_pad", type=int, default=0, help="Pre padding.")
    ap.add_argument("--fp16", action="store_true", help="Use FP16 inference (recommended on V100).")

    ap.add_argument("--jpg_quality", type=int, default=95, help="0-100 JPEG quality for outputs.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--no_recursive", action="store_true", help="Do not search input_dir recursively.")
    ap.add_argument("--warmup", action="store_true", help="Run a short warmup pass on each GPU.")

    return ap.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    # Choose GPUs. If CUDA_VISIBLE_DEVICES is set, torch will enumerate from 0..N-1.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script is meant to run on GPUs (e.g., V100).")

    if args.gpus.strip():
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        gpus = list(range(torch.cuda.device_count()))
    if not gpus:
        raise RuntimeError("No GPUs selected/found.")

    # Default weights: auto-download to cache
    weights_path = Path(args.weights).expanduser() if args.weights else (Path.home() / ".cache" / "realesrgan" / "RealESRGAN_x2plus.pth")
    # Known upstream release URL (if your HPC blocks outbound net, download manually and pass --weights)
    default_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth"
    ensure_weights(weights_path, default_url)

    files = list_images(input_dir, recursive=(not args.no_recursive))
    if not files:
        print("No .jpg/.jpeg files found.")
        return

    # Split files across GPUs (round-robin keeps work balanced)
    buckets = [[] for _ in gpus]
    for i, f in enumerate(files):
        buckets[i % len(gpus)].append(f)

    print(f"Found {len(files)} images. Using GPUs: {gpus}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"FP16:   {args.fp16} | tile={args.tile}")

    ctx = mp.get_context("spawn")
    args_dict = vars(args)
    args_dict["input_dir"] = str(input_dir)
    args_dict["output_dir"] = str(output_dir)
    args_dict["weights"] = str(weights_path)

    procs = []
    for rank, gpu_id in enumerate(gpus):
        p = ctx.Process(target=worker, args=(rank, gpu_id, buckets[rank], args_dict))
        p.start()
        procs.append(p)

    exit_code = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            exit_code = p.exitcode

    sys.exit(exit_code)

if __name__ == "__main__":
    main()
