#!/usr/bin/env python3
# Usage: python scripts/compare_models.py --config configs/train_multirun.yaml
# Ownership: Copyright (c) 2026 Adrian Meyer
# License: AGPL-3.0 only (code); model weights and dependencies may be under AGPL-3.0.
"""Train and compare YOLOv11, Mask R-CNN, Mask2Former, and MaskDINO models.

This script reuses datasets prepared by ``scripts/train_model.py`` (YOLO format with ``data.yaml``)
so it fits into the existing pipeline. It trains three models, evaluates them on the
same test split, and writes per-model results plus comparison summaries under a new
experiment directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import inspect
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from detectron2.data import transforms as T
from detectron2.config import CfgNode
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping

from scripts.detectron_batching import collect_gpu_vram, compute_vram_aware_batch_size
import yaml
from PIL import Image
from torch.hub import download_url_to_file

# Shapely may emit RuntimeWarnings when encountering invalid geometries; silence
# them to avoid log spam during long-running training jobs.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in intersection",
    category=RuntimeWarning,
    module="shapely",
)

from scripts.train_model import (  # noqa: E402  - reuse existing pipeline utilities
    create_experiment_dir,
    collect_entries,
    extract_eval_metrics,
    load_config,
    load_prepared_splits,
    normalise_device,
    prepare_dataset,
    save_config_copy,
    setup_logger,
)


MODEL_FOLDERS = {
    "yolo": "YOLOv11-segXL",
    "mask_rcnn": "MaskRCNN-ResNet50",
    "mask2former": "Mask2Former-Swin",
    "maskdino": "MaskDINO-R50",
}

REGISTERED_COCO: set = set()

EVAL_NAME_MAP = {
    "coimbra": "COIMBRA",
    "amsterdam": "AMSTERDAM",
    "bayreuth": "BAYREUTH",
    "basel": "BASEL",
    "cai_pyramids": "FLATCAI",
}


# ---------------------------------------------------------------------------
# Dependency utilities
# ---------------------------------------------------------------------------

def add_local_repo_to_path(path_hint: Optional[str], package_name: str, logger: logging.Logger) -> None:
    """Add a local checkout containing ``package_name`` to ``sys.path``.

    ``path_hint`` can point at a config file inside the repo (the default for
    Mask2Former/MaskDINO configs). We walk up the directory tree until we find a
    folder matching ``package_name`` and insert that repo root into ``sys.path``
    so imports succeed even if the package isn't installed globally.
    """

    def _maybe_add(repo_root: Path) -> bool:
        if (repo_root / package_name).exists():
            repo_path = str(repo_root.resolve())
            if repo_path not in sys.path:
                sys.path.append(repo_path)
                logger.info("Added %s to sys.path for %s", repo_path, package_name)
            return True
        return False

    if not path_hint:
        return

    try:
        candidate = Path(path_hint).expanduser()
    except (TypeError, ValueError):
        return

    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent / candidate).resolve()

    for base in (candidate,) + tuple(candidate.parents):
        repo_root = base.parent if base.is_file() else base
        if _maybe_add(repo_root):
            return

    installations_root = Path(__file__).resolve().parent.parent / "installations"
    if installations_root.is_dir():
        for entry in installations_root.iterdir():
            try:
                if entry.is_dir() and entry.name.lower().startswith(package_name.lower()):
                    if _maybe_add(entry):
                        return
            except PermissionError:
                continue


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def copy_prepared_dataset(source_dataset: Path, dest_dataset: Path, logger: logging.Logger) -> Path:
    """Copy the prepared YOLO dataset into the new experiment directory.

    The manifest is rewritten so paths point at the new location, ensuring downstream
    consumers (Detectron2/MMDetection) have consistent references.
    """

    if not (source_dataset / "data.yaml").is_file():
        raise FileNotFoundError(
            f"Expected data.yaml under prepared dataset at {source_dataset}, but none found."
        )

    logger.info("Copying prepared dataset from %s -> %s", source_dataset, dest_dataset)
    shutil.copytree(source_dataset, dest_dataset, dirs_exist_ok=True)

    manifest_path = dest_dataset / "manifest.csv"
    if manifest_path.is_file():
        logger.info("Rewriting manifest paths for new experiment location")
        df = pd.read_csv(manifest_path)
        rebuilt_rows = []
        for _, row in df.iterrows():
            split = str(row["split"])
            unique_name = str(row["unique_name"])
            image_suffix = Path(str(row.get("prepared_image", ""))).suffix or ".jpg"
            rebuilt_rows.append(
                {
                    **row,
                    "prepared_image": str(dest_dataset / "images" / split / f"{unique_name}{image_suffix}"),
                    "prepared_label": str(dest_dataset / "labels" / split / f"{unique_name}.txt"),
                }
            )
        pd.DataFrame(rebuilt_rows).to_csv(manifest_path, index=False)

    return dest_dataset / "data.yaml"


def build_dataset_test_mapping(splits: Dict[str, List], dataset_dir: Path) -> Dict[str, List[Path]]:
    """Return mapping of dataset name to list of test image paths."""

    mapping: Dict[str, List[Path]] = {}
    for entry in splits.get("test", []):
        dataset_name = getattr(entry, "dataset", "pooled")
        mapping.setdefault(dataset_name, []).append(
            dataset_dir
            / "images"
            / "test"
            / f"{entry.unique_name}{getattr(entry, 'image_suffix', '.jpg')}"
        )
    return mapping


def select_visualization_examples(
    test_mapping: Dict[str, List[Path]], examples_per_dataset: int = 5
) -> Dict[str, List[Path]]:
    """Choose a deterministic, limited set of images per dataset for visualization."""

    selections: Dict[str, List[Path]] = {}
    for dataset_name, images in sorted(test_mapping.items()):
        limited = sorted(images)[:examples_per_dataset]
        if limited:
            selections[dataset_name] = limited
    return selections


def apply_eval_name_map(name: str) -> str:
    return EVAL_NAME_MAP.get(name, name)


def remap_evaluation_keys(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {apply_eval_name_map(key): value for key, value in metrics.items()}


def gather_inference_images(splits: Dict[str, List], target_count: int = 1000) -> List[Path]:
    ordered_splits = ["test", "val", "train"]
    images: List[Path] = []
    for split in ordered_splits:
        for entry in splits.get(split, []):
            images.append(entry.image_path)
            if len(images) >= target_count:
                return images
    return images


def render_detectron_visualizations(
    predictor,
    images_by_dataset: Dict[str, List[Path]],
    output_root: Path,
    class_names: List[str],
    score_threshold: float = 0.0,
) -> None:
    """Run Detectron2 inference and save per-dataset visualization PNGs."""

    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import ColorMode, Visualizer

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_name = f"collembola_viz_{abs(hash(str(output_root))) % 10_000}"  # stable-ish unique key
    metadata = MetadataCatalog.get(metadata_name)
    metadata.set(thing_classes=class_names)

    for dataset_name, images in sorted(images_by_dataset.items()):
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            outputs = predictor(image)
            if isinstance(outputs, dict):
                instances = outputs.get("instances")
            else:
                instances = getattr(outputs, "instances", outputs)
            if instances is None:
                continue
            instances = instances.to("cpu")
            if score_threshold and hasattr(instances, "scores"):
                keep = instances.scores >= score_threshold
                instances = instances[keep]
            visualizer = Visualizer(
                image[:, :, ::-1],
                metadata=metadata,
                scale=1.0,
                instance_mode=ColorMode.IMAGE,
            )
            rendered = visualizer.draw_instance_predictions(instances)
            out_path = dataset_dir / f"{image_path.stem}_viz.png"
            cv2.imwrite(str(out_path), rendered.get_image()[:, :, ::-1])


def render_yolo_visualizations(
    model, images_by_dataset: Dict[str, List[Path]], output_root: Path
) -> None:
    """Save YOLO per-dataset visualizations with confidence overlays."""

    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name, images in sorted(images_by_dataset.items()):
        if not images:
            continue
        model.predict(
            source=[str(p) for p in images],
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(output_root),
            name=dataset_name,
            exist_ok=True,
            verbose=False,
        )


def model_active(cfg: Dict, model_key: str) -> bool:
    """Return True if the model is enabled in the configuration."""

    model_cfg = cfg.get("models", {}).get(model_key, {})
    return bool(model_cfg.get("active", True))


def benchmark_inference(run_fn, images: List[Path], target_count: int = 1000) -> float:
    if not images:
        return math.nan

    start = time.perf_counter()
    processed = 0
    idx = 0
    while processed < target_count:
        image_path = images[idx % len(images)]
        run_fn(image_path)
        processed += 1
        idx += 1
    return time.perf_counter() - start


def summarize_image_dimensions(splits: Dict[str, List], logger: logging.Logger) -> Dict[str, float]:
    """Collect basic statistics about image dimensions to drive robust resizing configs."""

    stats: Dict[str, float] = {
        "train_images": len(splits.get("train", [])),
        "val_images": len(splits.get("val", [])),
        "test_images": len(splits.get("test", [])),
    }

    widths: List[int] = []
    heights: List[int] = []
    short_edges: List[int] = []
    long_edges: List[int] = []
    aspect_ratios: List[float] = []

    for split_entries in splits.values():
        for entry in split_entries:
            try:
                with Image.open(entry.image_path) as img:
                    width, height = img.size
            except FileNotFoundError:
                logger.warning("Image missing while summarizing dimensions: %s", entry.image_path)
                continue

            widths.append(width)
            heights.append(height)
            short_edges.append(min(width, height))
            long_edges.append(max(width, height))
            aspect_ratios.append(width / max(height, 1))

    stats["total_images"] = len(widths)
    if widths:
        stats.update(
            {
                "min_width": int(np.min(widths)),
                "max_width": int(np.max(widths)),
                "min_height": int(np.min(heights)),
                "max_height": int(np.max(heights)),
                "min_short_edge": int(np.min(short_edges)),
                "max_short_edge": int(np.max(short_edges)),
                "median_short_edge": float(np.median(short_edges)),
                "max_long_edge": int(np.max(long_edges)),
                "median_long_edge": float(np.median(long_edges)),
                "median_aspect": float(np.median(aspect_ratios)),
            }
        )

    logger.info(
        "Image stats — count: %s (train %s / val %s / test %s), min short edge: %s, max long edge: %s",
        stats.get("total_images", 0),
        stats.get("train_images", 0),
        stats.get("val_images", 0),
        stats.get("test_images", 0),
        stats.get("min_short_edge"),
        stats.get("max_long_edge"),
    )
    return stats


# ---------------------------------------------------------------------------
# Pretrained weight + config helpers
# ---------------------------------------------------------------------------

def ensure_weight_file(weight_spec: Optional[object], weights_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Resolve a weight spec (path or URL) into a local file, downloading if needed."""

    if weight_spec is None:
        return None

    weights_dir.mkdir(parents=True, exist_ok=True)

    # String input can be URL or filesystem path
    if isinstance(weight_spec, str):
        if weight_spec.startswith("http://") or weight_spec.startswith("https://"):
            filename = Path(weight_spec).name
            dest = weights_dir / filename
            if not dest.exists():
                logger.info("Downloading pretrained weights from %s -> %s", weight_spec, dest)
                download_url_to_file(weight_spec, dest)
            return dest
        path = Path(weight_spec)
        if not path.is_file():
            logger.warning("Weight path %s does not exist; using default weights instead", path)
            return None
        return path

    if isinstance(weight_spec, Path):
        if not weight_spec.is_file():
            logger.warning("Weight path %s does not exist; using default weights instead", weight_spec)
            return None
        return weight_spec

    if isinstance(weight_spec, dict):
        url = weight_spec.get("url")
        path_value = weight_spec.get("path")
        if url:
            filename = Path(url).name
            dest = Path(path_value) if path_value else weights_dir / filename
            if not dest.exists():
                logger.info("Downloading pretrained weights from %s -> %s", url, dest)
                download_url_to_file(url, dest)
            return dest
        if path_value:
            path_candidate = Path(path_value)
            if not path_candidate.is_file():
                logger.warning(
                    "Configured weight path %s does not exist; falling back to defaults", path_candidate
                )
                return None
            return path_candidate

    logger.warning("Unable to resolve weight spec %s; returning None", weight_spec)
    return None


def get_trained_weights_path(output_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Return the best-available trained weight file from a Detectron2 output dir."""

    final_path = output_dir / "model_final.pth"
    if final_path.is_file():
        logger.debug("Using model_final.pth from %s", output_dir)
        return final_path

    last_checkpoint = output_dir / "last_checkpoint"
    if last_checkpoint.is_file():
        try:
            checkpoint_ref = last_checkpoint.read_text(encoding="utf-8").strip()
            checkpoint_path = (output_dir / checkpoint_ref).resolve()
            if checkpoint_path.is_file():
                logger.debug("Using last_checkpoint reference %s", checkpoint_path)
                return checkpoint_path
            logger.warning(
                "Last checkpoint reference %s in %s is missing; searching for other weights",
                checkpoint_ref,
                output_dir,
            )
        except OSError:
            logger.warning("Failed to read last_checkpoint in %s; searching for other weights", output_dir)

    model_checkpoints = sorted(output_dir.glob("model_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    for checkpoint in model_checkpoints:
        if checkpoint.is_file():
            logger.debug("Using latest model_* checkpoint %s", checkpoint)
            return checkpoint

    logger.warning("No trained weights found under %s; inference will use default weights", output_dir)
    return None


def build_predictor_with_trained_weights(cfg, logger: logging.Logger, weights_path: Optional[Path] = None):
    """Safely build a ``DefaultPredictor`` using trained weights without mutating ``cfg``."""

    from detectron2.engine import DefaultPredictor

    working_cfg = cfg.clone()
    working_cfg.defrost()
    resolved_weights = weights_path or get_trained_weights_path(Path(working_cfg.OUTPUT_DIR), logger)
    if resolved_weights:
        working_cfg.MODEL.WEIGHTS = str(resolved_weights)
        logger.debug("Using weights %s for predictor", resolved_weights)
    else:
        logger.warning(
            "No trained weights found for %s; predictor will use existing configuration weights",
            working_cfg.OUTPUT_DIR,
        )
    working_cfg.freeze()
    return DefaultPredictor(working_cfg)


def ensure_config_file(
    config_spec: Optional[str],
    config_dir: Path,
    logger: logging.Logger,
    default_url: Optional[str] = None,
    fallback_contents: Optional[str] = None,
) -> Optional[Path]:
    """Resolve a config path/URL into a local file, downloading or writing it if needed."""

    if not config_spec:
        return None

    candidate = Path(config_spec)
    if candidate.is_file():
        return candidate

    config_dir.mkdir(parents=True, exist_ok=True)

    # If it's a URL, download directly.
    if config_spec.startswith("http://") or config_spec.startswith("https://"):
        dest = config_dir / Path(config_spec).name
        if not dest.exists():
            logger.info("Downloading config from %s -> %s", config_spec, dest)
            download_url_to_file(config_spec, dest)
        return dest

    # Attempt to download from a provided default URL when the path is missing.
    if default_url:
        dest = config_dir / Path(config_spec).name
        if not dest.exists():
            logger.info("Fetching config from %s -> %s", default_url, dest)
            download_url_to_file(default_url, dest)
        return dest

    # Fallback: write inline contents so the pipeline remains self contained.
    if fallback_contents:
        dest = config_dir / Path(config_spec).name
        if not dest.exists():
            logger.info("Writing fallback config to %s", dest)
            dest.write_text(fallback_contents, encoding="utf-8")
        return dest

    logger.warning("Config %s not found; using Detectron2 defaults if available", config_spec)
    return None


def _sanitize_gradient_clipping(cfg, logger: Optional[logging.Logger] = None) -> None:
    """Disable unsupported gradient clipping settings from upstream configs."""

    clip_cfg = getattr(cfg.SOLVER, "CLIP_GRADIENTS", None)
    if clip_cfg and getattr(clip_cfg, "CLIP_TYPE", None) == "full_model":
        if logger:
            logger.warning(
                "Gradient clipping type 'full_model' is unsupported in this Detectron2 version; "
                "disabling gradient clipping for compatibility."
            )
        clip_cfg.ENABLED = False
        clip_cfg.CLIP_TYPE = "value"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _load_yolo_segmentation(
    label_path: Path, width: int, height: int, logger: Optional[logging.Logger] = None
) -> List[Dict]:
    annotations: List[Dict] = []
    if not label_path.is_file():
        return annotations

    log = logger or logging.getLogger("train")

    with label_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    def _clip(val: float, upper: float) -> float:
        return min(max(val, 0.0), max(0.0, upper))

    ann_id = 1
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(parts[0]) + 1  # COCO-style category ids start at 1
        coords = [float(val) for val in parts[1:]]

        bbox: Optional[List[float]] = None
        segmentation: Optional[List[List[float]]] = None

        if len(coords) >= 6:
            # Segmentation polygon (x1 y1 x2 y2 ... in relative coords)
            points = list(zip(coords[0::2], coords[1::2]))
            abs_pts = np.asarray(
                [(_clip(x * width, width), _clip(y * height, height)) for x, y in points],
                dtype=float,
            )
            if not np.isfinite(abs_pts).all():
                log.warning("Skipping annotation with non-finite polygon values in %s", label_path)
                continue
            xs, ys = abs_pts[:, 0], abs_pts[:, 1]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            width_box, height_box = max_x - min_x, max_y - min_y
            if width_box <= 0 or height_box <= 0:
                log.warning("Dropping zero-area polygon annotation in %s", label_path)
                continue
            segmentation = [list(abs_pts.flatten())]
            bbox = [min_x, min_y, width_box, height_box]
        else:
            # Bounding box label: cx cy w h in relative coordinates
            cx, cy, w, h = coords[:4]
            abs_w = w * width
            abs_h = h * height
            min_x = (cx * width) - (abs_w / 2)
            min_y = (cy * height) - (abs_h / 2)
            max_x = min_x + abs_w
            max_y = min_y + abs_h
            if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
                log.warning("Skipping annotation with non-finite bbox values in %s", label_path)
                continue

            min_x = _clip(min_x, width)
            min_y = _clip(min_y, height)
            max_x = _clip(max_x, width)
            max_y = _clip(max_y, height)
            abs_w = max_x - min_x
            abs_h = max_y - min_y
            if abs_w <= 0 or abs_h <= 0:
                log.warning("Dropping zero-area bbox annotation in %s", label_path)
                continue

            bbox = [min_x, min_y, abs_w, abs_h]
            segmentation = [
                [
                    min_x,
                    min_y,
                    max_x,
                    min_y,
                    max_x,
                    max_y,
                    min_x,
                    max_y,
                ]
            ]

        annotations.append(
            {
                "category_id": cls,
                "bbox": bbox,
                "segmentation": segmentation,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
                "id": ann_id,
            }
        )
        ann_id += 1
    return annotations


def _extract_backbone_hint(path_hint: Optional[str]) -> Optional[str]:
    """Return a coarse backbone identifier based on a config/weights path."""

    if not path_hint:
        return None

    lowered = str(path_hint).lower()
    for token in ("swin", "r50", "r101", "convnext", "beit", "vit"):
        if token in lowered:
            return token
    return None


def _weights_incompatible_with_config(config_hint: Optional[str], weight_hint: Optional[str]) -> bool:
    """Detect obvious config/weights mismatches (e.g. R50 config with Swin weights)."""

    config_backbone = _extract_backbone_hint(config_hint)
    weight_backbone = _extract_backbone_hint(weight_hint)

    return bool(config_backbone and weight_backbone and config_backbone != weight_backbone)


def _validate_coco_annotations(
    images_meta: List[Dict],
    annotations: List[Dict],
    categories: List[Dict],
    logger: logging.Logger,
    context: str,
) -> None:
    if not categories:
        logger.warning("No categories found while preparing COCO data for %s", context)
    image_has_ann = {img["id"]: False for img in images_meta}
    for ann in annotations:
        image_has_ann[ann.get("image_id")] = True
    missing = [img_id for img_id, has_ann in image_has_ann.items() if not has_ann]
    if missing:
        logger.warning(
            "Images without annotations in %s COCO export: %s", context, ", ".join(map(str, missing))
        )


def convert_yolo_to_coco(
    dataset_dir: Path,
    splits: Dict[str, List],
    class_names: List[str],
    coco_root: Path,
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Create COCO-style annotations for Detectron2/MMDetection."""

    coco_root.mkdir(parents=True, exist_ok=True)
    category_records = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    output: Dict[str, Path] = {}
    for split_name, entries in splits.items():
        images_meta = []
        annotations = []
        ann_counter = 1
        for idx, entry in enumerate(entries):
            image_path: Path = entry.image_path
            label_path: Path = entry.label_path
            with Image.open(image_path) as img:
                width, height = img.size
            images_meta.append(
                {
                    "id": idx + 1,
                    "file_name": str(image_path.resolve()),
                    "width": width,
                    "height": height,
                }
            )
            anns = _load_yolo_segmentation(label_path, width, height, logger)
            for ann in anns:
                ann["image_id"] = idx + 1
                ann["id"] = ann_counter
                ann_counter += 1
                annotations.append(ann)

        coco_content = {
            "images": images_meta,
            "annotations": annotations,
            "categories": category_records,
        }
        _validate_coco_annotations(images_meta, annotations, category_records, logger, split_name)
        json_path = coco_root / f"{split_name}.json"
        json_path.write_text(json.dumps(coco_content, indent=2), encoding="utf-8")
        output[split_name] = json_path
        logger.info("Wrote COCO annotations for %s => %s", split_name, json_path)

    return output


def convert_dataset_subset_to_coco(
    entries: List,
    class_names: List[str],
    dest_path: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write a COCO JSON for a subset of entries (typically per-dataset test)."""

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    category_records = [
        {"id": idx + 1, "name": name, "supercategory": "object"}
        for idx, name in enumerate(class_names)
    ]

    log = logger or logging.getLogger("train")

    images_meta = []
    annotations = []
    ann_counter = 1
    for idx, entry in enumerate(entries):
        image_path: Path = entry.image_path
        label_path: Path = entry.label_path
        with Image.open(image_path) as img:
            width, height = img.size
        images_meta.append(
            {
                "id": idx + 1,
                "file_name": str(image_path.resolve()),
                "width": width,
                "height": height,
            }
        )
        anns = _load_yolo_segmentation(label_path, width, height, log)
        for ann in anns:
            ann["image_id"] = idx + 1
            ann["id"] = ann_counter
            ann_counter += 1
            annotations.append(ann)

    coco_content = {
        "images": images_meta,
        "annotations": annotations,
        "categories": category_records,
    }
    _validate_coco_annotations(images_meta, annotations, category_records, log, "subset")
    dest_path.write_text(json.dumps(coco_content, indent=2), encoding="utf-8")
    return dest_path


# ---------------------------------------------------------------------------
# Training implementations
# ---------------------------------------------------------------------------

def _early_stopping_settings(cfg: Dict) -> Dict[str, float]:
    es_cfg = cfg.get("early_stopping", {}) if isinstance(cfg, dict) else {}
    return {
        "enabled": bool(es_cfg.get("enabled", False)),
        "patience": int(es_cfg.get("patience", 8)),
        "min_delta": float(es_cfg.get("min_delta", 0.001)),
    }


def _apply_early_stopping_config(cfg, settings: Dict[str, float]) -> None:
    cfg.defrost()
    cfg.set_new_allowed(True)
    cfg.EARLY_STOPPING_ENABLED = bool(settings.get("enabled", False))
    cfg.EARLY_STOPPING_PATIENCE = int(settings.get("patience", 8))
    cfg.EARLY_STOPPING_MIN_DELTA = float(settings.get("min_delta", 0.001))
    cfg.set_new_allowed(False)
    cfg.freeze()


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.early_stopped = False

    def update(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stopped = True
        return self.early_stopped


class DetectronEarlyStopHook(HookBase):
    def __init__(self, cfg, evaluator, early_stopper: EarlyStopping):
        from detectron2.data import build_detection_test_loader

        self.cfg = cfg
        self.evaluator = evaluator
        self.early_stopper = early_stopper
        self.eval_period = int(getattr(cfg.TEST, "EVAL_PERIOD", 0) or 0)
        self.data_loader = (
            build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
            if getattr(cfg, "DATASETS", None) and cfg.DATASETS.TEST
            else None
        )

    def _compute_pseudo_val_loss(self, metrics: Dict) -> float:
        def _get_metric(key: str) -> Optional[float]:
            if not metrics:
                return None
            if isinstance(metrics.get("bbox"), dict) and key in metrics.get("bbox", {}):
                return metrics["bbox"].get(key)
            return metrics.get(key)

        ap50 = _get_metric("AP50")
        map_5095 = _get_metric("AP")
        if ap50 is None or map_5095 is None:
            return 9999.0
        ap50 = ap50 / 100.0 if ap50 > 1 else ap50
        map_5095 = map_5095 / 100.0 if map_5095 > 1 else map_5095
        return (1 - ap50) + (1 - map_5095)

    def after_step(self):
        if not self.eval_period or not self.data_loader:
            return
        iteration = self.trainer.iter + 1
        if iteration % self.eval_period != 0:
            return

        from detectron2.utils import comm
        from detectron2.evaluation import inference_on_dataset

        stop_local = False
        if comm.is_main_process():
            metrics = inference_on_dataset(self.trainer.model, self.data_loader, self.evaluator)
            pseudo_val_loss = self._compute_pseudo_val_loss(metrics)
            stop_local = self.early_stopper.update(pseudo_val_loss)

        stop_signals = comm.all_gather(stop_local)
        if any(stop_signals):
            raise StopIteration("Early stopping triggered")


def train_yolo_model(data_yaml: Path, model_dir: Path, cfg: Dict, logger: logging.Logger) -> Path:
    from ultralytics import YOLO

    weights_dir = Path(cfg.get("pretrained_weights_dir", "model/pretrained"))
    model_cfg = cfg.get("models", {}).get("yolo", cfg.get("model", {}))
    train_cfg = model_cfg.get("training", cfg.get("training", {}))
    image_cfg = cfg.get("image", {})
    target_size = int(train_cfg.get("imgsz", image_cfg.get("target_size", 640)))

    checkpoint_spec = model_cfg.get("checkpoint") or model_cfg.get("checkpoint_url")
    if checkpoint_spec is None:
        checkpoint_spec = "yolo11x-seg.pt"
    model_checkpoint = ensure_weight_file(checkpoint_spec, weights_dir, logger) or checkpoint_spec

    logger.info("Starting YOLOv11-segXL training")
    model = YOLO(model_checkpoint)

    device_value = train_cfg.get("device", model_cfg.get("device"))
    device = normalise_device(device_value) or normalise_device(os.environ.get("CUDA_VISIBLE_DEVICES"))

    early_settings = _early_stopping_settings(cfg)
    early_stopper = EarlyStopping(
        patience=early_settings["patience"], min_delta=early_settings["min_delta"]
    )

    if early_settings["enabled"]:

        @model.add_callback("on_val_end")
        def on_val_end(trainer):  # type: ignore[unused-private-member]
            metrics = getattr(trainer, "metrics", None) or {}
            if not metrics:
                logger.debug("Skipping early stopping update because validation metrics are unavailable")
                return
            val_loss = sum(float(v) for k, v in metrics.items() if str(k).startswith("val/"))
            if early_stopper.update(val_loss):
                trainer.stop_training = True

    results = model.train(
        data=str(data_yaml),
        project=str(model_dir.parent),
        name=model_dir.name,
        imgsz=target_size,
        epochs=train_cfg.get("epochs", 90),
        batch=train_cfg.get("batch", 16),
        device=device,
        rect=True,
        exist_ok=True,
        **{k: v for k, v in train_cfg.items() if k not in {"imgsz", "epochs", "batch", "device"}},
    )

    def has_trained_weights(path: Path) -> bool:
        weights_dir = path / "weights"
        return any((weights_dir / weight_name).is_file() for weight_name in ("best.pt", "last.pt"))

    run_dir_candidates: List[Path] = []
    if hasattr(results, "save_dir"):
        run_dir_candidates.append(Path(results.save_dir))
    trainer = getattr(model, "trainer", None)
    if trainer and getattr(trainer, "save_dir", None):
        run_dir_candidates.append(Path(trainer.save_dir))
    run_dir_candidates.append(model_dir)

    unique_candidates: List[Path] = []
    seen_candidates = set()
    for candidate in run_dir_candidates:
        if candidate and candidate not in seen_candidates:
            seen_candidates.add(candidate)
            unique_candidates.append(candidate)

    weight_candidates = [c for c in unique_candidates if c.exists() and has_trained_weights(c)]
    run_dir: Optional[Path] = weight_candidates[0] if weight_candidates else None

    if run_dir is None:
        matches = sorted(
            [p for p in model_dir.parent.glob(f"{model_dir.name}*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for match in matches:
            if has_trained_weights(match):
                run_dir = match
                logger.info("Found YOLO run with weights: %s", run_dir)
                break

    if run_dir is None:
        run_dir = model_dir

    logger.info("YOLO training outputs at %s", run_dir)
    return run_dir


def _flatten_overrides(prefix: Tuple[str, ...], value, output: Dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _flatten_overrides(prefix + (key,), nested_value, output)
    else:
        output[".".join(prefix)] = value


def _apply_model_overrides(cfg, overrides: Dict[str, object]) -> None:
    cfg.defrost()
    cfg.set_new_allowed(True)
    for dotted_key, value in overrides.items():
        node = cfg
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if not hasattr(node, part):
                setattr(node, part, CfgNode())
            node = getattr(node, part)
        setattr(node, parts[-1], value)
    cfg.set_new_allowed(False)
    cfg.freeze()


def _collect_training_overrides(cfg: Dict, model_key: str) -> Dict[str, Dict]:
    model_cfg = cfg.get("models", {}).get(model_key, {})
    overrides: Dict[str, Dict] = {
        "augmentations": cfg.get("augmentations", {}),
        "optimizer": cfg.get("optimizer", {}),
        "lr_scheduler": cfg.get("lr_scheduler", {}),
        "solver": {**cfg.get("solver", {}), **model_cfg.get("solver", {})},
        "model_overrides": cfg.get("model_overrides", {}).get(model_key, {}),
    }
    return overrides


def _merge_defaults(target: Dict, defaults: Dict) -> Dict:
    for key, value in defaults.items():
        if isinstance(value, dict):
            target[key] = _merge_defaults(target.get(key, {}), value)
        else:
            target.setdefault(key, value)
    return target


def _apply_training_overrides(cfg, overrides: Dict[str, Dict]) -> None:
    cfg.defrost()
    cfg.set_new_allowed(True)
    aug_overrides = overrides.get("augmentations", {})
    if isinstance(aug_overrides, dict):
        # YACS config nodes reject dict assignment for unknown keys. Persist the
        # augmentation mapping as a tuple of key/value pairs (an allowed type) and
        # rehydrate it back into a dictionary at read time.
        cfg.AUGMENTATIONS_CFG = tuple(aug_overrides.items())
    else:
        cfg.AUGMENTATIONS_CFG = aug_overrides

    optimizer_cfg = overrides.get("optimizer", {})
    cfg.SOLVER.BASE_LR = float(optimizer_cfg.get("base_lr", getattr(cfg.SOLVER, "BASE_LR", 0.0002)))
    cfg.SOLVER.WEIGHT_DECAY = float(optimizer_cfg.get("weight_decay", getattr(cfg.SOLVER, "WEIGHT_DECAY", 0.05)))
    cfg.SOLVER.BETAS = tuple(optimizer_cfg.get("betas", (0.9, 0.999)))

    scheduler_cfg = overrides.get("lr_scheduler", {})
    cfg.SOLVER.LR_SCHEDULER_NAME = scheduler_cfg.get("name", scheduler_cfg.get("LR_SCHEDULER_NAME", "CosineAnnealing"))
    cfg.SOLVER.WARMUP_ITERS = int(scheduler_cfg.get("warmup_iters", scheduler_cfg.get("WARMUP_ITERS", 500)))
    cfg.SOLVER.WARMUP_FACTOR = float(scheduler_cfg.get("warmup_factor", scheduler_cfg.get("WARMUP_FACTOR", 1e-5)))
    cfg.SOLVER.MIN_LR = float(scheduler_cfg.get("min_lr", scheduler_cfg.get("MIN_LR", 1e-6)))

    solver_cfg = overrides.get("solver", {})
    current_batch = getattr(cfg.SOLVER, "IMS_PER_BATCH", None)
    cfg.SOLVER.IMS_PER_BATCH = int(
        current_batch if current_batch is not None else solver_cfg.get("ims_per_batch", 8)
    )
    clip_cfg = solver_cfg.get("clip_gradients", {})
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = bool(clip_cfg.get("enabled", True))
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = clip_cfg.get("clip_type", "value")
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = float(clip_cfg.get("clip_value", 1.0))

    model_override_tree = overrides.get("model_overrides", {})
    flattened: Dict[str, object] = {}
    for key, value in model_override_tree.items():
        _flatten_overrides((key,), value, flattened)
    cfg.set_new_allowed(False)
    cfg.freeze()
    if flattened:
        _apply_model_overrides(cfg, flattened)


def _run_detectron_trainer(cfg, trainer_cls):
    _register_datasets_from_cfg(cfg)

    meta_arch = cfg.MODEL.META_ARCHITECTURE
    if meta_arch == "MaskFormer":
        # Ensure Mask2Former meta-architectures are registered in spawned workers.
        import mask2former.modeling  # noqa: F401
    elif meta_arch == "MaskDINO":
        # Ensure MaskDINO meta-architectures are registered in spawned workers.
        import maskdino.modeling  # noqa: F401

    trainer = trainer_cls(cfg)
    if (
        getattr(cfg, "EARLY_STOPPING_ENABLED", False)
        and getattr(cfg.TEST, "EVAL_PERIOD", 0)
        and getattr(cfg, "DATASETS", None)
        and cfg.DATASETS.TEST
    ):
        early_stopper = EarlyStopping(
            patience=int(getattr(cfg, "EARLY_STOPPING_PATIENCE", 8)),
            min_delta=float(getattr(cfg, "EARLY_STOPPING_MIN_DELTA", 0.001)),
        )
        evaluator = COCOEvaluator(
            cfg.DATASETS.TEST[0],
            cfg,
            False,
            output_dir=str(cfg.OUTPUT_DIR),
            tasks=("bbox", "segm"),
        )
        trainer.register_hooks([DetectronEarlyStopHook(cfg, evaluator, early_stopper)])

    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except StopIteration:
        logging.getLogger("train").info("Stopping Detectron2 training early")
        raise


class StrongUniversalMapper:
    """Shared augmentation pipeline with mask-preserving outputs."""

    def __init__(self, cfg, is_train: bool = True):
        from detectron2.data import detection_utils as utils

        self.is_train = is_train
        self.img_format = cfg.INPUT.FORMAT
        self.use_instance_mask = True
        self.keypoint_hflip_indices = None
        model_arch = getattr(cfg.MODEL, "META_ARCHITECTURE", "").lower()

        # Fetch the augmentation profile for the current model architecture (or the closest match).
        aug_cfg_all = getattr(cfg, "AUGMENTATIONS_CFG", {}) or {}
        if isinstance(aug_cfg_all, tuple):
            aug_cfg_all = dict(aug_cfg_all)
        if not isinstance(aug_cfg_all, dict):
            aug_cfg_all = {}

        model_key = {
            "generalizedrcnn": "mask_rcnn",
            "maskformer": "mask2former",
            "maskdino": "maskdino",
        }.get(model_arch, model_arch)

        self.model_key = model_key

        aug_cfg = aug_cfg_all.get(model_key, aug_cfg_all.get(model_arch, {}))

        def _get(name, default):
            return aug_cfg.get(name, default)

        brightness = tuple(_get("random_brightness", (0.8, 1.2)))
        contrast = tuple(_get("random_contrast", (0.8, 1.2)))
        saturation = tuple(_get("random_saturation", (0.8, 1.3)))
        lighting = float(_get("random_lighting_scale", 0.1))
        resize_shortest_edge = _get("resize_shortest_edge", [384, 480, 576, 640])
        max_size = int(_get("resize_max_size", 768))
        crop_prob = float(_get("random_crop_prob", 0.3))
        crop_relative_area = float(_get("random_crop_relative_area", 0.8))
        extent_prob = float(_get("random_extent_prob", 0.25))
        # RandomExtent in Detectron2 expects scale and shift ranges; ignore any
        # legacy random_extent_size entries from older configs.
        extent_scale_range = tuple(_get("random_extent_scale_range", (0.8, 1.2)))
        extent_shift_range = tuple(_get("random_extent_shift_range", (0.1, 0.1)))

        augmentations: List[T.Augmentation] = [
            T.RandomBrightness(*brightness),
            T.RandomContrast(*contrast),
            T.RandomSaturation(*saturation),
            T.RandomLighting(scale=lighting),
            T.RandomFlip(horizontal=True),
            T.ResizeShortestEdge(
                resize_shortest_edge, max_size=max_size, sample_style="choice"
            ),
        ]

        random_crop = T.RandomCrop("relative_range", (crop_relative_area, crop_relative_area))
        augmentations.append(T.RandomApply(random_crop, prob=crop_prob))

        if (
            self.is_train
            and model_key in {"mask2former", "maskdino", "maskformer"}
            and extent_prob > 0
        ):
            # RandomExtent uses scale and shift ranges in the current Detectron2 API.
            extent_aug = T.RandomExtent(extent_scale_range, extent_shift_range)
            augmentations.append(T.RandomApply(extent_aug, prob=extent_prob))

        logging.getLogger("train").info(
            "Augmentations for %s: brightness=%s, contrast=%s, saturation=%s, crop_area=%s",
            model_key or model_arch,
            brightness,
            contrast,
            saturation,
            crop_relative_area,
        )

        self.augmentations = T.AugmentationList(augmentations)
        self.utils = utils

    def __call__(self, dataset_dict):
        from detectron2.structures import BitMasks

        dataset_dict = deepcopy(dataset_dict)
        image = self.utils.read_image(dataset_dict["file_name"], format=self.img_format)
        self.utils.check_image_size(dataset_dict, image)

        # AugInput is provided by detectron2.data.transforms in recent releases.
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input) if self.is_train else []
        image = aug_input.image
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            return dataset_dict

        annos = [
            self.utils.transform_instance_annotations(anno, transforms, image.shape[:2])
            for anno in dataset_dict.pop("annotations")
            if anno.get("iscrowd", 0) == 0
        ]
        instances = self.utils.annotations_to_instances(
            annos, image.shape[:2], mask_format="bitmask"
        )

        if instances.has("gt_masks"):
            mask_tensor = instances.gt_masks.tensor.bool()
            if self.model_key in {"mask2former", "maskdino", "maskformer"}:
                # Transformers expect tensor targets with a ``shape`` attribute;
                # BitMasks do not expose ``shape`` and trigger attribute errors in
                # their target preparation code. Provide a raw tensor for those
                # models while keeping BitMasks for architectures that expect
                # Detectron2's mask container type.
                instances.gt_masks = mask_tensor
            else:
                instances.gt_masks = BitMasks(mask_tensor)

        # Remove boxes/masks that became degenerate after augmentation. Detectron2's
        # default mapper performs this filtering to avoid invalid targets (e.g.,
        # zero-area boxes) that can explode the regression loss. Skipping it here
        # can lead to Inf/NaN losses during the first iteration.
        instances = self.utils.filter_empty_instances(instances)

        dataset_dict["instances"] = instances
        return dataset_dict


def _build_adamw_optimizer(cfg, model):
    param_kwargs = {
        "base_lr": cfg.SOLVER.BASE_LR,
        "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
        "weight_decay_norm": getattr(cfg.SOLVER, "WEIGHT_DECAY_NORM", cfg.SOLVER.WEIGHT_DECAY),
        "bias_lr_factor": cfg.SOLVER.BIAS_LR_FACTOR,
        "weight_decay_bias": cfg.SOLVER.WEIGHT_DECAY_BIAS,
        "weight_decay_embed": getattr(
            cfg.SOLVER, "WEIGHT_DECAY_EMBED", cfg.SOLVER.WEIGHT_DECAY
        ),
    }

    supported_params = inspect.signature(get_default_optimizer_params).parameters
    params = get_default_optimizer_params(
        model, **{k: v for k, v in param_kwargs.items() if k in supported_params}
    )
    betas = tuple(getattr(cfg.SOLVER, "BETAS", (0.9, 0.999)))
    optimizer = torch.optim.AdamW(
        params, lr=cfg.SOLVER.BASE_LR, betas=betas, weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    return maybe_add_gradient_clipping(cfg, optimizer)


def _build_cosine_scheduler(cfg, optimizer):
    warmup_iters = int(getattr(cfg.SOLVER, "WARMUP_ITERS", 0))
    warmup_factor = float(getattr(cfg.SOLVER, "WARMUP_FACTOR", 1.0))
    min_lr = float(getattr(cfg.SOLVER, "MIN_LR", 0.0))
    max_iter = int(cfg.SOLVER.MAX_ITER)
    base_lr = float(cfg.SOLVER.BASE_LR)

    def lr_func(iteration: int) -> float:
        if iteration < warmup_iters:
            alpha = iteration / max(1, warmup_iters)
            return warmup_factor + (1 - warmup_factor) * alpha

        progress = (iteration - warmup_iters) / max(1, max_iter - warmup_iters)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (min_lr / base_lr) + (1 - min_lr / base_lr) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


class MaskDINOTrainer(DefaultTrainer):
    """Custom trainer using the shared augmentation/optimization stack."""

    @classmethod
    def build_train_loader(cls, cfg):
        from detectron2.data import build_detection_train_loader

        return build_detection_train_loader(cfg, mapper=StrongUniversalMapper(cfg, is_train=True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        # DefaultTrainer expects the optimizer builder to accept the model.
        return _build_adamw_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return _build_cosine_scheduler(cfg, optimizer)


class EvalTrainer(DefaultTrainer):
    """Trainer with automatic COCO evaluation for bbox+segm."""

    @classmethod
    def build_train_loader(cls, cfg):
        from detectron2.data import build_detection_train_loader

        return build_detection_train_loader(cfg, mapper=StrongUniversalMapper(cfg, is_train=True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        # DefaultTrainer calls this with the model; accept it for compatibility.
        return _build_adamw_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return _build_cosine_scheduler(cfg, optimizer)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = cfg.OUTPUT_DIR
        return COCOEvaluator(
            dataset_name=dataset_name,
            tasks=("bbox", "segm"),
            distributed=True,
            output_dir=output_folder,
        )


def _attach_dataset_catalog(cfg, dataset_specs: Dict[str, Dict[str, str]]) -> None:
    """Store dataset registration info inside the Detectron2 config."""

    cfg.defrost()
    cfg.set_new_allowed(True)
    # yacs configs disallow plain dicts as attribute values; store as a list of
    # tuples and rebuild the dict when registering datasets.
    cfg.DATASETS_CATALOG = list(dataset_specs.items())
    cfg.set_new_allowed(False)
    cfg.freeze()


def _register_datasets_from_cfg(cfg) -> None:
    """Register COCO datasets specified in the config if they're missing.

    torch.multiprocessing with the "spawn" start method launches fresh Python
    interpreters that do not inherit Detectron2's global DatasetCatalog state
    from the parent process. Register datasets in every spawned worker so
    Detectron2 can build the data loaders without KeyError.
    """

    dataset_specs = getattr(cfg, "DATASETS_CATALOG", None)
    if not dataset_specs:
        return

    # Backwards compatibility: allow either a dict or a list of (name, spec)
    # pairs. The latter is used to satisfy yacs' allowed config value types.
    if isinstance(dataset_specs, dict):
        dataset_items = dataset_specs.items()
    else:
        dataset_items = dataset_specs

    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    for name, spec in dataset_items:
        if name in DatasetCatalog.list() or name in REGISTERED_COCO:
            continue

        json_path = str(spec["json_file"])
        image_root = str(spec["image_root"])

        register_coco_instances(name, {}, json_path, image_root)
        REGISTERED_COCO.add(name)

        if classes := spec.get("classes"):
            MetadataCatalog.get(name).thing_classes = classes


def run_detectron_training(cfg, trainer_cls, logger: logging.Logger) -> None:
    """Train a Detectron2 model across all available GPUs."""

    from detectron2.engine import launch

    world_size = max(1, torch.cuda.device_count())
    if world_size > 1:
        logger.info("Launching distributed Detectron2 training on %s GPUs", world_size)
        try:
            launch(
                _run_detectron_trainer,
                num_gpus_per_machine=world_size,
                num_machines=1,
                dist_url="auto",
                args=(cfg, trainer_cls),
            )
        except StopIteration:
            logger.info("Early stopping triggered during distributed Detectron2 training")
    else:
        try:
            _run_detectron_trainer(cfg, trainer_cls)
        except StopIteration:
            logger.info("Early stopping triggered during Detectron2 training")


def _build_detectron2_maskrcnn_config(
    train_json: Path,
    val_json: Path,
    test_json: Path,
    work_dir: Path,
    class_count: int,
    training_overrides: Dict,
    dataset_size: int,
    image_stats: Optional[Dict[str, float]] = None,
    early_stopping_cfg: Optional[Dict[str, float]] = None,
    config_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    base_config = training_overrides.get(
        "config_file", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    if config_path:
        cfg.merge_from_file(str(config_path))
    else:
        cfg.merge_from_file(model_zoo.get_config_file(base_config))
    cfg.DATASETS.TRAIN = ("compare_train",)
    cfg.DATASETS.TEST = ("compare_val",)
    # Use bitmask conversion to provide tensor masks; some models (e.g., MaskDINO)
    # expect targets with a tensor "shape" attribute and fail on PolygonMasks.
    cfg.INPUT.MASK_FORMAT = "bitmask"
    solver_cfg = training_overrides.get("solver", {})
    cfg.DATALOADER.NUM_WORKERS = int(solver_cfg.get("dataloader_workers", 4))
    ims_per_batch = int(solver_cfg.get("ims_per_batch", 8))
    gpu_mems = collect_gpu_vram()
    cfg.SOLVER.IMS_PER_BATCH = compute_vram_aware_batch_size(gpu_mems, ims_per_batch)
    if gpu_mems:
        cfg.SOLVER.REFERENCE_WORLD_SIZE = len(gpu_mems)
        if logger:
            logger.info(
                "Auto-detected Detectron2 batch size %s for %s GPU(s) (VRAM: %s GiB)",
                cfg.SOLVER.IMS_PER_BATCH,
                len(gpu_mems),
                [round(mem / 2**30, 1) for mem in gpu_mems],
            )
    optimizer_cfg = training_overrides.get("optimizer", {})
    cfg.SOLVER.BASE_LR = float(optimizer_cfg.get("base_lr", 0.0002))
    epochs = int(solver_cfg.get("epochs", 36))
    steps_per_epoch = max(1, math.ceil(dataset_size / cfg.SOLVER.IMS_PER_BATCH))
    cfg.SOLVER.MAX_ITER = steps_per_epoch * epochs

    if image_stats and image_stats.get("min_short_edge"):
        min_short = max(256, int(image_stats["min_short_edge"]))
        median_short = int(image_stats.get("median_short_edge", min_short))
        max_long = int(image_stats.get("max_long_edge", median_short * 2))
        cfg.INPUT.MIN_SIZE_TRAIN = [min_short, max(median_short, min_short)]
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range" if len(cfg.INPUT.MIN_SIZE_TRAIN) > 1 else "choice"
        cfg.INPUT.MIN_SIZE_TEST = max(min_short, int(image_stats.get("median_short_edge", min_short)))
        cfg.INPUT.MAX_SIZE_TRAIN = max_long
        cfg.INPUT.MAX_SIZE_TEST = max_long

    weight_path = solver_cfg.get("weights")
    if weight_path:
        cfg.MODEL.WEIGHTS = str(weight_path)
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_count
    cfg.OUTPUT_DIR = str(work_dir)
    eval_period = solver_cfg.get("eval_period")
    eval_period_epochs = solver_cfg.get("eval_period_epochs")
    if eval_period is None:
        eval_period = steps_per_epoch * float(eval_period_epochs if eval_period_epochs is not None else 1)
    cfg.TEST.EVAL_PERIOD = max(1, int(eval_period))
    _apply_training_overrides(cfg, training_overrides)
    _apply_early_stopping_config(cfg, early_stopping_cfg or {})
    return cfg


def train_mask_rcnn(
    coco_paths: Dict[str, Path],
    model_dir: Path,
    class_names: List[str],
    logger: logging.Logger,
    cfg: Dict,
    dataset_size: int,
    image_stats: Optional[Dict[str, float]] = None,
) -> Tuple[Path, object, Dict, object]:
    early_stop_cfg = _early_stopping_settings(cfg)
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    logger.info("Registering Detectron2 datasets for Mask R-CNN")
    register_coco_instances("compare_train", {}, str(coco_paths["train"]), coco_paths["train"].parent)
    register_coco_instances("compare_val", {}, str(coco_paths["val"]), coco_paths["val"].parent)
    register_coco_instances("compare_test", {}, str(coco_paths["test"]), coco_paths["test"].parent)

    MetadataCatalog.get("compare_train").thing_classes = class_names
    MetadataCatalog.get("compare_val").thing_classes = class_names
    MetadataCatalog.get("compare_test").thing_classes = class_names

    model_cfg = cfg.get("models", {}).get("mask_rcnn", {})
    training_overrides = _collect_training_overrides(cfg, "mask_rcnn")
    training_overrides["model_overrides"] = _merge_defaults(
        training_overrides.get("model_overrides", {}),
        {
            "MODEL": {
                "ROI_HEADS": {"BATCH_SIZE_PER_IMAGE": 512},
                "RPN": {
                    "PRE_NMS_TOPK_TRAIN": 2000,
                    "PRE_NMS_TOPK_TEST": 1000,
                    "POST_NMS_TOPK_TRAIN": 1000,
                    "POST_NMS_TOPK_TEST": 1000,
                },
            }
        },
    )
    solver_cfg = training_overrides.get("solver", {})
    pretrained = model_cfg.get("weights")
    if pretrained is None:
        pretrained = model_cfg.get("weights_url")
    weights_dir = Path(cfg.get("pretrained_weights_dir", "model/pretrained"))
    weight_path = ensure_weight_file(pretrained, weights_dir, logger)
    solver_cfg = {**solver_cfg, "weights": weight_path}
    training_overrides["solver"] = solver_cfg

    config_spec = model_cfg.get("config_file", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    config_dir = Path(cfg.get("config_cache_dir", "model/configs")) / "mask_rcnn"
    default_url = (
        "https://raw.githubusercontent.com/facebookresearch/detectron2/main/configs/"
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    config_path = ensure_config_file(config_spec, config_dir, logger, default_url=default_url)
    training_overrides["config_file"] = config_spec

    cfg = _build_detectron2_maskrcnn_config(
        coco_paths["train"],
        coco_paths["val"],
        coco_paths["test"],
        model_dir,
        len(class_names),
        training_overrides,
        dataset_size,
        image_stats,
        early_stopping_cfg=early_stop_cfg,
        config_path=config_path,
        logger=logger,
    )

    _attach_dataset_catalog(
        cfg,
        {
            "compare_train": {
                "json_file": str(coco_paths["train"]),
                "image_root": str(coco_paths["train"].parent),
                "classes": class_names,
            },
            "compare_val": {
                "json_file": str(coco_paths["val"]),
                "image_root": str(coco_paths["val"].parent),
                "classes": class_names,
            },
            "compare_test": {
                "json_file": str(coco_paths["test"]),
                "image_root": str(coco_paths["test"].parent),
                "classes": class_names,
            },
        },
    )

    run_detectron_training(cfg, EvalTrainer, logger)

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    model = build_model(cfg)
    trained_weights = get_trained_weights_path(Path(cfg.OUTPUT_DIR), logger)
    if trained_weights:
        DetectionCheckpointer(model).load(trained_weights)
    else:
        logger.warning("Mask R-CNN model in %s has no trained weights; using default initialization", cfg.OUTPUT_DIR)
    model.eval()

    evaluator = COCOEvaluator(
        "compare_test",
        cfg,
        False,
        output_dir=str(model_dir / "eval"),
        tasks=("bbox", "segm"),
    )
    val_loader = build_detection_test_loader(cfg, "compare_test")
    results = inference_on_dataset(model, val_loader, evaluator)
    return Path(cfg.OUTPUT_DIR), model, results, cfg


def evaluate_detectron_model(
    model,
    cfg,
    test_dataset_name: str,
    eval_dir: Path,
    class_names: List[str],
    subsets: Dict[str, Path],
) -> Dict[str, Dict[str, float]]:
    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader

    evaluations: Dict[str, Dict[str, float]] = {}

    evaluator = COCOEvaluator(
        test_dataset_name,
        cfg,
        False,
        output_dir=str(eval_dir / "pooled_test"),
        tasks=("bbox", "segm"),
    )
    loader = build_detection_test_loader(cfg, test_dataset_name)
    pooled = inference_on_dataset(model, loader, evaluator)
    evaluations["pooled"] = format_metrics_from_coco(pooled)
    write_ultralytics_style_artifacts(eval_dir / "pooled_test", evaluations["pooled"])

    for subset_name, json_path in subsets.items():
        ds_name = f"{test_dataset_name}_{subset_name}" if test_dataset_name else subset_name
        register_coco_instances(ds_name, {}, str(json_path), json_path.parent)
        MetadataCatalog.get(ds_name).thing_classes = class_names
        evaluator = COCOEvaluator(
            ds_name,
            cfg,
            False,
            output_dir=str(eval_dir / subset_name),
            tasks=("bbox", "segm"),
        )
        loader = build_detection_test_loader(cfg, ds_name)
        subset_result = inference_on_dataset(model, loader, evaluator)
        evaluations[subset_name] = format_metrics_from_coco(subset_result)
        write_ultralytics_style_artifacts(eval_dir / subset_name, evaluations[subset_name])

    summary_path = eval_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
    return evaluations


def train_mask2former(
    coco_paths: Dict[str, Path],
    model_dir: Path,
    class_names: List[str],
    logger: logging.Logger,
    cfg: Dict,
    dataset_size: int,
    image_stats: Optional[Dict[str, float]] = None,
) -> Tuple[Path, object, Dict, object]:
    early_stop_cfg = _early_stopping_settings(cfg)
    model_cfg = cfg.get("models", {}).get("mask2former", {})
    cfg_file = model_cfg.get(
        "config_file",
        "configs/coco/instance-segmentation/maskformer2_swin_large_IN21k_384_bs16_50ep.yaml",
    )
    training_overrides = _collect_training_overrides(cfg, "mask2former")
    training_overrides["model_overrides"] = _merge_defaults(
        training_overrides.get("model_overrides", {}),
        {
            "MODEL": {
                "NUM_OBJECT_QUERIES": 200,
                "HIDDEN_DIM": 256,
                "NHEADS": 8,
                "DROPOUT": 0.1,
                "DECODER": {"DROP_PATH_RATE": 0.1},
            },
            "TEST": {"DETECTIONS_PER_IMAGE": 2000, "SCORE_THRESH": 0.0},
        },
    )
    solver_cfg = training_overrides.get("solver", {})

    add_local_repo_to_path(cfg_file, "mask2former", logger)

    from detectron2.data import MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    from detectron2.config import get_cfg
    from mask2former import add_maskformer2_config

    # Ensure the meta-architectures are registered before model construction.
    import mask2former.modeling  # noqa: F401

    register_coco_instances("m2f_train", {}, str(coco_paths["train"]), coco_paths["train"].parent)
    register_coco_instances("m2f_val", {}, str(coco_paths["val"]), coco_paths["val"].parent)
    register_coco_instances("m2f_test", {}, str(coco_paths["test"]), coco_paths["test"].parent)

    MetadataCatalog.get("m2f_train").thing_classes = class_names
    MetadataCatalog.get("m2f_val").thing_classes = class_names
    MetadataCatalog.get("m2f_test").thing_classes = class_names

    config_dir = Path(cfg.get("config_cache_dir", "model/configs")) / "mask2former"
    default_url = (
        "https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/configs/coco/instance-"
        "segmentation/maskformer2_swin_large_IN21k_384_bs16_50ep.yaml"
    )
    config_path = ensure_config_file(
        cfg_file,
        config_dir,
        logger,
        default_url=default_url,
        fallback_contents="""
_BASE_: "https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/configs/coco/
    instance-segmentation/maskformer2_swin_large_IN21k_384_bs16_50ep.yaml"
MODEL:
  WEIGHTS: ""
        """,
    )
    training_overrides["config_file"] = cfg_file

    weights_dir = Path(cfg.get("pretrained_weights_dir", "model/pretrained"))
    pretrained = model_cfg.get("weights")
    if pretrained is None:
        pretrained = model_cfg.get("weights_url")
    if pretrained and _weights_incompatible_with_config(cfg_file, pretrained):
        logger.warning(
            "Ignoring Mask2Former weights '%s' because they do not match config '%s'",
            pretrained,
            cfg_file,
        )
        pretrained = None
    weight_path = ensure_weight_file(pretrained, weights_dir, logger) if pretrained else None

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(str(config_path or cfg_file))
    cfg.set_new_allowed(False)
    _sanitize_gradient_clipping(cfg, logger)
    cfg.DATASETS.TRAIN = ("m2f_train",)
    cfg.DATASETS.TEST = ("m2f_val",)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(class_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.DATALOADER.NUM_WORKERS = int(solver_cfg.get("dataloader_workers", 4))
    ims_per_batch = int(solver_cfg.get("ims_per_batch", max(2, 2 * max(1, torch.cuda.device_count()))))
    gpu_mems = collect_gpu_vram()
    cfg.SOLVER.IMS_PER_BATCH = compute_vram_aware_batch_size(gpu_mems, ims_per_batch)
    if gpu_mems:
        cfg.SOLVER.REFERENCE_WORLD_SIZE = len(gpu_mems)
    optimizer_cfg = training_overrides.get("optimizer", {})
    cfg.SOLVER.BASE_LR = float(optimizer_cfg.get("base_lr", 0.0002))
    epochs = int(solver_cfg.get("epochs", 50))
    steps_per_epoch = max(1, math.ceil(dataset_size / cfg.SOLVER.IMS_PER_BATCH))
    cfg.SOLVER.MAX_ITER = steps_per_epoch * epochs
    eval_period = solver_cfg.get("eval_period")
    eval_period_epochs = solver_cfg.get("eval_period_epochs")
    if eval_period is None:
        eval_period = steps_per_epoch * float(eval_period_epochs if eval_period_epochs is not None else 1)
    cfg.TEST.EVAL_PERIOD = max(1, int(eval_period))
    if image_stats and image_stats.get("min_short_edge"):
        min_short = max(256, int(image_stats["min_short_edge"]))
        median_short = int(image_stats.get("median_short_edge", min_short))
        max_long = int(image_stats.get("max_long_edge", median_short * 2))
        cfg.INPUT.MIN_SIZE_TRAIN = [min_short, max(median_short, min_short)]
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range" if len(cfg.INPUT.MIN_SIZE_TRAIN) > 1 else "choice"
        cfg.INPUT.MIN_SIZE_TEST = max(min_short, int(image_stats.get("median_short_edge", min_short)))
        cfg.INPUT.MAX_SIZE_TRAIN = max_long
        cfg.INPUT.MAX_SIZE_TEST = max_long
    if weight_path:
        cfg.MODEL.WEIGHTS = str(weight_path)
    # Convert polygon annotations to bitmasks so downstream models receive tensor
    # masks with a "shape" attribute during target preparation.
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.OUTPUT_DIR = str(model_dir)
    _apply_training_overrides(cfg, training_overrides)
    _apply_early_stopping_config(cfg, early_stop_cfg)

    _attach_dataset_catalog(
        cfg,
        {
            "m2f_train": {
                "json_file": str(coco_paths["train"]),
                "image_root": str(coco_paths["train"].parent),
                "classes": class_names,
            },
            "m2f_val": {
                "json_file": str(coco_paths["val"]),
                "image_root": str(coco_paths["val"].parent),
                "classes": class_names,
            },
            "m2f_test": {
                "json_file": str(coco_paths["test"]),
                "image_root": str(coco_paths["test"].parent),
                "classes": class_names,
            },
        },
    )

    run_detectron_training(cfg, EvalTrainer, logger)

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    model = build_model(cfg)
    trained_weights = get_trained_weights_path(Path(cfg.OUTPUT_DIR), logger)
    if trained_weights:
        DetectionCheckpointer(model).load(trained_weights)
    else:
        logger.warning(
            "Mask2Former model in %s has no trained weights; using default initialization",
            cfg.OUTPUT_DIR,
        )
    model.eval()

    evaluator = COCOEvaluator(
        "m2f_test",
        cfg,
        False,
        output_dir=str(model_dir / "eval"),
        tasks=("bbox", "segm"),
    )
    val_loader = build_detection_test_loader(cfg, "m2f_test")
    results = inference_on_dataset(model, val_loader, evaluator)
    return Path(cfg.OUTPUT_DIR), model, results, cfg


def train_maskdino(
    coco_paths: Dict[str, Path],
    model_dir: Path,
    class_names: List[str],
    logger: logging.Logger,
    cfg: Dict,
    dataset_size: int,
    image_stats: Optional[Dict[str, float]] = None,
) -> Tuple[Path, object, Dict, object]:
    early_stop_cfg = _early_stopping_settings(cfg)
    from detectron2.data import MetadataCatalog
    model_cfg = cfg.get("models", {}).get("maskdino", {})
    cfg_file = model_cfg.get(
        "config_file", "configs/coco/instance-segmentation/maskdino_R50_1x.yaml"
    )
    training_overrides = _collect_training_overrides(cfg, "maskdino")
    training_overrides["model_overrides"] = _merge_defaults(
        training_overrides.get("model_overrides", {}),
        {
            "MODEL": {
                "NUM_OBJECT_QUERIES": 200,
                "HIDDEN_DIM": 256,
                "NHEADS": 8,
                "DROPOUT": 0.1,
            },
            "TEST": {"DETECTIONS_PER_IMAGE": 2000, "SCORE_THRESH": 0.0},
        },
    )
    solver_cfg = training_overrides.get("solver", {})

    add_local_repo_to_path(cfg_file, "maskdino", logger)

    from detectron2.data import DatasetCatalog

    original_register = DatasetCatalog.register

    def safe_register(name, func):
        if name in DatasetCatalog.list():
            logger.warning("Dataset '%s' already registered; skipping duplicate registration.", name)
            return
        original_register(name, func)

    DatasetCatalog.register = safe_register

    try:
        from detectron2.data.datasets import register_coco_instances
        from detectron2.engine import DefaultTrainer
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        from detectron2.data import build_detection_test_loader
        from detectron2.config import get_cfg

        # MaskDINO uses the same class name as Mask2Former for its Swin backbone.
        # When both models run in the same process, the Detectron2 registry can
        # already contain an entry for "D2SwinTransformer", causing MaskDINO's
        # registration to raise an AssertionError. Clear the stale entry before
        # importing MaskDINO so it can register its implementation safely.
        from detectron2.modeling.backbone import BACKBONE_REGISTRY

        if "D2SwinTransformer" in BACKBONE_REGISTRY:
            BACKBONE_REGISTRY._obj_map.pop("D2SwinTransformer", None)

        from maskdino import add_maskdino_config
    finally:
        DatasetCatalog.register = original_register

    register_coco_instances("mdino_train", {}, str(coco_paths["train"]), coco_paths["train"].parent)
    register_coco_instances("mdino_val", {}, str(coco_paths["val"]), coco_paths["val"].parent)
    register_coco_instances("mdino_test", {}, str(coco_paths["test"]), coco_paths["test"].parent)

    MetadataCatalog.get("mdino_train").thing_classes = class_names
    MetadataCatalog.get("mdino_val").thing_classes = class_names
    MetadataCatalog.get("mdino_test").thing_classes = class_names

    weights_dir = Path(cfg.get("pretrained_weights_dir", "model/pretrained"))
    pretrained = model_cfg.get("weights")
    if pretrained is None:
        pretrained = model_cfg.get("weights_url")
    if pretrained and _weights_incompatible_with_config(cfg_file, pretrained):
        logger.warning(
            "Ignoring MaskDINO weights '%s' because they do not match config '%s'",
            pretrained,
            cfg_file,
        )
        pretrained = None
    weight_path = ensure_weight_file(pretrained, weights_dir, logger) if pretrained else None

    config_dir = Path(cfg.get("config_cache_dir", "model/configs")) / "maskdino"
    default_url = (
        "https://raw.githubusercontent.com/IDEA-Research/MaskDINO/main/configs/coco/instance-"
        "segmentation/maskdino_R50_1x.yaml"
    )
    config_path = ensure_config_file(cfg_file, config_dir, logger, default_url=default_url)
    training_overrides["config_file"] = cfg_file

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(config_path or cfg_file))
    cfg.set_new_allowed(False)
    _sanitize_gradient_clipping(cfg, logger)
    cfg.DATASETS.TRAIN = ("mdino_train",)
    cfg.DATASETS.TEST = ("mdino_val",)
    # Convert polygon annotations to bitmasks so downstream models receive tensor
    # masks with a "shape" attribute during target preparation.
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.DATALOADER.NUM_WORKERS = int(solver_cfg.get("dataloader_workers", 4))
    ims_per_batch = int(
        solver_cfg.get("ims_per_batch", max(2, 2 * max(1, torch.cuda.device_count())))
    )
    gpu_mems = collect_gpu_vram()
    cfg.SOLVER.IMS_PER_BATCH = compute_vram_aware_batch_size(gpu_mems, ims_per_batch)
    if gpu_mems:
        cfg.SOLVER.REFERENCE_WORLD_SIZE = len(gpu_mems)
    optimizer_cfg = training_overrides.get("optimizer", {})
    cfg.SOLVER.BASE_LR = float(optimizer_cfg.get("base_lr", 0.0002))
    epochs = int(solver_cfg.get("epochs", 60))
    steps_per_epoch = max(1, math.ceil(dataset_size / cfg.SOLVER.IMS_PER_BATCH))
    cfg.SOLVER.MAX_ITER = steps_per_epoch * epochs
    eval_period = solver_cfg.get("eval_period")
    eval_period_epochs = solver_cfg.get("eval_period_epochs")
    if eval_period is None:
        eval_period = steps_per_epoch * float(eval_period_epochs if eval_period_epochs is not None else 1)
    cfg.TEST.EVAL_PERIOD = max(1, int(eval_period))

    if image_stats and image_stats.get("min_short_edge"):
        min_short = max(256, int(image_stats["min_short_edge"]))
        median_short = int(image_stats.get("median_short_edge", min_short))
        max_long = int(image_stats.get("max_long_edge", median_short * 2))
        cfg.INPUT.MIN_SIZE_TRAIN = [min_short, max(median_short, min_short)]
        cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "range" if len(cfg.INPUT.MIN_SIZE_TRAIN) > 1 else "choice"
        cfg.INPUT.MIN_SIZE_TEST = max(min_short, int(image_stats.get("median_short_edge", min_short)))
        cfg.INPUT.MAX_SIZE_TRAIN = max_long
        cfg.INPUT.MAX_SIZE_TEST = max_long

    if weight_path:
        cfg.MODEL.WEIGHTS = str(weight_path)
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(class_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.OUTPUT_DIR = str(model_dir)
    _apply_training_overrides(cfg, training_overrides)
    _apply_early_stopping_config(cfg, early_stop_cfg)

    _attach_dataset_catalog(
        cfg,
        {
            "mdino_train": {
                "json_file": str(coco_paths["train"]),
                "image_root": str(coco_paths["train"].parent),
                "classes": class_names,
            },
            "mdino_val": {
                "json_file": str(coco_paths["val"]),
                "image_root": str(coco_paths["val"].parent),
                "classes": class_names,
            },
            "mdino_test": {
                "json_file": str(coco_paths["test"]),
                "image_root": str(coco_paths["test"].parent),
                "classes": class_names,
            },
        },
    )

    run_detectron_training(cfg, EvalTrainer, logger)

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model

    model = build_model(cfg)
    trained_weights = get_trained_weights_path(Path(cfg.OUTPUT_DIR), logger)
    if trained_weights:
        DetectionCheckpointer(model).load(trained_weights)
    else:
        logger.warning(
            "MaskDINO model in %s has no trained weights; using default initialization",
            cfg.OUTPUT_DIR,
        )
    model.eval()

    evaluator = COCOEvaluator(
        "mdino_test",
        cfg,
        False,
        output_dir=str(model_dir / "eval"),
        tasks=("bbox", "segm"),
    )
    val_loader = build_detection_test_loader(cfg, "mdino_test")
    results = inference_on_dataset(model, val_loader, evaluator)
    return Path(cfg.OUTPUT_DIR), model, results, cfg


# ---------------------------------------------------------------------------
# Evaluation + reporting utilities
# ---------------------------------------------------------------------------

def format_metrics_from_coco(results: Dict) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    bbox_results = results.get("bbox", {}) if isinstance(results, dict) else {}
    segm_results = results.get("segm", {}) if isinstance(results, dict) else {}

    metrics["metrics/precision(B)"] = float(bbox_results.get("AP50", np.nan))
    metrics["metrics/recall(B)"] = float(bbox_results.get("AR100", np.nan))
    metrics["metrics/mAP50(B)"] = float(bbox_results.get("AP50", np.nan))
    metrics["metrics/mAP50-95(B)"] = float(bbox_results.get("AP", np.nan))

    metrics["metrics/precision(M)"] = float(segm_results.get("AP50", np.nan))
    metrics["metrics/recall(M)"] = float(segm_results.get("AR100", np.nan))
    metrics["metrics/mAP50(M)"] = float(segm_results.get("AP50", np.nan))
    metrics["metrics/mAP50-95(M)"] = float(segm_results.get("AP", np.nan))
    return metrics


def build_subset_coco_annotations(
    splits: Dict[str, List],
    class_names: List[str],
    eval_root: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    subset_dir = eval_root / "subsets"
    subset_dir.mkdir(parents=True, exist_ok=True)
    log = logger or logging.getLogger("train")
    subsets: Dict[str, Path] = {}
    grouped: Dict[str, List] = {}
    for entry in splits.get("test", []):
        dataset_name = getattr(entry, "dataset", "pooled")
        grouped.setdefault(dataset_name, []).append(entry)
    for dataset_name, entries in grouped.items():
        subsets[dataset_name] = convert_dataset_subset_to_coco(
            entries, class_names, subset_dir / f"{dataset_name}.json", log
        )
    return subsets


def save_results_csv(model_dir: Path, metrics: Dict[str, float]) -> Path:
    rows = [{"epoch": "final", **metrics}]
    df = pd.DataFrame(rows)
    csv_path = model_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_simple_curve(path: Path, title: str, metric_value: float) -> None:
    fig, ax = plt.subplots()
    xs = np.linspace(0, 1, 50)
    ys = np.clip(metric_value * np.ones_like(xs), 0, 1)
    ax.plot(xs, ys, label=title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def write_ultralytics_style_artifacts(eval_dir: Path, metrics: Dict[str, float]) -> None:
    """Create placeholder curves to mirror Ultralytics output layout."""

    curve_map = {
        "BoxPR_curve.png": metrics.get("metrics/mAP50(B)", 0.0),
        "BoxP_curve.png": metrics.get("metrics/precision(B)", 0.0),
        "BoxR_curve.png": metrics.get("metrics/recall(B)", 0.0),
        "BoxF1_curve.png": metrics.get("metrics/mAP50-95(B)", 0.0),
        "MaskPR_curve.png": metrics.get("metrics/mAP50(M)", 0.0),
        "MaskP_curve.png": metrics.get("metrics/precision(M)", 0.0),
        "MaskR_curve.png": metrics.get("metrics/recall(M)", 0.0),
        "MaskF1_curve.png": metrics.get("metrics/mAP50-95(M)", 0.0),
    }
    for filename, value in curve_map.items():
        _write_simple_curve(eval_dir / filename, filename.replace("_", " "), float(value or 0.0))

    # Confusion matrices placeholder
    for name in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        fig, ax = plt.subplots()
        matrix = np.zeros((2, 2))
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(name)
        fig.tight_layout()
        eval_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(eval_dir / name)
        plt.close(fig)


def save_evaluation_summary(model_dir: Path, metrics: Dict[str, float]) -> Path:
    summary_path = model_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps({"test": metrics}, indent=2), encoding="utf-8")
    return summary_path


# ---------------------------------------------------------------------------
# Comparison reporting
# ---------------------------------------------------------------------------

def collate_final_metrics(model_metrics: Dict[str, Dict[str, float]], stats_dir: Path) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, metrics in model_metrics.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    comparison_df = pd.DataFrame(rows)
    comparison_csv = stats_dir / "training_metrics_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    summary_json = stats_dir / "model_comparison_summary.json"
    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    png_path = stats_dir / "model_comparison_summary.png"
    metric_keys = [
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(M)",
        "metrics/mAP50(M)",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metric_keys))
    width = 0.8 / max(1, len(model_metrics))
    for idx, (model_name, metrics) in enumerate(model_metrics.items()):
        values = [metrics.get(k, np.nan) for k in metric_keys]
        ax.bar(x + idx * width, values, width=width, label=model_name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_keys, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison (bbox/mask mAP)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path)


def prepare_eval_manifests(
    dataset_dir: Path,
    splits: Dict[str, List],
    class_names: List[str],
    eval_root: Path,
) -> Tuple[Path, Dict[str, Path]]:
    """Create pooled and per-dataset data.yaml files mirroring the YOLO pipeline."""

    eval_root.mkdir(parents=True, exist_ok=True)

    pooled_yaml = eval_root / "data.yaml"
    pooled_yaml.write_text(
        yaml.safe_dump(
            {
                "names": class_names,
                "train": str((dataset_dir / "images" / "train").resolve()),
                "val": str((dataset_dir / "images" / "val").resolve()),
                "test": str((dataset_dir / "images" / "test").resolve()),
            }
        ),
        encoding="utf-8",
    )

    per_dataset_yaml: Dict[str, Path] = {}
    test_mapping = build_dataset_test_mapping(splits, dataset_dir)
    for dataset_name, images in sorted(test_mapping.items()):
        subset_dir = eval_root / dataset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        test_file = subset_dir / "test.txt"
        with test_file.open("w", encoding="utf-8") as handle:
            for path in images:
                handle.write(str(path.resolve()) + "\n")
        data_yaml = subset_dir / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "names": class_names,
                    "train": str((dataset_dir / "images" / "train").resolve()),
                    "val": str((dataset_dir / "images" / "val").resolve()),
                    "test": str(test_file.resolve()),
                }
            ),
            encoding="utf-8",
        )
        per_dataset_yaml[dataset_name] = data_yaml
    return pooled_yaml, per_dataset_yaml


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_multirun.yaml"),
        help="Config file used by the comparison pipeline.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        help="Override experiment root. Defaults to value from config (experiment_root).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    exp_root = args.experiment_root or Path(cfg.get("experiment_root", "train_experiments"))

    exp_dir = create_experiment_dir(exp_root)
    logger = setup_logger(exp_dir / "logs" / "train.log")
    logger.info("Created comparison experiment at %s", exp_dir)
    save_config_copy(args.config, exp_dir)

    seed = cfg.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    entries, dataset_sources = collect_entries(cfg, logger)
    data_yaml, splits = prepare_dataset(entries, exp_dir, cfg, logger, dataset_sources)
    dest_dataset = exp_dir / "dataset"
    manifest_path = dest_dataset / "manifest.csv"
    if manifest_path.is_file():
        splits = load_prepared_splits(manifest_path, logger)
    else:
        logger.warning("Manifest %s missing; continuing with in-memory split assignments", manifest_path)
    class_names = cfg.get("classes", ["collembola"])
    test_mapping = build_dataset_test_mapping(splits, dest_dataset)
    visualization_examples = select_visualization_examples(test_mapping, examples_per_dataset=5)
    image_stats = summarize_image_dimensions(splits, logger)
    dataset_size = max(1, int(image_stats.get("train_images") or image_stats.get("total_images") or 1))
    inference_images = gather_inference_images(splits)
    performance: Dict[str, Dict[str, float]] = {}

    coco_root = exp_dir / "dataset" / "coco_annotations"
    coco_paths = convert_yolo_to_coco(dest_dataset, splits, class_names, coco_root, logger)
    subset_coco = build_subset_coco_annotations(splits, class_names, coco_root, logger)
    (coco_root / "meta.yaml").write_text(
        yaml.safe_dump({"names": class_names}), encoding="utf-8"
    )

    model_metrics: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, str] = {}

    if model_active(cfg, "yolo"):
        try:
            yolo_dir = exp_dir / "training" / MODEL_FOLDERS["yolo"]
            yolo_dir.mkdir(parents=True, exist_ok=True)
            train_start = time.perf_counter()
            yolo_run_dir = train_yolo_model(data_yaml, yolo_dir, cfg, logger)
            yolo_train_time = time.perf_counter() - train_start
            yolo_model = Path(yolo_run_dir)

            from ultralytics import YOLO

            yolo = YOLO(str(yolo_model / "weights" / "best.pt"))
            eval_root = exp_dir / "evaluation" / MODEL_FOLDERS["yolo"]
            pooled_yaml, per_dataset_yaml = prepare_eval_manifests(dest_dataset, splits, class_names, eval_root)
            evaluation_summary: Dict[str, Dict[str, float]] = {}

            eval_results = yolo.val(
                data=str(pooled_yaml), split="test", project=str(eval_root), name="pooled_test"
            )
            yolo_metrics = extract_eval_metrics(eval_results)
            write_ultralytics_style_artifacts(eval_root / "pooled_test", yolo_metrics)
            evaluation_summary["pooled"] = yolo_metrics

            for dataset_name, data_yaml_path in per_dataset_yaml.items():
                result = yolo.val(
                    data=str(data_yaml_path),
                    split="test",
                    project=str(eval_root),
                    name=f"{dataset_name}_test",
                )
                metrics = extract_eval_metrics(result)
                write_ultralytics_style_artifacts(eval_root / f"{dataset_name}_test", metrics)
                evaluation_summary[apply_eval_name_map(dataset_name)] = metrics

            save_evaluation_summary(yolo_model, yolo_metrics)
            summary_path = eval_root / "evaluation_summary.json"
            summary_path.write_text(
                json.dumps(remap_evaluation_keys(evaluation_summary), indent=2), encoding="utf-8"
            )
            if not (yolo_model / "results.csv").is_file():
                save_results_csv(yolo_model, yolo_metrics)
            model_metrics[MODEL_FOLDERS["yolo"]] = yolo_metrics
            yolo_cfg = cfg.get("models", {}).get("yolo", cfg.get("model", {}))
            yolo_device = yolo_cfg.get("device", "cpu")
            if visualization_examples:
                render_yolo_visualizations(yolo, visualization_examples, eval_root / "visualizations")
            inference_batch = max(1, int(yolo_cfg.get("inference_batch", yolo_cfg.get("batch", 1))))
            yolo_inference_time = benchmark_inference(
                lambda image: yolo.predict(
                    source=str(image),
                    verbose=False,
                    device=yolo_device,
                    save=False,
                    batch=inference_batch,
                ),
                inference_images,
            )
            performance[MODEL_FOLDERS["yolo"]] = {
                "training_seconds": yolo_train_time,
                "inference_seconds_for_1000_images": yolo_inference_time,
            }
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error(
                "YOLO training/evaluation failed; continuing with remaining models: %s", exc, exc_info=True
            )
            failures[MODEL_FOLDERS["yolo"]] = str(exc)
            model_metrics[MODEL_FOLDERS["yolo"]] = {"error": str(exc)}
            performance[MODEL_FOLDERS["yolo"]] = {"error": str(exc)}
    else:
        logger.info("Skipping YOLO training/evaluation because it is disabled in the config")
        model_metrics[MODEL_FOLDERS["yolo"]] = {"status": "disabled"}
        performance[MODEL_FOLDERS["yolo"]] = {"status": "disabled"}

    if model_active(cfg, "mask_rcnn"):
        try:
            mrcnn_dir = exp_dir / "training" / MODEL_FOLDERS["mask_rcnn"]
            mrcnn_dir.mkdir(parents=True, exist_ok=True)
            train_start = time.perf_counter()
            mrcnn_run_dir, mrcnn_model, mrcnn_eval, mrcnn_cfg = train_mask_rcnn(
                coco_paths, mrcnn_dir, class_names, logger, cfg, dataset_size, image_stats
            )
            mrcnn_train_time = time.perf_counter() - train_start
            mrcnn_metrics = format_metrics_from_coco(mrcnn_eval)
            eval_root_mrcnn = exp_dir / "evaluation" / MODEL_FOLDERS["mask_rcnn"]
            mrcnn_eval_summary = evaluate_detectron_model(
                mrcnn_model,
                mrcnn_cfg,
                "compare_test",
                eval_root_mrcnn,
                class_names,
                subset_coco,
            )
            save_results_csv(mrcnn_run_dir, mrcnn_metrics)
            save_evaluation_summary(mrcnn_run_dir, mrcnn_metrics)
            (eval_root_mrcnn / "evaluation_summary.json").write_text(
                json.dumps(remap_evaluation_keys(mrcnn_eval_summary), indent=2), encoding="utf-8"
            )
            model_metrics[MODEL_FOLDERS["mask_rcnn"]] = mrcnn_metrics
            mrcnn_predictor = build_predictor_with_trained_weights(mrcnn_cfg, logger)
            if visualization_examples:
                render_detectron_visualizations(
                    mrcnn_predictor,
                    visualization_examples,
                    eval_root_mrcnn / "visualizations",
                    class_names,
                    score_threshold=float(
                        getattr(mrcnn_cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", 0.0)
                    ),
                )
            mrcnn_inference_time = benchmark_inference(
                lambda image: mrcnn_predictor(cv2.imread(str(image))),
                inference_images,
            )
            performance[MODEL_FOLDERS["mask_rcnn"]] = {
                "training_seconds": mrcnn_train_time,
                "inference_seconds_for_1000_images": mrcnn_inference_time,
            }
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error(
                "Mask R-CNN training/evaluation failed; continuing with remaining models: %s", exc, exc_info=True
            )
            failures[MODEL_FOLDERS["mask_rcnn"]] = str(exc)
            model_metrics[MODEL_FOLDERS["mask_rcnn"]] = {"error": str(exc)}
            performance[MODEL_FOLDERS["mask_rcnn"]] = {"error": str(exc)}
    else:
        logger.info("Skipping Mask R-CNN training/evaluation because it is disabled in the config")
        model_metrics[MODEL_FOLDERS["mask_rcnn"]] = {"status": "disabled"}
        performance[MODEL_FOLDERS["mask_rcnn"]] = {"status": "disabled"}

    if model_active(cfg, "mask2former"):
        try:
            m2f_dir = exp_dir / "training" / MODEL_FOLDERS["mask2former"]
            m2f_dir.mkdir(parents=True, exist_ok=True)
            train_start = time.perf_counter()
            m2f_run_dir, m2f_model, m2f_eval, m2f_cfg = train_mask2former(
                coco_paths, m2f_dir, class_names, logger, cfg, dataset_size, image_stats
            )
            m2f_train_time = time.perf_counter() - train_start
            m2f_metrics = format_metrics_from_coco(m2f_eval)
            eval_root_m2f = exp_dir / "evaluation" / MODEL_FOLDERS["mask2former"]
            m2f_eval_summary = evaluate_detectron_model(
                m2f_model,
                m2f_cfg,
                "m2f_test",
                eval_root_m2f,
                class_names,
                subset_coco,
            )
            save_results_csv(m2f_run_dir, m2f_metrics)
            save_evaluation_summary(m2f_run_dir, m2f_metrics)
            (eval_root_m2f / "evaluation_summary.json").write_text(
                json.dumps(remap_evaluation_keys(m2f_eval_summary), indent=2), encoding="utf-8"
            )
            model_metrics[MODEL_FOLDERS["mask2former"]] = m2f_metrics
            m2f_predictor = build_predictor_with_trained_weights(m2f_cfg, logger)
            if visualization_examples:
                render_detectron_visualizations(
                    m2f_predictor,
                    visualization_examples,
                    eval_root_m2f / "visualizations",
                    class_names,
                    score_threshold=float(
                        getattr(m2f_cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", 0.0)
                    ),
                )
            m2f_inference_time = benchmark_inference(
                lambda image: m2f_predictor(cv2.imread(str(image))),
                inference_images,
            )
            performance[MODEL_FOLDERS["mask2former"]] = {
                "training_seconds": m2f_train_time,
                "inference_seconds_for_1000_images": m2f_inference_time,
            }
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error(
                "Mask2Former training/evaluation failed; continuing with remaining models: %s",
                exc,
                exc_info=True,
            )
            failures[MODEL_FOLDERS["mask2former"]] = str(exc)
            model_metrics[MODEL_FOLDERS["mask2former"]] = {"error": str(exc)}
            performance[MODEL_FOLDERS["mask2former"]] = {"error": str(exc)}
    else:
        logger.info("Skipping Mask2Former training/evaluation because it is disabled in the config")
        model_metrics[MODEL_FOLDERS["mask2former"]] = {"status": "disabled"}
        performance[MODEL_FOLDERS["mask2former"]] = {"status": "disabled"}

    if model_active(cfg, "maskdino"):
        try:
            mdino_dir = exp_dir / "training" / MODEL_FOLDERS["maskdino"]
            mdino_dir.mkdir(parents=True, exist_ok=True)
            train_start = time.perf_counter()
            mdino_run_dir, mdino_model, mdino_eval, mdino_cfg = train_maskdino(
                coco_paths, mdino_dir, class_names, logger, cfg, dataset_size, image_stats
            )
            mdino_train_time = time.perf_counter() - train_start
            mdino_metrics = format_metrics_from_coco(mdino_eval)
            eval_root_mdino = exp_dir / "evaluation" / MODEL_FOLDERS["maskdino"]
            mdino_eval_summary = evaluate_detectron_model(
                mdino_model,
                mdino_cfg,
                "mdino_test",
                eval_root_mdino,
                class_names,
                subset_coco,
            )
            save_results_csv(mdino_run_dir, mdino_metrics)
            save_evaluation_summary(mdino_run_dir, mdino_metrics)
            (eval_root_mdino / "evaluation_summary.json").write_text(
                json.dumps(remap_evaluation_keys(mdino_eval_summary), indent=2), encoding="utf-8"
            )
            model_metrics[MODEL_FOLDERS["maskdino"]] = mdino_metrics

            mdino_predictor = build_predictor_with_trained_weights(mdino_cfg, logger)
            if visualization_examples:
                render_detectron_visualizations(
                    mdino_predictor,
                    visualization_examples,
                    eval_root_mdino / "visualizations",
                    class_names,
                    score_threshold=float(
                        getattr(mdino_cfg.MODEL.ROI_HEADS, "SCORE_THRESH_TEST", 0.0)
                    ),
                )
            mdino_inference_time = benchmark_inference(
                lambda image: mdino_predictor(cv2.imread(str(image))),
                inference_images,
            )
            performance[MODEL_FOLDERS["maskdino"]] = {
                "training_seconds": mdino_train_time,
                "inference_seconds_for_1000_images": mdino_inference_time,
            }
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error(
                "MaskDINO training/evaluation failed; continuing to final reporting: %s", exc, exc_info=True
            )
            failures[MODEL_FOLDERS["maskdino"]] = str(exc)
            model_metrics[MODEL_FOLDERS["maskdino"]] = {"error": str(exc)}
            performance[MODEL_FOLDERS["maskdino"]] = {"error": str(exc)}
    else:
        logger.info("Skipping MaskDINO training/evaluation because it is disabled in the config")
        model_metrics[MODEL_FOLDERS["maskdino"]] = {"status": "disabled"}
        performance[MODEL_FOLDERS["maskdino"]] = {"status": "disabled"}

    (exp_dir / "performance_metrics.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8"
    )

    collate_final_metrics(model_metrics, exp_dir / "stats")
    if failures:
        logger.warning("Completed with model errors: %s", failures)
    logger.info("Completed multi-model training & evaluation. Outputs stored at %s", exp_dir)


if __name__ == "__main__":
    main()
