#!/usr/bin/env python3
# Usage: python scripts/train_model.py --config configs/train.yaml
# Ownership: Copyright (c) 2026 Adrian Meyer
# License: MIT (code); model weights and dependencies may be under AGPL-3.0.
"""Structured training pipeline for Collembola instance segmentation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image


@dataclass
class DatasetEntry:
    dataset: str
    image_path: Path
    label_path: Path
    unique_name: str
    image_suffix: str
    source_stem: str


@dataclass
class DatasetSource:
    name: str
    images_dir: Path
    labels_dir: Path

    @property
    def reserve_file(self) -> Path:
        return self.images_dir / "tst_reserve.json"


@dataclass
class SizeMismatchInfo:
    count: int = 0
    examples: List[str] = field(default_factory=list)


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def sanitize_name(name: str) -> str:
    keep = [c if c.isalnum() or c in {"_", "-"} else "_" for c in name]
    sanitized = "".join(keep)
    sanitized = sanitized.strip("_") or "sample"
    return sanitized


def create_experiment_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = root / f"exp_train_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"exp_train_{timestamp}_{suffix:02d}"
        suffix += 1

    subdirs = [
        candidate / "logs",
        candidate / "dataset" / "images" / "train",
        candidate / "dataset" / "images" / "val",
        candidate / "dataset" / "images" / "test",
        candidate / "dataset" / "labels" / "train",
        candidate / "dataset" / "labels" / "val",
        candidate / "dataset" / "labels" / "test",
        candidate / "stats",
        candidate / "training",
    ]
    for sub in subdirs:
        sub.mkdir(parents=True, exist_ok=True)
    return candidate


def iter_image_files(folder: Path) -> Iterable[Path]:
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    for pattern in patterns:
        yield from folder.glob(pattern)


def build_unique_name(base: str, used: set) -> str:
    candidate = base
    counter = 1
    while candidate in used:
        candidate = f"{base}_{counter:03d}"
        counter += 1
    used.add(candidate)
    return candidate


def derive_dataset_name(raw_name: str) -> str:
    if not raw_name:
        return sanitize_name("dataset")

    cleaned = raw_name.strip()
    prefixes = ("train-labelme-", "labelme-")
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break
    elif "-" in cleaned:
        cleaned = cleaned.split("-")[-1]

    return sanitize_name(cleaned)


def collect_entries(
    cfg: Dict, logger: logging.Logger
) -> Tuple[List[DatasetEntry], Dict[str, DatasetSource]]:
    datasets_cfg = cfg.get("datasets", [])
    if not datasets_cfg:
        raise ValueError("No datasets defined in configuration.")

    used_names = set()
    entries: List[DatasetEntry] = []
    dataset_sources: Dict[str, DatasetSource] = {}

    for dataset_cfg in datasets_cfg:
        raw_name = dataset_cfg.get("name", "dataset")
        ds_name = derive_dataset_name(raw_name)
        original_name = ds_name
        suffix = 1
        while ds_name in dataset_sources:
            ds_name = sanitize_name(f"{original_name}_{suffix:02d}")
            suffix += 1
        images_dir = Path(dataset_cfg["images"]).expanduser().resolve()
        labels_dir = Path(dataset_cfg["labels"]).expanduser().resolve()

        if not images_dir.is_dir():
            logger.warning(f"Images directory missing: {images_dir}")
            continue
        if not labels_dir.is_dir():
            logger.warning(f"Labels directory missing: {labels_dir}")
            continue

        dataset_sources[ds_name] = DatasetSource(ds_name, images_dir, labels_dir)

        for image_path in sorted(iter_image_files(images_dir)):
            stem = image_path.stem
            label_path = labels_dir / f"{stem}.json"
            if not label_path.is_file():
                label_path = labels_dir / f"{stem}.JSON"
            if not label_path.is_file():
                logger.warning(f"Missing label for {image_path}")
                continue

            unique_base = sanitize_name(f"{ds_name}_{stem}")
            unique_name = build_unique_name(unique_base, used_names)
            entries.append(
                DatasetEntry(
                    dataset=ds_name,
                    image_path=image_path,
                    label_path=label_path,
                    unique_name=unique_name,
                    image_suffix=image_path.suffix.lower() or ".jpg",
                    source_stem=stem,
                )
            )

    if not entries:
        raise RuntimeError("No valid image/label pairs found across datasets.")

    logger.info(f"Collected {len(entries)} samples from {len(datasets_cfg)} dataset(s).")
    return entries, dataset_sources


def convert_labelme_to_yolo(
    label_path: Path,
    class_map: Dict[str, int],
    image_width: int,
    image_height: int,
    logger: logging.Logger | None = None,
) -> str:
    try:
        data = json.loads(label_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive logging
        if logger:
            logger.warning("Failed to parse label file %s: %s", label_path, exc)
        return ""

    shapes = data.get("shapes", [])
    lines: List[str] = []
    def resolve_label(raw_label: str | None) -> str | None:
        """Map a raw Labelme label to a known class name."""
        if not raw_label:
            return None

        if raw_label in class_map:
            return raw_label

        for base_label in class_map:
            if raw_label.startswith(base_label):
                return base_label

        return None

    for shp in shapes:
        label = resolve_label(shp.get("label"))
        if label is None:
            continue

        pts = shp.get("points", [])
        shape_type = (shp.get("shape_type") or "polygon").lower()

        # Labelme rectangles store only two corner points; convert them into a polygon
        if shape_type == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            pts = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ]

        if not pts or len(pts) < 3:
            if logger:
                logger.warning(
                    "Skipping shape in %s with insufficient points (type=%s, points=%s)",
                    label_path,
                    shape_type,
                    len(pts),
                )
            continue

        coords: List[str] = [str(class_map[label])]
        for (x, y) in pts:
            x_norm = float(x) / float(image_width) if image_width else 0.0
            y_norm = float(y) / float(image_height) if image_height else 0.0
            coords.append(f"{x_norm:.6f}")
            coords.append(f"{y_norm:.6f}")
        lines.append(" ".join(coords))
    return "\n".join(lines)


def normalise_device(device_value) -> str | None:
    if device_value is None:
        return None
    if isinstance(device_value, (list, tuple)):
        return ",".join(str(item) for item in device_value)
    if isinstance(device_value, (int, float)):
        return str(int(device_value))
    return str(device_value)


def manage_test_reserve(
    dataset_name: str,
    entries: List[DatasetEntry],
    reserve_file: Path,
    test_ratio: float,
    rng: random.Random,
    logger: logging.Logger,
) -> Tuple[List[DatasetEntry], List[DatasetEntry]]:
    """Return (test_entries, remaining_entries) for a dataset."""
    stems = [entry.source_stem for entry in entries]
    available = {entry.source_stem: entry for entry in entries}

    selected_stems: List[str] = []
    if reserve_file.is_file():
        try:
            stored = json.loads(reserve_file.read_text(encoding="utf-8"))
            if not isinstance(stored, list):
                raise ValueError("Reserve file must contain a list of stems")
        except Exception as exc:  # pragma: no cover - defensive log
            logger.warning(
                "Failed to read TST reserve for %s (%s). Recreating.",
                dataset_name,
                exc,
            )
            stored = []

        selected_stems = [stem for stem in stored if stem in available]
        removed = set(stored) - set(selected_stems)
        if removed:
            logger.warning(
                "Dropping %s missing TST samples for %s", len(removed), dataset_name
            )
            reserve_file.write_text(
                json.dumps(selected_stems, indent=2), encoding="utf-8"
            )

    desired = int(round(len(entries) * max(test_ratio, 0.0)))
    desired = min(desired, len(entries))

    if len(selected_stems) < desired:
        candidates = [stem for stem in stems if stem not in selected_stems]
        rng.shuffle(candidates)
        needed = desired - len(selected_stems)
        selected_stems.extend(candidates[:needed])
        reserve_file.write_text(json.dumps(selected_stems, indent=2), encoding="utf-8")
        logger.info(
            "Reserved %s TST samples for dataset %s at %s",
            len(selected_stems),
            dataset_name,
            reserve_file,
        )
    elif not reserve_file.exists():
        reserve_file.write_text(json.dumps(selected_stems, indent=2), encoding="utf-8")

    selected_set = set(selected_stems)
    test_entries = [available[stem] for stem in selected_stems if stem in available]
    remaining_entries = [entry for entry in entries if entry.source_stem not in selected_set]
    return test_entries, remaining_entries


def prepare_dataset(
    entries: List[DatasetEntry],
    exp_dir: Path,
    cfg: Dict,
    logger: logging.Logger,
    dataset_sources: Dict[str, DatasetSource],
) -> Tuple[Path, Dict[str, List[DatasetEntry]]]:
    dataset_dir = exp_dir / "dataset"
    image_cfg = cfg.get("image", {})
    target_size = image_cfg.get("target_size")
    expected_size = int(target_size) if target_size is not None else None

    class_names = cfg.get("classes", ["collembola"])
    class_map = {name: idx for idx, name in enumerate(class_names)}

    if "collembola" in class_map and "insect" not in class_map:
        class_map["insect"] = class_map["collembola"]

    splits_cfg = cfg.get("splits", {"train": 0.7, "val": 0.2, "test": 0.1})
    if not math.isclose(sum(splits_cfg.values()), 1.0, rel_tol=1e-3):
        logger.warning("Splits do not sum to 1. Using proportional allocation with remainder to test.")

    rng = random.Random(cfg.get("random_seed", 42))

    dataset_to_entries: Dict[str, List[DatasetEntry]] = defaultdict(list)
    for entry in entries:
        dataset_to_entries[entry.dataset].append(entry)
    for subset_entries in dataset_to_entries.values():
        rng.shuffle(subset_entries)

    split_assignments: Dict[str, List[DatasetEntry]] = {"train": [], "val": [], "test": []}
    train_ratio = splits_cfg.get("train", 0.0)
    val_ratio = splits_cfg.get("val", 0.0)
    test_ratio = splits_cfg.get("test", 0.0)
    train_val_total = train_ratio + val_ratio

    for dataset_name, subset_entries in dataset_to_entries.items():
        source = dataset_sources.get(dataset_name)
        if source is None:
            logger.warning("Missing dataset source metadata for %s", dataset_name)
            continue

        test_entries, remaining_entries = manage_test_reserve(
            dataset_name,
            subset_entries,
            source.reserve_file,
            test_ratio,
            rng,
            logger,
        )

        remaining_count = len(remaining_entries)
        if train_val_total <= 0:
            train_entries = remaining_entries
            val_entries: List[DatasetEntry] = []
        else:
            train_share = train_ratio / train_val_total
            train_count = int(round(remaining_count * train_share))
            train_entries = remaining_entries[:train_count]
            val_entries = remaining_entries[train_count:]

        split_assignments["train"].extend(train_entries)
        split_assignments["val"].extend(val_entries)
        split_assignments["test"].extend(test_entries)

        logger.info(
            "Dataset %s => train %s, val %s, test %s",
            dataset_name,
            len(train_entries),
            len(val_entries),
            len(test_entries),
        )

    logger.info(
        "Split counts => train: %s, val: %s, test: %s",
        len(split_assignments["train"]),
        len(split_assignments["val"]),
        len(split_assignments["test"]),
    )

    manifest_rows = []

    size_mismatch_tracker: Dict[Tuple[int, int], SizeMismatchInfo] = {}
    max_warning_examples = 5

    for split, samples in split_assignments.items():
        logger.info(f"Preparing {len(samples)} samples for split '{split}'.")
        for entry in samples:
            dest_img = (
                dataset_dir
                / "images"
                / split
                / f"{entry.unique_name}{entry.image_suffix}"
            )
            dest_lbl = dataset_dir / "labels" / split / f"{entry.unique_name}.txt"
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            dest_lbl.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(entry.image_path) as img:
                orig_w, orig_h = img.size

            if expected_size is not None and (orig_w != expected_size or orig_h != expected_size):
                key = (orig_w, orig_h)
                entry_info = size_mismatch_tracker.setdefault(key, SizeMismatchInfo())
                entry_info.count += 1
                if len(entry_info.examples) < max_warning_examples:
                    entry_info.examples.append(str(entry.image_path))

            shutil.copy2(entry.image_path, dest_img)

            yolo_txt = convert_labelme_to_yolo(
                entry.label_path,
                class_map=class_map,
                image_width=orig_w,
                image_height=orig_h,
                logger=logger,
            )
            dest_lbl.write_text(yolo_txt, encoding="utf-8")

            manifest_rows.append(
                {
                    "split": split,
                    "unique_name": entry.unique_name,
                    "source_dataset": entry.dataset,
                    "source_image": str(entry.image_path),
                    "source_label": str(entry.label_path),
                    "prepared_image": str(dest_img),
                    "prepared_label": str(dest_lbl),
                }
            )

    if size_mismatch_tracker:
        total_mismatches = sum(info.count for info in size_mismatch_tracker.values())
        logger.info(
            "Detected %s images that differ from expected size %s. Proceeding without resizing.",
            total_mismatches,
            expected_size,
        )
        for (width, height), info in sorted(
            size_mismatch_tracker.items(), key=lambda item: item[1].count, reverse=True
        ):
            examples = info.examples
            suffix = "..." if info.count > len(examples) else ""
            logger.debug(
                " • %s images sized %sx%s. Examples: %s%s",
                info.count,
                width,
                height,
                ", ".join(examples),
                suffix,
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = dataset_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    logger.info(f"Saved manifest => {manifest_path}")

    split_summary = {split: len(samples) for split, samples in split_assignments.items()}
    (dataset_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )

    data_yaml = dataset_dir / "data.yaml"
    content = {
        "train": str((dataset_dir / "images" / "train").resolve()),
        "val": str((dataset_dir / "images" / "val").resolve()),
        "test": str((dataset_dir / "images" / "test").resolve()),
        "names": class_names,
    }
    data_yaml.write_text(yaml.safe_dump(content), encoding="utf-8")
    logger.info(f"Wrote data.yaml => {data_yaml}")

    return data_yaml, split_assignments


def load_prepared_splits(
    manifest_path: Path, logger: logging.Logger
) -> Dict[str, List[DatasetEntry]]:
    df = pd.read_csv(manifest_path)
    required_cols = {"split", "unique_name", "source_dataset", "prepared_image"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Manifest at {manifest_path} is missing required columns: {sorted(missing)}"
        )

    splits: Dict[str, List[DatasetEntry]] = defaultdict(list)
    for _, row in df.iterrows():
        split = str(row["split"])
        unique_name = str(row["unique_name"])
        dataset = str(row["source_dataset"])
        prepared_image = Path(row["prepared_image"])
        image_suffix = prepared_image.suffix or ".jpg"

        source_image = row.get("source_image")
        source_stem = (
            Path(source_image).stem if isinstance(source_image, str) and source_image else unique_name
        )
        prepared_label = row.get("prepared_label")
        label_path = (
            Path(prepared_label)
            if isinstance(prepared_label, str) and prepared_label
            else prepared_image.with_suffix(".txt")
        )

        entry = DatasetEntry(
            dataset=dataset,
            image_path=prepared_image,
            label_path=label_path,
            unique_name=unique_name,
            image_suffix=image_suffix,
            source_stem=source_stem,
        )
        splits[split].append(entry)

    for split_name in ("train", "val", "test"):
        splits.setdefault(split_name, [])

    logger.info("Loaded %s manifest entries from %s", len(df), manifest_path)
    return splits


def run_training(
    cfg: Dict,
    exp_dir: Path,
    data_yaml: Path,
    logger: logging.Logger,
) -> Path:
    from ultralytics import YOLO

    model_cfg = cfg.get("model", {})
    training_cfg = cfg.get("training", {})
    image_cfg = cfg.get("image", {})
    target_size = int(image_cfg.get("target_size", 640))

    model_checkpoint = model_cfg.get("checkpoint")
    if not model_checkpoint:
        raise ValueError("Model checkpoint path/name must be provided in configuration.")

    logger.info(f"Loading model checkpoint: {model_checkpoint}")
    model = YOLO(model_checkpoint)

    project_dir = exp_dir / "training"
    run_name = cfg.get("output", {}).get("project_name", "yolo")

    device_value = training_cfg.get("device", model_cfg.get("device"))
    normalised_device = normalise_device(device_value)

    train_kwargs = {
        "data": str(data_yaml),
        "epochs": training_cfg.get("epochs", 90),
        "batch": training_cfg.get("batch", 16),
        "imgsz": training_cfg.get("imgsz", target_size),
        "project": str(project_dir),
        "name": run_name,
    }

    if normalised_device:
        train_kwargs["device"] = normalised_device

    # Pass remaining parameters directly to YOLO.train
    excluded = {"epochs", "batch", "imgsz", "device"}
    for key, value in training_cfg.items():
        if key in excluded:
            continue
        train_kwargs[key] = value

    logger.info("Starting training with parameters:")
    for key, value in sorted(train_kwargs.items()):
        if key == "data":
            continue
        logger.info(f"  {key}: {value}")

    results = model.train(**train_kwargs)

    run_dir = project_dir / run_name
    if hasattr(results, "save_dir"):
        run_dir = Path(results.save_dir)
    logger.info(f"Training artifacts saved to {run_dir}")
    return run_dir


def summarise_training(
    run_dir: Path,
    exp_dir: Path,
    splits: Dict[str, List[DatasetEntry]],
    cfg: Dict,
    logger: logging.Logger,
) -> None:
    stats_dir = exp_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    results_csv = run_dir / "results.csv"
    summary: Dict[str, object] = {
        "run_directory": str(run_dir.resolve()),
        "best_weights": str((run_dir / "weights" / "best.pt").resolve()),
        "last_weights": str((run_dir / "weights" / "last.pt").resolve()),
        "split_counts": {split: len(samples) for split, samples in splits.items()},
        "classes": cfg.get("classes", ["collembola"]),
    }

    if results_csv.is_file():
        df = pd.read_csv(results_csv)
        copied_csv = stats_dir / "training_metrics.csv"
        df.to_csv(copied_csv, index=False)
        logger.info(f"Copied training metrics => {copied_csv}")

        metric_cols = [col for col in df.columns if col.startswith("metrics/")]
        best_metrics = {}
        if metric_cols:
            for col in metric_cols:
                best_metrics[col] = float(df[col].max())
        summary["metrics"] = best_metrics

        last_row = df.iloc[-1].to_dict()
        summary["final_epoch_metrics"] = last_row
    else:
        logger.warning(f"results.csv not found in {run_dir}")

    summary_path = stats_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"Summary written to {summary_path}")


def extract_eval_metrics(result_obj) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if result_obj is None:
        return metrics

    candidate = getattr(result_obj, "results_dict", None)
    if isinstance(candidate, dict):
        return candidate

    metrics_obj = getattr(result_obj, "metrics", None)
    if metrics_obj is not None:
        dict_candidate = getattr(metrics_obj, "results_dict", None)
        if isinstance(dict_candidate, dict):
            return dict_candidate

    if isinstance(result_obj, dict):
        return result_obj

    return metrics


def run_evaluation(
    cfg: Dict,
    exp_dir: Path,
    run_dir: Path,
    splits: Dict[str, List[DatasetEntry]],
    logger: logging.Logger,
    dataset_dir: Path | None = None,
) -> None:
    from ultralytics import YOLO

    weights_path = run_dir / "weights" / "best.pt"
    if not weights_path.is_file():
        logger.warning("Best weights not found at %s. Skipping evaluation.", weights_path)
        return

    class_names = cfg.get("classes", ["collembola"])
    dataset_dir = dataset_dir or exp_dir / "dataset"
    eval_root = exp_dir / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    evaluation_summary: Dict[str, Dict[str, float]] = {}
    overall_data_yaml = dataset_dir / "data.yaml"

    logger.info("Running pooled test evaluation on reserved TST data.")
    pooled_results = model.val(
        data=str(overall_data_yaml),
        split="test",
        project=str(eval_root),
        name="pooled_test",
    )
    evaluation_summary["pooled"] = extract_eval_metrics(pooled_results)

    test_entries = splits.get("test", [])
    per_dataset: Dict[str, List[DatasetEntry]] = defaultdict(list)
    for entry in test_entries:
        per_dataset[entry.dataset].append(entry)

    for dataset_name, dataset_entries in sorted(per_dataset.items()):
        if not dataset_entries:
            continue

        subset_dir = eval_root / dataset_name
        subset_dir.mkdir(parents=True, exist_ok=True)
        test_file = subset_dir / "test.txt"
        with test_file.open("w", encoding="utf-8") as handle:
            for entry in dataset_entries:
                image_path = (
                    dataset_dir
                    / "images"
                    / "test"
                    / f"{entry.unique_name}{entry.image_suffix}"
                )
                handle.write(str(image_path.resolve()) + "\n")

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

        logger.info("Evaluating reserved TST data for dataset %s", dataset_name)
        result = model.val(
            data=str(data_yaml),
            split="test",
            project=str(eval_root),
            name=f"{dataset_name}_test",
        )
        evaluation_summary[dataset_name] = extract_eval_metrics(result)

    summary_path = eval_root / "evaluation_summary.json"
    summary_path.write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")
    logger.info("Saved evaluation summary to %s", summary_path)


def save_config_copy(cfg_path: Path, exp_dir: Path) -> None:
    destination = exp_dir / "config_used.yaml"
    shutil.copy2(cfg_path, destination)


def main(config_path: Path, eval_only: Path | None = None) -> None:
    cfg = load_config(config_path)
    exp_root = Path(cfg.get("experiment_root", "train_experiments")).expanduser()

    if eval_only:
        source_exp = eval_only.expanduser().resolve()
        if not source_exp.is_dir():
            raise FileNotFoundError(f"Eval-only experiment directory not found: {source_exp}")

        exp_dir = create_experiment_dir(exp_root)
        logger = setup_logger(exp_dir / "logs" / "train.log")

        logger.info("Evaluation-only mode. Source experiment: %s", source_exp)
        save_config_copy(config_path, exp_dir)

        dataset_dir = source_exp / "dataset"
        manifest_path = dataset_dir / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found at {manifest_path} in eval-only mode.")

        splits = load_prepared_splits(manifest_path, logger)
        if not splits.get("test"):
            logger.warning("No test samples found in manifest for evaluation-only run.")

        summary_path = source_exp / "stats" / "summary.json"
        run_dir: Path | None = None
        if summary_path.is_file():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                candidate = summary_data.get("run_directory")
                if candidate:
                    run_dir_candidate = Path(candidate)
                    if run_dir_candidate.is_dir():
                        run_dir = run_dir_candidate
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to read summary.json for run directory: %s", exc)

        if run_dir is None:
            default_run = cfg.get("output", {}).get("project_name", "yolo")
            run_dir = source_exp / "training" / default_run

        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found for evaluation-only mode: {run_dir}")

        run_evaluation(cfg, exp_dir, run_dir, splits, logger, dataset_dir=dataset_dir)
        logger.info("Evaluation-only pipeline finished successfully. Outputs stored at %s", exp_dir)
        return

    exp_dir = create_experiment_dir(exp_root)
    logger = setup_logger(exp_dir / "logs" / "train.log")

    logger.info(f"Experiment directory: {exp_dir}")
    save_config_copy(config_path, exp_dir)

    seed = cfg.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    entries, dataset_sources = collect_entries(cfg, logger)
    data_yaml, splits = prepare_dataset(entries, exp_dir, cfg, logger, dataset_sources)
    run_dir = run_training(cfg, exp_dir, data_yaml, logger)
    summarise_training(run_dir, exp_dir, splits, cfg, logger)
    run_evaluation(cfg, exp_dir, run_dir, splits, logger)
    logger.info("Training pipeline finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collembola training pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to the training configuration file.",
    )
    parser.add_argument(
        "-evalonly",
        "--evalonly",
        type=Path,
        help=(
            "Re-run evaluation using an existing experiment directory. "
            "Outputs are written to a new experiment folder."
        ),
    )
    args = parser.parse_args()
    main(args.config, args.evalonly)
