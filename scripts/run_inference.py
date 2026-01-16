#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Usage: python scripts/run_inference.py --config configs/inference.yaml
# Ownership: Copyright (c) 2026 Adrian Meyer
# License: MIT (code); model weights and dependencies may be under AGPL-3.0.

import os
import math
import json
import csv
import glob
import pickle
import logging
from pathlib import Path
import sys
import select
import shutil
from collections import defaultdict
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import warnings

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

import torch
torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, Point, box, MultiPoint
from shapely.ops import unary_union
from shapely.validation import make_valid

import yaml
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import cv2

def find_main_circle(img_path):
    """
    Detect the main dish/glass circle.

    Strategy:
    1) Segment the muddy liquid region in HSV space (S high, V medium).
    2) Morphologically clean the mask.
    3) Compute the mask's center of mass.
    4) Run HoughCircles on the grayscale image masked by this region.
    5) From all Hough circles, keep only those
       - with radius in a plausible range, and
       - whose center is close to the mask center.
    6) Choose the candidate closest to the mask center, enlarge radius slightly.
    """
    if img_path is None:
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    mindim = min(h, w)

    # --- 1) Color-based segmentation of the muddy liquid ----------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # mask: reasonably saturated and not too bright or too dark
    mask = (s_ch > ROI_S_MIN) & (v_ch > ROI_V_MIN) & (v_ch < ROI_V_MAX)
    mask_u8 = mask.astype("uint8") * 255

    # morphological closing/opening to fill small gaps and remove noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ROI_KERNEL_SIZE, ROI_KERNEL_SIZE)
    )
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        # segmentation failed => no ROI for this slide
        return None

    cx_mask = float(xs.mean())
    cy_mask = float(ys.mean())

    # --- 2) Hough on masked edges --------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    gray_masked = cv2.bitwise_and(gray_blur, gray_blur, mask=mask_u8)

    min_radius = int(mindim * 0.30)
    max_radius = int(mindim * 0.60)

    circles = cv2.HoughCircles(
        gray_masked,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=h / 2,
        param1=120,
        param2=20,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    candidates = []
    for x_f, y_f, r_f in circles[0]:
        x = int(round(x_f))
        y = int(round(y_f))
        r = int(round(r_f))

        # keep only reasonably central centers
        if not (w * CIRCLE_CENTER_MARGIN <= x <= w * (1.0 - CIRCLE_CENTER_MARGIN)):
            continue
        if not (h * CIRCLE_CENTER_MARGIN <= y <= h * (1.0 - CIRCLE_CENTER_MARGIN)):
            continue

        rel_r = r / float(mindim)
        if not (CIRCLE_RADIUS_MIN_REL <= rel_r <= CIRCLE_RADIUS_MAX_REL):
            continue

        # distance of this center to the mask center (normalized by image size)
        dist_norm = math.hypot(x - cx_mask, y - cy_mask) / float(mindim)
        if dist_norm > 0.25:
            # too far away from mask center => likely spurious arc
            continue

        candidates.append((x, y, r, dist_norm))

    if not candidates:
        # nothing reliable => no ROI masking
        return None

    # choose the candidate closest to the mask center, prefer larger radius if tie
    x, y, r, _ = min(candidates, key=lambda c: (c[3], -c[2]))

    r = int(r * CIRCLE_RADIUS_SCALE)
    return int(x), int(y), int(r)


def point_inside_circle(px, py, circle):
    if circle is None:
        return True
    cx, cy, r = circle
    return (px - cx)**2 + (py - cy)**2 <= r*r


def filter_polygons_by_circle(polys, circle):
    if circle is None:
        return polys
    filtered = []
    for p in polys:
        geom = p.get("polygon")
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        if point_inside_circle(c.x, c.y, circle):
            filtered.append(p)
    return filtered

# ------------------------------------------------------------------------------
#                         CONFIGURATION & PATH SETUP
# ------------------------------------------------------------------------------

CONFIG_PATH = Path("configs/inference.yaml")
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

IMAGES_FOLDER = Path(CFG["images_folder"])               # folder with .jpg images
EXPERIMENTS_DIR = Path(CFG.get("experiments_dir", "exps"))

MODEL_DIR = Path(CFG["model"]["folder"])
MODEL_ARCHIVE = CFG["model"].get("archive")
MODEL_WEIGHTS_NAME = CFG["model"]["weights"]
FUSE_MODEL = CFG["model"].get("fuse", False)

TILE_SIZE = CFG["tile"]["size"]
SHIFT = CFG["tile"].get("shift", TILE_SIZE // 2)

N_GPUS = CFG["parallel"]["n_gpus"]
N_JOBS = CFG["parallel"]["n_jobs"]
BATCH_SIZE = CFG["parallel"].get("batch_size", 16)

BEST_CONF = CFG["fusion"]["best_conf"]
BEST_IOU = CFG["fusion"]["best_iou"]
ALPHA = CFG["fusion"].get("alpha", 45.0)
BEST_METHOD = "graphcut"
SKIP_BRIGHT = CFG.get("skip_bright", True)
SKIP_BLUE = CFG.get("skip_blue", True)
CIRCLE_CENTER_MARGIN = 0.20   # acceptable center deviation
CIRCLE_RADIUS_MIN_REL = 0.35  # min radius relative to min(image_dim)
CIRCLE_RADIUS_MAX_REL = 0.52  # max radius relative to min(image_dim)
CIRCLE_RADIUS_SCALE   = 1.05  # enlarge radius to avoid losing valid area
ROI_S_MIN = 40                # minimum saturation
ROI_V_MIN = 60                # lower bound for value
ROI_V_MAX = 210               # upper bound for value
ROI_KERNEL_SIZE = 21          # morphology kernel size (pixels)

# suppress noisy runtime warnings from Shapely intersections
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in intersection",
    category=RuntimeWarning,
)


def create_experiment_dir(root: Path) -> Path:
    root.mkdir(exist_ok=True)
    existing = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("exp_")]
    nums = [int(p.name.split("_")[1]) for p in existing if p.name.split("_")[1].isdigit()]
    next_num = max(nums, default=0) + 1
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"exp_{next_num:04d}_{dt}"
    exp_dir = root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "tiles" / "ori").mkdir(parents=True, exist_ok=True)
    (exp_dir / "tiles" / "shift").mkdir(parents=True, exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "final_viz").mkdir(exist_ok=True)
    (exp_dir / "final_viz_ori").mkdir(exist_ok=True)
    (exp_dir / "final_viz_shift").mkdir(exist_ok=True)
    (exp_dir / "stats").mkdir(exist_ok=True)
    (exp_dir / "labelme" / "ori").mkdir(parents=True, exist_ok=True)
    (exp_dir / "labelme" / "shift").mkdir(parents=True, exist_ok=True)
    return exp_dir


if "EXP_DIR" in os.environ:
    EXP_DIR = Path(os.environ["EXP_DIR"])
    _MAIN_PROC = False
else:
    EXP_DIR = create_experiment_dir(EXPERIMENTS_DIR)
    os.environ["EXP_DIR"] = str(EXP_DIR)
    _MAIN_PROC = True

TILE_FOLDER = EXP_DIR / "tiles"
RESULTS_FOLDER = EXP_DIR / "results"
INFERENCE_SOURCE_FOLDER = RESULTS_FOLDER
FINAL_VIZ = EXP_DIR / "final_viz"
FINAL_VIZ_ORI = EXP_DIR / "final_viz_ori"
FINAL_VIZ_SHIFT = EXP_DIR / "final_viz_shift"
LOG_FILE = EXP_DIR / "logs" / "pipeline.log"
OUTPUT_CSV = EXP_DIR / "stats" / "pipeline_results.csv"
STATS_DIR = EXP_DIR / "stats"
TILES_SOURCE_FILE = EXP_DIR / "tiles_source.txt"
INFERENCE_SOURCE_FILE = EXP_DIR / "inference_source.txt"
LABELME_DIR = EXP_DIR / "labelme"
TILE_META_PICKLE = EXP_DIR / "tile_meta.pkl"

FINAL_VIZ.mkdir(exist_ok=True)
FINAL_VIZ_ORI.mkdir(exist_ok=True)
FINAL_VIZ_SHIFT.mkdir(exist_ok=True)

# record source image folder for potential tile reuse (only once)
if _MAIN_PROC:
    (EXP_DIR / "source_folder.txt").write_text(str(IMAGES_FOLDER))

# store original image paths by index
IMAGE_PATHS = {}


def resolve_image_path(img_idx):
    """Return the image path for a slide id, trying common int/str variants."""
    candidates = []

    def _add_if_new(val):
        if val not in candidates:
            candidates.append(val)

    _add_if_new(img_idx)

    if isinstance(img_idx, str):
        sanitized = sanitize_index(img_idx)
        _add_if_new(sanitized)
        if sanitized.isdigit():
            _add_if_new(int(sanitized))
    elif isinstance(img_idx, (int, np.integer)):
        _add_if_new(str(int(img_idx)))

    for cand in candidates:
        if cand in IMAGE_PATHS:
            return IMAGE_PATHS[cand]
    return None


def collect_image_paths():
    """Populate IMAGE_PATHS with all original images."""
    all_imgs = sorted(
        list(Path(IMAGES_FOLDER).rglob("*.jpg"))
        + list(Path(IMAGES_FOLDER).rglob("*.JPG"))
    )
    for imgp in all_imgs:
        # Normalise the image stem in the same way as during tiling so that
        # lookups in ``IMAGE_PATHS`` match the slide identifiers produced by
        # the tiling and polygon‑building steps. Without this sanitisation the
        # final visualisation stage fails to locate the source image and skips
        # generating overlay files.
        stem = sanitize_index(imgp.stem.replace("_counted", ""))
        try:
            idx_ = int(stem)
        except Exception:
            idx_ = stem
        IMAGE_PATHS[idx_] = imgp

# counter for invalid intersection warnings
INVALID_INTERSECTIONS = 0

################################################################################
#                         SETUP LOGGING
################################################################################
def setup_logger(logfile):
    """Simple logger config."""
    logger = logging.getLogger("PipelineLogger")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # File handler
    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger

LOGGER = setup_logger(LOG_FILE)


def ask_with_timeout(prompt: str, timeout: int = 10, default: str = "n") -> str:
    """Read user input with timeout; return default if no input."""
    print(prompt, end="", flush=True)
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if r:
        return sys.stdin.readline().strip().lower()
    else:
        print()
        return default.lower()

def ensure_model_weights(model_dir: Path, archive_name: str, weights_name: str) -> Path:
    """Ensure the YOLO weight file exists, extracting a .7z archive if needed."""
    weights_path = model_dir / weights_name

    # If the weights are already present we are done.
    if weights_path.is_file():
        LOGGER.info(f"[MODEL] Using existing weights at {weights_path}")
        return weights_path

    # If an archive was provided, attempt extraction.
    if archive_name:
        # Support multi‑part archives such as 'best.pt.7z.001'.
        first_part = model_dir / (archive_name + ".001")
        archive_path = first_part if first_part.is_file() else (model_dir / archive_name)

        if archive_path.is_file():
            LOGGER.info(f"[MODEL] Extracting {archive_path.name} into {model_dir}")
            import py7zr
            with py7zr.SevenZipFile(archive_path, mode="r") as z:
                z.extractall(path=model_dir)

    # After extraction the weight file must exist.
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    return weights_path


MODEL_WEIGHTS_PATH = ensure_model_weights(MODEL_DIR, MODEL_ARCHIVE, MODEL_WEIGHTS_NAME)


################################################################################
#                        STEP 1: TILING
################################################################################

def is_bright(img_pil, brightness_threshold=128):
    arr = np.array(img_pil.convert("L"), dtype=np.float32)
    return arr.mean() > brightness_threshold

def is_blueish_hue(img_pil, hue_range=(180, 290), sat_threshold=60):
    hsv_img = img_pil.convert("HSV")
    arr = np.array(hsv_img, dtype=np.uint8).reshape(-1, 3)
    mean_h = arr[:, 0].mean()
    mean_s = arr[:, 1].mean()
    mean_h_deg = (mean_h / 255.) * 360.
    if (hue_range[0] <= mean_h_deg <= hue_range[1]) and (mean_s >= sat_threshold):
        return True
    return False

def should_skip_tile(tile_path):
    """Return reason for skipping tile if brightness/hue filters trigger."""
    with Image.open(tile_path) as img:
        if SKIP_BRIGHT and is_bright(img):
            return "bright"
        if SKIP_BLUE and is_blueish_hue(img):
            return "blue"
    return None

def sanitize_index(idx):
    """Replace underscores with dashes so parsing by underscores works."""
    return str(idx).replace("_", "-")


NATURAL_TOKEN_RE = re.compile(r"(\d+)")


def _natural_tokenize(value_str: str):
    """Return a tuple capturing alternating text/numeric chunks for stable ordering."""
    tokens = []
    for chunk in NATURAL_TOKEN_RE.split(value_str):
        if not chunk:
            continue
        if chunk.isdigit():
            tokens.append((0, int(chunk)))
        else:
            tokens.append((1, chunk.lower()))
    if not tokens:
        tokens.append((1, value_str.lower()))
    return tuple(tokens)


def slide_sort_key(value):
    """Sort slide identifiers, handling integers, dotted numbers, and text naturally."""
    if isinstance(value, (int, np.integer)):
        return (0, (int(value),))

    value_str = str(value).strip()

    if value_str.isdigit():
        return (0, (int(value_str),))

    if "." in value_str:
        dot_parts = value_str.split(".")
        if all(part.isdigit() for part in dot_parts):
            return (0, tuple(int(part) for part in dot_parts))

    return (1, _natural_tokenize(value_str))

def tile_image(img_path: Path, img_idx, out_subfolder: str):
    """
    Cut a single image into TILE_SIZE squares.
    out_subfolder = "ori" => offset (0,0)
    out_subfolder = "shift" => offset (SHIFT, SHIFT)
    """
    with Image.open(img_path) as im:
        w, h = im.size

    offset_x, offset_y = (0, 0) if out_subfolder == "ori" else (SHIFT, SHIFT)
    x_tiles = (w - offset_x) // TILE_SIZE
    y_tiles = (h - offset_y) // TILE_SIZE

    out_dir = Path(TILE_FOLDER) / out_subfolder
    out_dir.mkdir(exist_ok=True, parents=True)

    img_idx = sanitize_index(img_idx)
    for xx in range(x_tiles):
        for yy in range(y_tiles):
            x0 = offset_x + xx*TILE_SIZE
            y0 = offset_y + yy*TILE_SIZE
            x1 = x0 + TILE_SIZE
            y1 = y0 + TILE_SIZE
            with Image.open(img_path) as im:
                crop = im.crop((x0, y0, x1, y1))
            # ``img_idx`` may be a string for images that do not follow a
            # numeric naming convention. Formatting such values with the
            # ``:03d`` specifier raises ``ValueError``. Convert the index to
            # string and zero‑pad numeric indices to preserve the previous
            # behaviour while avoiding errors for non‑numeric names.
            idx_fmt = str(img_idx).zfill(3)
            tile_name = f"{out_subfolder}_{idx_fmt}_{xx:03d}_{yy:03d}.jpg"
            tile_path = out_dir / tile_name
            crop.save(tile_path)

def create_all_tiles_if_needed():
    """For each JPG in IMAGES_FOLDER, check if tiles exist; if not, tile them."""
    Path(TILE_FOLDER).mkdir(exist_ok=True)

    all_imgs = sorted(list(Path(IMAGES_FOLDER).rglob("*.jpg")) +
                      list(Path(IMAGES_FOLDER).rglob("*.JPG")))
    needed = []

    for imgp in all_imgs:
        with Image.open(imgp) as im:
            w, h = im.size

        # ``stem`` may contain underscores from the original filename. Replace
        # them with dashes so that later underscore-based parsing splits into
        # the expected grid/slide/x/y parts.
        stem = sanitize_index(imgp.stem.replace("_counted", ""))
        try:
            idx_ = int(stem)
        except:
            idx_ = stem
        idx_ = sanitize_index(idx_)

        xt_o = w // TILE_SIZE
        yt_o = h // TILE_SIZE
        expected_ori = xt_o * yt_o

        xt_s = (w - SHIFT) // TILE_SIZE
        yt_s = (h - SHIFT) // TILE_SIZE
        expected_shift = xt_s * yt_s

        # ``idx_`` can be either int or str; use the same formatting strategy as
        # in ``tile_image`` to ensure globs match the generated tile names.
        idx_glob = str(idx_).zfill(3)
        found_ori = len(list(Path(TILE_FOLDER, "ori").glob(f"ori_{idx_glob}_*.jpg")))
        found_sft = len(list(Path(TILE_FOLDER, "shift").glob(f"shift_{idx_glob}_*.jpg")))

        if (found_ori < expected_ori) or (found_sft < expected_shift):
            needed.append((imgp, idx_))

    if not needed:
        LOGGER.info("[TILING] All images appear tiled => skip.")
        return
    else:
        LOGGER.info(f"[TILING] Re-tiling {len(needed)} incomplete images…")

    def process_one(args):
        tifp, idx = args
        tile_image(tifp, idx, "ori")
        tile_image(tifp, idx, "shift")

    with tqdm_joblib(tqdm(total=len(needed), desc="Tiling")):
        Parallel(n_jobs=N_JOBS)(delayed(process_one)(tp) for tp in needed)


def maybe_reuse_previous_tiles():
    """Prompt to reuse tiles from the most recent previous experiment.

    Returns True if tiles from a previous experiment are reused, otherwise False.
    """

    global TILE_FOLDER

    exps = sorted(
        [p for p in EXPERIMENTS_DIR.glob("exp_*") if p.is_dir() and p != EXP_DIR]
    )

    if exps:
        prev_exp = exps[-1]

        # Determine the actual tile directory of the previous experiment. If that
        # experiment already reused tiles, follow its recorded source path.
        prev_src_file = prev_exp / "tiles_source.txt"
        if prev_src_file.is_file():
            prev_tiles = Path(prev_src_file.read_text().strip())
        else:
            prev_tiles = prev_exp / "tiles"

        if prev_tiles.exists() and any(prev_tiles.rglob("*.jpg")):
            prev_src = prev_exp / "source_folder.txt"
            src_path = prev_src.read_text().strip() if prev_src.is_file() else "unknown"
            resp = ask_with_timeout(
                f"Existing tiles found in the last experiment ({prev_exp.name}) derived from folder ({src_path}). "
                "Do you want to reuse them [y] or regenerate new ones [n]? ",
                timeout=10,
                default="n",
            )

            if resp and resp[0] == 'y':
                LOGGER.info(f"[TILING] Reusing tiles from {prev_exp.name}")
                TILE_FOLDER = prev_tiles
                TILES_SOURCE_FILE.write_text(str(TILE_FOLDER))
                return True

    # No reuse; record that tiles come from the current experiment directory
    TILES_SOURCE_FILE.write_text(str(TILE_FOLDER))
    return False


def maybe_reuse_previous_inference():
    """Prompt to reuse inference JSONs from the most recent experiment."""
    exps = sorted(
        [p for p in EXPERIMENTS_DIR.glob("exp_*") if p.is_dir() and p != EXP_DIR]
    )
    if exps:
        prev_exp = exps[-1]
        prev_results = prev_exp / "results"
        if prev_results.exists() and any(prev_results.glob("*.json")):
            resp = ask_with_timeout(
                f"Existing inference found in the last experiment ({prev_exp.name}). "
                "Do you want to reuse them [y] or regenerate new ones [n]? ",
                timeout=10,
                default="n",
            )
            if resp and resp[0] == 'y':
                global INFERENCE_SOURCE_FOLDER
                INFERENCE_SOURCE_FOLDER = prev_results
                INFERENCE_SOURCE_FILE.write_text(str(INFERENCE_SOURCE_FOLDER))
                json_files = list(prev_results.glob("*.json"))
                LOGGER.info(
                    f"[INFER] Using {len(json_files)} inference files from {prev_exp.name}"
                )
                return True
    INFERENCE_SOURCE_FILE.write_text(str(INFERENCE_SOURCE_FOLDER))
    return False

################################################################################
#                 STEP 2: YOLO INFERENCE (PARALLEL, MULTI-GPU)
################################################################################

# We'll store YOLO models per GPU ID in this global.
LOCAL_MODELS = {}


def export_labelme_tile(tile_path: Path, res) -> None:
    """Export inference results for a tile in labelme format."""
    try:
        relative = tile_path.relative_to(TILE_FOLDER)
    except ValueError:
        relative = Path(tile_path.name)

    dest_image_path = LABELME_DIR / relative
    dest_image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copy2(tile_path, dest_image_path)
    except FileNotFoundError:
        return

    shapes = []
    confidences = []
    if res.boxes is not None and getattr(res.boxes, "conf", None) is not None:
        confidences = [float(c) for c in res.boxes.conf.cpu().tolist()]
    if res.masks is not None and getattr(res.masks, "xy", None) is not None:
        polys = res.masks.xy
        for idx, coords in enumerate(polys):
            conf_val = confidences[idx] if idx < len(confidences) else None
            if conf_val is not None and conf_val < BEST_CONF:
                continue

            pts = [[float(x), float(y)] for x, y in coords]
            shape_entry = {
                "label": "collembola",
                "points": pts,
                "group_id": None,
                "description": None,
                "shape_type": "polygon",
                "flags": {},
            }
            if conf_val is not None:
                shape_entry["flags"]["confidence"] = conf_val
            shapes.append(shape_entry)

    if getattr(res, "orig_shape", None) is not None:
        height, width = res.orig_shape[:2]
    else:
        height = width = TILE_SIZE

    labelme_payload = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": dest_image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }

    out_json_path = dest_image_path.with_suffix(".json")
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(labelme_payload, f, indent=2)


def get_local_model(gpu_id):
    """
    Returns a YOLO model pinned to the given GPU. 
    Each child process loads the model exactly once, 
    stored in a 'LOCAL_MODELS' dict.
    """
    global LOCAL_MODELS

    if gpu_id in LOCAL_MODELS:
        # Already loaded in this worker process
        return LOCAL_MODELS[gpu_id]

    # Lazily import ultralytics to avoid conflicts if needed
    from ultralytics import YOLO
    device_name = f"cuda:{gpu_id}"
    LOGGER.info(f"[ChildProc] Loading YOLO => {MODEL_WEIGHTS_PATH} on {device_name}")
    model = YOLO(MODEL_WEIGHTS_PATH)
    model.to(device_name).eval()

    # Optionally fuse Conv+BN layers for faster inference
    if FUSE_MODEL:
        try:
            model.fuse()
        except Exception:
            pass

    # Dummy forward pass to allocate GPU memory
    dummy = torch.zeros((1,3,TILE_SIZE,TILE_SIZE), device=device_name)
    _ = model.predict(dummy, conf=0.01, iou=0.01, verbose=False)

    LOCAL_MODELS[gpu_id] = model
    return model


def save_predictions(res, tile_path):
    detections = []
    if res.masks and res.masks.xy is not None:
        nm = len(res.masks.xy)
        for i in range(nm):
            conf_ = float(res.boxes.conf[i])
            coords2d = res.masks.xy[i]
            pts = [(float(x), float(y)) for (x, y) in coords2d]
            detections.append({"points": pts, "confidence": conf_})
    out_json = Path(RESULTS_FOLDER, tile_path.stem + ".json")
    out_json.write_text(json.dumps({"detections": detections}))
    export_labelme_tile(tile_path, res)


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def inference_worker(gpu_id, tiles, batch_size):
    if not tiles:
        return
    model = get_local_model(gpu_id)
    device = f"cuda:{gpu_id}"
    n_batches = math.ceil(len(tiles) / batch_size)
    for batch in tqdm(
        chunked(tiles, batch_size),
        total=n_batches,
        desc=f"GPU {gpu_id}",
        position=gpu_id,
    ):
        paths = [str(p) for p in batch]
        res_list = model.predict(
            paths,
            conf=0.00,
            iou=0.7,
            device=device,
            verbose=False,
            batch=min(batch_size, len(paths)),
        )
        for res, tilep in zip(res_list, batch):
            save_predictions(res, tilep)

def parallel_inference():
    """
    Collect all tiles. For each tile, do_inference_on_tile(...).
    *IMPORTANT* => we limit n_jobs to exactly N_GPUS (not -1) 
    so we have at most N_GPUS worker processes, each pinned to a GPU. 
    This avoids OOM from reloading the model too many times.
    """
    ori_tiles = sorted((Path(TILE_FOLDER, "ori")).glob("*.jpg"))
    shift_tiles = sorted((Path(TILE_FOLDER, "shift")).glob("*.jpg"))
    all_tiles = ori_tiles + shift_tiles

    def precheck(tilep):
        outj = Path(RESULTS_FOLDER, tilep.stem + ".json")
        if outj.is_file():
            return None
        skip_reason = should_skip_tile(tilep)
        if skip_reason:
            outj.write_text(json.dumps({"detections": [], "skip_reason": skip_reason}))
            LOGGER.info(f"[INFER] Skip tile ({skip_reason}) => {tilep.stem}")
            return None
        return tilep

    with tqdm_joblib(tqdm(total=len(all_tiles), desc="Pre-check tiles")):
        pending = [p for p in Parallel(n_jobs=N_JOBS)(delayed(precheck)(tp) for tp in all_tiles) if p]

    if not pending:
        print("[INFER] All tiles done => skip.")
        return
    else:
        print(f"[INFER] => {len(pending)} tiles need inference across {N_GPUS} GPU(s).")
    chunks = [pending[i::N_GPUS] for i in range(N_GPUS)]

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        p = multiprocessing.Process(
            target=inference_worker, args=(gpu_id, chunk, BATCH_SIZE)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


################################################################################
#               STEP 3: BUILD RAW POLYGON PICKLE FROM JSON
################################################################################

RAW_PICKLE = Path(RESULTS_FOLDER, "raw_polygons.pkl")

def safe_polygon(coords):
    """
    Create a valid Shapely polygon from coords, ignoring degenerate cases.
    """
    # remove consecutive duplicates
    cleaned = []
    for (x,y) in coords:
        if not cleaned or (x,y) != cleaned[-1]:
            cleaned.append((x,y))
    if len(cleaned)<3:
        return None
    # ensure ring closure
    if cleaned[0]!= cleaned[-1]:
        cleaned.append(cleaned[0])
    # drop polygons with non-finite coords
    for x, y in cleaned:
        if not math.isfinite(x) or not math.isfinite(y):
            return None

    try:
        poly = Polygon(cleaned)
        if poly.is_empty:
            return None
        if not poly.is_valid:
            poly = make_valid(poly)
        if isinstance(poly, GeometryCollection):
            # union all pieces that have area
            poly = unary_union([g for g in poly.geoms if g.area>0])
        if poly.is_empty:
            return None
        return poly
    except:
        return None

TILE_META_CACHE = None


def get_tile_meta():
    global TILE_META_CACHE
    if TILE_META_CACHE is not None:
        return TILE_META_CACHE
    if TILE_META_PICKLE.is_file():
        with open(TILE_META_PICKLE, "rb") as f:
            TILE_META_CACHE = pickle.load(f)
    else:
        TILE_META_CACHE = {}
    return TILE_META_CACHE


def get_tile_meta_for_slide(img_idx):
    meta = get_tile_meta()
    if not meta:
        return {}
    if img_idx in meta:
        return meta[img_idx]
    str_idx = str(img_idx)
    if str_idx in meta:
        return meta[str_idx]
    if isinstance(img_idx, str):
        sanitized = sanitize_index(img_idx)
        if sanitized in meta:
            return meta[sanitized]
        if sanitized.isdigit():
            as_int = int(sanitized)
            if as_int in meta:
                return meta[as_int]
    if isinstance(img_idx, int):
        as_str = str(img_idx)
        if as_str in meta:
            return meta[as_str]
    return {}


def build_raw_polygons(force=False):
    """
    For each tile-level JSON in INFERENCE_SOURCE_FOLDER, offset polygons => global coords,
    store them in a dict => pickled as RAW_PICKLE. If RAW_PICKLE exists & not forced => skip.
    """
    need_meta = not TILE_META_PICKLE.is_file()

    if RAW_PICKLE.is_file() and not force and not need_meta:
        LOGGER.info(f"[BUILD] Raw polygons => already exist at {RAW_PICKLE}, skipping.")
        return
    elif RAW_PICKLE.is_file() and need_meta:
        LOGGER.info("[BUILD] Rebuilding tile metadata from inference JSON files.")

    jsfiles = sorted(Path(INFERENCE_SOURCE_FOLDER).glob("*.json"))

    def process_json(jf):
        stem = jf.stem  # e.g. "ori_slideA_003_002"
        parts = stem.split("_")
        if len(parts) < 4:
            return None

        grid = parts[0]
        xs, ys = parts[-2], parts[-1]
        sidx = "_".join(parts[1:-2])
        idx = int(sidx) if sidx.isdigit() else sidx
        if isinstance(idx, str):
            idx = sanitize_index(idx)
            if idx.isdigit():
                idx = int(idx)
        x_i = int(xs)
        y_i = int(ys)
        if grid == "ori":
            offx = x_i * TILE_SIZE
            offy = y_i * TILE_SIZE
        else:
            offx = SHIFT + x_i * TILE_SIZE
            offy = SHIFT + y_i * TILE_SIZE

        tile_bbox = (
            int(offx),
            int(offy),
            int(offx + TILE_SIZE),
            int(offy + TILE_SIZE),
        )

        dd = json.loads(jf.read_text())
        dets = dd.get("detections", [])
        skip_reason = dd.get("skip_reason")
        polys = []
        for dt in dets:
            conf_ = dt["confidence"]
            if conf_ is None:
                continue
            cpoints = [(px + offx, py + offy) for (px, py) in dt["points"]]
            poly_ = safe_polygon(cpoints)
            if poly_ is not None:
                polys.append({"polygon": poly_, "confidence": conf_, "grid": grid})

        return {
            "idx": idx,
            "polys": polys,
            "grid": grid,
            "tile_bbox": tile_bbox,
            "skip_reason": skip_reason,
        }

    with tqdm_joblib(tqdm(total=len(jsfiles), desc="Collect polygons => dict")):
        results = Parallel(n_jobs=N_JOBS)(delayed(process_json)(jf) for jf in jsfiles)

    all_dict = {}
    tile_registry = defaultdict(lambda: {"ori": set(), "shift": set()})
    skip_registry = defaultdict(lambda: {"ori": [], "shift": []})
    for res in results:
        if res is None:
            continue
        idx = res["idx"]
        polys = res["polys"]
        grid = res["grid"]
        tile_bbox = res["tile_bbox"]
        skip_reason = res["skip_reason"]
        if idx not in all_dict:
            all_dict[idx] = []
        all_dict[idx].extend(polys)
        tile_registry[idx][grid].add(tile_bbox)
        if skip_reason:
            skip_registry[idx][grid].append({"bbox": tile_bbox, "reason": skip_reason})

    with open(RAW_PICKLE, "wb") as f:
        pickle.dump(all_dict, f)

    serializable_meta = {}
    for idx, grids in tile_registry.items():
        serializable_meta[idx] = {
            "ori": sorted(list(grids["ori"])),
            "shift": sorted(list(grids["shift"])),
        }
        skip_info = skip_registry.get(idx)
        if skip_info:
            serialized_skips = {}
            for grid_name, entries in skip_info.items():
                if entries:
                    serialized_skips[grid_name] = entries
            if serialized_skips:
                serializable_meta[idx]["skipped"] = serialized_skips

    with open(TILE_META_PICKLE, "wb") as f:
        pickle.dump(serializable_meta, f)

    global TILE_META_CACHE
    TILE_META_CACHE = serializable_meta

    LOGGER.info(
        f"[BUILD] Stored raw polygons => {RAW_PICKLE}. #slides={len(all_dict)}"
    )
    LOGGER.info(
        f"[BUILD] Recorded tile metadata for {len(serializable_meta)} slides => {TILE_META_PICKLE}"
    )


def load_raw_dict():
    """
    Load the raw polygons dictionary => {slide_idx: [ {polygon, confidence}, ... ]}.
    """
    if not RAW_PICKLE.is_file():
        raise RuntimeError(f"[ERROR] {RAW_PICKLE} not found. Did you run build_raw_polygons() ?")
    with open(RAW_PICKLE, "rb") as f:
        data = pickle.load(f)
    return data



################################################################################
#                 STEP 4: FUSION (GRAPHCUT) with iou=0.101
################################################################################

def polygon_iou(a, b):
    global INVALID_INTERSECTIONS
    if a is None or b is None:
        return 0
    if not a.is_valid or a.is_empty:
        a = make_valid(a)
        if a.is_empty:
            return 0
    if not b.is_valid or b.is_empty:
        b = make_valid(b)
        if b.is_empty:
            return 0
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="invalid value encountered in intersection"
            )
            inter = a.intersection(b).area
    except RuntimeWarning:
        INVALID_INTERSECTIONS += 1
        return 0
    union = a.area + b.area - inter
    if union <= 0 or math.isnan(union):
        return 0
    return inter / union

def fuse_graphcut(polys, iou_th=0.3, alpha=45.0):
    """
    GraphCut fusion approach. 
    1) Build adjacency for polygons with IOU > iou_th.
    2) For each connected component => gather exterior coords 
       => concave hull => final shape. 
    3) Confidence => max of the component.
    """
    if not polys:
        return []

    n = len(polys)
    adj = [[] for _ in range(n)]
    for i in range(n):
        gi = polys[i]["polygon"]
        if gi is None:
            continue
        for j in range(i + 1, n):
            gj = polys[j]["polygon"]
            if gj is None:
                continue
            if polygon_iou(gi, gj) > iou_th:
                adj[i].append(j)
                adj[j].append(i)

    visited = set()
    out = []
    for i in range(n):
        if i in visited or polys[i]["polygon"] is None:
            continue
        stack = [i]
        comp = []
        while stack:
            v = stack.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            stack.extend(adj[v])

        coords = []
        confs = []
        for cidx in comp:
            pg = polys[cidx]["polygon"]
            confs.append(polys[cidx]["confidence"])
            # if multipolygon or geometrycollection => gather sub-polygons
            if isinstance(pg, (MultiPolygon, GeometryCollection)):
                for subg in pg.geoms:
                    if subg.geom_type == 'Polygon' and not subg.is_empty:
                        coords.extend(list(subg.exterior.coords))
            else:
                # normal polygon
                coords.extend(list(pg.exterior.coords))

        hull = concave_hull(coords, alpha=alpha)
        if hull and not hull.is_empty:
            out.append({
                "polygon": hull,
                "confidence": max(confs) if confs else 0
            })

    return out

def concave_hull(coords, alpha=45.0):
    """
    Use Shapely >=2.0's alpha_shape; fallback to convex_hull on errors.
    """
    if not coords:
        return None
    mp = MultiPoint(coords)
    if mp.is_empty:
        return None
    try:
        hull = mp.alpha_shape(alpha)
        if hull.area>0:
            return hull
        else:
            return mp.convex_hull
    except:
        return mp.convex_hull


################################################################################
#            STEP 5: VISUALIZATION + OPTIONAL CSV => STATS
################################################################################

def load_manual_points(img_idx):
    """
    Look for CSV named "{stem}.csv" or "{stem}_counted.csv" next to the image path.
    Returns list of tuples (point_id, x, y).
    """
    img_path = resolve_image_path(img_idx)
    if img_path is None:
        return []

    cfile1 = img_path.with_suffix(".csv")
    cfile2 = img_path.with_name(f"{img_path.stem}_counted.csv")
    chosen = cfile1 if cfile1.is_file() else cfile2 if cfile2.is_file() else None
    if chosen is None:
        return []

    rows = []
    with open(chosen, "r") as f:
        rr = csv.reader(f)
        hdr = next(rr, None)
        for row in rr:
            if len(row) < 7:
                continue
            try:
                mid = int(row[0])
                x_ = float(row[5])
                y_ = float(row[6])
                rows.append((mid, x_, y_))
            except Exception:
                pass
    return rows


def match_polygons_to_manual(polys, manual_pts):
    """
    Match polygons to manual points to derive TP/FP/FN statistics.
    Returns dict with counts, index lists and point lists.
    """
    from shapely.geometry import Point

    used = set()
    tp_polys = []
    tp_points = []
    fn_points = []
    for (mid, xm, ym) in manual_pts:
        pm = Point(xm, ym)
        found = None
        for pid, pp in enumerate(polys):
            if pid in used:
                continue
            if pp["polygon"].contains(pm):
                found = pid
                break
        if found is not None:
            used.add(found)
            tp_polys.append(found)
            tp_points.append((mid, xm, ym))
        else:
            fn_points.append((mid, xm, ym))

    fp_polys = [pid for pid in range(len(polys)) if pid not in used]

    tp = len(tp_polys)
    fp = len(fp_polys)
    fn = len(fn_points)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "tp_polys": tp_polys,
        "fp_polys": fp_polys,
        "tp_points": tp_points,
        "fn_points": fn_points,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def iter_subpolygons(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out = []
        for sg in geom.geoms:
            if sg.geom_type == "Polygon" and not sg.is_empty:
                out.append(sg)
        return out
    return []


def draw_tile_boundaries(draw, tile_meta, grids=None):
    if not tile_meta:
        return
    colors = {
        "ori": (0, 255, 0, 200),
        "shift": (0, 128, 255, 200),
    }
    target_grids = grids if grids is not None else ("ori", "shift")
    for grid in target_grids:
        boxes = tile_meta.get(grid, []) if tile_meta else []
        color = colors.get(grid)
        if not color:
            continue
        for bbox in boxes:
            draw.rectangle(bbox, outline=color, width=2)


def draw_skipped_tiles(draw, tile_meta, grids=None):
    if not tile_meta:
        return
    skip_info = tile_meta.get("skipped")
    if not skip_info:
        return
    colors = {
        "ori": (0, 255, 0, 220),
        "shift": (0, 128, 255, 220),
    }
    target_grids = grids if grids is not None else ("ori", "shift")
    for grid in target_grids:
        tiles = skip_info.get(grid, []) if isinstance(skip_info, dict) else []
        if not tiles:
            continue
        color = colors.get(grid)
        if not color:
            continue
        for tile in tiles:
            if isinstance(tile, dict):
                bbox = tile.get("bbox") or tile.get("tile_bbox")
            else:
                bbox = tile
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            draw.line([(x0, y0), (x1, y1)], fill=color, width=4)
            draw.line([(x0, y1), (x1, y0)], fill=color, width=4)


def visualize_raw_detection_grids(img_idx, raw_polygons, conf_, circle=None):
    img_path = resolve_image_path(img_idx)
    if img_path is None or not img_path.is_file():
        return

    grouped = defaultdict(list)
    for pol in raw_polygons:
        grid = pol.get("grid", "ori")
        grouped[grid].append(pol)

    with Image.open(img_path) as im:
        base_img = im.convert("RGB")

    tile_meta = get_tile_meta_for_slide(img_idx)
    grid_to_dir = {
        "ori": FINAL_VIZ_ORI,
        "shift": FINAL_VIZ_SHIFT,
    }

    for grid, outdir in grid_to_dir.items():
        grid_polys = grouped.get(grid, [])
        has_tiles = bool(tile_meta.get(grid))
        if not grid_polys and not has_tiles:
            continue

        canvas = base_img.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")
        if circle:
            cx, cy, r = circle
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)],
                outline=(255, 255, 0, 255),
                width=3,
            )
        for pol in grid_polys:
            for subp in iter_subpolygons(pol["polygon"]):
                coords = list(subp.exterior.coords)
                draw.polygon(coords, outline=(255, 215, 0, 255), fill=(255, 215, 0, 60))

        draw_tile_boundaries(draw, tile_meta, grids=[grid])
        draw_skipped_tiles(draw, tile_meta, grids=[grid])

        outdir.mkdir(parents=True, exist_ok=True)
        outf = outdir / f"{img_idx}_{grid}_raw_conf{conf_}.jpg"
        canvas.save(outf, "JPEG")


def visualize_overlays(img_idx, fused, match_info, conf_, iou_th, method, circle=None):
    """
    Visualize TP polygons in green, FP polygons in red,
    FN points in blue and TP points in medium green.
    """
    img_path = resolve_image_path(img_idx)
    if img_path is None or not img_path.is_file():
        return

    with Image.open(img_path) as im:
        base = im.convert("RGB")
        draw = ImageDraw.Draw(base, "RGBA")
        font = ImageFont.load_default()

        if circle:
            cx, cy, r = circle
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)],
                outline=(255, 255, 0, 255),
                width=4,
            )

        for pid, pol_ in enumerate(fused):
            geom = pol_["polygon"]
            conf_val = pol_["confidence"]
            subs = iter_subpolygons(geom)

            color_fill = (0, 255, 0, 80) if pid in match_info["tp_polys"] else (255, 0, 0, 80)
            color_line = (0, 255, 0, 255) if pid in match_info["tp_polys"] else (255, 0, 0, 255)

            for subp in subs:
                coords = list(subp.exterior.coords)
                draw.polygon(coords, fill=color_fill, outline=color_line)
                cent = subp.centroid
                lbl = f"{pid}:{conf_val:.2f}"
                draw.text((cent.x, cent.y), lbl, fill=(255, 255, 255, 255), font=font)

        for (_, xm, ym) in match_info.get("tp_points", []):
            r = 4
            draw.ellipse(((xm - r, ym - r), (xm + r, ym + r)), fill=(0, 128, 0, 255))

        for (_, xm, ym) in match_info["fn_points"]:
            r = 4
            draw.ellipse(((xm - r, ym - r), (xm + r, ym + r)), fill=(0, 0, 255, 255))

        tile_meta = get_tile_meta_for_slide(img_idx)
        draw_tile_boundaries(draw, tile_meta)
        draw_skipped_tiles(draw, tile_meta)

        outdir = Path(FINAL_VIZ, method)
        outdir.mkdir(parents=True, exist_ok=True)
        outf = outdir / f"{img_idx}_{method}_conf{conf_}_iou{iou_th}.jpg"
        base.save(outf, "JPEG")


def process_slide(sid, allpols):
    img_path = resolve_image_path(sid)
    circle = find_main_circle(img_path)

    filtered = [d for d in allpols if d["confidence"] >= BEST_CONF]

    # apply ROI filtering only if circle is valid
    filtered = filter_polygons_by_circle(filtered, circle)

    fused = fuse_graphcut(filtered, iou_th=BEST_IOU, alpha=ALPHA)

    manual_pts = load_manual_points(sid)
    if manual_pts:
        match_info = match_polygons_to_manual(fused, manual_pts)
    else:
        match_info = {
            "tp_polys": [],
            "fp_polys": list(range(len(fused))),
            "tp_points": [],
            "fn_points": [],
            "tp": 0,
            "fp": len(fused),
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
        }

    visualize_raw_detection_grids(sid, filtered, BEST_CONF, circle)
    visualize_overlays(sid, fused, match_info, BEST_CONF, BEST_IOU, BEST_METHOD, circle)

    return {
        "slide": sid,
        "human_count": len(manual_pts),
        "detected_count": len(fused),
        "tp": match_info["tp"],
        "fp": match_info["fp"],
        "fn": match_info["fn"],
        "precision": match_info["precision"],
        "recall": match_info["recall"],
        "f1": match_info["f1"],
    }

################################################################################
#                     STEP 6: MAIN PIPELINE
################################################################################


def run_pipeline():
    collect_image_paths()

    # 1) Tiling
    LOGGER.info("======== 1) TILING ========")
    reused = maybe_reuse_previous_tiles()
    if not reused:
        create_all_tiles_if_needed()

    # 2) YOLO Inference
    LOGGER.info("======== 2) YOLO Inference ========")
    reused_inf = maybe_reuse_previous_inference()
    if not reused_inf:
        parallel_inference()

    # 3) Build raw polygons
    LOGGER.info("======== 3) Build Raw Polygons ========")
    build_raw_polygons(force=False)

    # 4) Final fusion and statistics
    LOGGER.info("======== 4) Final Fusion with GraphCut ========")
    raw_data = load_raw_dict()

    slides = sorted(raw_data.keys(), key=slide_sort_key)
    tasks = [(sid, raw_data[sid]) for sid in slides]

    with tqdm_joblib(tqdm(total=len(tasks), desc="Fusion+viz")):
        results_rows = Parallel(n_jobs=N_JOBS)(
            delayed(process_slide)(sid, polys) for sid, polys in tasks
        )
    if INVALID_INTERSECTIONS:
        LOGGER.warning(
            f"Encountered {INVALID_INTERSECTIONS} invalid polygon intersections"
        )

    df = pd.DataFrame(results_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    LOGGER.info(f"Final results => {OUTPUT_CSV}")

    if "human_count" in df.columns:
        df_ann = df[df["human_count"] > 0]
        if not df_ann.empty:
            r2 = r2_score(df_ann["human_count"], df_ann["detected_count"])
            LOGGER.info(
                f"R2 correlation between human and fused counts: {r2:.4f}"
            )

            plt.figure()
            plt.scatter(
                df_ann["human_count"], df_ann["detected_count"], c="purple"
            )
            xs = np.array(df_ann["human_count"])
            ys = np.array(df_ann["detected_count"])
            m, b = np.polyfit(xs, ys, 1)
            x_line = np.linspace(xs.min(), xs.max(), 100)
            y_line = m * x_line + b
            plt.plot(
                x_line,
                y_line,
                color="orange",
                label=f"y={m:.2f}x+{b:.2f}",
            )
            plt.xlabel("Human count")
            plt.ylabel("Fused count")
            plt.title(f"Counts correlation (R2={r2:.3f})")
            plt.legend()
            plt.savefig(STATS_DIR / "counts_r2.png")

            total_tp = int(df_ann["tp"].sum())
            total_fp = int(df_ann["fp"].sum())
            total_fn = int(df_ann["fn"].sum())
            overall_prec = (
                total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            )
            overall_rec = (
                total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            )
            overall_f1 = (
                2 * overall_prec * overall_rec / (overall_prec + overall_rec)
                if (overall_prec + overall_rec) > 0
                else 0
            )
            summary = {
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
                "overall_precision": overall_prec,
                "overall_recall": overall_rec,
                "overall_f1": overall_f1,
                "r2": float(r2),
            }
            with open(STATS_DIR / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    LOGGER.info("=== PIPELINE FINISHED ===")


if __name__=="__main__":
    # recommended to set start method to 'spawn' for joblib + GPU usage
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    run_pipeline()
