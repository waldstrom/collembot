#!/usr/bin/env python3
# Usage: python scripts/gradio_app.py (for Colab helper modules).
# Ownership: Copyright (c) 2026 Adrian Meyer
# License: AGPL-3.0 only (code); model weights and dependencies may be under AGPL-3.0.
"""Gradio-ready inference helper for Google Colab.

This module mirrors the essential inference utilities from ``scripts/run_inference.py``
while keeping the surface area small enough for quick interactive use. The
primary entry point is :func:`launch_app`, which spins up a Gradio interface
that lets users upload a slide image, run tiled inference (with both original
and shifted grids), perform graph-cut fusion, and visualise detections together
with the final collembola count.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import yaml

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Point, Polygon
from shapely.strtree import STRtree
from shapely.validation import make_valid
import torch

# Enable the fastest convolution algorithms for the current hardware. This helps
# reduce Gradio inference latency on GPU-backed runtimes (e.g., Colab).
torch.backends.cudnn.benchmark = True

# -----------------------------------------------------------------------------
# Configuration defaults (kept close to the main inference script values)
# -----------------------------------------------------------------------------
TILE_SIZE = 576
SHIFT = 288
# Keep prediction batches modest but large enough to utilise the GPU effectively.
PREDICT_BATCH = 8
DEFAULT_DEVICE = "auto"
BEST_CONF = 0.450
BEST_IOU = 0.101
ALPHA = 45.0
CIRCLE_CENTER_MARGIN = 0.20
CIRCLE_RADIUS_MIN_REL = 0.35
CIRCLE_RADIUS_MAX_REL = 0.52
CIRCLE_RADIUS_SCALE = 1.05
ROI_S_MIN = 40
ROI_V_MIN = 60
ROI_V_MAX = 210
ROI_KERNEL_SIZE = 21
CPU_WORKERS = max(1, os.cpu_count() or 1)


def _load_config_overrides(config_path: Path = Path("configs/inference.yaml")) -> None:
    """Override core thresholds from ``configs/inference.yaml`` when available.

    The Colab helper should mirror the main ``scripts/run_inference.py`` behaviour. Reading
    the shared config keeps fusion thresholds, tiling sizes, and optional shifts
    aligned without hard-coding values in two places. Failures are silent to
    avoid breaking notebook imports when the config is absent.
    """

    global TILE_SIZE, SHIFT, PREDICT_BATCH, BEST_CONF, BEST_IOU, ALPHA

    if not config_path.is_file():
        return

    try:
        cfg = yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return

    tile_cfg = cfg.get("tile", {})
    TILE_SIZE = int(tile_cfg.get("size", TILE_SIZE))
    SHIFT = int(tile_cfg.get("shift", SHIFT))

    par_cfg = cfg.get("parallel", {})
    PREDICT_BATCH = int(par_cfg.get("batch_size", PREDICT_BATCH))

    fusion_cfg = cfg.get("fusion", {})
    BEST_CONF = float(fusion_cfg.get("best_conf", BEST_CONF))
    BEST_IOU = float(fusion_cfg.get("best_iou", BEST_IOU))
    ALPHA = float(fusion_cfg.get("alpha", ALPHA))


_load_config_overrides()

# ----------------------------------------------------------------------------
# Circle detection helpers (ported from scripts/run_inference.py)
# ----------------------------------------------------------------------------

def _to_bgr_image(img_source):
    """Return a BGR numpy array from a path or PIL image."""
    import cv2  # imported lazily to keep the module light when unused

    if img_source is None:
        return None

    if isinstance(img_source, (str, os.PathLike, Path)):
        return cv2.imread(str(img_source))

    if isinstance(img_source, Image.Image):
        arr = np.array(img_source.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return None


def find_main_circle(img_source):
    """Detect the main dish/glass circle in an image.

    The logic mirrors ``scripts/run_inference.py`` to keep behaviour consistent between the
    HPC and Colab variants, but downsamples large inputs to avoid excessive
    latency in Colab notebooks (e.g., 6000x4000 images).
    """
    import cv2  # imported lazily to keep the module light when unused

    img = _to_bgr_image(img_source)
    if img is None:
        return None

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    # Downscale very large images to keep HoughCircles performant. Keep track
    # of the scale factor to map the detected circle back to the original size.
    max_dim = max(h, w)
    TARGET_MAX_DIM = 1600
    scale = max(1.0, max_dim / float(TARGET_MAX_DIM))
    if scale > 1.0:
        new_w = int(round(w / scale))
        new_h = int(round(h / scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    else:
        scale = 1.0

    mindim = min(h, w)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    mask = (s_ch > ROI_S_MIN) & (v_ch > ROI_V_MIN) & (v_ch < ROI_V_MAX)
    mask_u8 = mask.astype("uint8") * 255

    kernel_size = max(3, int(round(ROI_KERNEL_SIZE / scale)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    ys, xs = np.nonzero(mask_u8)
    if xs.size == 0:
        return None

    cx_mask = float(xs.mean())
    cy_mask = float(ys.mean())

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

        if not (w * CIRCLE_CENTER_MARGIN <= x <= w * (1.0 - CIRCLE_CENTER_MARGIN)):
            continue
        if not (h * CIRCLE_CENTER_MARGIN <= y <= h * (1.0 - CIRCLE_CENTER_MARGIN)):
            continue

        rel_r = r / float(mindim)
        if not (CIRCLE_RADIUS_MIN_REL <= rel_r <= CIRCLE_RADIUS_MAX_REL):
            continue

        dist_norm = math.hypot(x - cx_mask, y - cy_mask) / float(mindim)
        if dist_norm > 0.25:
            continue

        candidates.append((x, y, r, dist_norm))

    if not candidates:
        return None

    x, y, r, _ = min(candidates, key=lambda c: (c[3], -c[2]))
    r = int(r * CIRCLE_RADIUS_SCALE)

    if scale != 1.0:
        x = int(round(x * scale))
        y = int(round(y * scale))
        r = int(round(r * scale))

    return int(x), int(y), int(r)


def point_inside_circle(px: float, py: float, circle):
    if circle is None:
        return True
    cx, cy, r = circle
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def filter_polygons_by_circle(polys: Sequence[dict], circle):
    if circle is None:
        return polys

    filtered = []
    for poly in polys:
        geom = poly.get("polygon")
        if geom is None or geom.is_empty:
            continue

        c = geom.centroid
        if point_inside_circle(c.x, c.y, circle):
            filtered.append(poly)

    return filtered


# ----------------------------------------------------------------------------
# Geometry utilities (graph-cut fusion)
# ----------------------------------------------------------------------------

def polygon_iou(a: Polygon, b: Polygon) -> float:
    return _polygon_iou_internal(a, b, assume_valid=False)


def _clean_polygon(poly: Polygon):
    """Return a valid, non-empty polygon or ``None``."""
    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.is_empty:
            return None
    return poly


def _polygon_iou_internal(a: Polygon, b: Polygon, *, assume_valid: bool) -> float:
    if a is None or b is None:
        return 0.0
    if not assume_valid:
        a = _clean_polygon(a)
        b = _clean_polygon(b)
        if a is None or b is None:
            return 0.0
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    if union <= 0 or math.isnan(union):
        return 0.0
    return inter / union


def concave_hull(coords: Sequence[Tuple[float, float]], alpha: float = ALPHA):
    if not coords:
        return None
    mp = MultiPoint(coords)
    if mp.is_empty:
        return None
    try:
        hull = mp.alpha_shape(alpha)
        return hull if hull.area > 0 else mp.convex_hull
    except Exception:
        return mp.convex_hull


def _build_adjacency(polys: Sequence[dict], iou_th: float):
    """Efficiently build adjacency using a spatial index and validated polygons."""

    n = len(polys)
    adj = [[] for _ in range(n)]

    prepared = []
    for idx, poly in enumerate(polys):
        geom = _clean_polygon(poly.get("polygon"))
        if geom is not None:
            prepared.append((idx, geom))

    if len(prepared) < 2:
        return adj, {idx: geom for idx, geom in prepared}

    indices = [idx for idx, _ in prepared]
    geoms = [geom for _, geom in prepared]
    geom_by_idx = {idx: geom for idx, geom in prepared}

    tree = STRtree(geoms)

    # ``query_bulk`` is available in Shapely 2.x. Google Colab often ships an
    # older 1.x build, so fall back to a manual query loop when needed to avoid
    # ``AttributeError``.
    if hasattr(tree, "query_bulk"):
        pairs = tree.query_bulk(geoms, predicate="intersects")
    else:  # Shapely < 2.0
        idx_by_id = {id(g): i for i, g in enumerate(geoms)}
        src_idx = []
        dst_idx = []
        for src, geom in enumerate(geoms):
            for candidate in tree.query(geom):
                dst = idx_by_id.get(id(candidate))
                if dst is not None:
                    src_idx.append(src)
                    dst_idx.append(dst)
        pairs = np.array([src_idx, dst_idx], dtype=int)

    if pairs.size == 0:
        return adj, geom_by_idx

    seen = set()
    for src, dst in zip(pairs[0], pairs[1]):
        if src == dst:
            continue
        a_idx = indices[src]
        b_idx = indices[dst]
        key = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
        if key in seen:
            continue
        seen.add(key)

        a_geom = geoms[src]
        b_geom = geoms[dst]
        if _polygon_iou_internal(a_geom, b_geom, assume_valid=True) > iou_th:
            adj[a_idx].append(b_idx)
            adj[b_idx].append(a_idx)

    return adj, geom_by_idx


def fuse_graphcut(polys: Sequence[dict], iou_th: float = BEST_IOU, alpha: float = ALPHA):
    if not polys:
        return []

    n = len(polys)
    adj, geom_by_idx = _build_adjacency(polys, iou_th)

    visited = set()
    out = []
    for i in range(n):
        if i in visited or geom_by_idx.get(i) is None:
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
            pg = geom_by_idx.get(cidx)
            confs.append(polys[cidx].get("confidence", 0))
            if isinstance(pg, (MultiPolygon, GeometryCollection)):
                coords.extend([tuple(pt) for subg in pg.geoms if subg.geom_type == "Polygon" for pt in subg.exterior.coords])
            else:
                coords.extend(list(pg.exterior.coords))

        hull = concave_hull(coords, alpha=alpha)
        if hull and not hull.is_empty:
            out.append({"polygon": hull, "confidence": max(confs) if confs else 0})
    return out


# ----------------------------------------------------------------------------
# Tiling and model inference
# ----------------------------------------------------------------------------


def _resolve_device(preferred: Optional[str] = None) -> str:
    """Return the runtime device, preferring CUDA when available."""

    if preferred and preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clear_cuda_cache():
    """Aggressively free cached CUDA memory before inference batches."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# ----------------------------------------------------------------------------

def _tile_coordinates(width: int, height: int, offset: Tuple[int, int]):
    off_x, off_y = offset
    x_tiles = (width - off_x) // TILE_SIZE
    y_tiles = (height - off_y) // TILE_SIZE
    for xx in range(x_tiles):
        for yy in range(y_tiles):
            x0 = off_x + xx * TILE_SIZE
            y0 = off_y + yy * TILE_SIZE
            yield (x0, y0, x0 + TILE_SIZE, y0 + TILE_SIZE)


def build_tiles(img: Image.Image, *, max_workers: Optional[int] = None):
    width, height = img.size
    tasks = []
    for label, offset in (("ori", (0, 0)), ("shift", (SHIFT, SHIFT))):
        for bbox in _tile_coordinates(width, height, offset):
            tasks.append((label, bbox))

    workers = max(1, max_workers or CPU_WORKERS)

    def _crop(task):
        label, bbox = task
        x0, y0, x1, y1 = bbox
        crop = img.crop((x0, y0, x1, y1)).convert("RGB")
        return {"tile": crop, "offset": (x0, y0), "tag": label}

    if len(tasks) <= 1 or workers == 1:
        return [_crop(t) for t in tasks]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_crop, tasks))


def _polygon_from_result(res, idx: int, offset: Tuple[int, int], grid: str):
    ox, oy = offset
    polygons = []
    if res.masks and res.masks.xy is not None:
        coords2d = res.masks.xy[idx]
        pts = [(float(x) + ox, float(y) + oy) for (x, y) in coords2d]
        unique_pts = []
        seen = set()
        for pt in pts:
            if pt in seen:
                continue
            seen.add(pt)
            unique_pts.append(pt)
        if len(unique_pts) >= 3:
            geom = Polygon(unique_pts)
            geom = make_valid(geom) if (not geom.is_valid or geom.is_empty) else geom
            if geom and not geom.is_empty:
                polygons.append({"polygon": geom, "confidence": float(res.boxes.conf[idx]), "grid": grid})
    else:
        box = res.boxes.xyxy[idx].tolist()
        x0, y0, x1, y1 = box
        pts = [(x0 + ox, y0 + oy), (x1 + ox, y0 + oy), (x1 + ox, y1 + oy), (x0 + ox, y1 + oy)]
        geom = Polygon(pts)
        polygons.append({"polygon": geom, "confidence": float(res.boxes.conf[idx]), "grid": grid})
    return polygons


def run_model_on_tiles(
    model,
    device: str,
    tiles: Sequence[dict],
    *,
    batch_size: int = PREDICT_BATCH,
    use_half: Optional[bool] = None,
    progress: Optional[Callable[[float, str], None]] = None,
    progress_range: Tuple[float, float] = (0.0, 1.0),
):
    detections: List[dict] = []
    # Half precision is disabled to avoid dtype mismatches during fusion.
    # Keep the parameter for backward compatibility but ignore the caller's preference.
    half = False

    _clear_cuda_cache()

    total_batches = max(1, math.ceil(len(tiles) / float(batch_size)))
    start_frac, end_frac = progress_range

    for batch_idx, batch_start in enumerate(range(0, len(tiles), batch_size)):
        batch = tiles[batch_start : batch_start + batch_size]
        imgs = [t["tile"] for t in batch]
        offsets = [t["offset"] for t in batch]
        grids = [t.get("tag", "ori") for t in batch]
        if progress:
            frac = start_frac + (batch_idx / total_batches) * (end_frac - start_frac)
            progress(frac, desc=f"Running model batch {batch_idx + 1}/{total_batches}")

        with torch.inference_mode():
            results = model.predict(
                imgs,
                conf=0.01,
                iou=0.7,
                device=device,
                verbose=False,
                half=half,
            )
        results = [res.cpu() for res in results]
        for res, offset, grid in zip(results, offsets, grids):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            for idx in range(len(res.boxes)):
                detections.extend(_polygon_from_result(res, idx, offset, grid))
    if progress:
        progress(end_frac, desc="Inference complete")
    return detections


def load_model(
    weights_path: str,
    fuse: bool = False,
    *,
    preferred_device: Optional[str] = None,
    use_half: Optional[bool] = None,
):
    from ultralytics import YOLO

    device = _resolve_device(preferred_device)
    # Half precision is disabled to avoid dtype mismatches during fusion.
    # Keep the parameter for backward compatibility but ignore the caller's preference.
    half = False
    _clear_cuda_cache()
    model = YOLO(weights_path)
    model.to(device).eval()
    if fuse:
        try:
            model.fuse()
        except Exception:
            pass
    return model, device, half


# ----------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------

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


def _build_tile_metadata(tiles: Sequence[dict]):
    meta = {"ori": [], "shift": []}
    for t in tiles:
        tag = t.get("tag", "ori")
        if tag not in meta:
            meta[tag] = []
        ox, oy = t.get("offset", (0, 0))
        meta[tag].append((ox, oy, ox + TILE_SIZE, oy + TILE_SIZE))
    return meta


def _draw_tile_boundaries(draw, tile_meta, grids=None):
    if not tile_meta:
        return
    colors = {"ori": (0, 255, 0, 200), "shift": (0, 128, 255, 200)}
    targets = grids if grids is not None else tile_meta.keys()
    for grid in targets:
        boxes = tile_meta.get(grid, [])
        color = colors.get(grid)
        if not color:
            continue
        for bbox in boxes:
            draw.rectangle(bbox, outline=color, width=2)


def _annotate_polygons(draw, polys, *, font, label_ids: bool):
    for idx, poly in enumerate(polys):
        geom = poly.get("polygon")
        conf = poly.get("confidence", 0.0)
        if geom is None or geom.is_empty:
            continue
        for subp in iter_subpolygons(geom):
            coords = list(subp.exterior.coords)
            draw.polygon(coords, outline=(255, 215, 0, 255), fill=(255, 215, 0, 60))
            if label_ids:
                cent = subp.centroid
                lbl = f"{idx}:{conf:.2f}"
                draw.text((cent.x, cent.y), lbl, fill=(255, 255, 255, 255), font=font)


def _draw_circle(draw, circle, *, width=3):
    if not circle:
        return
    cx, cy, r = circle
    draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(255, 255, 0, 255), width=width)


def compose_visualisations(
    base: Image.Image,
    tile_meta: dict,
    raw_polys: Sequence[dict],
    fused_polys: Sequence[dict],
    circle,
):
    font = ImageFont.load_default()
    base_rgb = base.convert("RGB")

    def _panel(polys, *, grids=None, label_ids=False):
        canvas = base_rgb.copy()
        draw = ImageDraw.Draw(canvas, "RGBA")
        _draw_circle(draw, circle, width=4 if label_ids else 3)
        _draw_tile_boundaries(draw, tile_meta, grids=grids)
        _annotate_polygons(draw, polys, font=font, label_ids=label_ids)
        return canvas

    raw_ori = _panel([p for p in raw_polys if p.get("grid") == "ori"], grids=["ori"], label_ids=False)
    raw_shift = _panel([p for p in raw_polys if p.get("grid") == "shift"], grids=["shift"], label_ids=False)
    fused = _panel(fused_polys, grids=None, label_ids=True)

    w, h = base_rgb.size
    combined = Image.new("RGB", (w * 2, h * 2), color=(0, 0, 0))
    combined.paste(fused, (0, 0))
    combined.paste(raw_ori, (w, 0))
    combined.paste(raw_shift, (0, h))

    info = Image.new("RGB", (w, h), color=(20, 20, 20))
    draw_info = ImageDraw.Draw(info)
    summary_lines = [
        "Fused detections",
        f"count: {len(fused_polys)}",
        "",
        "Grid detections",
        f"ori: {len([p for p in raw_polys if p.get('grid') == 'ori'])}",
        f"shift: {len([p for p in raw_polys if p.get('grid') == 'shift'])}",
    ]
    if circle:
        cx, cy, r = circle
        summary_lines.append("")
        summary_lines.append(f"circle: ({cx}, {cy}, r={r})")
    y = 10
    for line in summary_lines:
        draw_info.text((10, y), line, fill=(255, 255, 255), font=font)
        y += 18
    combined.paste(info, (w, h))

    return combined


def draw_polygons(
    base: Image.Image,
    polygons: Sequence[dict],
    circle,
    *,
    label_ids: bool = False,
):
    """Draw polygons (and optionally the detected circle) onto an image.

    This lightweight helper mirrors the original Colab walkthrough behaviour and
    keeps backward compatibility for existing notebooks. It overlays the
    provided polygons on ``base`` without the extra grid/tile summaries produced
    by :func:`compose_visualisations`.
    """

    font = ImageFont.load_default()
    canvas = base.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas, "RGBA")
    _draw_circle(draw, circle)
    _annotate_polygons(draw, polygons, font=font, label_ids=label_ids)
    return canvas


def _serialise_polygon(geom):
    polys = []
    for subp in iter_subpolygons(geom):
        polys.append([[float(x), float(y)] for (x, y) in subp.exterior.coords])
    return polys


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def infer_image(
    image: Image.Image,
    model,
    device: str,
    *,
    use_half: bool,
    batch_size: int = PREDICT_BATCH,
    progress: Optional[Callable[[float, str], None]] = None,
):
    if progress:
        progress(0.05, desc="Tiling image")
    tiles = build_tiles(image, max_workers=CPU_WORKERS)
    tile_meta = _build_tile_metadata(tiles)
    raw_polys = run_model_on_tiles(
        model,
        device,
        tiles,
        batch_size=batch_size,
        use_half=use_half,
        progress=progress,
        progress_range=(0.15, 0.80),
    )
    filtered_raw = [p for p in raw_polys if p.get("confidence", 0.0) >= BEST_CONF]
    circle_source = getattr(image, "filename", None) or image
    circle = find_main_circle(circle_source)
    filtered_roi = filter_polygons_by_circle(filtered_raw, circle)
    if progress:
        progress(0.82, desc="Fusing detections")
    fused = fuse_graphcut(filtered_roi, iou_th=BEST_IOU, alpha=ALPHA)
    fused = filter_polygons_by_circle(fused, circle)
    viz = compose_visualisations(image, tile_meta, filtered_roi, fused, circle)
    if progress:
        progress(0.98, desc="Done")
    return fused, viz, circle, filtered_roi


def launch_app(
    model_path: str,
    share: bool = False,
    fuse: bool = False,
    example_image: Optional[str] = None,
    preferred_device: str = DEFAULT_DEVICE,
    predict_batch_size: int = PREDICT_BATCH,
    use_half: Optional[bool] = None,
):
    if share:
        # Public sharing is disabled for safety. Always run locally.
        share = False
    model, device, half = load_model(
        model_path,
        fuse=fuse,
        preferred_device=preferred_device,
        use_half=use_half,
    )

    default_img = None
    examples: List[List[str]] = []
    if example_image:
        try:
            img_path = Path(example_image)
            if img_path.exists():
                default_img = Image.open(img_path)
                default_img.filename = str(img_path)
                examples = [[str(img_path)]]
        except Exception:
            default_img = None
            examples = []

    def _predict(pil_img: Image.Image, progress=gr.Progress(track_tqdm=True)):
        if pil_img is None:
            return None, "No image provided"
        fused, viz, circle, raw_filtered = infer_image(
            pil_img,
            model,
            device,
            use_half=half,
            batch_size=predict_batch_size,
            progress=progress,
        )
        detections = []
        for idx, pol in enumerate(fused):
            geom = pol.get("polygon")
            detections.append(
                {
                    "id": idx,
                    "confidence": round(float(pol.get("confidence", 0.0)), 4),
                    "polygons": _serialise_polygon(geom) if geom is not None else [],
                }
            )

        grid_counts = {
            "ori": len([p for p in raw_filtered if p.get("grid") == "ori"]),
            "shift": len([p for p in raw_filtered if p.get("grid") == "shift"]),
        }

        details = {
            "fused_count": len(fused),
            "grid_counts": grid_counts,
            "circle": circle,
            "detections": detections,
        }
        return viz, json.dumps(details, indent=2)

    desc_md = """
    **Collembola GPU inference**

    1. Upload a slide image.
    2. The image is tiled with both the original and shifted grids.
    3. YOLO predictions above conf {best_conf:.3f} are fused with the graph-cut strategy.
    4. Final detections are filtered by the detected main circle (if available).
    5. Preferred device: {device}
    6. Half precision: disabled
    7. Batch size: {predict_batch_size}
    """.format(device=device, predict_batch_size=predict_batch_size, best_conf=BEST_CONF)

    with gr.Blocks() as demo:
        gr.Markdown(desc_md)
        with gr.Row():
            inp = gr.Image(type="pil", label="Upload image", value=default_img)
        if examples:
            gr.Examples(label="Try the hosted example", examples=examples, inputs=inp)
        with gr.Row():
            out_img = gr.Image(label="Detections", type="pil")
            out_json = gr.Code(label="Detection summary", language="json")
        btn = gr.Button("Run inference", variant="primary")
        btn.click(fn=_predict, inputs=inp, outputs=[out_img, out_json])

    demo.launch(share=share, debug=True)

    return demo


__all__ = [
    "launch_app",
    "infer_image",
    "find_main_circle",
    "fuse_graphcut",
    "filter_polygons_by_circle",
    "compose_visualisations",
    "draw_polygons",
]
