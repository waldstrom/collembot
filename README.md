# Collembola Counting Toolkit

This repository hosts a complete pipeline for counting *Collembola* organisms in ecotoxicity soil slides. It bundles a reproducible YOLO11x-seg inference workflow, training utilities for refreshing the model on new datasets, and helper scripts for model comparison and batching. The current model was trained on 1,202 labeled tiles (21,072 instances) and achieves 0.97 bounding-box F1, 0.96 segmentation F1, and 0.99 mAP@0.5 at a 0.519 confidence threshold.

## Contents
- **inference.py** — end-to-end tiling, YOLO inference, polygon fusion, visualization, and metric reporting.
- **train.py** — structured training pipeline that builds YOLO datasets from multiple sources, trains with the Ultralytics CLI, and re-runs evaluation.
- **train-compare-models.py** — lightweight evaluator to benchmark multiple trained checkpoints on held-out data.
- **detectron_batching.py** — utility to queue Detectron2-style inference batches when experimenting with alternative backbones.
- **config.yaml / train-config.yaml / config_multitrain.yaml** — configuration entry points for inference, single-run training, and multi-run training sweeps.

## Hardware and Operating System Requirements
- CUDA-capable GPU (the provided configs target multi-GPU use; CPU-only inference is not yet optimized).
- Unix-like operating system (Linux recommended; Windows users should prefer WSL).
- Anaconda/Miniconda for environment management.
- At least 8 GB GPU VRAM and 16 GB system RAM to handle dense batches of 576×576 tiles.

## Installation
1. Install [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create and activate an environment:
   ```bash
   conda create -n collembola python=3.10
   conda activate collembola
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify the GPU is visible to PyTorch:
   ```bash
   python - <<'PY'
   import torch
   print('CUDA available:', torch.cuda.is_available())
   PY
   ```

## Image Acquisition Guidelines
To get the most reliable counts:
- Use a DSLR (APS-C or larger sensor) with a ring flash or diffuse lighting; keep exposure under 1/100 s.
- Prefer low ISO and darker substrates to maximize contrast with the white *Collembola* bodies.
- Save microscope slides as JPEG files.
- Optional: export ImageJ point annotations as CSVs (press **M**, then **Save as...**) with the same basename as the slide image to enable human/prediction comparisons.

## Data Organization
Place full-resolution slide images inside the folder referenced by `images_folder` in `config.yaml`. A typical layout is:
```
project_root/
├── config.yaml
├── data/
│   └── <study_name>/
│       ├── slide_001.jpg
│       ├── slide_001.csv   # optional ImageJ points
│       ├── slide_002.jpg
│       └── ...
```
Tiles are automatically created at 576×576 pixels with an optional 288-pixel shift for overlap.

## Inference Configuration (`config.yaml`)
Key parameters for `inference.py`:
```yaml
images_folder: data/jpgs_bart      # where source .jpg/.JPG slides live
experiments_dir: exps              # parent folder for experiment runs
model:
  folder: model                    # directory holding the checkpoint archive
  archive: best.pt.7z              # optional .7z to auto-extract
  weights: best.pt                 # final weight file name
  fuse: false                      # fuse model layers for speed
tile:
  size: 576
  shift: 288
parallel:
  n_gpus: 4                        # GPUs pinned to YOLO workers
  n_jobs: 16                       # CPU workers for tiling/fusion
  batch_size: 32                   # batch size per GPU during inference
fusion:
  best_conf: 0.732                 # detection confidence for fusion
  best_iou: 0.101                  # IoU threshold during graph-cut fusion
  alpha: 45.0                      # angular prior for polygon fusion
skip_bright: true                  # drop overly bright tiles
skip_blue: true                    # drop tiles dominated by blue hue
```
Adjust paths and parallelism to match local resources. The script extracts missing `.pt` weights from the archive automatically and records the originating image directory for reproducibility.

## Running Inference (`inference.py`)
Run the full pipeline with:
```bash
python inference.py
```
Key behaviors:
1. **Experiment folders** — Each run creates `exps/exp_XXXX_<timestamp>/` (or uses `EXP_DIR` if set) with subfolders for tiles, results, visualizations, logs, and stats. Previous tiles and inference JSONs can be reused interactively to avoid recomputation.
2. **Tiling & filtering** — Slides are split into `ori` and `shift` grids of 576×576 tiles. Bright or blueish tiles are skipped to reduce false positives; skipped reasons are recorded alongside tile metadata for later inspection.
3. **YOLO inference** — Multi-GPU, multi-processing inference loads one model per GPU, supports optional layer fusion, and can export tile-level LabelMe JSONs (`labelme/`) with confidences preserved in the `flags` section.
4. **Polygon assembly** — Tile detections are offset back to slide coordinates, cached in `raw_polygons.pkl`, and enriched with tile bounding boxes and skip reasons to speed up re-runs.
5. **Fusion & statistics** — Graph-cut fusion merges overlapping polygons using the configured confidence/IoU/angle priors. The pipeline writes per-slide counts, precision, recall, F1, and R² against CSV annotations to `stats/pipeline_results.csv` and a JSON summary. Annotated overlays are saved to `final_viz/`, `final_viz_ori/`, and `final_viz_shift/`.

### Useful runtime tips
- Set `EXP_DIR=/path/to/exp_dir` to direct outputs to a specific location (useful for deterministic reruns).
- Delete or move `raw_polygons.pkl` if fusion thresholds change; the build stage will regenerate polygons from tile JSONs.
- Use the interactive prompts to reuse tiles or inference JSONs from the latest experiment when only the fusion stage needs tweaking.

## Training Pipeline (`train.py`)
`train.py` constructs a YOLO dataset from one or more labeled sources and launches Ultralytics training:
- **Config-driven**: accepts a YAML file (default `train-config.yaml`) describing datasets, class names, image sizing, and hyperparameters such as optimizer, augmentation, and device selection.
- **Dataset handling**: images/labels can live in separate directories per dataset; filenames are sanitised and deduplicated. Optional `tst_reserve.json` files reserve specific images for testing.
- **Splitting**: train/val/test splits are generated per dataset using the configured ratios, with logs detailing sample counts and any size mismatches between images and labels.
- **Training**: runs the Ultralytics CLI with the assembled `data.yaml`, respecting multi-GPU `device` entries and mixed-precision/augmentation settings. Checkpoints and metrics are stored under `train_experiments/<timestamp>/`.
- **Evaluation**: after training, the script re-runs evaluation and summarises metrics alongside the split manifest for reproducibility.

Run training with:
```bash
python train.py --config train-config.yaml
```
To re-evaluate an existing experiment without retraining:
```bash
python train.py --evalonly path/to/train_experiments/exp_xxxx
```

### Multi-run training
`config_multitrain.yaml` defines lists of models, seeds, or datasets to sweep. Pass it to `train-compare-models.py` to launch multiple training runs and gather comparative metrics across checkpoints.

## Outputs at a glance
- **Tiles & provenance**: `tiles/` plus `tiles_source.txt` showing where tiles were sourced from.
- **Inference artifacts**: raw JSON detections (`results/`), polygon cache (`raw_polygons.pkl`), and LabelMe exports (`labelme/`).
- **Visual checks**: overlays in `final_viz/`, plus grid-specific variants for `ori` and `shift` tiles.
- **Metrics**: `stats/pipeline_results.csv` and `stats/summary.json` for slide-level metrics; training runs include Ultralytics logs and metrics under their experiment folders.

## Citation
If this workflow contributes to scientific publications, please cite the accompanying article and acknowledge the use of this repository.
