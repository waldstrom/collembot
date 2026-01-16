# COLLEMBOT — Collembola Counting Toolkit (v0.1.0)

COLLEMBOT is a reproducible pipeline for counting *Collembola* organisms in ecotoxicity soil slide images. It includes YOLOv11-seg inference, training utilities for curated datasets, and comparative benchmarking for segmentation backbones. Release: **v0.1.0**.

## Repository layout
- `configs/`
  - `inference.yaml` — inference configuration.
  - `train.yaml` — single-run training configuration.
  - `train_multirun.yaml` — multi-run training & model comparison configuration.
- `scripts/`
  - `run_inference.py` — tiling, YOLO inference, fusion, visualization, and metrics.
  - `train_model.py` — dataset preparation + YOLO training and evaluation.
  - `compare_models.py` — multi-model benchmarking (YOLO, Mask R-CNN, Mask2Former, MaskDINO).
  - `detectron_batching.py` — VRAM-aware batch sizing for Detectron2.
  - `upscale_images.py` — optional slide upscaling helper.
  - `gradio_app.py` — Colab-ready Gradio helper.
- `data/` — CC-BY-4.0 datasets and examples (see `data/README.md`).
- `notebooks/` — Colab walkthroughs and launchers.

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

## Data organization
Inference images live in the folder referenced by `images_folder` in `configs/inference.yaml`. A typical layout is:
```
project_root/
├── configs/
│   └── inference.yaml
├── data/
│   ├── inference-examples/
│   ├── train-labelme-amsterdam/
│   ├── train-labelme-basel/
│   ├── train-labelme-bayreuth/
│   ├── train-labelme-cai_pyramids/
│   └── train-labelme-coimbra/
```
Tiles are automatically created at 576×576 pixels with an optional 288-pixel shift for overlap.

## Inference configuration (`configs/inference.yaml`)
Key parameters for `scripts/run_inference.py`:
```yaml
images_folder: data/inference-examples
experiments_dir: exps
model:
  folder: model
  archive: best.pt.7z
  weights: best.pt
  fuse: false
```
Adjust paths and parallelism to match local resources. The script extracts missing `.pt` weights from the archive automatically and records the originating image directory for reproducibility.

## Running inference
```bash
python scripts/run_inference.py --config configs/inference.yaml
```
Key behaviors:
1. **Experiment folders** — Each run creates `exps/exp_XXXX_<timestamp>/` (or uses `EXP_DIR` if set) with subfolders for tiles, results, visualizations, logs, and stats.
2. **Tiling & filtering** — Slides are split into `ori` and `shift` grids of 576×576 tiles. Bright or blueish tiles are skipped to reduce false positives; skip reasons are recorded alongside tile metadata.
3. **YOLO inference** — Multi-GPU, multi-processing inference loads one model per GPU, supports optional layer fusion, and can export tile-level LabelMe JSONs (`labelme/`) with confidences preserved in the `flags` section.
4. **Polygon assembly** — Tile detections are offset back to slide coordinates, cached in `raw_polygons.pkl`, and enriched with tile bounding boxes and skip reasons to speed up re-runs.
5. **Fusion & statistics** — Graph-cut fusion merges overlapping polygons using the configured confidence/IoU/angle priors. The pipeline writes per-slide counts, precision, recall, F1, and R² against CSV annotations to `stats/pipeline_results.csv` and a JSON summary. Annotated overlays are saved to `final_viz/`, `final_viz_ori/`, and `final_viz_shift/`.

## Training pipeline (`scripts/train_model.py`)
`scripts/train_model.py` builds a YOLO dataset from one or more labeled sources and launches Ultralytics training:
- **Config-driven**: accepts a YAML file (default `configs/train.yaml`) describing datasets, class names, image sizing, and hyperparameters.
- **Dataset handling**: images/labels can live in separate directories per dataset; filenames are sanitised and deduplicated. Optional `tst_reserve.json` files reserve specific images for testing.
- **Splitting**: train/val/test splits are generated per dataset using the configured ratios, with logs detailing sample counts and any size mismatches between images and labels.
- **Training**: runs the Ultralytics CLI with the assembled `data.yaml`, respecting multi-GPU `device` entries and mixed-precision/augmentation settings. Checkpoints and metrics are stored under `train_experiments/<timestamp>/`.
- **Evaluation**: after training, the script re-runs evaluation and summarises metrics alongside the split manifest for reproducibility.

Run training with:
```bash
python scripts/train_model.py --config configs/train.yaml
```
To re-evaluate an existing experiment without retraining:
```bash
python scripts/train_model.py --evalonly path/to/train_experiments/exp_xxxx
```

### Multi-run training
`configs/train_multirun.yaml` defines lists of models, seeds, or datasets to sweep. Pass it to `scripts/compare_models.py` to launch multiple training runs and gather comparative metrics across checkpoints.

## Model weights
The published model weights are **AGPL-3.0 only** and can be obtained from Zenodo: **doi: 10.5281/zenodo.17987887**.

## Dependencies
This project depends on **YOLOv11 by Ultralytics**, which is licensed under **AGPL-3.0**. The dependency is not bundled with this repository.

## Licensing
- **Code**: MIT License (see `LICENSE`).
- **Data**: CC-BY-4.0 (see `data/README.md`).
- **Model weights**: AGPL-3.0 only (see Zenodo DOI above).

## Citation (required)
Author attribution is required for academic or derivative use (see `CITATION.md`):

**Wehrli & Meyer & Souza da Silva et al.; 2026; COLLEMBOT: AI-based counting of Collembola for OECD 232 tests (in preparation)**

Authors (* shared first authors): Micha Wehrli*, Adrian Meyer*, Éverton Souza da Silva*, Sam van Loon, Bart G. van Hall, Cornelis A. M. van Gestel, Tiago Natal-da-Luz, Max V. R. Döring, Heike Feldhaar, Magdalena Mair, Denis Jordan, Miriam Langer

Affiliations: Eawag (CH) · FHNW (CH) · University of Zurich (CH) · University of Bayreuth (DE) · Vrije Universiteit Amsterdam (NL) · Cloverstrategy Lda (PT)

## Contact
Adrian Meyer — adrian.meyer@fhnw.ch
