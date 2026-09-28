# GRIT-OBB

GRIT-OBB is an oriented object detector for medical images. It combines a
ResNet-50 feature pyramid backbone, a horizontal region proposal network,
supervised proposal correction, rotated position-sensitive alignment,
geometry-guided regional interaction, and two-stage localization supervision.

Model code and training, evaluation, and prediction commands are provided.
Trained weights and datasets are not included.

## Contents

- `obb_fusion/`: model architecture, geometry, sampling, regional interaction,
  rotated post-processing, configuration, and checkpoint utilities.
- `runtime/`: dataset reader, deterministic preprocessing, and evaluation code.
- `run.py`: training, split evaluation, and single-image prediction.

The runner supports individual training runs; multi-seed, ablation, and
robustness experiments require separate configuration.

## Environment

Use Python 3.10 or newer. The source environment uses PyTorch 2.7.0 and
torchvision 0.22.0. Install a matching PyTorch/torchvision build for your
hardware, then install NumPy, Pillow and OpenCV. For example, for CUDA 12.6:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==2.1.2 Pillow==11.0.0 opencv-python
python run.py --help
```

The backbone, FPN, and RPN use torchvision. The detection architecture builds on
RoI Transformer; the relation geometry is adapted from OrientedFormer.

## Data format

`--data-root` is the parent of the dataset directory:

| Key | Directory | Dataset |
| --- | --- | --- |
| `focus` | `FOCUS_OBB` | FOCUS detection annotations |
| `ocd` | `OCD_OBB` | OCD |
| `au_opg` | `AU_OPG_OBB` | AU-OPG |

```text
data/
  FOCUS_OBB/
    images/train/...
    images/val/...
    images/test/...
    annotations/instances_train_obb.json
    annotations/instances_val_obb.json
    annotations/instances_test_obb.json
```

Each split uses the following JSON format:

```json
{
  "images": [{"id": 1, "file_name": "example.png", "width": 640, "height": 480}],
  "categories": [{"id": 0, "name": "example_structure"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 0,
                   "obb": [100, 100, 180, 100, 180, 140, 100, 140]}]
}
```

Class IDs must be contiguous and zero-based, with identical class ordering
across splits. Each `obb` contains four ordered polygon vertices in original
image pixel coordinates. Keep participant-level partitions fixed. This release
does not provide the additional FOCUS anatomical annotations.

Images are converted to RGB, resized with preserved aspect ratio to a longest
side of 1,024 pixels, and zero-padded on the right and bottom to 1,024 x 1,024.
ImageNet normalization is applied inside the model. The reader transforms
oriented annotations consistently with the research preprocessing. No random
augmentation is applied.

## Train

```bash
python run.py train --dataset focus --data-root ./data --output ./runs/focus_seed0 --device cuda --seed 0 --epochs 50
```

Training uses batch size 1, AdamW (learning rate 0.0001, weight decay 0.0001),
gradient clipping at 1, and a 50-epoch cosine schedule. `best.pt` is selected by
validation AP50; `last.pt` is saved each epoch. Use an empty output directory.
Resume and early stopping are not supported.

ImageNet backbone weights are downloaded if not cached. Use `--no-pretrained`
for random initialization.

## Evaluate

```bash
python run.py evaluate --dataset focus --data-root ./data --checkpoint ./runs/focus_seed0/best.pt --split test --output ./results/test.json --device cuda
```

Evaluation exports rotated AP50 (0--1), ground truth, and predictions in original
image coordinates. Additional metrics are available in `runtime/frozen_metric.py`
and `runtime/evaluation.py`. Select operating thresholds on validation data.

## Predict

```bash
python run.py predict --checkpoint ./runs/focus_seed0/best.pt --image ./example.png --output ./results/prediction.json --device cuda --score-threshold 0.5
```

Use a checkpoint produced by this runner. Output contains class IDs, names,
scores, and four-vertex bounding polygons in original image coordinates.
The display threshold defaults to 0.5.

Exact polygon-based rotated NMS may be slow with large candidate sets.
