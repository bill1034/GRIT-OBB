# GRIT-OBB

GRIT-OBB is an oriented object detector for medical images. It combines a
ResNet-50 feature pyramid backbone, a horizontal region proposal network,
supervised proposal correction, rotated position-sensitive alignment,
geometry-guided regional interaction, and two-stage localization supervision.

This repository provides **model source code, a portable running entrypoint,
and documentation**. Trained GRIT-OBB weights, medical images, annotations,
and experiment outputs are not distributed.

## Contents

- `obb_fusion/`: model architecture, geometry, sampling, regional interaction,
  rotated post-processing, configuration, and checkpoint utilities.
- `runtime/`: dataset reader, deterministic preprocessing, and evaluation code.
- `run.py`: training, split evaluation, and single-image prediction.

The model implementation is preserved from the research code. The public
entrypoint is a simplified portable runner, not the original experiment
orchestration system. It does not reproduce the complete multi-seed,
ablation, or robustness study automatically.

## Environment

Use Python 3.10 or newer. The source environment uses PyTorch 2.7.0 and
torchvision 0.22.0. Install a matching PyTorch/torchvision build for your
hardware, then install NumPy, Pillow and OpenCV. For example, for CUDA 12.6:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==2.1.2 Pillow==11.0.0 opencv-python
python run.py --help
```

The backbone, FPN and horizontal RPN use torchvision. Custom rotated geometry,
sampling and post-processing are implemented in the model package. No MMDetection
or MMRotate installation is required by this release.

## Data format

Prepare your own appropriately authorized data. `--data-root` is the parent of
the dataset directory. Dataset keys map to directories as follows:

| Key | Directory | Dataset |
| --- | --- | --- |
| `d02` | `D02_OBB` | FOCUS detection annotations |
| `ocd` | `OCD_OBB` | OCD |
| `au_opg` | `AU_OPG_OBB` | AU-OPG |

```text
data/
  D02_OBB/
    images/train/...
    images/val/...
    images/test/...
    annotations/instances_train_obb.json
    annotations/instances_val_obb.json
    annotations/instances_test_obb.json
```

Each JSON file has `images`, `annotations`, and `categories` arrays. A minimal
schema example (illustrative only) is:

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
python run.py train --dataset d02 --data-root ./data --output ./runs/focus_seed0 --device cuda --seed 0 --epochs 50
```

The runner uses batch size 1, AdamW (learning rate 0.0001, weight decay 0.0001),
gradient clipping at 1, and a cosine learning-rate schedule with a 50-epoch
horizon. It saves `best.pt` according to validation AP50 and `last.pt` after each
epoch. It does not use the test split for checkpoint selection. Use an empty
output directory for each run. This simplified runner has no resume or early
stopping support; 50 epochs is its default run length.

By default, torchvision downloads ImageNet backbone initialization if it is
not cached. These are generic backbone weights, not trained GRIT-OBB weights.
Use `--no-pretrained` to initialize without downloading backbone weights.

## Evaluate

```bash
python run.py evaluate --dataset d02 --data-root ./data --checkpoint ./runs/focus_seed0/best.pt --split test --output ./results/test.json --device cuda
```

The entrypoint reports rotated AP50 and exports ground truth and predictions
in original image coordinates. AP50 is returned on the evaluator's native
0-to-1 scale. It is not the complete manuscript metric table. Additional
metric functions are available in `runtime/frozen_metric.py` and
`runtime/evaluation.py`; operating thresholds must be selected using validation
data only.

## Predict

```bash
python run.py predict --checkpoint ./runs/focus_seed0/best.pt --image ./example.png --output ./results/prediction.json --device cuda --score-threshold 0.5
```

Prediction requires a checkpoint produced by this runner. Output includes
zero-based class IDs, class names, scores and four-vertex bounding polygons in
original image coordinates. The example threshold is for displaying predictions;
it is not a validated clinical operating threshold. No trained checkpoint is
included in this repository.

The native exact polygon-based rotated NMS can be slow when many candidates
survive score filtering. Parameter count or FLOPs alone should not be interpreted
as end-to-end inference speed. This is research code for oriented localization.
