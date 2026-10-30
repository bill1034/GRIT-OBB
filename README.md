<div align="center">

# [Bioengineering · Accepted] GRIT-OBB

### Geometry-Guided Oriented Object Detection for Fetal Cardiac Structures in Ultrasound

**Geometry-guided Regional Interaction and Transformation for Oriented Bounding Box Detection**

**English** | [简体中文](README_zh-CN.md)

[Publication](#publication) · [Framework](#framework) · [Results](#results) · [Reproduction](#reproduction) · [Citation](#citation)

![Journal: Bioengineering](https://img.shields.io/badge/Journal-Bioengineering-2375ad)
![Status: Accepted](https://img.shields.io/badge/Status-Accepted-2e8b57)
![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776ab)
![PyTorch: 2.7.0](https://img.shields.io/badge/PyTorch-2.7.0-ee4c2c)

Jiachen Yuan, Wenzheng Huang, Anteng Cai, Qiujin Wang, Zihan Ye, Ning Xin, Camilla Fazi, Federica Fontanella, and Netzahualcoyotl Hernandez-Cruz

</div>

## Publication

| Item | Details |
| --- | --- |
| Title | GRIT-OBB: Geometry-Guided Oriented Object Detection for Fetal Cardiac Structures in Ultrasound |
| Journal | [Bioengineering](https://www.mdpi.com/journal/bioengineering) |
| Status | Accepted for publication on 29 October 2026 |
| Citation files | [BibTeX](CITATION.bib) · [Citation File Format](CITATION.cff) |

The final article link, DOI, and bibliographic details will be added after publication.

## Overview

GRIT-OBB localizes fetal cardiac structures with oriented bounding boxes. It combines a ResNet-50 feature pyramid backbone and horizontal region proposals with:

- **Supervised proposal correction** to provide geometry-aligned reference boxes.
- **Rotated position-sensitive alignment** to extract orientation-aware regional features.
- **Geometry-guided regional interaction** to combine appearance and relative box geometry.
- **Two-stage localization supervision** using equivalent-box regression and rotated IoU losses.

The repository provides model code and training, evaluation, and prediction commands. Datasets and trained GRIT-OBB checkpoints are not included.

## Framework

![GRIT-OBB framework](grit-obb-framework.svg?v=paper-figure2)

Figure 2 from the accepted manuscript. The central diagram shows the detection workflow, with modules (A)–(C) expanded in panels (a)–(c): (a) proposal correction and alignment; (b) geometry-guided relation residual; and (c) two-stage localization supervision, used only during training. Teal dashed arrows indicate detached geometry, while brown dashed arrows connect the corrected and final predictions to their supervision.

## Results

The following values are means across five runs from the accepted manuscript, expressed as percentages. Each dataset uses its own trained model.

| Dataset | AP50 ↑ | AP75 ↑ | mAP50:95 ↑ |
| --- | ---: | ---: | ---: |
| FOCUS | 47.74 | 8.53 | 16.81 |
| OCD | 73.14 | 18.97 | 30.77 |
| AU-OPG | 82.65 | 59.66 | 50.66 |

The paper also reports variability, localization analyses, ablations, and robustness experiments. The public runner supports individual runs; it does not automate the full experimental campaign.

## Repository Structure

```text
GRIT-OBB/
├── obb_fusion/              # Model, geometry, sampling, relation, and checkpoints
├── runtime/                 # Data adapter, preprocessing, and evaluation
├── run.py                   # Train / evaluate / predict entrypoint
├── grit-obb-framework.svg    # Figure 2 from the accepted manuscript
├── README.md                # English guide
├── README_zh-CN.md           # Chinese guide
├── CITATION.bib             # Accepted-paper BibTeX
└── CITATION.cff             # Machine-readable citation metadata
```

## Reproduction

### 1. Installation

```bash
git clone https://github.com/bill1034/GRIT-OBB.git
cd GRIT-OBB
```
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

### 2. Data format

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

### 3. Train

```bash
python run.py train --dataset focus --data-root ./data --output ./runs/focus_seed0 --device cuda --seed 0 --epochs 50
```

Training uses batch size 1, AdamW (learning rate 0.0001, weight decay 0.0001),
gradient clipping at 1, and a 50-epoch cosine schedule. `best.pt` is selected by
validation AP50; `last.pt` is saved each epoch. Use an empty output directory.
Resume and early stopping are not supported. The cosine schedule is fixed to 50 epochs in the current implementation, even when `--epochs` is changed. The paper's broader training and experiment protocol requires additional configuration; this command is a single-run reference workflow.

ImageNet backbone weights are downloaded if not cached. Use `--no-pretrained`
for random initialization.

### 4. Evaluate

```bash
python run.py evaluate --dataset focus --data-root ./data --checkpoint ./runs/focus_seed0/best.pt --split test --output ./results/test.json --device cuda
```

Evaluation exports rotated AP50 (0--1), ground truth, and predictions in original
image coordinates. Additional metrics are available in `runtime/frozen_metric.py`
and `runtime/evaluation.py`. Select operating thresholds on validation data.

### 5. Predict

```bash
python run.py predict --checkpoint ./runs/focus_seed0/best.pt --image ./example.png --output ./results/prediction.json --device cuda --score-threshold 0.5
```

Use a checkpoint produced by this runner. Output contains class IDs, names,
scores, and four-vertex bounding polygons in original image coordinates.
The display threshold defaults to 0.5.

Exact polygon-based rotated NMS may be slow with large candidate sets.

## Data and Checkpoints

| Dataset | Original data provider | Train / validation / test images | Classes |
| --- | --- | ---: | ---: |
| FOCUS | [Zenodo](https://zenodo.org/records/14597550) | 200 / 50 / 50 | 15 |
| OCD | [Deep-Cell-Tracking-EBB](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB) | 72 / 24 / 24 | 1 |
| AU-OPG | [Hugging Face](https://huggingface.co/datasets/YSFF/AU-OPG) | 630 / 135 / 136 | 6 |

The original datasets are available from their respective providers, subject to their access conditions. The additional FOCUS annotations generated in this study may be available for research purposes upon request to the corresponding author. Dataset access does not imply that the study-specific annotations or splits are included in this repository.

Trained GRIT-OBB checkpoints are not distributed in this release.

## Citation

If you use this work, please cite the accepted paper. Bibliographic details will be updated after publication.

```bibtex
@article{yuan2026gritobb,
  author = {Yuan, Jiachen and Huang, Wenzheng and Cai, Anteng and Wang, Qiujin and Ye, Zihan and Xin, Ning and Fazi, Camilla and Fontanella, Federica and Hernandez-Cruz, Netzahualcoyotl},
  title = {{GRIT-OBB}: Geometry-Guided Oriented Object Detection for Fetal Cardiac Structures in Ultrasound},
  journal = {Bioengineering},
  year = {2026},
  note = {Accepted for publication}
}
```

## Acknowledgements

This implementation uses [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision), builds on [RoI Transformer](https://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.html), and adapts relation geometry from [OrientedFormer](https://github.com/wokaikaixinxin/OrientedFormer). We thank the original dataset providers for making their resources available.

