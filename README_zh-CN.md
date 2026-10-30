<div align="center">

# [Bioengineering · 已录用] GRIT-OBB

### Geometry-Guided Oriented Object Detection for Fetal Cardiac Structures in Ultrasound

**面向胎儿心脏结构超声图像的几何引导旋转目标检测**

[English](README.md) | **简体中文**

![Journal: Bioengineering](https://img.shields.io/badge/Journal-Bioengineering-2375ad)
![Status: Accepted](https://img.shields.io/badge/Status-Accepted-2e8b57)

Jiachen Yuan, Wenzheng Huang, Anteng Cai, Qiujin Wang, Zihan Ye, Ning Xin, Camilla Fazi, Federica Fontanella, and Netzahualcoyotl Hernandez-Cruz

</div>

## 论文信息

| 项目 | 内容 |
| --- | --- |
| 论文标题 | GRIT-OBB: Geometry-Guided Oriented Object Detection for Fetal Cardiac Structures in Ultrasound |
| 期刊 | [Bioengineering](https://www.mdpi.com/journal/bioengineering) |
| 状态 | 已录用，录用日期为 2026 年 10 月 29 日 |
| 引用文件 | [BibTeX](CITATION.bib) · [Citation File Format](CITATION.cff) |

正式文章链接、DOI 和出版信息将在论文上线后补充。

## 方法概述

GRIT-OBB（Geometry-guided Regional Interaction and Transformation for Oriented Bounding Box Detection）采用旋转边界框定位胎儿心脏结构。模型以 ResNet-50 + FPN 和水平区域建议网络为基础，包含：

- **监督式候选框校正**：生成与目标几何结构对齐的参考框。
- **旋转位置敏感对齐**：提取具有方向感知能力的区域特征。
- **几何引导区域交互**：结合外观特征与候选框之间的相对几何关系。
- **两阶段定位监督**：采用等价框回归和旋转 IoU 损失。

框架流程图见[英文首页](README.md#framework)。采样前对校正后的参考框停止梯度传播。仓库提供模型及训练、评估、预测入口，不包含数据集和训练好的 GRIT-OBB 权重。

## 主要结果

下表来自已录用论文，为五次运行的均值，单位为百分比。三个数据集分别训练模型。

| 数据集 | AP50 ↑ | AP75 ↑ | mAP50:95 ↑ |
| --- | ---: | ---: | ---: |
| FOCUS | 47.74 | 8.53 | 16.81 |
| OCD | 73.14 | 18.97 | 30.77 |
| AU-OPG | 82.65 | 59.66 | 50.66 |

论文还报告了结果变异性、定位误差、消融及鲁棒性实验。公开入口用于单次运行，完整多种子实验、消融和鲁棒性实验需要额外配置。

## 仓库结构

- [`obb_fusion/`](obb_fusion/)：模型、几何运算、特征采样、区域交互、后处理、配置和检查点。
- [`runtime/`](runtime/)：数据读取、确定性预处理和评估。
- [`run.py`](run.py)：训练、数据划分评估和单图预测。
- [`CITATION.bib`](CITATION.bib) / [`CITATION.cff`](CITATION.cff)：论文引用信息。

## 运行说明

### 1. 安装环境

使用 Python 3.10 或以上版本。源环境使用 PyTorch 2.7.0 和 torchvision 0.22.0，请按硬件选择匹配的安装包。CUDA 12.6 示例：

```bash
git clone https://github.com/bill1034/GRIT-OBB.git
cd GRIT-OBB
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy==2.1.2 Pillow==11.0.0 opencv-python
python run.py --help
```

### 2. 准备数据

`--data-root` 指向数据集目录的父目录。类别 ID 必须从 0 开始连续编号，并在所有划分中保持一致。`obb` 为原图像素坐标中的四个有序顶点，共八个数值。JSON 示例见[英文说明](README.md#2-data-format)。

| 命令参数 | 数据目录 | 训练 / 验证 / 测试图像数 | 类别数 |
| --- | --- | ---: | ---: |
| `focus` | `FOCUS_OBB` | 200 / 50 / 50 | 15 |
| `ocd` | `OCD_OBB` | 72 / 24 / 24 | 1 |
| `au_opg` | `AU_OPG_OBB` | 630 / 135 / 136 | 6 |

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

保持受试者级划分固定。图像转为 RGB，保持纵横比缩放至最长边 1024 像素，再在右侧和底部补零至 1024 × 1024。模型内部应用 ImageNet 归一化，标注随图像一致变换；读取器不使用随机增强。

### 3. 训练

```bash
python run.py train --dataset focus --data-root ./data --output ./runs/focus_seed0 --device cuda --seed 0 --epochs 50
```

当前入口采用 batch size 1、AdamW（学习率 0.0001，权重衰减 0.0001）、梯度裁剪 1 和 50 轮余弦调度。`best.pt` 根据验证集 AP50 选择，`last.pt` 每轮保存；输出目录必须为空。当前不支持恢复训练和早停。余弦调度在代码中固定为 50 轮，即使修改 `--epochs` 也不会自动改变调度周期。

论文完整训练与实验协议需要额外配置，上述命令提供单次运行参考。ImageNet 骨干权重在未缓存时自动下载；使用 `--no-pretrained` 可随机初始化。

### 4. 评估

```bash
python run.py evaluate --dataset focus --data-root ./data --checkpoint ./runs/focus_seed0/best.pt --split test --output ./results/test.json --device cuda
```

命令导出旋转 AP50（范围 0–1）、原图坐标中的真实框和预测结果。其他指标见 [`runtime/frozen_metric.py`](runtime/frozen_metric.py) 和 [`runtime/evaluation.py`](runtime/evaluation.py)。操作阈值应在验证集上确定。

### 5. 单图预测

```bash
python run.py predict --checkpoint ./runs/focus_seed0/best.pt --image ./example.png --output ./results/prediction.json --device cuda --score-threshold 0.5
```

使用本入口保存的检查点。输出包含类别 ID、名称、置信度和原图坐标中的四顶点框。显示阈值默认为 0.5。候选框较多时，精确多边形旋转 NMS 可能较慢。

## 数据与权重

- FOCUS 原始数据：[Zenodo](https://zenodo.org/records/14597550)。
- OCD 原始数据：[Deep-Cell-Tracking-EBB](https://github.com/LucasKirsten/Deep-Cell-Tracking-EBB)。
- AU-OPG 原始数据：[Hugging Face](https://huggingface.co/datasets/YSFF/AU-OPG)。

原始数据遵循各提供方的访问条件。本研究新增的 FOCUS 标注可向通讯作者申请用于研究，具体提供情况以申请结果为准。原始数据可访问并不代表本研究的新增标注和具体划分已包含在仓库中。当前不提供训练好的 GRIT-OBB 权重。

## 引用

如本工作对您的研究有帮助，请引用已录用论文。正式出版后将更新完整书目信息。

```bibtex
@article{yuan2026gritobb,
  author = {Yuan, Jiachen and Huang, Wenzheng and Cai, Anteng and Wang, Qiujin and Ye, Zihan and Xin, Ning and Fazi, Camilla and Fontanella, Federica and Hernandez-Cruz, Netzahualcoyotl},
  title = {{GRIT-OBB}: Geometry-Guided Oriented Object Detection for Fetal Cardiac Structures in Ultrasound},
  journal = {Bioengineering},
  year = {2026},
  note = {Accepted for publication}
}
```

## 致谢

本实现使用 [PyTorch](https://pytorch.org/) 和 [torchvision](https://github.com/pytorch/vision)，基于 [RoI Transformer](https://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Learning_RoI_Transformer_for_Oriented_Object_Detection_in_Aerial_Images_CVPR_2019_paper.html)，并参考 [OrientedFormer](https://github.com/wokaikaixinxin/OrientedFormer) 的关系几何设计。感谢原始数据集提供者。
