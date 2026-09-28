"""Read-only adapter for the frozen ``preprocessed_obb_v3`` medical OBB data.

The adapter uses the authoritative per-split OBB JSON files. Preprocessing
matches the frozen OrientedFormer formal pipeline: MMDetection keep-ratio
resize with rounded integer dimensions, OpenCV bilinear interpolation, then
zero padding on the right and bottom. Boxes use ``(cx, cy, long_side,
short_side, theta)`` with theta in radians and ``[-pi/2, pi/2)``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset


_DATASETS = {
    "d02": "D02_OBB",
    "ocd": "OCD_OBB",
    "au_opg": "AU_OPG_OBB",
}
_DATA_ROOT = Path("data")


def _canonicalize_longsidele90(box: np.ndarray) -> np.ndarray:
    cx, cy, width, height, angle = map(float, box)
    if width < height:
        width, height = height, width
        angle += math.pi / 2
    angle = (angle + math.pi / 2) % math.pi - math.pi / 2
    return np.asarray([cx, cy, width, height, angle], dtype=np.float32)


def polygon_to_longsidele90(polygons: Tensor) -> Tensor:
    """Convert N quadrilaterals to long-side-le90 rotated rectangles."""
    array = polygons.detach().cpu().numpy().astype(np.float32, copy=False)
    result = []
    for polygon in array.reshape(-1, 4, 2):
        (cx, cy), (width, height), angle_deg = cv2.minAreaRect(polygon)
        result.append(_canonicalize_longsidele90(
            np.asarray([cx, cy, width, height, math.radians(angle_deg)])))
    if not result:
        return torch.empty((0, 5), dtype=torch.float32, device=polygons.device)
    return torch.as_tensor(np.stack(result), dtype=torch.float32, device=polygons.device)


def longsidele90_to_polygon(boxes: Tensor) -> Tensor:
    """Convert long-side-le90 boxes to quadrilaterals."""
    device = boxes.device
    result = []
    for cx, cy, width, height, angle in boxes.detach().cpu().numpy():
        result.append(cv2.boxPoints(((float(cx), float(cy)),
                                    (float(width), float(height)),
                                    math.degrees(float(angle)))))
    if not result:
        return torch.empty((0, 4, 2), dtype=torch.float32, device=device)
    return torch.as_tensor(np.stack(result), dtype=torch.float32, device=device)


def rescale_rboxes_formal(boxes: Tensor, scale_factor: tuple[float, float]) -> Tensor:
    """Match MMRotate ``RotatedBoxes.rescale_`` and le90 regularization."""
    if not len(boxes):
        return boxes.clone()
    scale_x, scale_y = scale_factor
    cx, cy, width_box, height_box, angle = boxes.unbind(dim=-1)
    cos_value, sin_value = torch.cos(angle), torch.sin(angle)
    cx = cx * scale_x
    cy = cy * scale_y
    width_box = width_box * torch.sqrt(
        (scale_x * cos_value) ** 2 + (scale_y * sin_value) ** 2)
    height_box = height_box * torch.sqrt(
        (scale_x * sin_value) ** 2 + (scale_y * cos_value) ** 2)
    angle = torch.atan2(scale_x * sin_value, scale_y * cos_value)
    swap = width_box < height_box
    long_side = torch.where(swap, height_box, width_box)
    short_side = torch.where(swap, width_box, height_box)
    angle = torch.where(swap, angle + math.pi / 2, angle)
    angle = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    return torch.stack((cx, cy, long_side, short_side, angle), dim=-1)


def _formal_resize_and_pad(array: np.ndarray, polygons: Tensor, size: int):
    """Match MMDetection Resize(keep_ratio=True)+Pad(size), without augments."""
    height, width = array.shape[:2]
    scale = min(size / width, size / height)
    # mmcv rescale_size uses int(value * scale + 0.5).
    resized_width = min(size, int(width * scale + 0.5))
    resized_height = min(size, int(height * scale + 0.5))
    scale_x = resized_width / width
    scale_y = resized_height / height
    resized = cv2.resize(array, (resized_width, resized_height),
                         interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:resized_height, :resized_width] = resized
    image = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1)

    # OrientedFormer converts qboxes to rboxes before Resize. Its RotatedBoxes
    # rescale_ applies the realized x/y factors to centers, sides, and angle.
    boxes = rescale_rboxes_formal(
        polygon_to_longsidele90(polygons), (scale_x, scale_y))
    metadata = {
        "scale_factor": [float(scale_x), float(scale_y)],
        "padding": [0.0, 0.0],
        "original_size": [int(height), int(width)],
        "resized_size": [int(resized_height), int(resized_width)],
        "input_size": [int(size), int(size)],
        "resize_backend": "opencv_INTER_LINEAR_uint8",
        "pad_placement": "right_bottom",
    }
    return image, boxes, metadata


class MedicalDataset(Dataset):
    """Frozen real-data reader with no augmentation or model-side behavior."""

    def __init__(self, dataset_key: str, split: str, size: int = 1024, data_root=None):
        if dataset_key not in _DATASETS:
            raise ValueError(f"dataset_key must be one of {tuple(_DATASETS)}")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a positive integer")
        self.dataset_key = dataset_key
        self.split = split
        self.size = size
        self.root = Path(data_root or _DATA_ROOT) / _DATASETS[dataset_key]
        annotation_path = self.root / "annotations" / f"instances_{split}_obb.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = sorted(data["categories"], key=lambda row: int(row["id"]))
        category_ids = [int(row["id"]) for row in categories]
        if category_ids != list(range(len(categories))):
            raise ValueError("Canonical category ids must be contiguous and zero-based")
        self.class_names = [str(row["name"]) for row in categories]
        grouped = {int(row["id"]): [] for row in data["images"]}
        for annotation in data["annotations"]:
            image_id = int(annotation["image_id"])
            if image_id not in grouped:
                raise ValueError("Annotation references an unknown image id")
            grouped[image_id].append(annotation)
        self.samples = []
        seen_ids = set()
        for row in data["images"]:
            image_id = int(row["id"])
            if image_id in seen_ids:
                raise ValueError("Duplicate image id in split")
            seen_ids.add(image_id)
            self.samples.append({
                "image_id": image_id,
                "file_name": str(row["file_name"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "annotations": grouped[image_id],
            })

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def to_original_polygons(polygons: Tensor, metadata: dict) -> Tensor:
        padding = polygons.new_tensor(metadata["padding"])
        return (polygons - padding) / polygons.new_tensor(metadata["scale_factor"])

    def __getitem__(self, index: int):
        sample = self.samples[index]
        path = (self.root / "images" / self.split / sample["file_name"]).resolve()
        expected_root = (self.root / "images" / self.split).resolve()
        if not path.is_relative_to(expected_root):
            raise ValueError("Image path escapes the canonical split directory")
        with Image.open(path) as source:
            array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        if array.shape[:2] != (sample["height"], sample["width"]):
            raise ValueError(f"Decoded dimensions differ from annotation: {path}")
        polygons = []
        labels = []
        for annotation in sample["annotations"]:
            polygon = np.asarray(annotation["obb"], dtype=np.float32).reshape(4, 2)
            if not np.isfinite(polygon).all():
                raise ValueError(f"Non-finite OBB coordinates: {path}")
            polygons.append(polygon)
            labels.append(int(annotation["category_id"]))
        original = torch.as_tensor(np.stack(polygons), dtype=torch.float32) if polygons else torch.empty((0, 4, 2))
        image, boxes, metadata = _formal_resize_and_pad(array, original, self.size)
        metadata.update({"file_name": sample["file_name"], "dataset_key": self.dataset_key,
                         "split": self.split})
        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": sample["image_id"],
            "original_polygons": original,
            "metadata": metadata,
        }
        return image, target
