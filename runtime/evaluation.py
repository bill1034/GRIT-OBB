"""Validation/test adapters around the frozen v3 rotated evaluator."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import json
import os
import random
import sys

import numpy as np
import torch

from obb_fusion.geometry import rbox_to_corners
from .data_adapter import MedicalDataset


FROZEN_EVALUATOR_PATH = Path(__file__).resolve().parent / "frozen_metric.py"
_spec = importlib.util.spec_from_file_location("frozen_obb_repro_metric", FROZEN_EVALUATOR_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load frozen evaluator: {FROZEN_EVALUATOR_PATH}")
frozen_metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen_metric)


@contextmanager
def preserve_rng_state():
    """Make stochastic validation loss observational with respect to RNG."""
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state); np.random.set_state(np_state); torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _size(target, fallback):
    return tuple(target.get("metadata", {}).get("original_size", fallback))


def target_rows(target: dict, image_size: tuple[int, int]) -> list[dict]:
    height, width = _size(target, image_size)
    polygons = target.get("original_polygons")
    if polygons is None:
        polygons = rbox_to_corners(target["boxes"])
    return [{"image_id": str(target["image_id"]), "class_id": int(label),
             "bbox": polygon.detach().cpu().flatten().tolist(),
             "image_width": int(width), "image_height": int(height)}
            for polygon, label in zip(polygons, target["labels"])]


def prediction_rows(prediction: dict, image_id, image_size: tuple[int, int],
                    metadata: dict | None = None) -> list[dict]:
    height, width = tuple(metadata.get("original_size", image_size)) if metadata else image_size
    polygons = rbox_to_corners(prediction["boxes"])
    if metadata is not None:
        polygons = MedicalDataset.to_original_polygons(polygons, metadata)
    return [{"image_id": str(image_id), "class_id": int(label),
             "bbox": polygon.detach().cpu().flatten().tolist(), "score": float(score),
             "image_width": int(width), "image_height": int(height)}
            for polygon, label, score in zip(polygons, prediction["labels"], prediction["scores"])]


def evaluate(model, loader, device, class_names, *, compute_loss=True,
             heartbeat_path: Path | None=None, phase="validation", started=None):
    """Return deterministic val loss, frozen rotated AP50 and auditable raw rows."""
    was_training = model.training
    losses, ground_truth, predictions, image_count = [], [], [], 0
    with preserve_rng_state(), torch.no_grad():
        try:
            for batch_index,(images, targets) in enumerate(loader):
                if heartbeat_path is not None and (batch_index == 0 or batch_index % 10 == 0):
                    now=__import__("time").time(); path=Path(heartbeat_path)
                    payload={"state":"running","phase":phase,"heartbeat":now,"started":started,
                             "pid":os.getpid(),"images_completed":batch_index,
                             "peak_reserved_bytes":int(torch.cuda.max_memory_reserved(device)) if device.type=="cuda" else 0}
                    temporary=path.with_suffix(path.suffix+".tmp")
                    temporary.write_text(json.dumps(payload,indent=2),encoding="utf-8"); os.replace(temporary,path)
                images = [image.to(device) for image in images]
                moved = [{**t, "boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
                if compute_loss:
                    values = model.forward_train(images, moved)
                    total = sum(values.values())
                    if not torch.isfinite(total):
                        raise FloatingPointError("non-finite validation loss")
                    losses.append(float(total.detach().cpu()))
                # Target-free inference is a separate call; targets are never passed.
                batch_predictions = model.predict(images)
                for image, target, prediction in zip(images, targets, batch_predictions):
                    image_count += 1
                    size = tuple(image.shape[-2:])
                    ground_truth.extend(target_rows(target, size))
                    predictions.extend(prediction_rows(prediction, target["image_id"], size,
                                                       target.get("metadata")))
        finally:
            model.train(was_training)
    iou_fn = frozen_metric._batched_iou_lookup(ground_truth, predictions)
    ap50 = frozen_metric.compute_validation_map50(ground_truth, predictions, class_names, iou_fn=iou_fn)
    events, _, matches = frozen_metric._match_predictions(
        ground_truth, predictions, iou_threshold=.5, iou_fn=iou_fn)
    return {"val_loss": float(np.mean(losses)) if losses else None, "val_AP50": float(ap50),
            "matched_predictions_at_50": len(matches), "prediction_count": len(events),
            "image_count": image_count,
            "ground_truth": ground_truth, "predictions": predictions}


def full_metrics(rows: dict, class_names, operating_threshold: float,
                 small_area_threshold: float, medium_area_threshold: float):
    """Offline full suite; caller must enforce best-checkpoint/test gates."""
    gt, pred = rows["ground_truth"], rows["predictions"]
    return frozen_metric.evaluate_detections(gt, pred, class_names,
        operating_threshold=operating_threshold, small_area_threshold=small_area_threshold,
        medium_area_threshold=medium_area_threshold, num_images=rows["image_count"],
        iou_fn=frozen_metric._batched_iou_lookup(gt, pred))
