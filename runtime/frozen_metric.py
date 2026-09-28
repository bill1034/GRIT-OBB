"""Rotated validation and final-test metrics for the v3 OBB campaign.

The validation path intentionally computes only rotated mAP50.  The complete
metric suite is an offline/test-time operation after validation has selected a
checkpoint and frozen the confidence threshold.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np


def _is_vendored_gra_runtime() -> bool:
    """Avoid importing the separate MMRotate package inside the GRA fork.

    GRA vendors an MMDetection OBB registry.  In the legacy environment its
    registry object is shared with the installed MMRotate 0.3 package, so
    importing MMRotate after the official GRA detector has registered its
    coders raises a duplicate-registration error.  The public evaluator does
    not need MMRotate for this path: its OpenCV convex-polygon IoU fallback is
    the same rotated geometry used by the other legacy runs.
    """

    try:
        import mmdet

        return "gra_orcnn_r50" in Path(mmdet.__file__).resolve().as_posix().lower()
    except Exception:
        return False


torch = None  # type: ignore[assignment]
box_iou_rotated = None
BaseMetric = object  # type: ignore[assignment,misc]
METRICS = None
QuadriBoxes = None
RotatedBoxes = None

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    from mmcv.ops import box_iou_rotated
except ImportError:
    box_iou_rotated = None

try:
    from mmengine.evaluator import BaseMetric
except ImportError:
    BaseMetric = object  # type: ignore[assignment,misc]

if not _is_vendored_gra_runtime():
    try:
        from mmrotate.registry import METRICS
        from mmrotate.structures.bbox import QuadriBoxes, RotatedBoxes
    except Exception:
        # Legacy MMRotate 0.3.x does not expose the MMEngine registry API, and
        # some forked legacy trees share its registry with their own coders.
        # The evaluator remains fully usable through its polygon IoU path.
        METRICS = None
        QuadriBoxes = None
        RotatedBoxes = None


MetricRow = dict[str, Any]
IoUFunction = Callable[[MetricRow, MetricRow], float]
IOU_THRESHOLDS = tuple(0.50 + 0.05 * index for index in range(10))


def _polygon_array(box: Sequence[float]) -> np.ndarray:
    array = np.asarray(box, dtype=np.float32)
    if array.shape != (8,) or not np.isfinite(array).all():
        raise ValueError("bbox must contain eight finite qbox coordinates")
    return array.reshape(4, 2)


def _polygon_area(box: Sequence[float]) -> float:
    return abs(float(cv2.contourArea(_polygon_array(box))))


def _polygon_center(box: Sequence[float]) -> np.ndarray:
    return _polygon_array(box).mean(axis=0)


def _opencv_iou(left: Sequence[float], right: Sequence[float]) -> float:
    left_points, right_points = _polygon_array(left), _polygon_array(right)
    left_area = _polygon_area(left)
    right_area = _polygon_area(right)
    intersection, _ = cv2.intersectConvexConvex(left_points, right_points)
    union = left_area + right_area - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    if torch is None or QuadriBoxes is None or box_iou_rotated is None:
        return _opencv_iou(left, right)
    qboxes = torch.tensor([left, right], dtype=torch.float32)
    rboxes = QuadriBoxes(qboxes).convert_to("rbox").tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    overlap = box_iou_rotated(rboxes[:1].to(device), rboxes[1:].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return float(overlap[0, 0].detach().cpu())


def _row_iou(prediction: MetricRow, target: MetricRow) -> float:
    return _iou(prediction["bbox"], target["bbox"])


def _batched_iou_lookup(
    ground_truth: Sequence[MetricRow], predictions: Sequence[MetricRow]
) -> IoUFunction:
    """Build one rotated-IoU matrix per class/image pair."""
    if torch is None or QuadriBoxes is None or box_iou_rotated is None:
        return _row_iou
    grouped_targets: dict[tuple[int, str], list[MetricRow]] = defaultdict(list)
    grouped_predictions: dict[tuple[int, str], list[MetricRow]] = defaultdict(list)
    for row in ground_truth:
        grouped_targets[(int(row["class_id"]), str(row["image_id"]))].append(row)
    for row in predictions:
        grouped_predictions[(int(row["class_id"]), str(row["image_id"]))].append(row)
    pair_values: dict[tuple[int, int], float] = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for key, pred_rows in grouped_predictions.items():
        target_rows = grouped_targets.get(key, [])
        if not target_rows:
            continue
        pred_qboxes = torch.tensor(
            [row["bbox"] for row in pred_rows], dtype=torch.float32, device=device
        )
        target_qboxes = torch.tensor(
            [row["bbox"] for row in target_rows], dtype=torch.float32, device=device
        )
        pred_rboxes = QuadriBoxes(pred_qboxes).convert_to("rbox").tensor
        target_rboxes = QuadriBoxes(target_qboxes).convert_to("rbox").tensor
        overlaps = box_iou_rotated(pred_rboxes, target_rboxes).detach().cpu().numpy()
        for pred_index, prediction in enumerate(pred_rows):
            for target_index, target in enumerate(target_rows):
                pair_values[(id(prediction), id(target))] = float(
                    overlaps[pred_index, target_index]
                )

    def lookup(prediction: MetricRow, target: MetricRow) -> float:
        return pair_values.get((id(prediction), id(target)), 0.0)

    return lookup


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def _average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    recall_levels = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(precision[recall >= level].max())
        if np.any(recall >= level)
        else 0.0
        for level in recall_levels
    ]
    return float(np.mean(interpolated))


def _class_ap(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    class_id: int,
    iou_threshold: float,
    iou_fn: IoUFunction,
) -> float:
    gt_by_image: dict[str, list[MetricRow]] = defaultdict(list)
    for row in ground_truth:
        if int(row["class_id"]) == class_id:
            gt_by_image[str(row["image_id"])].append(row)
    gt_count = sum(map(len, gt_by_image.values()))
    if gt_count == 0:
        return 0.0
    used: dict[str, set[int]] = defaultdict(set)
    ordered = sorted(
        (row for row in predictions if int(row["class_id"]) == class_id),
        key=lambda row: float(row["score"]),
        reverse=True,
    )
    true_positives: list[float] = []
    false_positives: list[float] = []
    for prediction in ordered:
        image_id = str(prediction["image_id"])
        targets = gt_by_image.get(image_id, [])
        overlaps = [
            iou_fn(prediction, target) if index not in used[image_id] else -1.0
            for index, target in enumerate(targets)
        ]
        best = int(np.argmax(overlaps)) if overlaps else -1
        matched = best >= 0 and overlaps[best] >= iou_threshold
        if matched:
            used[image_id].add(best)
        true_positives.append(float(matched))
        false_positives.append(float(not matched))
    if not ordered:
        return 0.0
    tp = np.cumsum(np.asarray(true_positives, dtype=np.float64))
    fp = np.cumsum(np.asarray(false_positives, dtype=np.float64))
    recall = tp / gt_count
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return _average_precision(recall, precision)


def _match_predictions(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    *,
    iou_threshold: float,
    iou_fn: IoUFunction,
    score_threshold: float | None = None,
    max_dets_per_image: int | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, int]], list[tuple[MetricRow, MetricRow, float]]]:
    """Greedy class-aware matching shared by fixed, AR, FROC, and LRP."""
    gt_by_class_image: dict[tuple[int, str], list[MetricRow]] = defaultdict(list)
    for row in ground_truth:
        gt_by_class_image[(int(row["class_id"]), str(row["image_id"]))].append(row)
    selected = [
        row for row in predictions
        if score_threshold is None or float(row["score"]) >= score_threshold
    ]
    if max_dets_per_image is not None:
        by_image: dict[str, list[MetricRow]] = defaultdict(list)
        for row in selected:
            by_image[str(row["image_id"])].append(row)
        selected = [
            row
            for rows in by_image.values()
            for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True)[
                :max_dets_per_image
            ]
        ]
    ordered = sorted(selected, key=lambda row: float(row["score"]), reverse=True)
    used: dict[tuple[int, str], set[int]] = defaultdict(set)
    events: list[dict[str, Any]] = []
    counts: dict[int, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    matches: list[tuple[MetricRow, MetricRow, float]] = []
    for prediction in ordered:
        class_id = int(prediction["class_id"])
        image_id = str(prediction["image_id"])
        targets = gt_by_class_image.get((class_id, image_id), [])
        overlaps = [
            iou_fn(prediction, target) if index not in used[(class_id, image_id)] else -1.0
            for index, target in enumerate(targets)
        ]
        best = int(np.argmax(overlaps)) if overlaps else -1
        matched = best >= 0 and overlaps[best] >= iou_threshold
        if matched:
            used[(class_id, image_id)].add(best)
            iou = float(overlaps[best])
            counts[class_id]["tp"] += 1
            matches.append((prediction, targets[best], iou))
        else:
            iou = 0.0
            counts[class_id]["fp"] += 1
        events.append(
            {
                "score": float(prediction["score"]),
                "tp": int(matched),
                "fp": int(not matched),
                "iou": iou,
            }
        )
    for class_id in range(
        max(
            [int(row["class_id"]) for row in ground_truth]
            + [int(row["class_id"]) for row in predictions]
            + [0]
        )
        + 1
    ):
        gt_count = sum(
            len(rows) for (row_class, _), rows in gt_by_class_image.items()
            if row_class == class_id
        )
        counts[class_id]["fn"] = gt_count - counts[class_id]["tp"]
    return events, dict(counts), matches


def _fixed_counts(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    num_classes: int,
    threshold: float,
    iou_fn: IoUFunction,
    match_iou_threshold: float,
) -> tuple[list[dict[str, int]], list[tuple[MetricRow, MetricRow, float]]]:
    _, raw_counts, matches = _match_predictions(
        ground_truth,
        predictions,
        iou_threshold=match_iou_threshold,
        iou_fn=iou_fn,
        score_threshold=threshold,
    )
    return (
        [raw_counts.get(class_id, {"tp": 0, "fp": 0, "fn": 0}) for class_id in range(num_classes)],
        matches,
    )


def _macro_fixed_metrics(counts: Sequence[dict[str, int]]) -> tuple[float, float, float]:
    precisions, recalls, f1s = [], [], []
    for row in counts:
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return _mean(precisions), _mean(recalls), _mean(f1s)


def _micro_fixed_metrics(counts: Sequence[dict[str, int]]) -> tuple[float, float, float]:
    tp = sum(row["tp"] for row in counts)
    fp = sum(row["fp"] for row in counts)
    fn = sum(row["fn"] for row in counts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def select_operating_threshold(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    num_classes: int,
    *,
    iou_fn: IoUFunction | None = None,
    match_iou_threshold: float = 0.5,
) -> float:
    """Select one validation-only score threshold by macro F1, then recall."""
    if not predictions:
        return 1.0
    iou_fn = iou_fn or _row_iou
    candidates = sorted({float(row["score"]) for row in predictions}, reverse=True)
    best: tuple[float, float, float] | None = None
    best_threshold = 1.0
    for threshold in candidates:
        counts, _ = _fixed_counts(
            ground_truth,
            predictions,
            num_classes,
            threshold,
            iou_fn,
            match_iou_threshold,
        )
        _, recall, f1 = _macro_fixed_metrics(counts)
        candidate = (f1, recall, threshold)
        if best is None or candidate > best:
            best, best_threshold = candidate, threshold
    return float(best_threshold)


def compute_validation_map50(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    class_names: Sequence[str],
    *,
    iou_fn: IoUFunction | None = None,
) -> float:
    """Return only the validation checkpoint-selection metric."""
    iou_fn = iou_fn or _batched_iou_lookup(ground_truth, predictions)
    return _mean(
        _class_ap(ground_truth, predictions, class_id, 0.5, iou_fn)
        for class_id in range(len(class_names))
    )


def _dimensions(row: MetricRow, dimensions: dict[str, tuple[float, float]]) -> tuple[float, float]:
    image_id = str(row["image_id"])
    width = float(row.get("image_width", dimensions.get(image_id, (1.0, 1.0))[0]))
    height = float(row.get("image_height", dimensions.get(image_id, (1.0, 1.0))[1]))
    return max(width, 1.0), max(height, 1.0)


def _normalized_area(row: MetricRow, dimensions: dict[str, tuple[float, float]]) -> float:
    width, height = _dimensions(row, dimensions)
    return _polygon_area(row["bbox"]) / (width * height)


def _size_rows(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    label: str,
    small_area_threshold: float,
    medium_area_threshold: float,
) -> tuple[list[MetricRow], list[MetricRow]]:
    dimensions = {
        str(row["image_id"]): (float(row["image_width"]), float(row["image_height"]))
        for row in ground_truth
        if "image_width" in row and "image_height" in row
    }

    def keep(row: MetricRow) -> bool:
        area = _normalized_area(row, dimensions)
        if label == "S":
            return area < small_area_threshold
        if label == "M":
            return small_area_threshold <= area < medium_area_threshold
        if label == "L":
            return area >= medium_area_threshold
        raise ValueError(f"unknown size label: {label}")

    return [row for row in ground_truth if keep(row)], [
        row for row in predictions if keep(row)
    ]


def _ar_at_100(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    iou_fn: IoUFunction,
    *,
    iou_threshold: float,
) -> float:
    if not ground_truth:
        return 0.0
    _, counts, _ = _match_predictions(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        iou_fn=iou_fn,
        max_dets_per_image=100,
    )
    return sum(row["tp"] for row in counts.values()) / len(ground_truth)


def _froc_metrics(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    iou_fn: IoUFunction,
    *,
    iou_threshold: float,
    num_images: int,
) -> dict[str, float]:
    targets = (0.25, 0.5, 1.0, 2.0)
    total_gt = len(ground_truth)
    image_count = max(int(num_images), 1)
    if not predictions or total_gt == 0:
        sensitivities = {target: 0.0 for target in targets}
        return {
            **{
                f"rotated/FROC_sensitivity@{target:g}_fp_per_image": value
                for target, value in sensitivities.items()
            },
            "rotated/FROC_mean_sensitivity": 0.0,
            "rotated/FROC_AUC_normalized_0_2": 0.0,
        }
    events, _, _ = _match_predictions(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        iou_fn=iou_fn,
    )
    x = [0.0]
    y = [0.0]
    tp = fp = 0
    for event in events:
        tp += int(event["tp"])
        fp += int(event["fp"])
        x.append(fp / image_count)
        y.append(tp / total_gt)
    unique_x: list[float] = []
    unique_y: list[float] = []
    for current_x, current_y in zip(x, y):
        if unique_x and math.isclose(current_x, unique_x[-1]):
            unique_y[-1] = max(unique_y[-1], current_y)
        else:
            unique_x.append(current_x)
            unique_y.append(current_y)

    def sensitivity(target: float) -> float:
        if target <= unique_x[0]:
            return unique_y[0]
        if target >= unique_x[-1]:
            return unique_y[-1]
        return float(np.interp(target, unique_x, unique_y))

    sensitivities = {target: sensitivity(target) for target in targets}
    grid = np.linspace(0.0, 2.0, 201)
    auc = float(np.trapz([sensitivity(float(value)) for value in grid], grid) / 2.0)
    return {
        **{
            f"rotated/FROC_sensitivity@{target:g}_fp_per_image": value
            for target, value in sensitivities.items()
        },
        "rotated/FROC_mean_sensitivity": _mean(sensitivities.values()),
        "rotated/FROC_AUC_normalized_0_2": auc,
    }


def _lrp_metrics(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    iou_fn: IoUFunction,
    *,
    iou_threshold: float,
) -> dict[str, float]:
    total_gt = len(ground_truth)
    if total_gt == 0:
        return {
            "rotated/moLRP": 0.0,
            "rotated/moLRP_localization": 0.0,
            "rotated/moLRP_false_positive": 0.0,
            "rotated/moLRP_false_negative": 0.0,
            "rotated/moLRP_operating_threshold": 1.0,
        }
    events, _, _ = _match_predictions(
        ground_truth,
        predictions,
        iou_threshold=iou_threshold,
        iou_fn=iou_fn,
    )
    candidate_indices = [-1]
    candidate_indices.extend(
        index for index in range(len(events))
        if index == len(events) - 1 or events[index + 1]["score"] != events[index]["score"]
    )
    best: tuple[float, float, float, float, float] | None = None
    for index in candidate_indices:
        active = events[: index + 1]
        tp = sum(int(event["tp"]) for event in active)
        fp = sum(int(event["fp"]) for event in active)
        fn = total_gt - tp
        localization_sum = sum(1.0 - float(event["iou"]) for event in active if event["tp"])
        denominator = max(tp + fp + fn, 1)
        lrp = (localization_sum + fp + fn) / denominator
        loc = localization_sum / max(tp, 1)
        fp_component = fp / max(tp + fp, 1)
        fn_component = fn / max(tp + fn, 1)
        threshold = float(events[index]["score"]) if index >= 0 else 1.0
        candidate = (lrp, loc, fp_component, fn_component, threshold)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return {
        "rotated/moLRP": float(best[0]),
        "rotated/moLRP_localization": float(best[1]),
        "rotated/moLRP_false_positive": float(best[2]),
        "rotated/moLRP_false_negative": float(best[3]),
        "rotated/moLRP_operating_threshold": float(best[4]),
    }


def evaluate_detections(
    ground_truth: Sequence[MetricRow],
    predictions: Sequence[MetricRow],
    class_names: Sequence[str],
    *,
    operating_threshold: float | None,
    small_area_threshold: float,
    medium_area_threshold: float | None = None,
    match_iou_threshold: float = 0.5,
    num_images: int | None = None,
    iou_fn: IoUFunction | None = None,
) -> dict[str, float]:
    """Compute the complete Section 7 suite using qbox/rbox overlap."""
    num_classes = len(class_names)
    if medium_area_threshold is None:
        medium_area_threshold = float(small_area_threshold)
    iou_fn = iou_fn or _batched_iou_lookup(ground_truth, predictions)
    if operating_threshold is None:
        operating_threshold = select_operating_threshold(
            ground_truth,
            predictions,
            num_classes,
            iou_fn=iou_fn,
            match_iou_threshold=match_iou_threshold,
        )
    operating_threshold = float(operating_threshold)
    num_images = num_images or len(
        {str(row["image_id"]) for row in ground_truth}
        | {str(row["image_id"]) for row in predictions}
    )

    ap_by_threshold = {
        threshold: [
            _class_ap(ground_truth, predictions, class_id, threshold, iou_fn)
            for class_id in range(num_classes)
        ]
        for threshold in (0.25, 0.50, 0.75)
    }
    per_class_ap50_95 = [
        _mean(
            _class_ap(ground_truth, predictions, class_id, threshold, iou_fn)
            for threshold in IOU_THRESHOLDS
        )
        for class_id in range(num_classes)
    ]
    counts, matches = _fixed_counts(
        ground_truth,
        predictions,
        num_classes,
        operating_threshold,
        iou_fn,
        match_iou_threshold,
    )
    macro_precision, macro_recall, macro_f1 = _macro_fixed_metrics(counts)
    micro_precision, micro_recall, micro_f1 = _micro_fixed_metrics(counts)
    matched_ious = [match[2] for match in matches]
    center_errors = []
    for prediction, target, _ in matches:
        diagonal = math.hypot(float(target["image_width"]), float(target["image_height"]))
        error = float(
            np.linalg.norm(
                _polygon_center(prediction["bbox"]) - _polygon_center(target["bbox"])
            )
        )
        center_errors.append(error / diagonal if diagonal else 0.0)

    result: dict[str, float] = {
        "rotated/AP25": _mean(ap_by_threshold[0.25]),
        "rotated/AP50": _mean(ap_by_threshold[0.50]),
        "rotated/AP75": _mean(ap_by_threshold[0.75]),
        "rotated/mAP25": _mean(ap_by_threshold[0.25]),
        "rotated/mAP50": _mean(ap_by_threshold[0.50]),
        "rotated/mAP75": _mean(ap_by_threshold[0.75]),
        "rotated/mAP50_95": _mean(per_class_ap50_95),
        "rotated/AR@100": _mean(
            _ar_at_100(
                ground_truth,
                predictions,
                iou_fn,
                iou_threshold=threshold,
            )
            for threshold in IOU_THRESHOLDS
        ),
        "rotated/macro_precision": macro_precision,
        "rotated/macro_recall": macro_recall,
        "rotated/macro_sensitivity": macro_recall,
        "rotated/macro_f1": macro_f1,
        "rotated/micro_precision": micro_precision,
        "rotated/micro_recall": micro_recall,
        "rotated/micro_sensitivity": micro_recall,
        "rotated/micro_f1": micro_f1,
        "rotated/precision": macro_precision,
        "rotated/recall": macro_recall,
        "rotated/f1": macro_f1,
        "rotated/mean_matched_iou": _mean(matched_ious),
        "rotated/normalized_center_error": _mean(center_errors),
        "rotated/operating_threshold": operating_threshold,
    }
    for class_id, class_name in enumerate(class_names):
        class_counts = counts[class_id]
        tp, fp, fn = class_counts["tp"], class_counts["fp"], class_counts["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        prefix = f"rotated/per_class/{class_name}"
        result[f"{prefix}/AP50_95"] = float(per_class_ap50_95[class_id])
        result[f"{prefix}/AP50"] = float(ap_by_threshold[0.50][class_id])
        result[f"{prefix}/AP75"] = float(ap_by_threshold[0.75][class_id])
        result[f"{prefix}/Precision"] = float(precision)
        result[f"{prefix}/Recall"] = float(recall)
        result[f"{prefix}/Sensitivity"] = float(recall)
        result[f"{prefix}/F1"] = float(f1)

    for size_label in ("S", "M", "L"):
        size_gt, size_predictions = _size_rows(
            ground_truth,
            predictions,
            size_label,
            float(small_area_threshold),
            float(medium_area_threshold),
        )
        size_ap = _mean(
            _class_ap(size_gt, size_predictions, class_id, threshold, iou_fn)
            for class_id in range(num_classes)
            for threshold in IOU_THRESHOLDS
        )
        size_ar = _mean(
            _ar_at_100(
                size_gt,
                size_predictions,
                iou_fn,
                iou_threshold=threshold,
            )
            for threshold in IOU_THRESHOLDS
        )
        result[f"rotated/AP_{size_label}"] = float(size_ap)
        result[f"rotated/AR_{size_label}"] = float(size_ar)
        if size_label == "S":
            result["rotated/small_mAP50"] = _mean(
                _class_ap(size_gt, size_predictions, class_id, 0.5, iou_fn)
                for class_id in range(num_classes)
            )

    result.update(
        _froc_metrics(
            ground_truth,
            predictions,
            iou_fn,
            iou_threshold=match_iou_threshold,
            num_images=num_images,
        )
    )
    result.update(
        _lrp_metrics(
            ground_truth,
            predictions,
            iou_fn,
            iou_threshold=match_iou_threshold,
        )
    )
    return {key: float(value) for key, value in result.items()}


def serialize_prediction_rows(
    path: Path, ground_truth: Sequence[MetricRow], predictions: Sequence[MetricRow]
) -> None:
    """Write one JSON object per image so test inference is auditable/replayable."""
    by_image: dict[str, dict[str, list[MetricRow]]] = defaultdict(
        lambda: {"ground_truth": [], "predictions": []}
    )
    for row in ground_truth:
        by_image[str(row["image_id"])]["ground_truth"].append(row)
    for row in predictions:
        by_image[str(row["image_id"])]["predictions"].append(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for image_id in sorted(by_image):
            handle.write(
                json.dumps(
                    {"image_id": image_id, **by_image[image_id]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        handle.flush()
    temporary.replace(path)


def load_prediction_rows(path: Path) -> tuple[list[MetricRow], list[MetricRow]]:
    ground_truth: list[MetricRow] = []
    predictions: list[MetricRow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            ground_truth.extend(row.get("ground_truth", []))
            predictions.extend(row.get("predictions", []))
    return ground_truth, predictions


class MedicalOBBMetric(BaseMetric):  # type: ignore[misc]
    """MMEngine metric wrapper with a deliberately small validation mode."""

    default_prefix = None

    def __init__(
        self,
        operating_threshold: float | None = None,
        small_area_threshold: float = 0.0,
        medium_area_threshold: float | None = None,
        metric_key: str = "rotated/mAP50",
        mode: str = "validation",
        match_iou_threshold: float = 0.5,
        collect_device: str = "cpu",
        prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        if metric_key not in {"rotated/mAP50", "rotated/mAP50_95"}:
            raise ValueError(f"unsupported checkpoint metric key: {metric_key}")
        if mode not in {"validation", "test"}:
            raise ValueError(f"unsupported metric mode: {mode}")
        self.metric_key = metric_key
        self.mode = mode
        self.operating_threshold = operating_threshold
        self.small_area_threshold = float(small_area_threshold)
        self.medium_area_threshold = (
            float(medium_area_threshold)
            if medium_area_threshold is not None
            else float(small_area_threshold)
        )
        self.match_iou_threshold = float(match_iou_threshold)
        if BaseMetric is not object:
            super().__init__(collect_device=collect_device, prefix=prefix)
        else:
            self.results: list[Any] = []

    @staticmethod
    def _as_qboxes(boxes: Any) -> np.ndarray:
        if hasattr(boxes, "convert_to"):
            boxes = boxes.convert_to("qbox").tensor
        elif hasattr(boxes, "tensor"):
            boxes = boxes.tensor
        if (
            torch is not None
            and RotatedBoxes is not None
            and hasattr(boxes, "shape")
            and len(boxes.shape) == 2
            and boxes.shape[-1] == 5
        ):
            tensor = boxes if isinstance(boxes, torch.Tensor) else torch.as_tensor(boxes)
            boxes = RotatedBoxes(tensor).convert_to("qbox").tensor
        if hasattr(boxes, "detach"):
            boxes = boxes.detach().cpu().numpy()
        return np.asarray(boxes, dtype=np.float32).reshape(-1, 8)

    @staticmethod
    def _as_numpy(values: Any) -> np.ndarray:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        return np.asarray(values)

    def process(self, data_batch: Any, data_samples: Iterable[Any]) -> None:
        for sample in data_samples:
            if isinstance(sample, dict):
                image_id = str(sample["img_id"])
                shape = sample["ori_shape"]
                gt_instances = sample["gt_instances"]
                pred_instances = sample["pred_instances"]
                gt_bboxes, gt_label_values = gt_instances["bboxes"], gt_instances["labels"]
                pred_bboxes = pred_instances["bboxes"]
                pred_label_values = pred_instances["labels"]
                pred_score_values = pred_instances["scores"]
            else:
                image_id = str(sample.img_id)
                shape = sample.ori_shape
                gt_bboxes, gt_label_values = sample.gt_instances.bboxes, sample.gt_instances.labels
                pred_bboxes = sample.pred_instances.bboxes
                pred_label_values = sample.pred_instances.labels
                pred_score_values = sample.pred_instances.scores
            height, width = int(shape[0]), int(shape[1])
            gt_boxes = self._as_qboxes(gt_bboxes)
            gt_labels = self._as_numpy(gt_label_values)
            pred_boxes = self._as_qboxes(pred_bboxes)
            pred_labels = self._as_numpy(pred_label_values)
            pred_scores = self._as_numpy(pred_score_values)
            targets = [
                {
                    "image_id": image_id,
                    "class_id": int(label),
                    "bbox": box.tolist(),
                    "image_width": width,
                    "image_height": height,
                }
                for box, label in zip(gt_boxes, gt_labels)
            ]
            predictions = [
                {
                    "image_id": image_id,
                    "class_id": int(label),
                    "bbox": box.tolist(),
                    "score": float(score),
                    "image_width": width,
                    "image_height": height,
                }
                for box, label, score in zip(pred_boxes, pred_labels, pred_scores)
            ]
            self.results.append((targets, predictions))

    def compute_metrics(self, results: list[Any]) -> dict[str, float]:
        ground_truth = [row for targets, _ in results for row in targets]
        predictions = [row for _, detections in results for row in detections]
        class_names = tuple(self.dataset_meta["classes"])
        iou_fn = _batched_iou_lookup(ground_truth, predictions)
        if self.mode == "validation":
            ap50 = [
                _class_ap(ground_truth, predictions, class_id, 0.5, iou_fn)
                for class_id in range(len(class_names))
            ]
            return {"rotated/mAP50": _mean(ap50)}
        return evaluate_detections(
            ground_truth,
            predictions,
            class_names,
            operating_threshold=self.operating_threshold,
            small_area_threshold=self.small_area_threshold,
            medium_area_threshold=self.medium_area_threshold,
            match_iou_threshold=self.match_iou_threshold,
            num_images=len(results),
            iou_fn=iou_fn,
        )


if METRICS is not None:
    METRICS.register_module(module=MedicalOBBMetric, force=True)
