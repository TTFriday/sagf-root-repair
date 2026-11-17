from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

from .constants import IGNORE_INDEX, NUM_CLASSES, ROOT_CLASSES
from .phenotyping import skeleton_length, skeleton_path_length


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, num_classes: int = NUM_CLASSES, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes) & (prediction >= 0) & (prediction < num_classes)
    indices = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def class_metrics_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    for cls in range(matrix.shape[0]):
        tp = float(matrix[cls, cls])
        fp = float(matrix[:, cls].sum() - tp)
        fn = float(matrix[cls, :].sum() - tp)
        gt_empty, pred_empty = tp + fn == 0, tp + fp == 0
        dice = 1.0 if gt_empty and pred_empty else 2 * tp / max(1.0, 2 * tp + fp + fn)
        iou = 1.0 if gt_empty and pred_empty else tp / max(1.0, tp + fp + fn)
        precision = 1.0 if pred_empty and gt_empty else tp / max(1.0, tp + fp)
        recall = 1.0 if gt_empty else tp / max(1.0, tp + fn)
        values.update({f"dice_class_{cls}": dice, f"iou_class_{cls}": iou, f"precision_class_{cls}": precision, f"recall_class_{cls}": recall})
    return values


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    denominator = prediction.sum() + target.sum()
    return 1.0 if denominator == 0 else float(2 * np.logical_and(prediction, target).sum() / denominator)


def cldice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return 1.0
    if not prediction.any() or not target.any():
        return 0.0
    pred_skeleton, target_skeleton = skeletonize(prediction), skeletonize(target)
    precision = np.logical_and(pred_skeleton, target).sum() / max(1, pred_skeleton.sum())
    sensitivity = np.logical_and(target_skeleton, prediction).sum() / max(1, target_skeleton.sum())
    return float(2 * precision * sensitivity / max(1e-12, precision + sensitivity))


def connected_component_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labels, pred_count = ndimage.label(prediction.astype(bool), structure)
    target_labels, target_count = ndimage.label(target.astype(bool), structure)
    component_error = float(abs(pred_count - target_count))
    if target_count == 0:
        largest_recall = 1.0 if pred_count == 0 else 0.0
    else:
        sizes = np.bincount(target_labels.ravel())[1:]
        largest_id = int(np.argmax(sizes)) + 1
        largest = target_labels == largest_id
        largest_recall = float(np.logical_and(prediction, largest).sum() / max(1, largest.sum()))
    return {"component_count_error": component_error, "largest_component_recall": largest_recall}


def _point_masks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _skeleton_points(skeletonize(mask.astype(bool)))


def _skeleton_points(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    endpoints = skeleton & (count == 1)
    branches = skeleton & (count >= 3)
    branches = ndimage.label(branches, np.ones((3, 3), dtype=np.uint8))[0]
    branch_centers = np.zeros_like(skeleton)
    for index in range(1, branches.max() + 1):
        coordinates = np.argwhere(branches == index)
        if len(coordinates):
            branch_centers[tuple(np.rint(coordinates.mean(0)).astype(int))] = True
    return endpoints, branch_centers


def tolerant_point_metrics(predicted_points: np.ndarray, target_points: np.ndarray, tolerance: float = 5.0) -> dict[str, float]:
    predicted, target = np.argwhere(predicted_points), np.argwhere(target_points)
    if not len(predicted) and not len(target):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "count_error": 0.0}
    if not len(predicted) or not len(target):
        return {"precision": 0.0 if len(predicted) else 1.0, "recall": 0.0 if len(target) else 1.0, "f1": 0.0, "count_error": float(abs(len(predicted) - len(target)))}
    distances = cdist(predicted, target)
    rows, columns = linear_sum_assignment(distances)
    true_positive = int(np.sum(distances[rows, columns] <= tolerance))
    precision = true_positive / len(predicted)
    recall = true_positive / len(target)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "count_error": float(abs(len(predicted) - len(target)))}


def surface_distances(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    prediction, target = prediction.astype(bool), target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0, 0.0
    if not prediction.any() or not target.any():
        return math.nan, math.nan
    structure = np.ones((3, 3), dtype=bool)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure, border_value=0)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure, border_value=0)
    target_distance = ndimage.distance_transform_edt(~target_surface)
    pred_distance = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate([target_distance[pred_surface], pred_distance[target_surface]])
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def compute_metrics(prediction: np.ndarray, target: np.ndarray, point_tolerance: float = 5.0) -> dict[str, float]:
    prediction, target = np.asarray(prediction).copy(), np.asarray(target)
    prediction[target == IGNORE_INDEX] = 0
    valid_target = target.copy()
    matrix = confusion_matrix(prediction, valid_target)
    result = class_metrics_from_confusion(matrix)
    valid = target != IGNORE_INDEX
    root_pred = np.isin(prediction, ROOT_CLASSES) & valid
    root_target = np.isin(target, ROOT_CLASSES) & valid
    result["complete_root_dice"] = binary_dice(root_pred, root_target)
    result["complete_root_cldice"] = cldice_score(root_pred, root_target)
    result.update(connected_component_metrics(root_pred, root_target))
    target_length = skeleton_length(root_target)
    result["skeleton_length_error_px"] = float(skeleton_length(root_pred) - target_length)
    result["skeleton_length_absolute_error_px"] = abs(result["skeleton_length_error_px"])
    pred_end, pred_branch = _point_masks(root_pred)
    target_end, target_branch = _point_masks(root_target)
    for name, pred_points, target_points in (("endpoint", pred_end, target_end), ("branchpoint", pred_branch, target_branch)):
        for metric, value in tolerant_point_metrics(pred_points, target_points, point_tolerance).items():
            result[f"{name}_{metric}"] = value
    result["root_hd95_px"], result["root_asd_px"] = surface_distances(root_pred, root_target)
    return result


@dataclass
class TargetFeatures:
    """Prediction-independent parts of compute_metrics, computed once per label image."""

    target: np.ndarray
    valid: np.ndarray
    root: np.ndarray
    skeleton: np.ndarray
    component_count: int
    largest: np.ndarray | None
    length: float
    endpoints: np.ndarray
    branch_centers: np.ndarray
    surface_distance: np.ndarray | None
    scored: np.ndarray
    codes: np.ndarray


def target_features(target: np.ndarray) -> TargetFeatures:
    target = np.asarray(target)
    valid = target != IGNORE_INDEX
    root = np.isin(target, ROOT_CLASSES) & valid
    skeleton = skeletonize(root)
    labels, count = ndimage.label(root, np.ones((3, 3), dtype=np.uint8))
    largest = None
    if count:
        largest = labels == int(np.argmax(np.bincount(labels.ravel())[1:])) + 1
    endpoints, branch_centers = _skeleton_points(skeleton)
    surface_distance = None
    if root.any():
        surface = root ^ ndimage.binary_erosion(root, structure=np.ones((3, 3), dtype=bool), border_value=0)
        surface_distance = ndimage.distance_transform_edt(~surface)
    scored = np.flatnonzero(valid & (target >= 0) & (target < NUM_CLASSES))
    codes = NUM_CLASSES * target.ravel()[scored].astype(np.int64)
    return TargetFeatures(target, valid, root, skeleton, int(count), largest, skeleton_path_length(skeleton), endpoints,
                          branch_centers, surface_distance, scored, codes)


def _window(mask: np.ndarray, margin: int = 2) -> tuple[slice, slice]:
    """Bounding box of mask grown by margin, with an even origin so that rounding of centroids is unchanged."""
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if not len(rows):
        return slice(0, 1), slice(0, 1)
    y0, x0 = max(0, rows[0] - margin) // 2 * 2, max(0, cols[0] - margin) // 2 * 2
    return slice(y0, rows[-1] + margin + 1), slice(x0, cols[-1] + margin + 1)


def compute_metrics_fast(prediction: np.ndarray, features: TargetFeatures, point_tolerance: float = 5.0) -> dict[str, float]:
    """Same values as compute_metrics, reusing the target features and restricting root metrics to the root window."""
    prediction = np.asarray(prediction).copy()
    prediction[~features.valid] = 0
    if prediction.size and prediction.min() >= 0 and prediction.max() < NUM_CLASSES:
        indices = features.codes + prediction.ravel()[features.scored].astype(np.int64)
        matrix = np.bincount(indices, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
    else:
        matrix = confusion_matrix(prediction, features.target)
    result = class_metrics_from_confusion(matrix)
    root_full = np.isin(prediction, ROOT_CLASSES) & features.valid
    box = _window(root_full | features.root)
    root_pred, root_target = root_full[box], features.root[box]
    target_skeleton = features.skeleton[box]
    result["complete_root_dice"] = binary_dice(root_pred, root_target)
    pred_skeleton = skeletonize(root_pred)
    if not root_pred.any() and not root_target.any():
        result["complete_root_cldice"] = 1.0
    elif not root_pred.any() or not root_target.any():
        result["complete_root_cldice"] = 0.0
    else:
        precision = np.logical_and(pred_skeleton, root_target).sum() / max(1, pred_skeleton.sum())
        sensitivity = np.logical_and(target_skeleton, root_pred).sum() / max(1, target_skeleton.sum())
        result["complete_root_cldice"] = float(2 * precision * sensitivity / max(1e-12, precision + sensitivity))
    pred_count = ndimage.label(root_pred, np.ones((3, 3), dtype=np.uint8))[1]
    result["component_count_error"] = float(abs(pred_count - features.component_count))
    if features.largest is None:
        result["largest_component_recall"] = 1.0 if pred_count == 0 else 0.0
    else:
        largest = features.largest[box]
        result["largest_component_recall"] = float(np.logical_and(root_pred, largest).sum() / max(1, largest.sum()))
    result["skeleton_length_error_px"] = float(skeleton_path_length(pred_skeleton) - features.length)
    result["skeleton_length_absolute_error_px"] = abs(result["skeleton_length_error_px"])
    pred_end, pred_branch = _skeleton_points(pred_skeleton)
    for name, pred_points, target_points in (("endpoint", pred_end, features.endpoints[box]),
                                             ("branchpoint", pred_branch, features.branch_centers[box])):
        for metric, value in tolerant_point_metrics(pred_points, target_points, point_tolerance).items():
            result[f"{name}_{metric}"] = value
    if not root_pred.any() and not root_target.any():
        result["root_hd95_px"], result["root_asd_px"] = 0.0, 0.0
    elif not root_pred.any() or not root_target.any():
        result["root_hd95_px"], result["root_asd_px"] = math.nan, math.nan
    else:
        target_distance = features.surface_distance[box]
        pred_surface = root_pred ^ ndimage.binary_erosion(root_pred, structure=np.ones((3, 3), dtype=bool), border_value=0)
        to_pred, _ = cKDTree(np.argwhere(pred_surface)).query(np.argwhere(target_distance == 0))
        distances = np.concatenate([target_distance[pred_surface], to_pred])
        result["root_hd95_px"], result["root_asd_px"] = float(np.percentile(distances, 95)), float(np.mean(distances))
    return result


def mean_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {key: float(np.nanmean([row.get(key, np.nan) for row in rows])) for key in numeric}
