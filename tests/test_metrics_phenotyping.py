import numpy as np
import pytest

from toposeg.metrics import compute_metrics, tolerant_point_metrics
from toposeg.phenotyping import phenotype, skeleton_length
from toposeg.point_matching import maximum_cardinality_metrics


def test_perfect_prediction_metrics_and_ignore():
    target = np.zeros((32, 32), dtype=np.uint8)
    target[5:25, 15] = 1
    target[15, 16:25] = 2
    target[0, 0] = 7
    prediction = target.copy()
    prediction[0, 0] = 6
    metrics = compute_metrics(prediction, target)
    assert metrics["complete_root_dice"] == 1.0
    assert metrics["complete_root_cldice"] == 1.0
    assert metrics["root_hd95_px"] == 0.0
    assert metrics["root_asd_px"] == 0.0
    assert metrics["endpoint_f1"] == 1.0
    assert metrics["branchpoint_f1"] == 1.0


def test_phenotype_pixel_length():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[5, 2:7] = 1
    assert skeleton_length(mask == 1) == 4.0
    values = phenotype(mask)
    assert values["main_root_length_px"] == 4.0
    assert values["lateral_root_length_px"] == 0.0
    assert values["complete_root_length_px"] == 4.0
    assert values["root_endpoints"] == 2


def point_mask(points):
    mask = np.zeros((12, 12), dtype=bool)
    for point in points:
        mask[point] = True
    return mask


def test_maximum_cardinality_recovers_feasible_pairs_rejected_by_hungarian():
    predicted = point_mask([(5, 5), (8, 9)])
    target = point_mask([(5, 5), (5, 1)])
    assert tolerant_point_metrics(predicted, target, tolerance=5.0)["f1"] == 0.5
    assert maximum_cardinality_metrics(predicted, target, tolerance=5.0) == {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "count_error": 0.0,
    }


@pytest.mark.parametrize("predicted,target,expected", [
    ([], [], {"precision": 1.0, "recall": 1.0, "f1": 1.0, "count_error": 0.0}),
    ([], [(5, 5)], {"precision": 1.0, "recall": 0.0, "f1": 0.0, "count_error": 1.0}),
    ([(5, 5)], [], {"precision": 0.0, "recall": 1.0, "f1": 0.0, "count_error": 1.0}),
    ([(0, 0)], [(11, 11)], {"precision": 0.0, "recall": 0.0, "f1": 0.0, "count_error": 0.0}),
])
def test_maximum_cardinality_empty_and_no_match(predicted, target, expected):
    assert maximum_cardinality_metrics(point_mask(predicted), point_mask(target), tolerance=5.0) == expected


def test_maximum_cardinality_is_one_to_one_with_unequal_counts():
    predicted = point_mask([(5, 4), (5, 6)])
    target = point_mask([(5, 5)])
    result = maximum_cardinality_metrics(predicted, target, tolerance=1.0)
    assert result == {"precision": 0.5, "recall": 1.0, "f1": 2 / 3, "count_error": 1.0}
    reversed_result = maximum_cardinality_metrics(target, predicted, tolerance=1.0)
    assert reversed_result == {"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "count_error": 1.0}
