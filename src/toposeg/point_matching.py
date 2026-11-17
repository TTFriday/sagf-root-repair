from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist


def maximum_cardinality_metrics(predicted_points: np.ndarray, target_points: np.ndarray,
                                tolerance: float = 5.0) -> dict[str, float]:
    predicted, target = np.argwhere(predicted_points), np.argwhere(target_points)
    if not len(predicted) and not len(target):
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "count_error": 0.0}
    if not len(predicted) or not len(target):
        return {"precision": 0.0 if len(predicted) else 1.0,
                "recall": 0.0 if len(target) else 1.0, "f1": 0.0,
                "count_error": float(abs(len(predicted) - len(target)))}
    adjacency = csr_matrix(cdist(predicted, target) <= tolerance)
    matching = maximum_bipartite_matching(adjacency, perm_type="column")
    true_positive = int(np.sum(matching >= 0))
    return {"precision": true_positive / len(predicted), "recall": true_positive / len(target),
            "f1": 2 * true_positive / (len(predicted) + len(target)),
            "count_error": float(abs(len(predicted) - len(target)))}
