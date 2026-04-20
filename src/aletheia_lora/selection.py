"""
Layer selection — rank layers by gradient norm and pick the top-k%.
"""

from typing import Dict, List


def select_layers(
    layer_scores: Dict[int, float],
    top_pct: int = 50,
) -> List[int]:
    """
    Select the top ``top_pct``% of layers by gradient norm.

    Args:
        layer_scores: Dict mapping layer index to gradient norm
            (as returned by :func:`gradient_probe`).
        top_pct: Percentage of layers to select (1–100). Default 50.

    Returns:
        Sorted list of selected layer indices.
    """
    num_layers = len(layer_scores)
    top_k = max(int(num_layers * top_pct / 100), 1)

    indexed = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    selected = sorted([idx for idx, _ in indexed[:top_k]])
    return selected
