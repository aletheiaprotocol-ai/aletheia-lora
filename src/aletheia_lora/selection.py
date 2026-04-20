"""Layer selection: rank layers by gradient norm and pick the top-k percent."""

from typing import Dict, List


def select_layers(
    layer_scores: Dict[int, float],
    top_pct: float = 50,
) -> List[int]:
    """Select the top ``top_pct`` percent of layers by gradient norm.

    Args:
        layer_scores: Mapping from layer index to gradient norm, as returned by gradient_probe().
        top_pct: Percentage of layers to select, inclusive range 1-100.

    Returns:
        Sorted list of selected layer indices.
    """
    if not layer_scores:
        raise ValueError("layer_scores must contain at least one layer score")
    if not 1 <= top_pct <= 100:
        raise ValueError("top_pct must be between 1 and 100")

    num_layers = len(layer_scores)
    top_k = max(int(num_layers * top_pct / 100), 1)

    indexed = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)
    selected = sorted([idx for idx, _ in indexed[:top_k]])
    return selected
