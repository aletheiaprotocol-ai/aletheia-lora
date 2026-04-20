"""
Gradient probe — measures per-layer gradient norms to assess task relevance.
"""

import gc
import math
from typing import Dict, List, Optional

import torch
import torch.utils.data


def _extract_layer_idx(name: str, max_layers: int) -> Optional[int]:
    """Extract transformer layer index from a parameter name."""
    import re
    for pattern in [r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.block\.(\d+)\."]:
        m = re.search(pattern, name)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < max_layers:
                return idx
    return None


def gradient_probe(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    num_layers: int,
    probe_batches: int = 5,
    chunk_size: int = 8,
    device: Optional[torch.device] = None,
    compute_dtype: torch.dtype = torch.float16,
    seed: int = 42,
) -> Dict[int, float]:
    """
    Memory-safe chunked gradient probe.

    Processes layers in groups of ``chunk_size``, computing accumulated gradient
    norms per layer on ``probe_batches`` mini-batches.

    Args:
        model: Pretrained causal LM (must be on ``device``).
        dataset: Tokenized dataset with ``input_ids`` and ``attention_mask``.
        num_layers: Number of transformer layers in the model.
        probe_batches: Number of batches to use for the probe (default 5).
        chunk_size: Number of layers to enable gradients for at once (default 8).
        device: CUDA device. Inferred from model if None.
        compute_dtype: Mixed-precision dtype (torch.float16 or torch.bfloat16).
        seed: Random seed for data sampling reproducibility.

    Returns:
        Dict mapping layer index to accumulated gradient norm.
    """
    if device is None:
        device = next(model.parameters()).device

    layer_norms: Dict[int, float] = {i: 0.0 for i in range(num_layers)}

    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0, generator=g
    )

    # Pre-collect probe batches
    probe_data: List[dict] = []
    for i, batch in enumerate(loader):
        if i >= probe_batches:
            break
        probe_data.append(batch)

    # Process layers in chunks for bounded memory
    for chunk_start in range(0, num_layers, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_layers)

        # Enable gradients only for layers in the current chunk
        for name, p in model.named_parameters():
            layer_idx = _extract_layer_idx(name, num_layers)
            if layer_idx is not None and chunk_start <= layer_idx < chunk_end and p.is_floating_point():
                p.requires_grad_(True)
            elif p.is_floating_point():
                p.requires_grad_(False)

        model.train()
        for batch in probe_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", dtype=compute_dtype):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids.clone(),
                )
            loss = outputs.loss.float()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    layer_idx = _extract_layer_idx(name, num_layers)
                    if layer_idx is not None:
                        layer_norms[layer_idx] += param.grad.float().norm().item()

            model.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()

    # Freeze all parameters after probing
    for p in model.parameters():
        if p.is_floating_point():
            p.requires_grad_(False)

    return layer_norms
