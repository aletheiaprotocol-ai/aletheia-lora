"""Gradient probe: measure per-layer gradient norms for task relevance."""

import gc
import re
from contextlib import nullcontext
from typing import Dict, List, Optional

import torch
import torch.utils.data


def _extract_layer_idx(name: str, max_layers: int) -> Optional[int]:
    """Extract transformer layer index from a parameter name."""
    for pattern in [r"\.layers\.(\d+)\.", r"\.h\.(\d+)\.", r"\.block\.(\d+)\."]:
        match = re.search(pattern, name)
        if match:
            idx = int(match.group(1))
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
    """Run a memory-bounded gradient probe.

    The probe processes layers in groups of ``chunk_size`` and accumulates the
    gradient norm for each layer over ``probe_batches`` mini-batches.

    Args:
        model: Pretrained causal LM. The model should already be on ``device``.
        dataset: Tokenized dataset with ``input_ids`` and ``attention_mask``.
        num_layers: Number of transformer layers in the model.
        probe_batches: Number of batches to use for the probe.
        chunk_size: Number of layers to enable gradients for at once.
        device: Device. Inferred from model if omitted.
        compute_dtype: Mixed-precision dtype used only when probing on CUDA.
        seed: Random seed for probe-batch sampling.

    Returns:
        Mapping from layer index to accumulated gradient norm.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if probe_batches <= 0:
        raise ValueError("probe_batches must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    layer_norms: Dict[int, float] = {i: 0.0 for i in range(num_layers)}

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )

    probe_data: List[dict] = []
    for i, batch in enumerate(loader):
        if i >= probe_batches:
            break
        probe_data.append(batch)

    for chunk_start in range(0, num_layers, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_layers)

        for name, param in model.named_parameters():
            layer_idx = _extract_layer_idx(name, num_layers)
            in_chunk = layer_idx is not None and chunk_start <= layer_idx < chunk_end
            if param.is_floating_point():
                param.requires_grad_(in_chunk)

        model.train()
        for batch in probe_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=compute_dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids.clone(),
                )
            loss = outputs.loss.float()
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                layer_idx = _extract_layer_idx(name, num_layers)
                if layer_idx is not None:
                    layer_norms[layer_idx] += param.grad.float().norm().item()

            model.zero_grad()

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for param in model.parameters():
        if param.is_floating_point():
            param.requires_grad_(False)

    return layer_norms
