"""
PEFT LoRA config generation with Aletheia layer selection.
"""

from typing import List, Optional

from peft import LoraConfig


def aletheia_lora_config(
    selected_layers: List[int],
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Build a PEFT LoraConfig that only adapts the selected layers.

    Args:
        selected_layers: Sorted list of layer indices to adapt
            (as returned by :func:`select_layers`).
        target_modules: LoRA target module names. Defaults to standard
            Llama/Qwen modules.
        r: LoRA rank (default 16).
        lora_alpha: LoRA alpha scaling (default 32).
        lora_dropout: Dropout probability for LoRA layers (default 0.0).
        task_type: PEFT task type (default ``"CAUSAL_LM"``).

    Returns:
        A :class:`peft.LoraConfig` with ``layers_to_transform`` set.
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        layers_to_transform=selected_layers,
        task_type=task_type,
        bias="none",
    )


def apply_aletheia_lora(
    model,
    selected_layers: List[int],
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Convenience wrapper: builds the LoRA config for Aletheia.
    Returns the config (caller applies via ``get_peft_model``).

    This is identical to :func:`aletheia_lora_config` — provided
    for a friendlier API name.
    """
    return aletheia_lora_config(
        selected_layers=selected_layers,
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type=task_type,
    )
