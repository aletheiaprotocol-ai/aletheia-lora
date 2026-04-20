"""
PEFT LoRA config generation with Aletheia layer selection.
"""

from typing import Dict, List, Optional, Tuple

from peft import LoraConfig


ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}
MLP_MODULES = {"gate_proj", "up_proj", "down_proj"}


def _build_asymmetric_patterns(
    target_modules: List[str],
    attention_r: Optional[int],
    mlp_r: Optional[int],
    attention_alpha: Optional[int],
    mlp_alpha: Optional[int],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build PEFT rank/alpha pattern dictionaries for module-level asymmetry."""
    rank_pattern: Dict[str, int] = {}
    alpha_pattern: Dict[str, int] = {}

    for module in target_modules:
        if module in ATTENTION_MODULES:
            if attention_r is not None:
                rank_pattern[module] = attention_r
            if attention_alpha is not None:
                alpha_pattern[module] = attention_alpha
        elif module in MLP_MODULES:
            if mlp_r is not None:
                rank_pattern[module] = mlp_r
            if mlp_alpha is not None:
                alpha_pattern[module] = mlp_alpha

    return rank_pattern, alpha_pattern


def aletheia_lora_config(
    selected_layers: List[int],
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    attention_r: Optional[int] = None,
    mlp_r: Optional[int] = None,
    attention_alpha: Optional[int] = None,
    mlp_alpha: Optional[int] = None,
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
        attention_r: Optional rank override for attention projections
            (q/k/v/o). Use with ``mlp_r`` to reproduce asymmetric recipes.
        mlp_r: Optional rank override for MLP projections
            (gate/up/down). For example, paper-style asymmetric configs can
            use ``attention_r=16`` and ``mlp_r=64``.
        attention_alpha: Optional alpha override for attention projections.
        mlp_alpha: Optional alpha override for MLP projections.
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

    rank_pattern, alpha_pattern = _build_asymmetric_patterns(
        target_modules=target_modules,
        attention_r=attention_r,
        mlp_r=mlp_r,
        attention_alpha=attention_alpha,
        mlp_alpha=mlp_alpha,
    )

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        layers_to_transform=selected_layers,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        task_type=task_type,
        bias="none",
    )


def apply_aletheia_lora(
    model,
    selected_layers: List[int],
    target_modules: Optional[List[str]] = None,
    r: int = 16,
    lora_alpha: int = 32,
    attention_r: Optional[int] = None,
    mlp_r: Optional[int] = None,
    attention_alpha: Optional[int] = None,
    mlp_alpha: Optional[int] = None,
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
        attention_r=attention_r,
        mlp_r=mlp_r,
        attention_alpha=attention_alpha,
        mlp_alpha=mlp_alpha,
        lora_dropout=lora_dropout,
        task_type=task_type,
    )
