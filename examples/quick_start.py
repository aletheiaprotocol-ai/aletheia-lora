"""Aletheia-LoRA reproducible quick-start example.

Demonstrates the public API without requiring a GPU:
  1. Simulated gradient probe scores
  2. Top-k layer selection
  3. Selective LoRA config, including paper-style asymmetric ranks

For real training, replace the simulated scores with ``gradient_probe()`` on
your model and tokenized dataset.
"""

import sys

sys.path.insert(0, "src")

from aletheia_lora import aletheia_lora_config, select_layers

# ------------------------------------------------------------------
# Step 1: Simulate gradient probe results
# ------------------------------------------------------------------
# In practice, you would call:
#   from aletheia_lora import gradient_probe
#   layer_scores = gradient_probe(model, dataset, num_layers=32)
#
# Here we use representative scores from a Qwen-style 36-layer probe.
layer_scores = {
    0: 0.12, 1: 0.08, 2: 0.31, 3: 0.45,
    4: 0.67, 5: 0.22, 6: 0.89, 7: 0.54,
    8: 0.33, 9: 0.71, 10: 0.15, 11: 0.92,
    12: 0.28, 13: 0.63, 14: 0.41, 15: 0.77,
    16: 0.19, 17: 0.85, 18: 0.36, 19: 0.58,
    20: 0.73, 21: 0.11, 22: 0.95, 23: 0.44,
    24: 0.66, 25: 0.29, 26: 0.81, 27: 0.52,
    28: 0.38, 29: 0.70, 30: 0.14, 31: 0.88,
    32: 0.47, 33: 0.21, 34: 0.76, 35: 0.60,
}
print(f"Total layers: {len(layer_scores)}")
print(f"Top-5 gradient layers: {sorted(layer_scores, key=layer_scores.get, reverse=True)[:5]}")

# ------------------------------------------------------------------
# Step 2: Select top 50% layers by gradient magnitude
# ------------------------------------------------------------------
selected = select_layers(layer_scores, top_pct=50)
print(f"\nSelected {len(selected)}/{len(layer_scores)} layers (top 50%):")
print(f"  Layers: {selected}")
print(f"  Skipped: {sorted(set(range(36)) - set(selected))}")

# ------------------------------------------------------------------
# Step 3: Build selective LoRA configs
# ------------------------------------------------------------------
config = aletheia_lora_config(
    selected_layers=selected,
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
print("\nUniform LoRA config:")
print(f"  r={config.r}, alpha={config.lora_alpha}")
print(f"  layers_to_transform={config.layers_to_transform}")
print(f"  target_modules={config.target_modules}")
print(f"  task_type={config.task_type}")

# Paper-style asymmetric attention/MLP rank allocation.
asymmetric_config = aletheia_lora_config(
    selected_layers=selected,
    attention_r=16,
    mlp_r=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
print("\nAsymmetric LoRA rank pattern:")
print(f"  q_proj rank={asymmetric_config.rank_pattern['q_proj']}")
print(f"  up_proj rank={asymmetric_config.rank_pattern['up_proj']}")

# ------------------------------------------------------------------
# Step 4: What you would do next (commented; needs GPU + model)
# ------------------------------------------------------------------
# from transformers import AutoModelForCausalLM
# from peft import get_peft_model
#
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
# model = get_peft_model(model, config)
# # Train with HF Trainer or a custom loop.
# # The paper reports 15-28% wall-clock speedup with bounded extra forgetting
# # on the evaluated benchmark pack; validate this on your own task/hardware.

print("\nQuick-start complete. The config above is ready for get_peft_model().")
