# Aletheia-LoRA

**Gradient-guided layer selection for efficient LoRA fine-tuning.**

[![arXiv](https://img.shields.io/badge/arXiv-2604.15351-b31b1b.svg)](https://arxiv.org/abs/2604.15351)
[![DOI](https://img.shields.io/badge/DOI-10.48550%2FarXiv.2604.15351-blue.svg)](https://doi.org/10.48550/arXiv.2604.15351)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Method%20Card-yellow.svg)](https://huggingface.co/aletheiaprotocol/aletheia-lora)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

Aletheia identifies the most informative transformer layers with a lightweight gradient probe, then applies LoRA adapters only to those layers. Across the finalized Research Line A evidence pack, this yields a **15-28% training speedup** with **bounded extra forgetting and broadly matched downstream behavior** across **14 successful models** from **8 architecture families** spanning **0.5B to 72B parameters**.

Paper: [Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures](https://arxiv.org/abs/2604.15351)

DOI: [10.48550/arXiv.2604.15351](https://doi.org/10.48550/arXiv.2604.15351)

Hugging Face card: [aletheiaprotocol/aletheia-lora](https://huggingface.co/aletheiaprotocol/aletheia-lora)

## Installation

Until the package is published to PyPI, install it from source:

```bash
git clone https://github.com/aletheiaprotocol-ai/aletheia-lora.git
cd aletheia-lora
python -m pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from aletheia_lora import gradient_probe, select_layers, apply_aletheia_lora
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

# 1. Measure per-layer importance with a short gradient probe.
layer_scores = gradient_probe(
    model=model,
    dataset=train_dataset,       # tokenized HF Dataset with input_ids + attention_mask
    num_layers=36,
    probe_batches=5,
    chunk_size=8,
)

# 2. Select the top 50% most informative layers.
selected_layers = select_layers(layer_scores, top_pct=50)

# 3. Apply LoRA only to the selected layers.
peft_config = apply_aletheia_lora(
    model=model,
    selected_layers=selected_layers,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=16,
    lora_alpha=32,
)
model = get_peft_model(model, peft_config)
```

To use the paper-style asymmetric attention/MLP allocation:

```python
peft_config = apply_aletheia_lora(
    model=model,
    selected_layers=selected_layers,
    attention_r=16,
    mlp_r=64,
)
```

## Supported Claim

Use this wording:

> Aletheia provides a reproducible training-efficiency improvement over standard LoRA across multiple model families and scales, with bounded extra forgetting on downstream benchmarks.

This package supports that finished narrow claim. It does not claim:

- universal replacement for every LoRA setup
- better quality on every model and every metric
- compression better than quantization
- proof of the full Aletheia or NEXUS paradigm

## How It Works

1. **Gradient probe**: computes per-layer gradient norms on a small probe batch set with bounded memory usage.
2. **Layer selection**: ranks layers by gradient magnitude and keeps the top-k percent.
3. **Selective LoRA**: applies standard PEFT `LoraConfig` only to the selected layers.

## Evidence Summary

- **81 experiment rows** across 14 successful models and 8 architecture families
- **100% per-model speed win rate** in the main speed campaign
- **23.1% mean speedup** in the strongest frozen campaign
- **Bounded extra forgetting** on MMLU, GSM8K, and HumanEval under the narrow-claim framing
- **Scale range** from 0.5B to 72B parameters, including a MoE model

## Repository Contents

| Path | Purpose |
|------|---------|
| `src/aletheia_lora/` | Core library: probe, layer selection, and PEFT config builder |
| `examples/quick_start.py` | CPU-safe usage demo with simulated layer scores |
| `tests/` | Unit tests for the public API |
| `MODEL_CARD.md` | Human-readable method card |
| `huggingface/README.md` | Hugging Face Hub-ready card |
| `CITATION.cff` | GitHub citation metadata |

## Reproducibility

The full empirical record is described in the paper. In the NEXUS monorepo, the raw Line A artifacts live under `aletheia_results/`, `aletheia_figures/`, `leo_slurm/`, and `paper/aletheia_paper.*`. If this package is released as a standalone repository, freeze those artifacts in a separate release asset or Hugging Face dataset rather than mixing large experiment outputs into the source package.

## Citation

```bibtex
@misc{saket2026aletheia,
  title={Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures},
  author={Saket, Abdulmalek},
  year={2026},
  eprint={2604.15351},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2604.15351},
  url={https://arxiv.org/abs/2604.15351}
}
```

## License

Apache 2.0
