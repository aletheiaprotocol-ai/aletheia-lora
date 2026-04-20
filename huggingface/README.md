---
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
language:
  - en
tags:
  - lora
  - peft
  - layer-selection
  - gradient-probe
  - parameter-efficient-fine-tuning
  - arxiv:2604.15351
---

# Aletheia-LoRA

Aletheia-LoRA is the public method card for:

**Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures**

- Paper: https://arxiv.org/abs/2604.15351
- DOI: https://doi.org/10.48550/arXiv.2604.15351
- Code: https://github.com/aletheiaprotocol-ai/aletheia-lora
- Hub: https://huggingface.co/aletheiaprotocol/aletheia-lora
- License: Apache-2.0

## What This Is

Aletheia uses a lightweight gradient probe to identify the most task-relevant transformer layers, then applies LoRA adapters only to those layers. The goal is cheaper LoRA fine-tuning while preserving downstream behavior under the evaluated benchmark pack.

This Hub card is intended for the method/package release. It does not include base model weights. If trained adapters are uploaded later, create one adapter repo per base model and link those adapter repos back to this method card.

## Published Claim

> Aletheia provides a reproducible training-efficiency improvement over standard LoRA across multiple model families and scales, with bounded extra forgetting on downstream benchmarks.

## Evidence Summary

| Metric | Value |
|--------|-------|
| arXiv ID | 2604.15351 |
| Successful models | 14 |
| Architecture families | 8 |
| Scale range | 0.5B-72B |
| Experiment rows | 81 |
| Mean speedup | 23.1% |
| Speedup range | 15-28% |
| Benchmarks | MMLU, GSM8K, HumanEval |

## Installation

```bash
git clone https://github.com/aletheiaprotocol-ai/aletheia-lora.git
cd aletheia-lora
pip install -e .
```

## Minimal Usage

```python
from aletheia_lora import gradient_probe, select_layers, apply_aletheia_lora
from peft import get_peft_model

layer_scores = gradient_probe(
    model=model,
    dataset=train_dataset,
    num_layers=36,
    probe_batches=5,
)

selected_layers = select_layers(layer_scores, top_pct=50)
peft_config = apply_aletheia_lora(model, selected_layers, r=16, lora_alpha=32)
model = get_peft_model(model, peft_config)
```

Paper-style asymmetric allocation:

```python
peft_config = apply_aletheia_lora(
    model,
    selected_layers,
    attention_r=16,
    mlp_r=64,
)
```

## Limitations

- Tested mainly on English-language evaluation tasks.
- The top-50% selection rule is a robust default, not a theorem.
- The current package exposes the method API; production Trainer integrations should be added separately.
- Wall-clock speedups were measured in the reported hardware/software setup and may vary on other systems.
- This card should not be used to imply that Aletheia improves every model or every metric.

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
