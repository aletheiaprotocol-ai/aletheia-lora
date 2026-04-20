---
library_name: peft
tags:
  - lora
  - layer-selection
  - gradient-probe
  - parameter-efficient-fine-tuning
  - aletheia
  - arxiv:2604.15351
license: apache-2.0
language:
  - en
---

# Aletheia-LoRA: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning

**Paper:** [Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures](https://arxiv.org/abs/2604.15351)

**DOI:** [10.48550/arXiv.2604.15351](https://doi.org/10.48550/arXiv.2604.15351)

**Author:** Abdulmalek Saket (Royal Fenice Kft.)

**License:** Apache-2.0

## Summary

Aletheia identifies the most task-relevant transformer layers via a lightweight gradient probe, then applies LoRA adapters only to those layers. The published Line A evidence pack reports **15-28% training speedup** (mean 23.1%) with **bounded extra forgetting** across 14 successful models from 8 architecture families spanning 0.5B to 72B parameters.

This card describes the Aletheia-LoRA method/package. It is not a base model weight release.

## Key Results

| Metric | Value |
|--------|-------|
| Successful models tested | 14 (8 architecture families) |
| Scale range | 0.5B-72B parameters |
| Mean speedup | 23.1% (p < 0.001) |
| Campaign 1 win rate | 100% per-model |
| Extra MMLU forgetting | bounded under the evaluated benchmark pack |
| Architectures | Qwen 2.5, Llama 3.1, Mistral, Gemma 2, Phi-3.5, StableLM 2, TinyLlama, Mixtral (MoE) |

## Method

1. **Gradient probe**: measure per-layer gradient norms on a small probe batch set.
2. **Layer selection**: keep the top percentage of layers by gradient magnitude.
3. **Selective LoRA**: build a PEFT `LoraConfig` with `layers_to_transform` set to the selected layers.

```python
from aletheia_lora import gradient_probe, select_layers, apply_aletheia_lora

layer_scores = gradient_probe(model, dataset, num_layers=36, probe_batches=5)
selected_layers = select_layers(layer_scores, top_pct=50)
peft_config = apply_aletheia_lora(model, selected_layers, r=16, lora_alpha=32)
```

## Installation

```bash
git clone https://github.com/aletheiaprotocol-ai/aletheia-lora.git
cd aletheia-lora
pip install -e ".[dev]"
```

## Supported Claim

> Aletheia provides a reproducible training-efficiency improvement over standard LoRA across multiple model families and scales, with bounded extra forgetting on downstream benchmarks.

## Non-Claims

- This is not a universal replacement for every LoRA setup.
- This does not prove better quality on every model, task, or metric.
- This does not prove compression better than quantization.
- This does not prove the broader NEXUS/Aletheia cognitive architecture.
- This does not include base model weights.

## Limitations

- Evaluated mainly on English-language tasks (MMLU, GSM8K, HumanEval).
- Top-50% selection is a fixed heuristic; optimal percentage may vary by task.
- Gradient probe assumes a causal LM objective; other objectives are not tested here.
- Speedup percentages are wall-clock measurements on the reported hardware; they may differ on other systems.
- The package currently exposes the core method API, not a one-command Trainer wrapper.

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
