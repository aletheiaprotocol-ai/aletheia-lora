# Changelog

## 0.1.0 (2026-04-20)

Initial public release aligned with arXiv:2604.15351.

- `gradient_probe()` - memory-safe chunked gradient probing for per-layer importance
- `select_layers()` - top-k% layer selection by gradient magnitude
- `apply_aletheia_lora()` / `aletheia_lora_config()` - PEFT LoraConfig with `layers_to_transform`
- Tested across 14 successful models, 8 families, 0.5B-72B parameters
- Quick-start example, Hugging Face card, citation metadata, and test suite included
