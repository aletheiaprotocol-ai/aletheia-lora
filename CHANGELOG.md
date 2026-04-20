# Changelog

## 0.1.2 (2026-04-20)

Release-hardening patch.

- Added GitHub Actions CI and release/publishing documentation.
- Fixed quick-start wording to avoid the unsupported "zero degradation" claim.
- Added input validation for layer selection.
- Made `gradient_probe()` CPU-safe by only using CUDA autocast/cache clearing on CUDA devices.
- Added a tutorial notebook and aggregate evidence-bundle workflow.

## 0.1.1 (2026-04-20)

Patch release.

- Added optional asymmetric rank and alpha allocation via `attention_r`, `mlp_r`, `attention_alpha`, and `mlp_alpha`.
- Preserved backward-compatible uniform-rank behavior for existing users.
- Added tests for paper-style attention/MLP rank patterns.

## 0.1.0 (2026-04-20)

Initial public release aligned with arXiv:2604.15351.

- `gradient_probe()` - memory-safe chunked gradient probing for per-layer importance
- `select_layers()` - top-k% layer selection by gradient magnitude
- `apply_aletheia_lora()` / `aletheia_lora_config()` - PEFT LoraConfig with `layers_to_transform`
- Tested across 14 successful models, 8 families, 0.5B-72B parameters
- Quick-start example, Hugging Face card, citation metadata, and test suite included
