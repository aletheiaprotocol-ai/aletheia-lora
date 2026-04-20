# Aletheia-LoRA Release Checklist

This checklist prepares Line A for GitHub, Hugging Face, PyPI, and evidence-bundle distribution.

## 1. Source Repo

- Public repo: `aletheiaprotocol-ai/aletheia-lora`.
- Keep source code, tests, docs, notebooks, and examples in the repo.
- Do not commit `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `build/`, `dist/`, or raw experiment outputs.
- Put evidence bundles in GitHub Releases, Hugging Face Datasets, or object storage.
- Use semantic tags such as `v0.1.2` for patch releases.

Recommended preflight:

```bash
cd aletheia-lora
python -m pytest tests -q
python examples/quick_start.py
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## 2. Required Public Files

- `README.md` - public package overview and quick start.
- `MODEL_CARD.md` - method/model-card style documentation.
- `huggingface/README.md` - Hugging Face Hub card source.
- `CITATION.cff` - GitHub citation metadata.
- `LICENSE` - Apache-2.0 license.
- `CHANGELOG.md` - release notes.
- `pyproject.toml` - package metadata.
- `.github/workflows/ci.yml` - public CI.
- `.github/workflows/publish-pypi.yml` - manual PyPI publish workflow using Trusted Publishing.

## 3. Hugging Face Hub

- Method card repo: `aletheiaprotocol/aletheia-lora`.
- Use `huggingface/README.md` as the Hub repo `README.md`.
- If adapters are released later, create separate adapter repos per base model.
- For adapter repos, include `adapter_config.json`, adapter weights, base model name, training recipe, and a link back to arXiv:2604.15351.

Suggested Hub layout for a method-only repo:

```text
README.md
LICENSE
CITATION.cff
```

Suggested Hub layout for a future adapter repo:

```text
README.md
adapter_config.json
adapter_model.safetensors
training_args.json
metrics.json
```

## 4. Evidence Bundle

For reproducibility, freeze a separate evidence bundle with:

- aggregate run inventory (`aggregate_runs.csv`)
- aggregate model summary (`model_summary.csv`)
- aggregate domain summary (`domain_summary.csv`)
- sanitized metadata summary (`summary.json`)
- selected final figures, if publication-safe

Avoid uploading raw local runs, private datasets, checkpoints, or adapter weights into the package repo.

## 5. Claim Boundary

Use the supported claim:

> Aletheia provides a reproducible training-efficiency improvement over standard LoRA across multiple model families and scales, with bounded extra forgetting on downstream benchmarks.

Do not claim:

- universal superiority on all LoRA tasks
- better final quality for every model
- compression better than quantization
- proof of the full NEXUS brain or Line H Cordyceps control
