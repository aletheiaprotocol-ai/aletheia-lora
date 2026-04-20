# Aletheia-LoRA Release Checklist

This checklist prepares Line A for GitHub, Hugging Face, and later package distribution.

## 1. Source Repo

- Confirm the public repo name: `aletheiaprotocol-ai/aletheia-lora`.
- Keep source code, tests, docs, and examples in the repo.
- Do not commit `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, `dist/`, or large raw experiment outputs.
- Put large evidence bundles in GitHub Releases, Hugging Face Datasets, or object storage.
- Tag the first release as `v0.1.0`.

Recommended commands:

```bash
cd aletheia-lora
python -m pytest tests -q
python -m pip install build
python -m build
git tag v0.1.0
```

## 2. Required Public Files

- `README.md` - public package overview and quick start.
- `MODEL_CARD.md` - method/model-card style documentation.
- `huggingface/README.md` - ready to upload as the Hugging Face Hub card.
- `CITATION.cff` - GitHub citation metadata.
- `LICENSE` - Apache-2.0 license.
- `CHANGELOG.md` - release notes.
- `pyproject.toml` - package metadata.

## 3. Hugging Face Hub

- Create a Hub repo for the method card, for example `<org-or-user>/aletheia-lora`.
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

- `paper/aletheia_paper.pdf`
- `paper/aletheia_paper.tex`
- `aletheia_figures/`
- selected aggregate CSV/JSON summaries from `aletheia_results/`
- SLURM scripts needed to reproduce the published runs

Avoid uploading every raw local run into the package repo unless it is intentionally a dataset release.

## 5. Claim Boundary

Use the supported claim:

> Aletheia provides a reproducible training-efficiency improvement over standard LoRA across multiple model families and scales, with bounded extra forgetting on downstream benchmarks.

Do not claim:

- universal superiority on all LoRA tasks
- better final quality for every model
- compression better than quantization
- proof of the full NEXUS brain or Line H Cordyceps control
