# Evidence Bundle Policy

The public package repository intentionally excludes raw HPC outputs, trained
adapters, checkpoints, and large result directories.

Recommended public evidence bundle contents:

- aggregate run inventory (`aggregate_runs.csv`)
- aggregate model summary (`model_summary.csv`)
- aggregate domain summary (`domain_summary.csv`)
- sanitized metadata summary (`summary.json`)
- figure thumbnails or final paper figures, if publication-safe

Do not include:

- credentials
- full local paths containing usernames
- checkpoints or adapter weights unless explicitly intended for release
- raw private datasets
- cloud bucket URLs requiring private access
