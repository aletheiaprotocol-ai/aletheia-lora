# PyPI Release Notes

The package name `aletheia-lora` was checked locally with:

```bash
python -m pip index versions aletheia-lora
```

At the time of the 2026-04-20 check, no matching PyPI distribution was found.

## Publish Path

The repository includes `.github/workflows/publish-pypi.yml`. The workflow is manual-only so GitHub Releases do not trigger PyPI publishing before credentials are configured.

To publish:

1. Create a PyPI account for the Aletheia/company identity.
2. Create a PyPI API token.
3. Add the token as a GitHub repository secret named `PYPI_API_TOKEN`.
4. Run the `Publish to PyPI` workflow manually.

Local preflight:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Do not paste PyPI tokens into chat or commit them into the repository.
