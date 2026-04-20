# PyPI Release Notes

The package name `aletheia-lora` was checked locally with:

```bash
python -m pip index versions aletheia-lora
```

At the time of the 2026-04-20 check, no matching PyPI distribution was found.

## Recommended Publish Path: Trusted Publisher

The repository includes `.github/workflows/publish-pypi.yml`. The workflow is manual-only and uses PyPI Trusted Publishing through GitHub OIDC. This avoids storing a long-lived PyPI token in GitHub.

To publish:

1. Create a PyPI account for the Aletheia/company identity.
2. Go to `https://pypi.org/manage/account/publishing/`.
3. Add a pending GitHub publisher with:
   - PyPI Project Name: `aletheia-lora`
   - Owner: `aletheiaprotocol-ai`
   - Repository name: `aletheia-lora`
   - Workflow name: `publish-pypi.yml`
   - Environment name: `pypi`
4. Run the `Publish to PyPI` workflow manually from GitHub Actions.

Important: a pending publisher does not reserve the name until the first successful publish, so run the workflow soon after adding it.

Local preflight:

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

Do not paste PyPI tokens into chat or commit them into the repository.

## Fallback Path: API Token

Only use this if Trusted Publishing fails. Create a PyPI API token, add it as a GitHub repository secret named `PYPI_API_TOKEN`, and change the workflow to pass:

```yaml
with:
  password: ${{ secrets.PYPI_API_TOKEN }}
```
