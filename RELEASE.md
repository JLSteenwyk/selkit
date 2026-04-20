# Release Automation

Selkit releases are automated with GitHub Actions via `.github/workflows/release.yml`.

## Supported release triggers

1. Push a version tag (recommended): `vX.Y.Z`
2. Manually run the `Release` workflow from GitHub Actions (optional `version` input)

## Behavior

The workflow will:

1. Validate that `selkit/version.py` matches the requested/tagged version.
2. Validate that the top entry of `CHANGELOG.md` matches `selkit/version.py`
   (same check that runs on every PR via `version-consistency.yml`).
3. Build source and wheel distributions with `python -m build`.
4. Run `twine check` on built artifacts.
5. Upload the distributions as workflow-run artifacts.
6. Create a GitHub Release and attach `dist/*` (sdist + wheel) to it.

The workflow does **not** publish to PyPI automatically — no PyPI credentials
are required to run it. To publish a release:

1. Let the workflow run and produce a GitHub Release.
2. Download the `sdist` (`.tar.gz`) and `wheel` (`.whl`) from the release page.
3. Locally: `twine upload selkit-<ver>.tar.gz selkit-<ver>-py3-none-any.whl`.

## Cutting a release

```
# 1. Bump the version
#    - selkit/version.py
#    - add a new "## X.Y.Z" entry to the top of CHANGELOG.md
make release-check   # runs check_version_sync.py locally

# 2. Commit + push the bump
git commit -am "chore: bump to vX.Y.Z"
git push origin main

# 3. Tag and push
git tag vX.Y.Z
git push origin vX.Y.Z

# The Release workflow handles the rest.
```
