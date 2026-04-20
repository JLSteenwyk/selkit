# Releasing selkit

Releases are published manually to PyPI. The flow follows ClipKIT.

## Pre-release checklist

1. Bump `selkit/version.py` → `__version__ = "X.Y.Z"`.
2. Add a new `## X.Y.Z` entry to the top of `CHANGELOG.md` describing what's in the release.
3. Verify the two are in sync: `make release-check` (or `python3 scripts/check_version_sync.py`). The `Version Consistency` GitHub Action also runs this on every PR and push.
4. Run the full test suite locally: `make test.fast`.
5. Commit the bump + changelog entry and push to `main`.

## Publish to PyPI

```
rm -rf dist
python3 setup.py sdist bdist_wheel --universal
twine upload dist/* -r pypi
```

Or simply: `make release` (which runs the version-sync check first).

This expects a configured `~/.pypirc` or `TWINE_USERNAME` / `TWINE_PASSWORD` environment variables for your PyPI account.

## Tag the release on GitHub

After the PyPI upload succeeds:

```
git tag vX.Y.Z
git push origin vX.Y.Z
```

Then create a GitHub Release for `vX.Y.Z` referencing the CHANGELOG entry.
