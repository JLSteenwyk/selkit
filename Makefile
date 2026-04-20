install:
	python3 -m pip install .

develop:
	python3 -m pip install -r requirements.txt
	python3 -m pip install -e .

test: test.unit test.integration

test.unit:
	python3 -m pytest tests/unit -v

test.integration:
	python3 -m pytest tests/integration -v

test.fast:
	python3 -m pytest tests/unit tests/integration -v

test.validation:
	# Full-fit PAML-comparison corpus (slow, ~15 min). CI uses the static
	# match test in tests/integration/test_paml_lnl_match.py instead.
	python3 -m pytest tests/validation -v -m validation

# used by GitHub actions during CI workflow
test.coverage:
	python3 -m pytest tests/unit tests/integration --cov=selkit --cov-report=xml --cov-report=term

release-check:
	python3 scripts/check_version_sync.py

# Manual release: run locally after bumping version + CHANGELOG, pushing to
# main, and tagging. Pushes to PyPI (requires ~/.pypirc or env credentials).
release: release-check
	rm -rf dist
	python3 setup.py sdist bdist_wheel --universal
	twine upload dist/* -r pypi

.PHONY: install develop test test.unit test.integration test.fast test.validation test.coverage release-check release
