# selkit

Pure-Python reimplementation of PAML's selection-analysis workflows (codeml, yn00).

v1 ships codeml site models (M0, M1a, M2a, M7, M8, M8a) with auto-LRTs, per-site NEB posteriors, model-level parallelism, multi-start optimization with a convergence gate, strict input validation, and reproducible `run.yaml` artifacts.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Typical CLI workflow

```bash
selkit codeml site-models \
    --alignment gene.fa \
    --tree species.nwk \
    --output results/gene1 \
    --threads 4
```

This fits M0, M1a, M2a, M7, M8, and M8a, computes M1a-vs-M2a, M7-vs-M8, and M8a-vs-M8 LRTs, and emits per-site NEB posteriors for M2a and M8.

Outputs in `results/gene1/`:

- `results.json` — canonical structured output (all fits, LRTs, BEB sites, warnings).
- `fits.tsv`, `lrts.tsv`, `beb_M2a.tsv`, `beb_M8.tsv` — tabular views.
- `run.yaml` — exact config; reproduce with `selkit rerun results/gene1/run.yaml`.

## Validation

```bash
selkit validate --alignment gene.fa --tree species.nwk
```

Runs every pre-flight check short of model fitting (length multiples of 3, stop codons, taxon match, foreground-label consistency).

## Library

```python
from selkit import codeml_site_models

result = codeml_site_models(
    alignment="gene.fa", tree="species.nwk", output_dir="out/",
    models=("M0", "M1a", "M2a"),
)
print(result.fits["M2a"].lnL)
for s in result.beb["M2a"]:
    if s.p_positive > 0.95:
        print(f"site {s.site}: P(omega > 1) = {s.p_positive:.3f}")
```

## Not yet in v1

- Branch models (one-ratio, two-ratios, free-ratios)
- Branch-site models (Model A, Model A null)
- yn00 pairwise dN/dS

These land in subsequent plans built on the same engine.

## Design

See `docs/superpowers/specs/2026-04-18-selkit-design.md`.
