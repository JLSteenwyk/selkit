# PAML-comparison validation corpus

Each subdirectory of `corpus/` is one case:

```
corpus/<case_id>/
├── alignment.fa          # FASTA codon alignment
├── tree.nwk              # Newick
├── expected.json         # PAML-derived ground truth
└── meta.yaml             # genetic_code, models to fit, bundle, tests, etc.
```

`expected.json` structure:

```json
{
  "fits": {
    "M0":  { "lnL": -1234.567, "params": { "omega": 0.234, "kappa": 2.12 } },
    "M1a": { ... },
    "M2a": { ... }
  },
  "beb": {
    "M2a": [ { "site": 17, "p_positive": 0.972 }, ... ]
  }
}
```

## Thresholds (v1)

- `|lnL_selkit − lnL_paml| ≤ 0.01`
- `|ω − ω_paml| ≤ 1e-3`
- `|branch_length − paml_bl| ≤ 1e-2`
- BEB site posteriors agree within `1e-3` on sites PAML identifies as positively selected.

## Adding a new case

1. Run PAML codeml externally on (alignment, tree).
2. Extract fitted lnLs, ω values, and (where relevant) BEB posteriors.
3. Write `expected.json`; commit alongside the inputs.
4. The test runner will pick it up automatically.
