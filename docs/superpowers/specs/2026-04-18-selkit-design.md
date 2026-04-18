# selkit — design spec

**Date:** 2026-04-18
**Author:** Jacob Steenwyk (with Claude brainstorming)
**Status:** Draft — pending user review

## 1. Purpose

`selkit` is a pure-Python reimplementation of the selection-analysis portion of PAML, with a modern CLI/library UX. It replaces the experience of hand-editing `codeml.ctl` files, running cryptic binaries, and parsing free-form `rst`/`rub`/`2NG.*` output with a single well-documented tool that emits canonical JSON + TSV and offers sensible defaults for the most common workflows (positive-selection detection on a gene; branch-specific dN/dS; pairwise dN/dS).

## 2. Scope

### In scope (v1)

- **codeml** — codon substitution models by ML:
  - Site models: **M0, M1a, M2a, M7, M8, M8a**
  - Branch models: **one-ratio, two-ratios, free-ratios**
  - Branch-site models: **Model A, Model A null**
- **yn00** — pairwise dN/dS (Yang & Nielsen 2000 closed-form method).
- An internal nucleotide-ML likelihood layer used as scaffolding for codeml — **not exposed as a user-facing `baseml` command**.

### Out of scope (v1)

- `baseml` as a user-facing command (nucleotide-ML is internal only).
- `mcmctree` (Bayesian divergence-time estimation).
- A user-facing sequence simulator ("evolver"). A minimal internal simulator exists for validation testing only.
- GPU acceleration / JAX backends.
- Intra-model parallelism (parallelizing within a single likelihood computation).

### Non-goals

- Being bit-for-bit identical to PAML. The correctness bar is numerical agreement within documented tolerances on a canonical test corpus (see §9).
- Replicating every PAML flag. selkit is designed around modern defaults; obscure PAML knobs are exposed only where they meaningfully affect results.
- Running PAML binaries under the hood. selkit is a reimplementation, not a wrapper.

## 3. Design principles

1. **Library-first, CLI thin.** The public Python API is the primary interface; the CLI is a shell over it. This keeps library / CLI / JSON output schema consistent by construction.
2. **Sensible defaults with escape hatches.** The common case — "is this gene under positive selection?" — should be one command with no flags. Power-user control lives behind flags, not in the default experience.
3. **Strict inputs, loud failures.** selkit errors (not warnings) on malformed input, with actionable messages pointing at file/line/taxon and suggesting the right flag. Silent coercion is a footgun.
4. **Structured outputs only.** Every result produced by every subcommand is a single dataclass graph serialized to JSON; TSVs and the CLI summary are rendered from it.
5. **Reproducibility is automatic, not a feature.** Every run writes a `run.yaml` alongside its results; `selkit rerun run.yaml` reproduces bit-for-bit.
6. **Correctness before speed.** Pure numpy/scipy first, validated against PAML; numba/JIT hotpath optimization is a documented follow-on, not a v1 requirement.

## 4. UX decisions (locked)

| Axis | Decision |
|---|---|
| CLI shape | Grouped subcommands (`selkit codeml site-models ...`) **and** declarative YAML (`selkit run analysis.yaml`). |
| Input formats | FASTA + Newick by default; Phylip accepted as a fallback (sniff-detected). |
| Output formats | Canonical `results.json` **plus** per-analysis TSVs. Both produced every run. |
| Tree labeling | Accept **all three**: `#1`/`$1` in Newick, name-based CLI flags (`--foreground`, `--foreground-tips`, `--clade-mrca`), **and** a labels TSV file. Collision is an error. |
| LRTs | Auto-run for "named bundles" (site-models, branch-models, branch-site); overridable with `--tests`. |
| Parallelism | Model-level (`--threads N`) **and** multi-gene batch mode (`--alignment-dir ...`). |
| Python API | Library-first; CLI is a thin wrapper. |
| Progress UX | `rich` progress bars per model; multi-start optimization (K=3 by default) with a convergence-agreement gate. |
| Validation | Strict by default; dedicated `selkit validate` subcommand. |
| Reproducibility | Auto-written `run.yaml` in output dir; `selkit rerun` replays it. |

## 5. Architecture

Four layers, one-way dependencies: **io → engine → services → cli**.

```
selkit/
├── selkit/
│   ├── __init__.py           # public library API
│   ├── __main__.py           # CLI entry
│   ├── cli.py                # argparse setup
│   ├── cli_registry.py       # subcommand registration (PhyKIT-style)
│   ├── services/             # orchestration layer
│   │   ├── codeml/
│   │   │   ├── site_models.py
│   │   │   ├── branch_models.py
│   │   │   ├── branch_site_models.py
│   │   │   └── lrt.py
│   │   ├── yn00/pairwise.py
│   │   └── validate.py
│   ├── engine/               # pure ML engine, no IO
│   │   ├── genetic_code.py   # codon tables, stops, translation
│   │   ├── rate_matrix.py    # Q(ω, κ, π), eigen-decomp, P(t)
│   │   ├── codon_model.py    # M0, M1a, M2a, M7, M8, M8a, branch, branch-site
│   │   ├── likelihood.py     # Felsenstein pruning; per-site & total lnL
│   │   ├── optimize.py       # multi-start L-BFGS-B, convergence gate
│   │   ├── beb.py            # Bayes Empirical Bayes posteriors
│   │   └── simulator.py      # internal sequence simulator (testing only)
│   ├── io/
│   │   ├── alignment.py      # FASTA/Phylip readers; codon validation
│   │   ├── tree.py           # Newick parser; label normalization
│   │   ├── config.py         # YAML load/save (run.yaml)
│   │   └── results.py        # JSON + TSV emitters; dataclasses
│   ├── progress/runner.py    # rich-based UI for parallel runs
│   ├── errors.py
│   └── version.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── validation/           # PAML-comparison corpus
├── pyproject.toml
└── README.md
```

### Layer contracts

- **`engine/`** is pure math. Takes in-memory arrays and `LabeledTree` objects; returns `ModelFit`. No file IO, no side effects, no logging to stdout. Trivially unit-testable; the same code runs from CLI, Jupyter, or a pipeline.
- **`io/`** is pure parsing / serialization. Never calls engine or services. Returns validated dataclasses.
- **`services/`** orchestrates: expands bundles, schedules parallel fits, runs LRTs, emits BEB. Calls engine + io; never called by engine.
- **`cli/`** is argparse glue. Parses argv / YAML into `RunConfig`, invokes a service, exits with the right code.

### Numerical stack (decision)

**numpy + scipy first, with a documented path to numba hotpath optimization.**
- numpy for rate matrices, pruning, site likelihoods.
- `scipy.optimize.minimize(method="L-BFGS-B")` for parameter fitting.
- Parameter transforms enforce constraints (softplus for non-negative; logit for 0-1).
- Multi-start by seeded random perturbation of starting values.
- **No JAX, no cython in v1.** If profiling (post-validation) shows ≥3× gap vs PAML on the representative workload, numba-JIT the identified hotpath(s) — a localized refactor, not an architectural change.

## 6. Core data types

```python
@dataclass(frozen=True)
class CodonAlignment:
    taxa: tuple[str, ...]
    codons: np.ndarray            # shape (n_taxa, n_codons), int codes
    genetic_code: str
    stripped_sites: tuple[int, ...]

@dataclass(frozen=True)
class LabeledTree:
    newick: str                   # canonical form, labels stripped
    labels: dict[str, int]        # branch_id -> label (0=background, 1+ = foreground)
    tip_order: tuple[str, ...]

@dataclass(frozen=True)
class ModelFit:
    model: str                    # "M0", "M1a", "M8", "branch-site-A", ...
    lnL: float
    n_params: int
    params: dict[str, float]
    branch_lengths: dict[str, float]
    starts: list[StartResult]
    converged: bool
    runtime_s: float

@dataclass(frozen=True)
class StartResult:
    seed: int
    final_lnL: float
    iterations: int
    params: dict[str, float]

@dataclass(frozen=True)
class LRTResult:
    null: str
    alt: str
    delta_lnL: float
    df: int
    p_value: float
    test_type: Literal["chi2", "mixed_chi2"]
    significant_at_0_05: bool

@dataclass(frozen=True)
class BEBSite:
    site: int                     # 1-indexed codon position
    p_positive: float             # posterior P(ω > 1)
    mean_omega: float

@dataclass(frozen=True)
class RunConfig:
    # full invocation, serialized to/from run.yaml
    alignment: Path                    # or alignment_dir for batch mode
    alignment_dir: Path | None
    tree: Path
    labels: BranchLabelSpec            # union: in-newick | flags | file
    subcommand: str                    # "codeml.site-models", ...
    models: tuple[str, ...]            # expanded from bundle, e.g. ("M0","M1a",...)
    tests: tuple[str, ...]             # expanded LRT pairs
    genetic_code: str                  # "standard", "vertebrate_mito", ...
    output_dir: Path
    threads: int
    seed: int
    n_starts: int                      # multi-start K (default 3)
    convergence_tol: float             # lnL disagreement threshold (default 0.5)
    strict: StrictFlags                # stop-codon / length-mismatch behavior
    selkit_version: str                # auto-filled
    git_sha: str | None                # auto-filled if available

@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    fits: dict[str, ModelFit]
    lrts: list[LRTResult]
    beb: dict[str, list[BEBSite]]
    warnings: list[str]
```

JSON output = `RunResult` serialized. TSV output = flattened views (one TSV for fits, one for LRTs, one BEB TSV per model). Library return value = `RunResult`.

## 7. Data flow — single run

Walkthrough of `selkit codeml site-models --alignment aln.fa --tree t.nwk --output out/ --threads 4`:

1. **Parse & resolve config.** `cli.py` maps argv to `RunConfig`. If `--config run.yaml` is given, YAML is loaded and merged (CLI flags override YAML).
2. **Validate inputs** (see §8).
3. **Plan the work.** Services layer expands the bundle (`site-models` → `[M0, M1a, M2a, M7, M8, M8a]`), determines LRTs (`M1a↔M2a`, `M7↔M8`, `M8a↔M8`), and marks which models need BEB (`M2a`, `M8`).
4. **Fit models in parallel.** `ProcessPoolExecutor(max_workers=threads)` dispatches one model per worker. Each worker runs K=3 multi-start L-BFGS-B optimizations; the best-lnL result wins, with `converged=False` flagged if top-2 starts disagree by >0.5 lnL units. Per-worker progress is streamed to the main process and rendered as stacked `rich` bars.
5. **Run LRTs.** `lrt.py` computes `2·(lnL_alt − lnL_null)`, applies the correct df, and uses mixed 50:50 χ²₀:χ²₁ for boundary cases (M8a-vs-M8, branch-site A-vs-null).
6. **Run BEB.** For models with a positive-selection ω class (M2a, M8, branch-site A), compute per-site posterior P(ω>1) per Yang, Wong & Nielsen 2005.
7. **Emit outputs.**
   - `out/results.json` — canonical `RunResult`.
   - `out/fits.tsv`, `out/lrts.tsv`, `out/beb_<model>.tsv`.
   - `out/run.yaml` — the exact `RunConfig` + selkit version + git-sha if available.
   - stdout: `rich` summary table (model / lnL / ω / converged / LRT p-value).
8. **Exit.** 0 on success, 1 on validation error, 2 on convergence failure (opt-out: `--allow-unconverged`), 3 on internal engine error.

## 8. Input validation

`services/validate.py` runs every check short of model fitting. Same code path powers the standalone `selkit validate` subcommand and the pre-flight validation inside `codeml` / `yn00`.

### Stop-codon handling

| Situation | Default | Escape hatch |
|---|---|---|
| Terminal stop on all taxa at the same final position | Auto-strip with INFO note | `--no-strip-terminal-stop` |
| Mid-sequence stop codon (any taxon) | Error (taxon + codon position) | `--strip-stop-codons` (drop any column where ≥1 taxon has a stop) or `--mask-stop-codons` (treat stops as missing data per taxon) |

### Sequence length

| Situation | Default | Escape hatch |
|---|---|---|
| Length % 3 ≠ 0 for any taxon | Error (taxon + length + remainder) | `--trim-trailing` (only valid if all taxa have the same remainder; drops final 1-2 bp uniformly) |

### Tree / alignment consistency

- Tip names in tree must exactly match taxa in alignment. Mismatch → error with the two sets. `--prune-unmatched` drops tips/taxa missing from the other side.
- Duplicate taxa in alignment → error.
- Empty sequences → error.

### Branch-label collision

If more than one of `--foreground`, `--foreground-tips`, `--labels-file`, or in-Newick `#1` markers is provided, error. (Explicit > implicit; we refuse to guess which source wins.)

### Error-message template

```
ERROR: <what>
  <key>: <value>
  ...
  hint: <actionable suggestion with flag name>
```

All errors emit the same structure. Error messages are also available machine-readably via `--json-errors` for CI integrations.

## 9. Testing & validation

Three CI tiers plus a simulation-based correctness check:

### Tier 1 — Unit tests (`tests/unit/`, every PR)

- `genetic_code.py`: codon ↔ AA mapping, stop detection across all 12 NCBI genetic codes.
- `rate_matrix.py`: Q row sums = 0; eigendecomp round-trip (`V·diag(λ)·V⁻¹ ≈ Q`); `P(t) = exp(Qt)` is stochastic for t ∈ {0, 0.01, 1, 100}.
- `likelihood.py`: pruning on 3-taxon toy trees with closed-form expected lnL.
- `codon_model.py`: mixture weights sum to 1 per model; parameter-to-Q mappings; constraint transforms are invertible.
- `optimize.py`: convergence gate fires on a deliberately bumpy synthetic likelihood.
- `io/`: FASTA / Phylip / Newick round-trips; label normalization collapses `#1`, CLI flags, and labels-file to identical internal state.

### Tier 2 — Integration tests (`tests/integration/`, every PR)

- End-to-end on tiny alignments; assert result JSON shape (not values), error codes, `run.yaml` round-trip.
- `selkit validate` catches every category in the error taxonomy.
- Parallel runs produce identical results to sequential (reproducibility under threading).

### Tier 3 — PAML-comparison validation corpus (`tests/validation/`, nightly + tagged releases)

The correctness gate. A directory of canonical `(alignment, tree, expected.json)` triples:

- Coverage: small / large / deep / shallow trees; every model family; tricky cases (high ω, ω≈1, long branches, many taxa).
- Each `expected.json` is produced from a one-time PAML run, checked into the repo.
- Per-fit thresholds:
  - `|lnL_selkit − lnL_paml| ≤ 0.01`
  - `|ω_selkit − ω_paml| ≤ 1e-3`
  - `|branch_length_selkit − branch_length_paml| ≤ 1e-2`
- BEB site-level posteriors agree within 1e-3 on sites PAML identifies as positively selected.
- Every codon-model family needs ≥3 triples in the corpus before that family is considered shipped.

### Simulation-based validation (`tests/validation/simulation/`, tagged releases only)

`engine/simulator.py` generates synthetic codon alignments under known ω. For each model family, N replicates; assert that fitting recovers ω within Monte-Carlo tolerance. Catches bugs independent of PAML itself (i.e., cases where both implementations could be wrong in the same way).

### Coverage targets

- `engine/`: 90% line coverage.
- `services/`, `io/`: 80%.
- `cli/`: smoke tests only; no coverage target.

### CI topology

- Every PR: unit + integration (~1 minute).
- Nightly + tagged releases: validation corpus (~10 minutes).
- Tagged releases only: simulation tests (~1 hour).

## 10. Error handling

Exception hierarchy in `errors.py`:

- `SelkitInputError` — malformed files, stop codons, taxon mismatch. Always actionable; see §8 template.
- `SelkitConfigError` — invalid model name, conflicting flags, unknown genetic code.
- `SelkitConvergenceWarning` — non-fatal. Multi-start lnL disagreement past threshold. Written to `RunResult.warnings`; surfaces in summary table.
- `SelkitEngineError` — numerical blow-up (singular Q, negative branch length optimizer drift). Ships with an issue-template link; users should never see this without context.

**Explicit non-behaviors:**
- No silent coercion anywhere in the pipeline.
- No `try/except/continue` inside engine code — engine either returns a valid `ModelFit` or raises.
- Services decide whether one model's failure aborts a bundle (default: record the failure, skip that model, exit non-zero).
- `--warnings-as-errors` flips every warning into a non-zero exit for CI use.

## 11. Build order (proposed, to be refined during planning)

1. `engine/genetic_code.py`, `engine/rate_matrix.py`, `io/alignment.py`, `io/tree.py` — foundation.
2. `engine/likelihood.py` + M0 — prove Felsenstein pruning against PAML on a 3-taxon toy case.
3. `engine/optimize.py` + multi-start — validate on M0 fits.
4. Site-model family (M1a, M2a, M7, M8, M8a) + LRT + BEB.
5. `services/codeml/site_models.py`, CLI, JSON/TSV/YAML IO — first shippable slice.
6. yn00 (closed-form; fast to add once alignment IO + genetic-code exist).
7. Branch models.
8. Branch-site models (requires labeled-tree infrastructure).

## 12. Open questions

None at the time of writing; to be raised during implementation planning.

## 13. Appendix — references

- Yang, Z. (2007). PAML 4: Phylogenetic Analysis by Maximum Likelihood. *Mol. Biol. Evol.* 24(8):1586-1591.
- Yang, Z. & Nielsen, R. (2000). Estimating synonymous and nonsynonymous substitution rates under realistic evolutionary models. *Mol. Biol. Evol.* 17(1):32-43.
- Yang, Z., Wong, W.S.W. & Nielsen, R. (2005). Bayes empirical Bayes inference of amino acid sites under positive selection. *Mol. Biol. Evol.* 22(4):1107-1118.
- Zhang, J., Nielsen, R. & Yang, Z. (2005). Evaluation of an improved branch-site likelihood method for detecting positive selection at the molecular level. *Mol. Biol. Evol.* 22(12):2472-2479.
- Felsenstein, J. (1981). Evolutionary trees from DNA sequences: a maximum likelihood approach. *J. Mol. Evol.* 17(6):368-376.
