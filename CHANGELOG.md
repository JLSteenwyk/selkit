# Changelog

## 0.3.0 (unreleased — in progress)

### Refactor (Phase 1)

This phase reshapes the service/CLI layer around the three model families (site, branch, branch-site) with **zero behavior change on PAML numerics**. It is the structural foundation for Phase 2 (branch models) and Phase 3 (true BEB), which build on the orchestrator pattern introduced here.

**Breaking changes** (pre-1.0, no compat aliases):

- CLI subcommand `selkit codeml site-models` renamed to `selkit codeml site`.
- New CLI subcommand `selkit codeml branch-site` for `ModelA` / `ModelA_null` (previously hosted under `site-models`).
- `selkit rerun` hard-fails on v0.2 `run.yaml` files (`subcommand: codeml.site-models`) with a migration message pointing at the two new subcommands.
- `BEBSite.mean_omega` field renamed to `posterior_mean_omega` to make the statistical meaning explicit.
- `RunResult` gains a required `family: Literal["site", "branch", "branch-site"]` field.
- `BEBSite` gains optional fields `p_class_2a`, `p_class_2b`, `beb_grid_size` (all `None` in Phase 1; populated by Phase 3).

**Internal refactor:**

- `engine/beb.py` promoted to `engine/beb/` package (`__init__.py`, `site.py`, `_grid.py` stub for Phase 3).
- `services/codeml/site_models.py` (~234 lines) split into `site_models.py` (~58 lines, M0–M8a only) + `branch_site.py` (~59 lines, ModelA/ModelA_null) + a shared `_orchestrator.py` (~330 lines) that owns the validate → fit → LRT → BEB pipeline.
- `services/codeml/_orchestrator.py:run_family(family, registry, default_lrts, default_beb_models, ...)` is the single execution path that all three family services delegate to. Phase 2's `branch_models.py` and yn00 (v0.4) reuse the same shape.
- Tagged-union fit dataclasses introduced: `SiteModelFit`, `BranchSiteModelFit` (with `class_proportions` for ModelA), and `BranchModelFit` (stub for Phase 2). Legacy `ModelFit` retained as a transitional alias.
- `MODEL_REGISTRY` dicts use named module-scope factory functions instead of lambdas, so they pickle cleanly through `ProcessPoolExecutor`.

**New library API:**

- `selkit.codeml_branch_site_models(...)` — calls `run_branch_site_models`; mirrors the existing `codeml_site_models` shape but requires `foreground`.
- `selkit.codeml_branch_models(...)` — stub raising `NotImplementedError("... Phase 2 ...")` so the import surface is stable across the upcoming phases.

**Output schema:**

- `results.json` gains a top-level `family` field per fit.
- `run.yaml` gains a `family` field derived from `subcommand` (informational; `subcommand` remains the source of truth).

**Migration from 0.2.x:**

```
# old (v0.2)
selkit codeml site-models --alignment X --tree Y --output Z \
    --models M0,M1a,M2a,M7,M8,M8a,ModelA,ModelA_null

# new (v0.3) — split into two runs
selkit codeml site        --alignment X --tree Y --output Z/site
selkit codeml branch-site --alignment X --tree Y --output Z/branch_site \
    --foreground <labels>
```

PAML numerical agreement preserved: all 12 corpus lnL-match cases (HIV 4-taxon and 13-taxon site models, lysozyme branch-site Model A and Model A null) still pass at PAML's reported point within `|ΔlnL| < 1e-3`.

### Branch models (Phase 2)

- Four Yang-1998 branch models: `TwoRatios` (K=1 foreground-vs-background ω),
  `TwoRatiosFixed` (ω_fg pinned at 1; null for the boundary LRT),
  `NRatios` (K ≥ 1; one ω per `#`-label class), and `FreeRatios` (one ω
  per branch, with the two root-adjacent branches merged per PAML
  convention).
- Shared engine helper `_build_n_ratios_qs` and generalised
  `scale_per_label_qs` replace the branch-site-specific
  `scale_branch_site_qs` (kept as a thin wrapper so the existing
  lysozyme branch-site corpus stays byte-for-byte).
- CLI: `selkit codeml branch --alignment ... --tree ... --output ...
  [--models M0,TwoRatios,TwoRatiosFixed,NRatios,FreeRatios]
  [--foreground ...]`. Defaults to `M0,TwoRatios,TwoRatiosFixed` —
  the canonical positive-selection trio.
- Library: `codeml_branch_models(...)` sibling to `codeml_site_models` /
  `codeml_branch_site_models`.
- `BranchModelFit` now carries a structured `per_branch_omega` array
  (`branch_id`, `tip_set`, `label`, `paml_node_id`, `omega`, `SE`). Output
  TSVs split into `fits_branch.tsv` (per-model summary) and
  `fits_branch_per_branch.tsv` (per-branch rows; `tip_set` pipe-delimited).
- Per-branch ω **standard errors** are now populated: `EngineFit` exposes
  `hess_inv_diag` (a per-parameter natural-space SE dict derived from the
  L-BFGS-B inverse-Hessian approximation with a delta-method Jacobian
  correction), and `_extract_per_branch_omega` routes it into the per-branch
  `SE` field. Requires `scipy>=1.0`; falls back to `null` with a warning on
  older scipy. Treat SEs as a guide, not a rigorous CI — see tutorial 06
  for caveats.
- LRT registry gains lazy-df specifiers (`"lazy_K"` and `"lazy_Bminus2"`)
  resolved against the tree at test-fire time. Registered branch-model
  LRTs: `M0 vs TwoRatios` (1 df, χ²), `TwoRatiosFixed vs TwoRatios`
  (1 df, mixed χ²), `M0 vs NRatios` (K df, χ²), and `M0 vs FreeRatios`
  (B−2 df, χ² with a "caution" warning).
- Two new PAML corpus cases: `lysozyme_two_ratios` (TwoRatios /
  TwoRatiosFixed on the colobine clade) and `hiv_4s_free_ratios`
  (FreeRatios on the 4-taxon HIV quartet). Both currently skipped
  pending Jacob's PAML reference outputs (`expected.json`); meta.yaml,
  alignment, and tree are committed.
- New tutorial: `docs/tutorials/06_branch_test.rst`, mirroring
  tutorial 05's structure.
- Output TSV emission now splits by family: `fits_site.tsv`,
  `fits_branch.tsv` + `fits_branch_per_branch.tsv`, and
  `fits_branch_site.tsv`. The legacy flat `fits.tsv` is removed.
- `LRTResult` gains an optional `warning: str | None` field.
- `selkit rerun` supports `subcommand: codeml.branch` manifests.

Phase 3 (true BEB) follows.

## 0.2.0

- **Branch-site Model A and Model A null.** The branch-site test of positive selection (Zhang et al. 2005; Yang et al. 2005) — tests whether a pre-designated foreground lineage experienced episodic adaptation at some codons. Fit via `--models ModelA,ModelA_null`. LRT `ModelA_null` vs `ModelA` is 1 df, mixed 50:50 χ²₀:χ²₁ (boundary test); included automatically when both models are fit.
- Engine gained per-label Q dispatch: `_prune_tree_partials` now accepts either a single ndarray (site models) or a `dict[int, ndarray]` keyed by branch label (branch-site). Backward compatible.
- `scale_branch_site_qs` scales each label's Qs by that label's class-averaged mean rate — matches PAML's convention that branch lengths are "expected substitutions per codon averaged over site classes on that branch". Verified against PAML 4.10.10 on the lysozyme dataset.
- New PAML corpus case `tests/validation/corpus/lysozyme_branchsite/` (19-taxon primate lysozyme, colobine clade foreground). Static lnL match to `<1e-3` for Model A and Model A null at PAML's reported point; end-to-end optimiser fit agrees with PAML to `<0.01` lnL.
- Tutorial 05 walks through the branch-site workflow end-to-end.
- Orchestrator refuses to fit `ModelA` / `ModelA_null` when no foreground is labelled on the tree (previously the fit would silently succeed as an ordinary site model).

Not yet in v1.1 (tracked for follow-ups):

- Per-site NEB posteriors for Model A (which sites on the foreground are positively selected). v0.2 reports the bulk `p2 = p2a + p2b` fraction; per-site posteriors land in a later release.
- Branch models with multiple ω ratios (Yang 1998 "free-ratios" etc.).
- `yn00` pairwise dN/dS.

## 0.1.0

- First public release. Pure-Python reimplementation of PAML's codeml site-model workflow.
- Site models implemented and PAML-validated: M0, M1a, M2a, M7, M8, M8a.
- Automated likelihood-ratio tests (M1a-vs-M2a, M7-vs-M8, M8a-vs-M8) with standard and mixed χ² null distributions.
- Per-site NEB posteriors for M2a and M8; full BEB (integration over hyperparameters) deferred to a later release.
- Multi-start L-BFGS-B optimization with a convergence-tolerance gate; model-level parallelism via `ProcessPoolExecutor`.
- Pure-Python FASTA and relaxed-Phylip readers with format auto-detect, BOM handling, and strict codon/stop-codon validation.
- Minimal Newick parser with PAML `#N` / `$N` branch-label syntax for foreground designation; `ForegroundSpec` normalisation across CLI flags and labels files.
- Reproducible `run.yaml` artifact + `selkit rerun` CLI for re-executing an analysis from its manifest.
- Structured outputs: JSON (canonical), per-table TSVs (`fits.tsv`, `lrts.tsv`, `beb_<model>.tsv`), and a rich-formatted summary on the console.
- Public library surface: `from selkit import codeml_site_models, RunResult, ModelFit, LRTResult, BEBSite, …`.
- Numerical agreement with PAML 4.10.10 verified on two corpus cases (4-taxon and 13-taxon HIV envelope subsets); `|ΔlnL| < 1e-3` at PAML's reported point across all five site models.

Not yet in v1 (tracked for follow-ups):

- `yn00` pairwise dN/dS.
- Branch models (one-ratio, two-ratios, free-ratios).
- Branch-site models (Model A, Model A null).
- True BEB integration over hyperparameters.
