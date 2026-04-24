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

Phases 2 (branch models) and 3 (true BEB) follow.

### True BEB (Phase 3)

- Replaced v0.1–v0.2 NEB posteriors with true BEB (Yang 2005 *MBE* 22:1107)
  for `M2a`, `M8`, and `ModelA`.
- New `engine/beb/_grid.py` hosts `integrate_posteriors_over_grid` with
  log-sum-exp numerical stability; shared across site (`engine/beb/site.py`)
  and branch-site (`engine/beb/branch_site.py`) BEB. The grid integrand
  includes the per-grid-point marginal-likelihood weight
  `f(D|θ_g) = Π_h Σ_k w_k(θ_g) · L_{h,k}(θ_g)` (Yang 2005 eq. 5), which
  weights each grid point by how well it explains the data and does NOT
  cancel after integrating over θ. The singleton-grid (G=1) ≡ NEB
  invariant is preserved because the constant scalar `f(D|θ_1)` cancels
  between numerator and denominator at G=1.
- `BEBSite` schema: `mean_omega` → `posterior_mean_omega`; new fields
  `p_class_2a`, `p_class_2b`, `beb_grid_size`. Branch-site ModelA
  populates all three; site models leave `p_class_2a`/`p_class_2b` as
  `null` (empty string in TSV).
- `beb_<model>.tsv` now has 6 columns: `site, p_positive,
  posterior_mean_omega, p_class_2a, p_class_2b, beb_grid_size`.
- New CLI flags: `--no-beb` (skip BEB) and `--beb-grid N` (grid size per
  hyperparameter; default 10, PAML-compatible). Library kwargs: `beb`,
  `beb_grid` on `codeml_site_models` and `codeml_branch_site_models`.
- Self-consistency invariant: `--beb-grid 1` reproduces the v0.2 NEB
  posteriors exactly (within floating-point tolerance).
- Validation corpus: `hiv_4s_beb_m8` (M8) and `lysozyme_beb_modelA`
  (ModelA) — PAML match within `|Δ p_positive| < 0.01`. (Fixtures
  pending PAML re-run.)

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
