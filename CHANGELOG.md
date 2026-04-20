# Changelog

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
