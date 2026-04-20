.. _change_log:


Change log
==========

Major changes to selkit are summarised here.

**0.2.0**

- **Branch-site Model A and Model A null.** The branch-site test of positive selection (Zhang *et al.* 2005; Yang *et al.* 2005) — tests whether a pre-designated foreground lineage experienced episodic adaptation at some codons. Fit via ``--models ModelA,ModelA_null``. The LRT is automatic, 1 df, mixed 50:50 χ² (boundary test).
- Engine gained per-label Q dispatch: a site class's Q can now be either a single ndarray (site models) or a ``dict[int, ndarray]`` keyed by branch label (branch-site). Backward compatible with every existing call site.
- ``scale_branch_site_qs`` scales each label's Qs by that label's class-averaged mean rate — matches PAML's convention that branch lengths are "expected substitutions per codon averaged over site classes on that branch". Verified against PAML 4.10.10 on the lysozyme dataset.
- New PAML corpus case ``tests/validation/corpus/lysozyme_branchsite/`` (19-taxon primate lysozyme, colobine clade foreground). Static lnL match to better than 10⁻³ at PAML's reported point for both Model A and Model A null; end-to-end optimiser fit agrees with PAML to better than 10⁻².
- Tutorial 05 walks through the branch-site workflow end-to-end with the lysozyme example.
- Orchestrator refuses to fit ``ModelA`` / ``ModelA_null`` when no foreground is labelled on the tree (previously the fit would silently succeed as an ordinary site model).

**0.1.0**

- First public release. Pure-Python reimplementation of PAML's codeml site-model workflow.
- Site models implemented and PAML-validated: M0, M1a, M2a, M7, M8, M8a.
- Automated likelihood-ratio tests: M1a-vs-M2a, M7-vs-M8 (standard χ²), M8a-vs-M8 (mixed 50:50 χ²).
- Per-site NEB posteriors for M2a and M8; full BEB (integration over hyperparameters) deferred to a later release.
- Multi-start L-BFGS-B optimisation with a convergence-tolerance gate; model-level parallelism via ``ProcessPoolExecutor``.
- Pure-Python FASTA and relaxed-Phylip readers with format auto-detect, BOM handling, and strict codon/stop-codon validation.
- Minimal Newick parser with PAML ``#N`` / ``$N`` branch-label syntax for foreground designation; ``ForegroundSpec`` normalisation across CLI flags and labels files.
- Reproducible ``run.yaml`` artefact + ``selkit rerun`` CLI for re-executing an analysis from its manifest.
- Structured outputs: JSON (canonical), per-table TSVs (``fits.tsv``, ``lrts.tsv``, ``beb_<model>.tsv``), and a rich-formatted summary on the console.
- Public library surface: ``from selkit import codeml_site_models, RunResult, ModelFit, LRTResult, BEBSite, …``.
- Numerical agreement with PAML 4.10.10 verified on two corpus cases (4-taxon and 13-taxon HIV envelope subsets); ``|Δ lnL| < 1e-3`` at PAML's reported point across all five site models.

Not yet in v1 (tracked for follow-ups):

- ``yn00`` pairwise dN/dS.
- Branch models (one-ratio, two-ratios, free-ratios).
- Branch-site models (Model A, Model A null).
- True BEB integration over hyperparameters.
