About
=====

**selkit** is a modern, pure-Python reimplementation of the selection-analysis workflows in PAML (``codeml``, ``yn00``), written to be fast, installable with ``pip``, easy to embed as a library, and reproducible by design.

The first release focuses on ``codeml`` **site models** — the models most commonly used to test for episodic positive selection across sites in a codon alignment. For a four-taxon HIV alignment, a full M0 / M1a / M2a / M7 / M8 / M8a fit plus LRTs plus BEB posteriors completes in under a minute on a laptop.


Why selkit?
-----------

The PAML distribution is a remarkable piece of scientific software and the gold standard for codon-model likelihood analysis, but it is distributed as a Fortran/C codebase driven by a text control file (``codeml.ctl``). Typical users wrap ``codeml`` in shell scripts, parse its text output, and manage model comparisons by hand.

selkit aims to preserve PAML's numerical behaviour while offering:

- **Library-first API.** ``from selkit import codeml_site_models`` — one call returns a structured ``RunResult`` dataclass containing every fit, LRT, and BEB site.
- **Sensible defaults.** Running ``selkit codeml site-models`` with just ``--alignment`` and ``--tree`` fits the standard bundle (M0, M1a, M2a, M7, M8, M8a) and computes the standard LRTs (M1a-vs-M2a, M7-vs-M8, M8a-vs-M8) without further configuration.
- **Strict validation.** Malformed input (non-multiple-of-3 lengths, duplicate taxa, mid-sequence stop codons, taxon mismatches between alignment and tree) fails fast with an actionable error message, not a silent misfit.
- **Reproducibility.** Every run emits a ``run.yaml`` manifest; ``selkit rerun run.yaml`` reproduces the same analysis. Seeds, the multi-start convergence tolerance, and genetic-code choice are all captured.
- **Parallelism.** Multiple models fit concurrently via ``ProcessPoolExecutor`` when ``--threads > 1``.
- **Structured outputs.** Canonical JSON + per-table TSVs (``fits.tsv``, ``lrts.tsv``, ``beb_<model>.tsv``) mean no regex parsing of text output.


Correctness
-----------

Numerical agreement with PAML is verified in the test suite:

- ``tests/integration/test_paml_lnl_match.py`` runs on every push — it evaluates selkit's log-likelihood at PAML's reported (branch-length, parameter) point and asserts absolute Δ lnL < 10⁻³ across all five site models on both corpus cases.
- ``tests/validation/test_paml_corpus.py`` (opt-in, ``pytest -m validation``) runs the optimiser end-to-end from scratch and confirms selkit converges to the same optimum PAML reports.

See :doc:`../validation/index` for the numerical details.


Implementation notes
--------------------

selkit is pure Python on top of NumPy + SciPy. Core design choices:

- **Engine layer** (``selkit.engine``) is IO-free: ``GeneticCode``, ``build_q``, ``prob_transition_matrix`` (via ``scipy.linalg.expm``), ``tree_log_likelihood_mixture``, site-model implementations, and L-BFGS-B optimisation.
- **IO layer** (``selkit.io``) owns parsing (FASTA, relaxed-Phylip, Newick), validation, and the dataclasses that survive to the user (``CodonAlignment``, ``LabeledTree``, ``RunConfig``, ``RunResult``, ``ModelFit``, ``LRTResult``, ``BEBSite``).
- **Services layer** (``selkit.services``) wraps the engine with pre-flight validation and the model-bundle orchestrator.
- **CLI layer** (``selkit.cli``) is a thin argparse shell over the library.

The four layers have one-way dependencies (``io → engine → services → cli``), which keeps the engine testable in isolation and lets the CLI evolve without touching numerical code.


The developers
--------------

selkit is developed and maintained by
`Jacob L. Steenwyk <https://jlsteenwyk.github.io/>`_
and contributors. It sits alongside the broader jlsteenwyk toolkit family:
`ClipKIT <https://github.com/JLSteenwyk/ClipKIT>`_ (alignment trimming),
`PhyKIT <https://github.com/JLSteenwyk/PhyKIT>`_ (phylogenomic utilities),
`BioKIT <https://github.com/JLSteenwyk/BioKIT>`_ (general sequence utilities),
and others.
