PAML validation
===============

selkit's central correctness claim is numerical agreement with the reference PAML implementation. This page summarises how that claim is made concrete in the test suite.


Corpus cases
------------

Two corpus cases are checked in under ``tests/validation/corpus/``. Both are drawn from the ``HIVNSsites`` example distributed with PAML 4.10.10 (Yang *et al.* 2000).

- ``hiv_4s`` — 4 taxa × 91 codons (the 4-taxon HIVenvSweden subset). Small, fast to run end-to-end.
- ``hiv_13s`` — 13 taxa × 91 codons (the full HIVenvSweden alignment). Larger, exercises the beta-discretised ω machinery in M7 and M8 on non-degenerate data.

Each corpus directory contains:

.. code-block:: text

    tests/validation/corpus/<case>/
    ├── alignment.fa          # FASTA codon alignment
    ├── tree.nwk              # Newick tree (topology; BLs are starting values)
    ├── meta.yaml             # case metadata + which models to fit
    └── expected.json         # PAML's reported lnL, per-model parameters,
                              # per-model optimised Newick (BLs in substitutions
                              # per codon averaged over site classes)

``expected.json`` captures PAML's reported fit at the precision of its ``mlc`` output (typically 6 decimals on lnL; 5 decimals on ω / κ / p0).


Static lnL match (fast, runs in CI)
-----------------------------------

``tests/integration/test_paml_lnl_match.py`` is parametrised over every (corpus-case, model) pair and runs on every push via the standard CI job. It:

1. Loads the alignment, estimates F3X4 equilibrium frequencies.
2. Parses the **per-model** Newick (with PAML's optimised branch lengths) from ``expected.json``.
3. Builds the site-model Q matrices at PAML's reported ω / κ / p0 / p_beta / q_beta.
4. Calls ``tree_log_likelihood_mixture`` (Felsenstein pruning with running per-node scaling).
5. Asserts ``|selkit.lnL - PAML.lnL| < 10⁻³``.

Current status: **10 / 10 pass in ~1.4 seconds** (5 site models × 2 corpus cases).

What this test catches: any regression in ``build_q`` (the GY94 codon Q construction), ``prob_transition_matrix`` (scipy's Padé-13 ``expm``), ``estimate_f3x4``, mixture-Q scaling, or Felsenstein pruning. In other words, the core likelihood calculation.

What it does *not* catch: optimiser regressions (convergence, starting-values). That is covered by the slower test below.


End-to-end fit match (opt-in, ~15 minutes locally)
--------------------------------------------------

``tests/validation/test_paml_corpus.py`` runs ``selkit codeml site-models`` against each corpus case from scratch — multi-start L-BFGS-B included — and asserts both the lnL and the fitted parameters agree with PAML within published tolerances:

- ``|Δ lnL| < 10⁻²``.
- ``|Δ ω| < max(10⁻³, 0.5 % relative)`` (relative tolerance because large-ω classes on flat likelihood ridges are optimiser-stopping-point dependent).

This test is marked ``pytest.mark.validation`` and is not run by default. Trigger it locally with:

.. code-block:: shell

    pytest tests/validation -v -m validation

or via the Makefile:

.. code-block:: shell

    make test.validation

CI does not run the end-to-end fit; the static lnL match is the authoritative automated gate.


Caveats and methodology notes
-----------------------------

**F3X4 codon frequencies** — selkit uses the same F3X4 estimator PAML uses (raw per-position nucleotide counts, with the product normalised over sense codons). An optional ``pseudocount`` keyword on ``estimate_f3x4`` is available for degenerate inputs where some nucleotide isn't observed at some codon position; the default is ``0.0``, matching PAML.

**Mixture Q scaling** — Under PAML's convention, branch lengths are in units of expected codon substitutions averaged over site classes. selkit scales all classes of a mixture by a single weighted-average mean rate, preserving rate heterogeneity between classes (a slow class with ω=0 runs slower than a neutral class with ω=1 along the same branch). Early development of selkit used per-class scaling and produced silently wrong lnLs on mixtures; the current implementation is verified against PAML.

**Matrix exponential** — selkit computes P(t) = exp(Q·t) via ``scipy.linalg.expm`` (Padé-13 with scaling-and-squaring). An earlier implementation used eigendecomposition + inversion and returned garbage on rank-deficient Q (e.g. when ω=0 zeroes out many off-diagonal entries). This has been fixed; the current implementation is numerically stable.

**BEB vs NEB** — selkit v0.3 emits true BEB (Yang 2005). See "True BEB" below.


True BEB (Yang 2005) -- v0.3
----------------------------

v0.1--v0.2 reported NEB (Naive Empirical Bayes) posteriors evaluated at the
MLE. v0.3 replaces these with true BEB: integration of the per-class posteriors
over a grid of hyperparameter values, following Yang 2005 (*MBE* 22:1107).

Supported models:

- ``M2a``, ``M8`` (site) -- grid over ``(p0, p1, omega2)`` and ``(p0, p_beta, q_beta, omega2)``.
- ``ModelA`` (branch-site) -- grid over ``(p0, p1, omega2)``; reports per-class
  ``p_class_2a`` and ``p_class_2b`` separately.

Validation corpus:

- ``tests/validation/corpus/hiv_4s_beb_m8/`` -- M8 BEB posteriors vs PAML on
  the 4-taxon HIV env V3 subset. Threshold ``|delta p_positive| < 0.01``.
- ``tests/validation/corpus/lysozyme_beb_modelA/`` -- Model A BEB per-site
  posteriors vs PAML on the 19-taxon primate lysozyme with colobine
  foreground. Same threshold.

The ``--beb-grid N`` CLI flag and ``beb_grid=N`` library kwarg control the
integration grid size per hyperparameter. Default is 10, matching PAML.


Extending the corpus
--------------------

Adding a new case is mechanical:

1. Run PAML's ``codeml`` on (alignment, tree) with your preferred ``codeml.ctl``.
2. Create ``tests/validation/corpus/<case>/`` and copy in ``alignment.fa`` and ``tree.nwk``.
3. Extract per-model lnL, parameters, and the optimised Newick (with branch lengths) from the ``mlc`` output, and write them to ``expected.json`` in the schema described above.
4. Write ``meta.yaml`` specifying ``models``, ``n_starts``, ``seed``, etc.
5. The static test (``test_paml_lnl_match``) automatically discovers the new case via parametrisation.
