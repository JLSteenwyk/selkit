Branch-site test: positive selection on specific lineages
==========================================================

Site models (tutorials 02 and 03) ask whether *any* codon in your alignment
evolves faster than neutral — averaged across every branch of the tree. That is
a useful first pass, but it conflates signal from many lineages and can miss
episodes of adaptation that are restricted to a single clade. **Branch-site
models** ask a sharper question: did a *pre-designated foreground lineage*
experience positive selection at some codons, even if the rest of the tree
(the background) did not?

The canonical example is primate lysozyme. Messier and Stewart (1997)
hypothesised that the colobine primate ancestor — monkeys that evolved foregut
fermentation to digest leaves — experienced positive selection on the stomach
lysozyme gene after its divergence from other primates. The branch leading to
all colobines is the foreground; every other branch is the background. Zhang
et al. (2005) introduced the formal branch-site test for this dataset.

**Dataset:** Yang 1998 / Zhang et al. 2005 primate lysozyme alignment —
19 sequences, 130 codons. The colobine clade is already labelled ``#1`` in the
tree file distributed with this tutorial.

**Hypothesis:** Did the colobine lineage experience episodic positive selection
at some codons of lysozyme after its divergence from other primates?


The two branch-site models
--------------------------

Model A null
~~~~~~~~~~~~

Model A null has four site classes that together cover every combination of
purifying/neutral behaviour on foreground and background branches:

- **Class 0** — ω = ω₀ ∈ (0, 1) on all branches (purifying everywhere).
- **Class 1** — ω = 1 on all branches (neutral everywhere).
- **Classes 2a and 2b** — ω = ω₀ (class 2a) or ω = 1 (class 2b) on background
  branches, and **ω₂ fixed at 1** on the foreground branches.

With ω₂ pinned to 1 the foreground branches cannot be faster than neutral.
This is the null hypothesis.

Model A (alternative)
~~~~~~~~~~~~~~~~~~~~~

Model A has the same four classes, but the foreground rate in classes 2a and 2b
is a *free* parameter ω₂ ≥ 1. When ω₂ > 1 those site classes represent positive
selection specifically on the foreground lineage.

The LRT statistic is 2 · (lnL\ :sub:`ModelA` − lnL\ :sub:`ModelA_null`).
Because ω₂ = 1 is a boundary of the allowed region (ω₂ ≥ 1), the null
distribution is a 50:50 mixture of χ²₀ (a point mass at zero) and χ²₁.
selkit detects the ModelA_null vs ModelA pair by name and applies this
mixed-χ² correction automatically — the same mechanism used for the M8a vs M8
boundary test in tutorial 03.


Designating the foreground
--------------------------

The colobine branch can be labelled three equivalent ways:

1. **Inline ``#1`` in the Newick** — the corpus tree ships this way; no extra
   flag is needed.

2. **``--foreground`` flag** — pass a comma-separated list of tip names whose
   MRCA defines the clade::

       --foreground 5.colobus_Cgu&Can,6.langur_Sen&Sve,7.langur_Tob&Tfr,8.Douc_langur_Pne,9.probiscis_Nla

3. **``--labels-file``** — a tab-delimited text file with a ``tip`` column and a
   ``label`` column; set ``label=1`` for each colobine tip and leave the others
   blank.

All three produce identical results. For the run below we use the pre-labelled
tree file, so no extra flag is needed.


Step 1: Download the dataset
-----------------------------

.. code-block:: shell

    mkdir lysozyme && cd lysozyme
    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/lysozyme_branchsite/alignment.fa
    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/lysozyme_branchsite/tree.nwk

Or with ``wget``:

.. code-block:: shell

    mkdir lysozyme && cd lysozyme
    wget https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/lysozyme_branchsite/alignment.fa
    wget https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/lysozyme_branchsite/tree.nwk

The tree already contains the ``#1`` label on the colobine clade. You can
inspect it before running to confirm which branches are foreground.


Step 2: Validate, then fit both branch-site models
----------------------------------------------------

First, confirm that the alignment and tree are consistent:

.. code-block:: shell

    selkit validate --alignment alignment.fa --tree tree.nwk
    # OK: 19 taxa, 130 codons, genetic code = standard

Then fit both models in a single command:

.. code-block:: shell

    selkit codeml site-models \
        --alignment alignment.fa \
        --tree tree.nwk \
        --output results/ \
        --models ModelA_null,ModelA \
        --n-starts 3 --seed 0 --threads 2

Flags used:

- ``--models ModelA_null,ModelA`` — fit the null and the alternative in one run.
- ``--n-starts 3`` — three independent optimisation starts; the best lnL is kept.
- ``--seed 0`` — deterministic starting values for reproducibility.
- ``--threads 2`` — run both models concurrently.

Expected runtime: approximately **5 minutes** on a laptop with 2 threads. Both
models are slower than M0 or M1a because each gradient step requires evaluating
four site-class rate matrices, and the MLE of ω₀ tends to sit near the (0, 1)
boundary, which slows convergence.


Step 3: Inspect the console output
------------------------------------

When the run finishes selkit prints a model-fits table and an LRT table:

.. code-block:: text

    ModelA_null ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ done
    ModelA      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ done
                      selkit codeml site-models
    ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
    ┃ model       ┃ lnL        ┃ omega (or omega2) ┃ converged ┃
    ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
    │ ModelA_null │ -1036.2674 │ 0 (boundary)      │ yes       │
    │ ModelA      │ -1035.5305 │ 4.8097            │ yes       │
    └─────────────┴────────────┴───────────────────┴───────────┘
                               LRTs
    ┏━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━┳━━━━━━━━┳━━━━━━┓
    ┃ null        ┃ alt     ┃ 2dlnL  ┃ df ┃ p      ┃ sig. ┃
    ┡━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━╇━━━━━━━━╇━━━━━━┩
    │ ModelA_null │ ModelA  │ 1.4738 │ 1  │ 0.1124 │      │
    └─────────────┴─────────┴────────┴────┴────────┴──────┘

Reading the fits table:

- **ModelA_null** — lnL = −1036.2674. The ω₀ estimate hits the lower boundary
  (0), meaning sites in classes 0 and 2a are inferred to be evolving at
  essentially zero nonsynonymous rate. Kappa = 4.0623, p0 = 0.2144,
  p1_frac = 0.2256.
- **ModelA** — lnL = −1035.5305. The foreground positive-selection rate
  ω₂ = 4.8097 is well above 1, and roughly 40 % of sites are estimated to fall
  in the positive class on the foreground (p0 = 0.3266, p1_frac = 0.4000,
  kappa = 4.1547). However, the lnL gain over the null is small.

Reading the LRT:

- **2·ΔlnL = 1.4738**, df = 1, mixed-χ² p = **0.1124**.
- The empty ``sig.`` column signals a non-significant result at α = 0.05.


Step 4: Interpret the result
-----------------------------

The branch-site test of positive selection on the colobine lysozyme lineage is
**not significant** (p ≈ 0.11, mixed χ², α = 0.05).

Interpretation contract:

- **p < 0.05** — evidence for positive selection at some codons specifically on
  the labelled (foreground) branches. The alternative Model A is preferred.
- **p ≥ 0.05** — cannot reject the null of no positive selection on the
  foreground. The signal, if any, is too weak to distinguish from neutral
  evolution at the ω = 1 boundary.

For the colobine lysozyme dataset the conclusion is: **the data do not provide
sufficient evidence to reject the null at conventional significance**. This
matches PAML 4.10.10, which reports p ≈ 0.1128 for this dataset.

Historical context: Zhang et al. (2005) originally found weak evidence for
positive selection on this lineage using the first version of the branch-site
test. Larger modern lysozyme alignments and the mixed-χ² boundary correction
have since softened that signal. The branch-site test is deliberately conservative
because mis-specification of the background ω process can produce false
positives; the mixed-χ² correction helps guard against this.


Step 5: Fraction of sites in the positive class
-------------------------------------------------

Even though the LRT is not significant, the parameter estimates are still
informative. Under Model A, the proportion of sites in the foreground-positive
classes (2a + 2b) is:

  p₂ = 1 − p₀ − (1 − p₀) · p1_frac = 1 − 0.3266 − 0.6734 · 0.4000 ≈ **0.40**

Roughly 40 % of sites are estimated to be under positive selection on the
foreground — but because the test statistic is only 1.47 (p = 0.11) we cannot
rule out that the optimiser is picking up noise near the ω = 1 boundary rather
than a genuine positive-selection signal.

.. note::

   selkit v0.1 does not yet compute the Bayes Empirical Bayes posterior of
   *which individual codons* on the foreground are under positive selection for
   Model A. Per-site posteriors under Model A will be added in a later release.
   For now, p₂ gives the estimated *fraction* of sites in the positive class
   averaged over the foreground lineage.


Numerical parity with PAML
---------------------------

selkit's Model A and Model A null log-likelihoods match PAML 4.10.10 to better
than ``1e-3`` lnL units on this dataset. Starting the optimiser from scratch,
selkit converges to the same optimum (``|ΔlnL| < 0.01``). These checks are
encoded in ``tests/integration/test_paml_lnl_match.py`` and run on every
release.


Python API equivalent
----------------------

For scripted or pipeline use, the same analysis runs entirely in Python:

.. code-block:: python

    from pathlib import Path
    from selkit import codeml_site_models
    from selkit.io.tree import ForegroundSpec

    result = codeml_site_models(
        alignment=Path("alignment.fa"),
        tree=Path("tree.nwk"),
        output_dir=Path("results/"),
        models=("ModelA_null", "ModelA"),
        # If the tree already has `#1` inline, this argument is optional;
        # otherwise pass the foreground spec explicitly.
        foreground=ForegroundSpec(mrca=(
            "5.colobus_Cgu&Can", "6.langur_Sen&Sve", "7.langur_Tob&Tfr",
            "8.Douc_langur_Pne", "9.probiscis_Nla",
        )),
        n_starts=3, seed=0, threads=2,
    )
    branchsite = next(
        l for l in result.lrts if l.null == "ModelA_null" and l.alt == "ModelA"
    )
    verdict = "significant" if branchsite.significant_at_0_05 else "not significant"
    print(f"branch-site test: p = {branchsite.p_value:.4g} ({verdict})")

``result.lrts`` is a list of ``LRTResult`` objects with the same fields as the
TSV file written to disk. The ``ForegroundSpec`` argument is optional when the
tree already carries the ``#1`` label; it is shown here to document the
programmatic equivalent of the ``--foreground`` CLI flag.


What next?
----------

- :doc:`02_detecting_positive_selection` — M1a vs M2a site-model test. Use this
  when your hypothesis is "are any sites under positive selection across the
  whole tree?" rather than "is a specific lineage under positive selection?".

- :doc:`04_library_workflow` — batch analysis across hundreds of genes using the
  Python API, including error handling and result aggregation.
