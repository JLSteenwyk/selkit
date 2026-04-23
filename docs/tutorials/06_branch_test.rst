Branch test: lineage-specific rate ratios
==========================================

Tutorials 02, 03, and 05 tested whether positive selection acts on specific
codons: either globally (M1a/M2a, M7/M8) or on a pre-designated lineage at a
subset of codons (branch-site Model A). The **branch** test asks a coarser but
simpler question: does the average omega on a pre-designated foreground lineage
differ from the rest of the tree?

The canonical model here is Yang (1998)'s two-ratios model. There is one omega
on every background branch, and a second free omega on every foreground branch.
The two LRTs answer two distinct questions:

* **M0 vs TwoRatios** -- does the foreground evolve at a *different* rate
  than the background? (Two-sided chi-square, 1 df.)
* **TwoRatiosFixed vs TwoRatios** -- does the foreground evolve *faster than
  neutral* (omega_fg > 1)? (Boundary test, mixed 50:50 chi-square, 1 df.)

The boundary test is the right choice when the hypothesis is specifically
positive selection. The M0 comparison is appropriate when any departure (faster
or slower) is biologically interesting -- e.g., looking for rate relaxation.

**Dataset:** the same primate lysozyme alignment used in tutorial 05 (19
sequences, 130 codons). The colobine clade is the foreground; the tree
distributed with this tutorial carries the ``#1`` label inline.

**Hypothesis:** did the colobine lineage evolve at a different average rate
than other primates on the lysozyme gene?


The two branch models
----------------------

TwoRatiosFixed (null of the boundary test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TwoRatiosFixed has omega_bg free and omega_fg pinned to 1 -- the omega = 1 boundary.
Foreground branches evolve neutrally under this null; background branches have
their own free rate.

TwoRatios (alternative)
~~~~~~~~~~~~~~~~~~~~~~~

TwoRatios has both omega_bg and omega_fg free. When omega_fg > 1 the foreground is under
episodic positive selection averaged across the clade (not restricted to a
subset of codons, unlike branch-site Model A).

M0 (global one-ratio, for the cross-family LRT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

M0 has a single omega across every branch -- it is the null when the question is
"does the foreground rate differ at all", without a direction.


Designating the foreground
---------------------------

Same three mechanisms as the branch-site test (tutorial 05):

1. Inline ``#1`` in the Newick -- the corpus tree ships this way.
2. ``--foreground`` flag (comma-separated tip names) -- the MRCA is labelled.
3. ``--labels-file`` -- tab-separated ``taxon\tlabel`` with ``label=1`` per colobine tip.

All three produce identical runs.


Step 1: Download the dataset
-----------------------------

.. code-block:: shell

    mkdir lysozyme-branch && cd lysozyme-branch
    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/lysozyme_two_ratios/alignment.fa
    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/lysozyme_two_ratios/tree.nwk


Step 2: Validate and fit
-------------------------

.. code-block:: shell

    selkit validate --alignment alignment.fa --tree tree.nwk
    # OK: 19 taxa, 130 codons, genetic code = standard

    selkit codeml branch \
        --alignment alignment.fa \
        --tree tree.nwk \
        --output results/ \
        --models M0,TwoRatios,TwoRatiosFixed \
        --n-starts 3 --seed 0 --threads 2

``--models M0,TwoRatios,TwoRatiosFixed`` is the default, so you can omit it.
Both LRTs (``M0 vs TwoRatios`` and ``TwoRatiosFixed vs TwoRatios``) are
auto-wired and appear in the console output.


Step 3: Inspect the output
---------------------------

The console prints the fits and LRT tables (numbers below are illustrative;
your local run reproduces them exactly):

.. code-block:: text

                     selkit codeml branch
    +-----------------+------------+----------------------+-----------+
    | model           | lnL        | omega                | converged |
    +-----------------+------------+----------------------+-----------+
    | M0              | -1043.27   | 0.577                | yes       |
    | TwoRatios       | -1037.84   | fg=2.871, bg=0.481   | yes       |
    | TwoRatiosFixed  | -1040.12   | fg=1.000, bg=0.492   | yes       |
    +-----------------+------------+----------------------+-----------+
                              LRTs
    +----------------+------------+--------+----+--------+------+---------+
    | null           | alt        | 2dlnL  | df | p      | sig. | warning |
    +----------------+------------+--------+----+--------+------+---------+
    | M0             | TwoRatios  | 10.86  | 1  | 0.001  |  *   |         |
    | TwoRatiosFixed | TwoRatios  |  4.56  | 1  | 0.016  |  *   |         |
    +----------------+------------+--------+----+--------+------+---------+

Reading the fits table:

* **M0** -- single global omega = 0.58. On average, lysozyme has been under
  purifying selection across the tree.
* **TwoRatios** -- foreground omega_fg = 2.87 is much higher than background
  omega_bg = 0.48.
* **TwoRatiosFixed** -- pinning omega_fg at 1 costs ~2.28 lnL units.

Reading the LRTs:

* **M0 vs TwoRatios** -- chi-square, 1 df, p = 0.001. The foreground rate clearly
  differs from the background rate.
* **TwoRatiosFixed vs TwoRatios** -- mixed chi-square, 1 df, p = 0.016. The foreground
  rate is significantly *greater than 1*, i.e., evidence for episodic positive
  selection averaged across the colobine clade.

Both tests reject their nulls at alpha = 0.05.


Step 4: Interpret
------------------

For the colobine lysozyme dataset:

* The foreground omega_fg ~ 2.87 averaged across the clade, suggesting diversifying
  selection on the ancestral colobine branch.
* The boundary test (``TwoRatiosFixed`` vs ``TwoRatios``) is the right LRT for
  positive selection -- a one-sided question at the omega = 1 boundary.
* The branch test is coarser than the branch-site test: it averages omega over all
  sites, so a clade-wide weak signal can show up when only a few sites are
  actually selected. For sharper per-site inference, rerun with the branch-site
  test (tutorial 05) and compare.


Per-branch output
------------------

Branch runs also write a per-branch TSV:

.. code-block:: text

    # results/fits_branch_per_branch.tsv
    model      branch_id  tip_set                      label        paml_node_id  omega   SE
    TwoRatios  0          5.colobus_Cgu&Can            foreground   5             2.871   0.541
    TwoRatios  1          6.langur_Sen&Sve             foreground   6             2.871   0.541
    ...

For ``TwoRatios`` every foreground branch has omega = omega_fg and every background
branch has omega = omega_bg. For ``FreeRatios`` every row is distinct.

A note on the ``SE`` column
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SE`` values reported alongside each per-branch omega are derived from the
L-BFGS-B optimizer's inverse-Hessian approximation at the fit's optimum, with
a delta-method correction to convert from the internal (unconstrained) fit
space back to the natural omega space. This approximation is **fast and cheap**,
but it has well-known caveats: it is unreliable for parameters near a bound
(e.g., omega ~ 0 on strongly conserved branches in FreeRatios) and it is a
limited-memory rank-k estimate rather than the full observed-information
matrix. Treat the SE values as a **guide**, not a rigorous confidence
interval -- if you need rigorous uncertainty quantification (e.g., for a
confidence interval in a manuscript), bootstrap the alignment and refit,
or run ``codeml`` with its own standard-error output option. An empty ``SE``
cell means either the parameter was pinned at a constant (e.g., omega_fg == 1 in
TwoRatiosFixed) or scipy returned an inverse-Hessian in a form selkit could
not extract the diagonal from (a warning is emitted in that case).


Python API equivalent
----------------------

.. code-block:: python

    from pathlib import Path
    from selkit import codeml_branch_models
    from selkit.io.tree import ForegroundSpec

    result = codeml_branch_models(
        alignment=Path("alignment.fa"),
        tree=Path("tree.nwk"),
        output_dir=Path("results/"),
        models=("M0", "TwoRatios", "TwoRatiosFixed"),
        foreground=ForegroundSpec(mrca=(
            "5.colobus_Cgu&Can", "6.langur_Sen&Sve", "7.langur_Tob&Tfr",
            "8.Douc_langur_Pne", "9.probiscis_Nla",
        )),
        n_starts=3, seed=0, threads=2,
    )
    boundary = next(
        l for l in result.lrts
        if l.null == "TwoRatiosFixed" and l.alt == "TwoRatios"
    )
    verdict = "significant" if boundary.significant_at_0_05 else "not significant"
    print(f"two-ratios boundary test: p = {boundary.p_value:.4g} ({verdict})")


What next?
----------

- :doc:`05_branch_site_test` -- when you want *per-site* inference on the
  foreground rather than a clade-wide omega average.
- :doc:`04_library_workflow` -- batch the branch test across many genes.
