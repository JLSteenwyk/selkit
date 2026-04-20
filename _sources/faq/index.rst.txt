Frequently asked questions
==========================


General
-------

**Does selkit give the same answers as PAML?**

Yes, to high precision on tested inputs. On the two PAML reference cases bundled in the repo (``tests/validation/corpus/hiv_4s`` and ``hiv_13s``), selkit's log-likelihood at PAML's reported (branch-length, parameter) point matches PAML 4.10.10 to under 10⁻³ lnL units for every site model. See :doc:`../validation/index`.

Small differences are expected on finite-sample runs for a different reason: both selkit and PAML are nonlinear optimisers, and on flat likelihood ridges (especially for large-ω classes in M2a / M8) the exact stopping point is optimiser-implementation-dependent. That's why the validation test compares ω values with a relative tolerance; the lnL agreement is the primary correctness signal.

**Why Python and not just use PAML directly?**

PAML is the reference. selkit is for the common case where you want to script or embed the analysis: looping over thousands of genes, wiring the output into downstream analysis, testing input-preparation changes without re-running hours of fits, or shipping a paper's analysis as a reproducible ``run.yaml``. PAML's control-file interface and free-form ``mlc`` output are workable but laborious in that setting.

**Does selkit replace PAML?**

No. PAML ships many more models (baseml, branch models with multiple ω ratios, clade models, ``yn00``, ``mcmctree``, ``evolver``, ``codeml``-based dating, ...). selkit currently implements ``codeml`` site models (M0, M1a, M2a, M7, M8, M8a) and the branch-site test (Model A, Model A null). For anything outside that scope, use PAML.


Inputs
------

**What alignment formats are accepted?**

FASTA and relaxed-Phylip. The reader auto-detects the format. Sequences must be codon-aligned (length is a multiple of 3) and contain only A, C, G, T, ``-``, ``N``, ``?``, or ``\n``. IUPAC ambiguity codes beyond ``N`` are silently treated as missing data.

**A universally-shared terminal stop codon column is stripped — why?**

PAML's and selkit's codon models are defined over sense codons only. An alignment-wide terminal stop is an artefact of how the alignment was extracted (many pipelines retain the stop codon); leaving it in would require the model to "explain" a state that isn't in its state space. Stripping it is numerically harmless and matches PAML's default behaviour.

**What about mid-sequence stop codons?**

Errors by default, because a mid-sequence stop is usually a sign the alignment is wrong (wrong reading frame, frameshift, pseudogene). Pass ``--strip-stop-codons`` to drop every codon column that contains a mid-sequence stop in any taxon.

**My taxon names don't match between alignment and tree.**

The validator will flag this explicitly. In v0.1 you have to reconcile them manually before running — a ``--prune-unmatched`` flag to drop mismatched taxa / tips is a planned follow-up.

**How do I label a foreground clade?**

Three mutually-exclusive options (pick one):

1. Inline in Newick: ``((a:0.1, b:0.1) #1, (c:0.1, d:0.1));`` — the clade labelled ``#1`` is foreground. (PAML also accepts ``$1`` on tip branches.)
2. ``--foreground TIP1,TIP2,...`` on the CLI — foreground = MRCA clade of the listed tips.
3. ``--foreground-tips TIP1,TIP2,...`` — foreground = exactly the listed tip branches, nothing else.
4. ``--labels-file labels.tsv`` — two-column TSV (``taxon<TAB>label``).

Foreground labels are ignored by the ordinary site models (M0 / M1a / M2a / M7 / M8 / M8a) — those models treat every branch the same. The **branch-site test** (``ModelA`` / ``ModelA_null``, :doc:`../tutorials/05_branch_site_test`) is the one that actually uses the foreground.


Models and statistics
---------------------

**Which LRTs does selkit compute?**

By default, all three standard site-model LRTs:

- **M1a vs M2a** — 2 degrees of freedom, standard χ². Tests for the presence of a positive-selection class.
- **M7 vs M8** — 2 degrees of freedom, standard χ². Tests whether a beta-distributed ω is enough or a positive-selection class improves fit.
- **M8a vs M8** — 1 degree of freedom, **mixed** 50:50 χ²₀:χ²₁. Boundary test; halves the nominal p-value because the null model sits on the parameter-space boundary.

**What is BEB? What is NEB? Which does selkit give me?**

Bayes Empirical Bayes (BEB) integrates the per-site posterior probabilities over the hyperparameters (p0, p_beta, q_beta, etc.) using a prior; Naïve Empirical Bayes (NEB) evaluates the posterior at the MLE of those hyperparameters. PAML's ``BEB`` option runs BEB; its ``NEB`` output is the cheaper naïve version.

selkit v0.1 currently ships **NEB**. In practice the two agree closely on data sets with strong signal; they can differ materially when the MLE sits near a boundary. True BEB (with numerical integration) is a scoped follow-up.

**Why does omega2 come out slightly different from PAML?**

On the HIV test cases we've seen selkit's M2a ω₂ differ from PAML's by ~0.03 % in relative terms (e.g. 6.8563 vs 6.85808). Large-ω classes sit on a flat likelihood ridge: the fit's predictive content (partition of sites into classes, LRT significance, BEB posteriors) is effectively identical even when the ω₂ point estimate differs at the fourth decimal. selkit's validation tolerance accommodates this.


Performance
-----------

**Is selkit fast enough for production?**

For a 4-taxon × 91-codon alignment, the full M0 / M1a / M2a fit plus LRTs plus BEB runs in about 25 seconds on a laptop with ``n_starts=3``. M7 / M8 with 10 beta-discretised ω classes are roughly 4-5× slower per model. For large datasets (>100 taxa or >>1000 codons) the optimiser's inner Q-matrix builds and matrix exponentials dominate runtime; parallelising at the gene level (one selkit invocation per alignment) is the recommended pattern.

**What's the memory footprint?**

Modest. Per-site Felsenstein partials are stored as ``(n_sites, n_sense)`` float64 — ~488 bytes per site per internal node. For a 1000-codon 50-taxon alignment, peak memory is under 50 MB.

**Can I run it on a cluster?**

Yes — each ``selkit`` invocation is independent and writes all state (``run.yaml`` plus outputs) to its ``--output`` directory. Submit one invocation per gene to a job scheduler. Within a single invocation, ``--threads`` parallelises across models.


Output
------

**Why does my LRT show ``delta_lnL = 10.5`` when the lnL diff is 5.25?**

Historical quirk in the field name. ``LRTResult.delta_lnL`` stores the **test statistic** 2·(lnL_alt - lnL_null), which is what gets fed to χ². A future release will rename this to ``lrt_stat``. The ``p_value`` and ``significant_at_0_05`` fields are correct in any case.

**Can I get a labelled Newick back?**

Not yet. The current ``LabeledTree.newick`` is the un-labelled canonical form of the input. ``_canonicalize`` in ``selkit.io.tree`` doesn't emit ``#N`` labels — a scoped follow-up. Labels are available on the ``LabeledTree.labels`` dict and per-node.


Development
-----------

**How do I run the tests?**

.. code-block:: shell

    make develop              # pip install -r requirements.txt + pip install -e .
    make test.fast            # unit + integration (~40 s)
    make test.validation      # opt-in PAML corpus full-fit (~15 min)

**How do I cut a release?**

See :doc:`../change_log/index` for the latest changes. The release process is documented in ``RELEASE.md`` in the repo root: bump ``selkit/version.py``, add a ``## X.Y.Z`` entry to the top of ``CHANGELOG.md``, run ``make release-check`` to confirm they agree, commit, tag ``vX.Y.Z``, then ``make release`` builds and uploads to PyPI.
