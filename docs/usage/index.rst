Usage
=====

selkit has three subcommands and one library entry point. The CLI is a thin wrapper around the library.


Command line
------------


``selkit validate``
~~~~~~~~~~~~~~~~~~~

Runs the pre-flight validation checks without fitting any models. Useful before long runs.

.. code-block:: shell

    selkit validate --alignment gene.fa --tree species.nwk

Output on success:

.. code-block:: text

    OK: 13 taxa, 91 codons, genetic code = standard

Exits 1 with a structured error on failure (taxon mismatch, length not a multiple of 3, mid-sequence stop codon, duplicate taxon, etc.).

Options:

- ``--alignment PATH`` — FASTA or relaxed-Phylip codon alignment. Format auto-detected.
- ``--tree PATH`` — Newick tree.
- ``--genetic-code NAME`` — ``standard`` (default) or ``vertebrate_mitochondrial``.
- ``--foreground TIP1,TIP2,...`` — MRCA clade becomes foreground.
- ``--foreground-tips TIP1,TIP2,...`` — exactly the listed tips become foreground.
- ``--labels-file PATH`` — TSV of ``taxon<TAB>label`` rows (only ``label=1`` rows supported in v0.1).
- ``--strip-stop-codons`` — drop any codon column containing a mid-sequence stop from every taxon.
- ``--no-strip-terminal-stop`` — do not automatically strip a universally-shared terminal stop codon column.


``selkit codeml site-models``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fits the site-model bundle, computes LRTs, and emits BEB posteriors for M2a and M8.

.. code-block:: shell

    selkit codeml site-models \
        --alignment gene.fa \
        --tree species.nwk \
        --output results/gene1 \
        --threads 4

Default bundle: ``M0, M1a, M2a, M7, M8, M8a``. Default LRTs: M1a-vs-M2a, M7-vs-M8, M8a-vs-M8 (mixed 50:50 χ²).

Key options:

- ``--models M0,M1a,M2a`` — fit only these models.
- ``--threads N`` — fit up to ``N`` models in parallel (default 1).
- ``--n-starts K`` — multi-start L-BFGS-B with ``K`` independent seeds per model (default 3).
- ``--seed N`` — top-level random seed; sub-seeds are derived deterministically.
- ``--allow-unconverged`` — exit 0 even if the multi-start convergence gate flags disagreement between the top-2 log-likelihoods.
- ``--foreground`` / ``--foreground-tips`` / ``--labels-file`` — same semantics as ``validate``. Note: v0.1 ships site models only, so foreground labels are accepted but currently unused by the fit. They will become load-bearing when branch and branch-site models land.

Exit codes:

- ``0`` — success, all models converged.
- ``1`` — input validation error.
- ``2`` — fits completed but one or more models failed the convergence gate. Use ``--allow-unconverged`` to downgrade to 0.


``selkit rerun``
~~~~~~~~~~~~~~~~

Re-executes a previous analysis from its ``run.yaml`` manifest, producing byte-identical fits (modulo optimiser float-precision drift).

.. code-block:: shell

    selkit rerun results/gene1/run.yaml --output results/gene1_rerun

The manifest captures every input path, flag, seed, and tolerance, so a rerun on a different machine or a year later reproduces the same analysis.


Library
-------

The library entry point is ``codeml_site_models``:

.. code-block:: python

    from pathlib import Path
    from selkit import codeml_site_models

    result = codeml_site_models(
        alignment=Path("gene.fa"),
        tree=Path("species.nwk"),
        output_dir=Path("out/"),
        models=("M0", "M1a", "M2a", "M7", "M8", "M8a"),
        n_starts=3,
        seed=0,
        threads=4,
    )

It returns a :class:`selkit.RunResult` with:

- ``result.fits`` — ``dict[str, ModelFit]`` keyed by model name. Each :class:`selkit.ModelFit` carries ``lnL``, ``params``, ``branch_lengths``, ``starts`` (per-start results for convergence diagnosis), ``converged``, ``runtime_s``.
- ``result.lrts`` — ``list[LRTResult]`` with null/alt names, ``delta_lnL`` (2·Δ lnL — the test statistic, despite the field name), ``p_value``, ``test_type`` (``"chi2"`` or ``"mixed_chi2"``), ``significant_at_0_05``.
- ``result.beb`` — ``dict[str, list[BEBSite]]`` for M2a and M8, each site carrying 1-indexed position, ``p_positive`` (posterior of ω > 1), and ``posterior_mean_omega``.
- ``result.warnings`` — unconverged models.
- ``result.config`` — the reconstructible :class:`selkit.RunConfig`.

Short worked example: flag sites with posterior > 0.95 in M2a:

.. code-block:: python

    sig_sites = [s for s in result.beb["M2a"] if s.p_positive > 0.95]
    print(f"{len(sig_sites)} sites with P(ω > 1) > 0.95 under M2a")


Inputs
------

**Codon alignment** — FASTA or relaxed-Phylip, auto-detected. Sequence length must be a multiple of 3. Stop codons:

- A universally shared terminal stop codon is automatically stripped (this preserves frame and does not bias the fit).
- Mid-sequence stop codons are an error by default. Pass ``--strip-stop-codons`` to drop any codon column containing a mid-sequence stop across any taxon.
- Gaps / ``N`` / ``?`` are encoded as missing data and marginalised in the pruning recursion.

**Tree** — Newick, optionally with PAML ``#N`` / ``$N`` branch-label syntax (``#1`` marks the foreground clade). Branch lengths in the input tree are used as starting values; they are re-optimised jointly with the model parameters.

**Foreground spec** — three mutually-exclusive ways to designate foreground branches:

- Inline ``#1`` in the Newick.
- ``--foreground TIP1,TIP2,...`` — foreground = MRCA clade of the listed tips.
- ``--foreground-tips TIP1,TIP2,...`` — foreground = exactly the listed tip branches.
- ``--labels-file labels.tsv`` — TSV of ``taxon<TAB>label`` rows.

Passing more than one is an error.


Outputs
-------

An output directory after a site-models run contains:

.. code-block:: text

    results/gene1/
    ├── results.json         # canonical, all fits + LRTs + BEB + warnings
    ├── fits.tsv             # model, lnL, n_params, converged, runtime_s, params
    ├── lrts.tsv             # null, alt, delta_lnL (2·ΔlnL), df, p, sig.
    ├── beb_M2a.tsv          # site, p_positive, posterior_mean_omega
    ├── beb_M8.tsv           # same columns, per site
    └── run.yaml             # full manifest for reproducibility

``results.json`` is the canonical output. The TSVs are convenience views — everything in them is also in the JSON.
