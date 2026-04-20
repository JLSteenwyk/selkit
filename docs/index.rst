selkit
======

**selkit** is a fast, pure-Python reimplementation of `PAML <http://abacus.gene.ucl.ac.uk/software/paml.html>`_'s selection-analysis workflows (``codeml``, ``yn00``).

v0.1 ships ``codeml`` site models — M0, M1a, M2a, M7, M8, and M8a — with automated likelihood-ratio tests, per-site posterior probabilities of positive selection, multi-start optimisation, model-level parallelism, strict input validation, and reproducible ``run.yaml`` artefacts.

**Numerical parity with PAML has been verified:** on two corpus cases (HIVenvSweden 4-taxon and 13-taxon subsets from the PAML distribution), selkit matches PAML 4.10.10 at PAML's reported (branch-length, parameter) point to absolute Δ lnL < 10⁻³ for every site model.


Quick start
-----------

.. code-block:: shell

    # install
    pip install selkit

    # run (fits M0, M1a, M2a, M7, M8, M8a; computes LRTs; emits BEB posteriors)
    selkit codeml site-models \
        --alignment gene.fa \
        --tree species.nwk \
        --output results/gene1 \
        --threads 4

Outputs land in ``results/gene1/``:

- ``results.json`` — canonical structured output (all fits, LRTs, BEB sites, warnings).
- ``fits.tsv``, ``lrts.tsv``, ``beb_M2a.tsv``, ``beb_M8.tsv`` — tabular views.
- ``run.yaml`` — exact config; reproduce with ``selkit rerun results/gene1/run.yaml``.


Installation
------------

**From PyPI (recommended):**

.. code-block:: shell

    pip install selkit

Requires Python 3.11+.

**From source:**

.. code-block:: shell

    git clone https://github.com/JLSteenwyk/selkit.git
    cd selkit
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    make install


Library usage
-------------

.. code-block:: python

    from selkit import codeml_site_models

    result = codeml_site_models(
        alignment="gene.fa",
        tree="species.nwk",
        output_dir="out/",
        models=("M0", "M1a", "M2a"),
    )

    print(result.fits["M2a"].lnL)
    for site in result.beb["M2a"]:
        if site.p_positive > 0.95:
            print(f"site {site.site}: P(omega > 1) = {site.p_positive:.3f}")


What's in v1
------------

Implemented and PAML-validated:

- **Site models:** M0 (one-ratio), M1a (nearly-neutral), M2a (positive-selection), M7 (beta), M8 (beta + positive), M8a (beta + ω=1 null).
- **LRTs:** automatic M1a-vs-M2a, M7-vs-M8 (standard χ²), M8a-vs-M8 (mixed 50:50 χ²).
- **Per-site posteriors:** NEB posterior probability of positive selection for M2a and M8.
- **Multi-start optimisation:** L-BFGS-B with soft-plus / logit transforms and a convergence-tolerance gate.
- **Parallelism:** model-level via ``ProcessPoolExecutor``.
- **Inputs:** FASTA + relaxed-Phylip readers with format auto-detect and strict codon/stop-codon validation; Newick parser with PAML ``#N`` / ``$N`` branch-label syntax for foreground designation.
- **Outputs:** canonical JSON, per-table TSV, and ``run.yaml`` manifest for full reproducibility.
- **CLI:** ``selkit validate``, ``selkit codeml site-models``, ``selkit rerun``.

Deferred to later releases:

- ``yn00`` (pairwise dN/dS).
- Branch models (one-ratio, two-ratios, free-ratios).
- Branch-site models (Model A, Model A null).
- True BEB (integration over hyperparameters; v0.1 uses NEB — posterior at the MLE of hyperparameters).


Citation
--------

If selkit is useful in your work, please cite the software and `the original PAML paper
<https://academic.oup.com/mbe/article/24/8/1586/1103731>`_:

Yang Z. (2007). PAML 4: Phylogenetic analysis by maximum likelihood. *Molecular Biology and Evolution*, 24(8):1586-91. doi: ``10.1093/molbev/msm088``.



.. toctree::
    :maxdepth: 2

    about/index
    tutorials/index
    usage/index
    validation/index
    api/index
    change_log/index
    faq/index
    other_software/index

