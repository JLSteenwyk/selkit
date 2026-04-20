Batch analysis & library use (many genes, mitochondrial code)
=============================================================

This tutorial answers two questions that come up once you have worked through the
single-gene CLI examples in tutorials 01–03:

1. *I have hundreds or thousands of genes. How do I run selkit across all of them
   and collect the results in a structured way?*
2. *My sequences are mitochondrial (or from another non-standard genetic code).
   How do I tell selkit?*

**Dataset:** Same 13-taxon HIV envelope V3 alignment used throughout the tutorial
series.  For the mitochondrial section the same file is reused as a pedagogical
stand-in — the point is the flag, not the biology.

**Audience:** Someone comfortable writing Python who wants to move beyond the CLI
for larger-scale or more customised analyses.


Why use the library instead of the CLI?
---------------------------------------

``selkit codeml site-models`` is a thin wrapper over the Python function
``codeml_site_models``.  For a handful of genes the CLI is perfectly fine.  For
hundreds or thousands of genes, calling the library directly is cheaper and more
flexible:

- You control exactly which models are fitted per gene.
- You collect results directly into Python objects (no round-trip through TSV/JSON).
- You can customise starting values, convergence tolerances, and error handling.
- You can feed results straight into pandas, write a custom summary, or branch on
  the output mid-script without shelling out.


Library entry point
-------------------

The entire public surface is importable from the top-level ``selkit`` package:

.. code-block:: python

    from selkit import codeml_site_models, RunResult, ModelFit, LRTResult, BEBSite

Brief description of each name:

- ``codeml_site_models`` — the main entry point; fits one or more site models for a
  single alignment/tree pair and returns a ``RunResult``.
- ``RunResult`` — dataclass holding ``fits``, ``lrts``, and ``beb`` for a completed
  run.
- ``ModelFit`` — dataclass for a single fitted model: log-likelihood, parameters,
  branch lengths, convergence flag.
- ``LRTResult`` — dataclass for one likelihood-ratio test: null/alt model names,
  test statistic, p-value, significance flag.
- ``BEBSite`` — dataclass for a single codon's Bayes Empirical Bayes result: site
  index (1-based), posterior probability of positive selection, posterior-mean ω.

Full signature of ``codeml_site_models``:

.. code-block:: python

    codeml_site_models(
        *,
        alignment: Path | str,
        tree: Path | str,
        output_dir: Path | str,
        models: Iterable[str] = ("M0", "M1a", "M2a", "M7", "M8", "M8a"),
        genetic_code: str = "standard",
        foreground: ForegroundSpec | None = None,
        n_starts: int = 3,
        seed: int = 0,
        threads: int = 1,
        convergence_tol: float = 0.5,
    ) -> RunResult

All arguments are keyword-only.  ``output_dir`` is created automatically if it does
not exist.  ``threads`` parallelises model fitting *within a single gene* (see
`Gene-level parallelism`_ below for the recommended batch pattern).


Single-gene library call
------------------------

Download the tutorial dataset if you have not already done so:

.. code-block:: shell

    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/hiv_13s/alignment.fa
    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/hiv_13s/tree.nwk

Then run selkit from Python:

.. code-block:: python

    from pathlib import Path
    from selkit import codeml_site_models

    result = codeml_site_models(
        alignment=Path("alignment.fa"),
        tree=Path("tree.nwk"),
        output_dir=Path("out/hiv_13s"),
        models=("M1a", "M2a", "M7", "M8", "M8a"),
        n_starts=3,
        seed=42,
        threads=2,
    )

    # Summary of fits
    for name, fit in result.fits.items():
        print(f"{name}: lnL = {fit.lnL:.2f}  converged = {fit.converged}")

    # LRT results
    for lrt in result.lrts:
        verdict = "significant" if lrt.significant_at_0_05 else "ns"
        print(f"{lrt.null:>3} vs {lrt.alt:<3}: p = {lrt.p_value:.3g}  ({verdict})")

    # Significant sites under M8
    sig_m8 = [s for s in result.beb.get("M8", []) if s.p_positive >= 0.95]
    print(f"M8 BEB sites with P >= 0.95: {[s.site for s in sig_m8]}")

Expected output:

.. code-block:: text

    M1a: lnL = -1114.64  converged = True
    M2a: lnL = -1106.45  converged = True
    M7:  lnL = -1115.40  converged = True
    M8:  lnL = -1106.39  converged = True
    M8a: lnL = -1114.58  converged = True
    M1a vs M2a: p = 0.000276  (significant)
    M7  vs M8 : p = 0.000123  (significant)
    M8a vs M8 : p = 2.59e-05  (significant)
    M8 BEB sites with P >= 0.95: [28, 51, 66, 87]

The same output files (``results.json``, ``fits.tsv``, ``lrts.tsv``, ``beb_*.tsv``,
``run.yaml``) are written to ``out/hiv_13s/`` regardless of whether you use the CLI
or the library — the two interfaces are identical in every respect except how you
invoke them.


Batch across many genes
-----------------------

Assume your genes are laid out as::

    genes/
    ├── gene_A/
    │   ├── alignment.fa
    │   └── tree.nwk
    ├── gene_B/
    │   ├── alignment.fa
    │   └── tree.nwk
    └── ...

A sequential loop with basic error handling:

.. code-block:: python

    import json
    from pathlib import Path
    from selkit import codeml_site_models
    from selkit.errors import SelkitInputError

    GENE_DIR = Path("genes")
    OUT_DIR = Path("batch_out")
    OUT_DIR.mkdir(exist_ok=True)

    summary_rows = []
    for gene in sorted(GENE_DIR.iterdir()):
        if not gene.is_dir():
            continue
        try:
            result = codeml_site_models(
                alignment=gene / "alignment.fa",
                tree=gene / "tree.nwk",
                output_dir=OUT_DIR / gene.name,
                models=("M1a", "M2a", "M7", "M8", "M8a"),
                n_starts=3, seed=0, threads=2,
            )
        except SelkitInputError as e:
            print(f"{gene.name}: skipped — {e}")
            continue

        m7_vs_m8 = next(l for l in result.lrts if l.null == "M7" and l.alt == "M8")
        sig_sites = [s.site for s in result.beb.get("M8", []) if s.p_positive >= 0.95]

        summary_rows.append({
            "gene": gene.name,
            "lnL_M0":  None,  # not fit in this run
            "lnL_M8":  result.fits["M8"].lnL,
            "m7_vs_m8_p": m7_vs_m8.p_value,
            "m7_vs_m8_sig": m7_vs_m8.significant_at_0_05,
            "sig_sites": sig_sites,
        })

    # Dump structured summary
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))

``SelkitInputError`` is raised for problems that can be detected before any
optimisation runs (malformed alignment, missing taxa, etc.).  Other exceptions
propagate normally — catch ``Exception`` if you need a broader safety net for
production pipelines.

.. note::

    ``threads`` parallelises model fitting **within a single gene** — at most six
    models run concurrently, so the ceiling is six threads.  For a batch of hundreds
    of genes, parallelising at the gene level (one selkit call per gene, ``threads=1``
    each) almost always gives better total throughput.  See the next section.


Gene-level parallelism
----------------------

For medium-sized batches (tens to low hundreds of genes) ``concurrent.futures``
is the simplest option:

.. code-block:: python

    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    GENE_DIR = Path("genes")
    OUT_DIR = Path("batch_out")

    def _fit_one(gene_dir: Path) -> dict:
        from selkit import codeml_site_models  # import inside worker process
        result = codeml_site_models(
            alignment=gene_dir / "alignment.fa",
            tree=gene_dir / "tree.nwk",
            output_dir=OUT_DIR / gene_dir.name,
            models=("M1a", "M2a", "M7", "M8", "M8a"),
            n_starts=3, seed=0, threads=1,
        )
        m7_vs_m8 = next(l for l in result.lrts if l.null == "M7" and l.alt == "M8")
        return {"gene": gene_dir.name, "p": m7_vs_m8.p_value}

    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_fit_one, [d for d in GENE_DIR.iterdir() if d.is_dir()]))

The import of ``codeml_site_models`` is placed inside the worker function to avoid
pickling issues on some platforms.  Each worker runs one gene at ``threads=1``; the
eight workers together keep eight cores busy without contention.

For large batches (thousands of genes) or HPC environments, use a job scheduler
(Slurm, Nextflow, Snakemake) rather than ``ProcessPoolExecutor``.  Schedulers
provide checkpointing, job accounting, and automatic retries — none of which
``ProcessPoolExecutor`` offers out of the box.


Non-standard genetic codes
--------------------------

selkit currently ships two genetic codes:

- ``standard`` — NCBI translation table 1 (the default).
- ``vertebrate_mitochondrial`` — NCBI translation table 2.

The codes differ in their stop-codon sets.  Under the vertebrate mitochondrial code,
AGA and AGG are stop codons (they encode Arg in the standard code) and TGA encodes
Trp (it is a stop in the standard code).

To analyse mitochondrial-encoded genes, pass ``--genetic-code`` on the CLI:

.. code-block:: shell

    selkit codeml site-models \
        --alignment mito_gene.fa \
        --tree species.nwk \
        --output out/ \
        --genetic-code vertebrate_mitochondrial

Or set ``genetic_code`` in the library call:

.. code-block:: python

    result = codeml_site_models(
        alignment="mito_gene.fa",
        tree="species.nwk",
        output_dir="out/",
        genetic_code="vertebrate_mitochondrial",
    )

Your alignment must already be translated correctly for the chosen code.
``selkit validate`` (or the implicit validation inside ``codeml_site_models``) will
raise ``SelkitInputError`` if any codon that is a stop under the selected code
appears in a non-terminal position.

If you need a different NCBI translation table — invertebrate mitochondrial,
echinoderm mitochondrial, etc. — file an issue on the selkit GitHub repository.
Additional codes are tracked in Appendix A of the documentation.


Reproducing a run later
-----------------------

Every call to ``codeml_site_models`` writes a ``run.yaml`` file inside
``output_dir``.  That file captures every input path, flag, seed, and tolerance
used for the run.  To reproduce it exactly:

.. code-block:: shell

    selkit rerun out/hiv_13s/run.yaml --output out/hiv_13s_rerun

A rerun on a different machine, or a year later, produces byte-identical log-
likelihoods (modulo optimiser float-precision drift beyond the fourth decimal place).
For archival purposes, committing ``run.yaml`` alongside your results is sufficient
to document the full analysis.


Accessing the raw dataclasses
------------------------------

If you want to skip JSON I/O entirely, work with the ``RunResult`` object directly:

.. code-block:: python

    from selkit import RunResult, ModelFit

    # `result` is the object returned by codeml_site_models()
    m2a_fit: ModelFit = result.fits["M2a"]
    print(type(m2a_fit))           # selkit.io.results.ModelFit
    print(m2a_fit.branch_lengths)  # dict keyed by bl_<node_id>

All fields are plain Python types (floats, dicts, lists) — no custom serialisation
required.  See :doc:`../api/index` for the complete schema of every dataclass.


What next?
----------

- :doc:`01_first_run` — M0 baseline and output file tour.
- :doc:`02_detecting_positive_selection` — M1a vs M2a LRT and BEB analysis.
- :doc:`03_beta_models_boundary_test` — M7/M8/M8a and the boundary test.
