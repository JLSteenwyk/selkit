API reference
=============

selkit exposes a small, stable public surface at the package root. All names below are importable as ``from selkit import ...``.


Entry point
-----------

.. py:function:: codeml_site_models(*, alignment, tree, output_dir, models=("M0", "M1a", "M2a", "M7", "M8", "M8a"), genetic_code="standard", foreground=None, n_starts=3, seed=0, threads=1, convergence_tol=0.5)

    Fit a bundle of codeml site models against a codon alignment and tree.

    Creates ``output_dir`` if it doesn't exist and returns a :class:`RunResult` carrying every fit, every LRT, BEB posteriors for M2a / M8 (where those models were fit), the ``RunConfig`` manifest, and any convergence warnings.

    :param alignment: Path to a FASTA or relaxed-Phylip codon alignment.
    :param tree: Path to a Newick tree. Optional PAML ``#N`` / ``$N`` branch labels.
    :param output_dir: Directory where ``run.yaml`` and intermediate artefacts will be written.
    :param models: Tuple of model names to fit. Available: ``M0``, ``M1a``, ``M2a``, ``M7``, ``M8``, ``M8a``.
    :param genetic_code: ``"standard"`` or ``"vertebrate_mitochondrial"``.
    :param foreground: Optional :class:`ForegroundSpec`. Accepted in v0.1 but not yet consumed by site models; reserved for branch / branch-site models.
    :param n_starts: Number of random starts per model for multi-start L-BFGS-B.
    :param seed: Top-level random seed. Sub-seeds for multi-start are derived deterministically.
    :param threads: If > 1, fits multiple models concurrently via ``ProcessPoolExecutor``.
    :param convergence_tol: lnL tolerance used to flag multi-start disagreement.

    :returns: :class:`RunResult`
    :raises selkit.errors.SelkitInputError: on alignment, tree, or configuration errors.


Dataclasses
-----------


.. py:class:: RunResult

    Top-level result returned by :func:`codeml_site_models`. Frozen.

    :var config: the :class:`RunConfig` used for this run (reconstructible via ``selkit rerun``).
    :var fits: ``dict[str, ModelFit]`` keyed by model name.
    :var lrts: ``list[LRTResult]``.
    :var beb: ``dict[str, list[BEBSite]]`` for M2a and M8 (if fit).
    :var warnings: ``list[str]`` of non-fatal convergence issues.


.. py:class:: ModelFit

    Per-model fit result. Frozen.

    :var model: Model name (``"M0"``, ``"M1a"``, …).
    :var lnL: Maximum log-likelihood (natural log).
    :var n_params: Total free parameters (model params + branch lengths).
    :var params: ``dict[str, float]`` — model-specific free parameters (``omega``, ``kappa``, ``omega0``, ``omega2``, ``p0``, ``p_beta``, ``q_beta``, ``p1_frac``).
    :var branch_lengths: ``dict[str, float]`` — per-branch lengths keyed by internal node id (``bl_1``, ``bl_2``, ...).
    :var starts: ``list[StartResult]`` — per-start result, useful for convergence diagnosis.
    :var converged: ``True`` if the top-2 starts' lnLs agree within ``convergence_tol``.
    :var runtime_s: Wall-clock time to fit this model.


.. py:class:: LRTResult

    Likelihood-ratio test result. Frozen.

    :var null: Null-model name (e.g. ``"M1a"``).
    :var alt: Alternative-model name (e.g. ``"M2a"``).
    :var delta_lnL: The LRT statistic 2·(lnL_alt - lnL_null). Despite the field name, this is the χ² statistic, not the raw lnL difference.
    :var df: Degrees of freedom.
    :var p_value: p-value under the test distribution.
    :var test_type: ``"chi2"`` (standard) or ``"mixed_chi2"`` (50:50 mixture of χ²₀ and χ²₁ for boundary tests like M8a-vs-M8).
    :var significant_at_0_05: ``p_value < 0.05``.


.. py:class:: BEBSite

    Per-site NEB posterior. Frozen.

    :var site: 1-indexed codon position.
    :var p_positive: P(ω > 1 | site data, MLE hyperparameters).
    :var mean_omega: Posterior-mean ω at this site.


.. py:class:: CodonAlignment

    Validated codon alignment. Frozen.

    :var taxa: ``tuple[str, ...]`` of taxon names in row order.
    :var codons: ``np.ndarray`` shape ``(n_taxa, n_codons)``, dtype ``int16``. Integer codon codes; gaps / ambiguous → ``-1``.
    :var genetic_code: Name of the genetic code used to encode (``"standard"``, ``"vertebrate_mitochondrial"``).
    :var stripped_sites: ``tuple[int, ...]`` — 0-indexed codon positions that were stripped (terminal stop, ``--strip-stop-codons`` drops).


.. py:class:: LabeledTree

    Parsed Newick tree with optional branch labels.

    :var root: Root :class:`selkit.io.tree.Node`.
    :var newick: Canonical Newick string (branch lengths preserved; labels not currently round-tripped).
    :var labels: ``dict[int, int]`` — ``{node_id: label}``; only entries with non-zero labels.
    :var tip_order: ``tuple[str, ...]`` — tip names in pre-order traversal.


.. py:class:: RunConfig

    Exact configuration of an analysis, serialisable to ``run.yaml``. Used by ``selkit rerun``.


.. py:class:: ForegroundConfig
.. py:class:: ForegroundSpec

    Normalised foreground-branch specification. ``ForegroundConfig`` is the serialisable form stored in ``RunConfig``; ``ForegroundSpec`` is the engine-side shape (tips, MRCA, explicit labels dict).


Errors
------


.. py:class:: selkit.errors.SelkitError

    Base class for all selkit exceptions.


.. py:class:: selkit.errors.SelkitInputError

    Malformed input files (alignment, tree, labels).


.. py:class:: selkit.errors.SelkitConfigError

    Invalid configuration (unknown model name, conflicting flags).


.. py:class:: selkit.errors.SelkitEngineError

    Numerical failure inside the ML engine.


.. py:class:: selkit.errors.SelkitConvergenceWarning

    Warning (not an exception) emitted when the multi-start convergence gate fails. Flagged on :attr:`RunResult.warnings`.
