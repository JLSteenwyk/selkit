Tutorials
=========

Worked examples for common selkit workflows. Each tutorial is self-contained: download the toy dataset from the link inside, paste the commands, reproduce the numbers.

All four tutorials use the same HIV envelope V3 alignment from the PAML distribution (``HIVNSsites`` example, Yang *et al.* 2000) — well-studied, small enough to run in minutes, and shows real positive selection.


.. figure:: /_static/figures/workflow.png
   :align: center
   :width: 90%

   selkit pipeline: inputs → validate → fit → LRTs + BEB → structured outputs.


Available tutorials
-------------------

.. toctree::
   :maxdepth: 1

   01_first_run
   02_detecting_positive_selection
   03_beta_models_boundary_test
   04_library_workflow
   05_branch_site_test


What you'll learn
-----------------

- :doc:`01_first_run` — Install, download toy data, validate inputs, and fit the M0 one-ratio model. Five minutes end-to-end.
- :doc:`02_detecting_positive_selection` — The most common selkit workflow: the M1a vs M2a LRT plus per-site BEB posteriors. Answers "are any codons under positive selection?".
- :doc:`03_beta_models_boundary_test` — The more statistically robust M7 / M8 / M8a workflow, including the mixed-χ² boundary test that prevents false positives when the ω class is really at ω = 1.
- :doc:`04_library_workflow` — Embed selkit in a script to batch-process many genes, run with non-standard genetic codes (mitochondrial), and work with selkit's dataclasses directly.
- :doc:`05_branch_site_test` — The branch-site test of positive selection: Model A / Model A null. Tests whether a pre-designated lineage (the "foreground") experienced episodic adaptation, using the PAML lysozyme dataset (Yang 1998; Zhang et al. 2005).


Dataset provenance
------------------

The HIV env V3 alignment (13 taxa × 91 codons) is bundled with PAML 4.10.10 under ``examples/HIVNSsites/HIVenvSweden.txt``. selkit's test suite pins PAML's reported fit at this alignment as the numerical-correctness reference — see :doc:`../validation/index` for details.
