Other software
==============

selkit is part of a family of bioinformatics tools developed and maintained by `Jacob L. Steenwyk <https://jlsteenwyk.github.io/>`_ and contributors. Each addresses a different stage of the sequence-analysis pipeline.


`ClipKIT <https://github.com/JLSteenwyk/ClipKIT>`_
--------------------------------------------------

Fast, flexible multiple-sequence alignment trimming. ClipKIT keeps phylogenetically informative sites and removes noisy ones, improving downstream tree inference. Drop-in replacement for Gblocks / trimAl in many pipelines.

Install: ``pip install clipkit``


`PhyKIT <https://github.com/JLSteenwyk/PhyKIT>`_
------------------------------------------------

A toolkit for processing and summarising phylogenomic datasets: alignment statistics, tree metrics (RF, quartet scores, bipartition support, etc.), orthologue selection, and assembly-quality checks.

Install: ``pip install phykit``


`BioKIT <https://github.com/JLSteenwyk/BioKIT>`_
------------------------------------------------

General-purpose utilities for processing sequence data: codon composition, GC content, gene statistics, alignment summaries, and other one-liners that usually require five shell scripts.

Install: ``pip install jlsteenwyk-biokit``


`orthofisher <https://github.com/JLSteenwyk/orthofisher>`_
----------------------------------------------------------

High-throughput ortholog identification across hundreds of genomes using a single HMMER profile.


`orthosnap <https://github.com/JLSteenwyk/orthosnap>`_
------------------------------------------------------

Algorithm for inferring single-copy orthologous groups from gene-family phylogenies — useful as an alignment-input step upstream of selkit.


Typical pipeline
----------------

For a codon-based selection analysis, a common workflow is:

.. code-block:: text

    raw CDS alignments
        │
        ▼  ClipKIT (trim noisy columns)
    cleaned codon alignments
        │
        ▼  PhyKIT + IQ-TREE / RAxML (infer gene trees)
    gene trees
        │
        ▼  selkit codeml site-models
    per-gene site-model fits, LRTs, BEB posteriors
        │
        ▼  Downstream: positive-selection screens,
           functional-enrichment analyses, etc.
