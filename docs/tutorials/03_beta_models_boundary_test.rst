Beta-discretised ω and the M8a boundary test
=============================================

Tutorial 02 used M1a and M2a to ask whether any sites are under positive
selection. M2a's three-class ω discretisation (one purifying, one neutral,
one positive) is convenient but crude. This tutorial introduces the more
flexible **M7 / M8 / M8a** family, which replaces the discrete purifying and
neutral classes with a continuous **beta distribution on (0, 1)**, and then
shows how to confirm that an inferred positive-selection class really has
ω > 1 rather than sitting at the ω = 1 boundary.

**Dataset:** same 13-taxon HIV envelope V3 alignment used in tutorials 01
and 02 (91 codons, 13 sequences).

**Hypotheses answered:**

1. Do the HIV sites' dN/dS values follow a smooth distribution (beta on
   (0, 1)) rather than two or three discrete classes?
2. Given evidence of a ω > 1 class, is that class *really* > 1 — or could
   it just be ω = 1 (neutral) with sampling noise?

.. figure:: /_static/figures/lrt_decision.png
   :align: center
   :width: 100%

   Decision flow for the three canonical site-model LRTs.


Background: M7, M8, and the boundary problem
---------------------------------------------

**M7** is "beta, no positive class". The ω distribution across sites is
modelled as a beta distribution on (0, 1) discretised into 10
equal-probability bins, each assigned the bin's median ω. Two free shape
parameters (p_beta and q_beta) control the distribution. When both are
small, the beta is U-shaped — strong purifying selection at many sites and
near-neutrality at others — without any mass above ω = 1. M7 is the null:
there is no positive selection.

**M8** adds a positive-selection spike on top of M7: a proportion (1 − p₀)
of sites is assigned a free parameter ω₂, constrained to ω₂ ≥ 1. The M7
vs M8 likelihood-ratio test (LRT) has 2 degrees of freedom and uses a
standard χ² critical value. Because the M7 null is more realistic than the
two-class M1a null used in tutorial 02, many researchers prefer this LRT as
their primary test.

**The boundary problem.** When M8 fits better than M7, ω₂ is reported as
the maximum-likelihood estimate — but that estimate could be sitting just
above 1 due to sampling noise. **M8a** is M8 with ω₂ pinned to exactly 1
(the lower boundary of the allowed region). The M8a vs M8 LRT asks: *given
that M8 beats M7, is the inferred positive class truly above 1, or is it
right at the neutral boundary?*

Because ω₂ = 1 is a **boundary** of the parameter space (ω₂ ≥ 1), the
standard χ²₁ distribution overstates significance by a factor of 2. The
correct null is a 50:50 mixture of a point mass at zero (χ²₀) and a
standard χ²₁. selkit detects the M8a vs M8 comparison automatically and
applies the mixed correction, so the reported p-value is already the right
one.


Step 1: Download the dataset
-----------------------------

.. code-block:: shell

    # with curl
    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/hiv_13s/alignment.fa
    curl -O https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/hiv_13s/tree.nwk

    # or with wget
    wget https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/hiv_13s/alignment.fa
    wget https://raw.githubusercontent.com/JLSteenwyk/selkit/main/tests/validation/corpus/hiv_13s/tree.nwk


Step 2: Fit the full model suite
---------------------------------

Run all six models in a single command. The ``--models`` flag accepts a
comma-separated list; selkit fits them in parallel across the requested
thread count.

.. code-block:: shell

    selkit codeml site-models \
        --alignment alignment.fa \
        --tree tree.nwk \
        --output results/ \
        --models M0,M1a,M2a,M7,M8,M8a \
        --n-starts 3 --seed 0 --threads 2

Flags used:

- ``--models M0,M1a,M2a,M7,M8,M8a`` — fit all six site models.
- ``--n-starts 3`` — three independent BFGS starts per model; the best is kept.
- ``--seed 0`` — deterministic starting values for reproducibility.
- ``--threads 2`` — run two models concurrently.

Expected runtime: approximately **12 minutes** on a laptop with 2 threads.
M7, M8, and M8a are each slower than M0/M1a/M2a because every BFGS
gradient step requires evaluating 10–11 rate matrices (one per beta bin plus
the positive class).


Step 3: Inspect ``fits.tsv``
-----------------------------

.. code-block:: text

    model   lnL             n_params  converged  runtime_s  params
    M0      -1137.688190    25        true       7.884      {"kappa": 2.47, "omega": 0.90}
    M1a     -1114.641736    26        true       25.875     {"kappa": 2.59, "omega0": 0.079, "p0": 0.484}
    M2a     -1106.445005    28        true       69.339     {"kappa": 2.79, "omega0": 0.060, "omega2": 3.63, "p0": 0.377, "p1_frac": 0.709}
    M7      -1115.395312    26        true       116.398    {"kappa": 2.56, "p_beta": 0.147, "q_beta": 0.118}
    M8      -1106.388268    28        true       264.471    {"kappa": 2.79, "omega2": 3.47, "p0": 0.80, "p_beta": 0.167, "q_beta": 0.149}
    M8a     -1114.579213    27        true       221.846    {"kappa": 2.59, "p0": 0.503, "p_beta": 1.105, "q_beta": 10.21}

Reading the beta models:

**M7** (lnL = −1115.40)
    p_beta = 0.147 and q_beta = 0.118 — both well below 1. When both shape
    parameters of a beta are less than 1 the distribution is U-shaped, with
    probability mass near 0 *and* near 1. Biologically: a substantial fraction
    of sites are strongly purifying (ω near 0) and another fraction are nearly
    neutral (ω approaching 1), with relatively few sites at intermediate ω.
    There is no mass above 1; M7 cannot model positive selection.

**M8** (lnL = −1106.39)
    Adds a positive-selection class: ω₂ = 3.47 with proportion 1 − p₀ = 1 −
    0.80 = **0.20** of sites. The log-likelihood gain over M7 (−1115.40 →
    −1106.39) is striking and points to a genuine positive-selection signal.

**M8a** (lnL = −1114.58)
    The constrained null for the boundary test. The "positive" class is pinned
    to ω₂ = 1. Notice that lnL (−1114.58) is much worse than M8 (−1106.39)
    and nearly identical to M7 (−1115.40) — the data strongly prefer a class
    above 1 rather than at 1.


Step 4: Interpret ``lrts.tsv``
--------------------------------

.. code-block:: text

    null  alt  delta_lnL    df  p_value         test_type   significant_at_0_05
    M1a   M2a  16.393463    2   0.000275553     chi2        true
    M7    M8   18.014088    2   0.000122544     chi2        true
    M8a   M8   16.381891    1   2.58888e-05     mixed_chi2  true

``delta_lnL`` is the test statistic 2·ΔlnL, not the raw lnL difference.

**M1a vs M2a** (p = 2.8 × 10⁻⁴)
    Site-rate heterogeneity with a positive class fits significantly better
    than the simple two-class nearly-neutral model. This reproduces the
    tutorial 02 conclusion.

**M7 vs M8** (p = 1.2 × 10⁻⁴)
    The same positive-selection conclusion holds under the more flexible beta
    null. Because M7 is a more realistic null than M1a for most genes —
    purifying pressure is rarely concentrated at a single ω value — **this
    LRT is the recommended primary positive-selection test** when you want
    statistical robustness.

**M8a vs M8** (p = 2.6 × 10⁻⁵, mixed χ²)
    The boundary test. The positive-selection class is firmly above ω = 1;
    it is not explained by sampling noise around neutrality. selkit applied the
    50:50 mixture correction automatically; the p-value shown is already the
    conservative, correct one.


Step 5: Why the mixed χ² matters
----------------------------------

When the null hypothesis lies on the *boundary* of the parameter space —
here, ω₂ = 1 is the lower bound of the region ω₂ ≥ 1 — the likelihood-ratio
statistic no longer follows a pure χ²₁ distribution. Under the null, the
maximum-likelihood estimate of ω₂ is constrained to be ≥ 1, so roughly half
the time (when the unconstrained optimum would be below 1) the test statistic
is exactly 0. The correct reference distribution is a 50:50 mixture of a
point mass at 0 and a standard χ²₁, which makes the critical value at α = 0.05
approximately 2.71 (the median of χ²₁) rather than 3.84 (the 95th percentile).

Using an uncorrected χ²₁ would make the p-value roughly **twice** as small as
it should be and inflate the false-positive rate. selkit detects the M8a vs M8
pair by name and switches to the mixture distribution automatically. No extra
flags are needed.


Step 6: Bayes empirical Bayes site posteriors from M8
------------------------------------------------------

M8 produces its own BEB table. Because the M8 null (beta distribution) is
smoother than M2a's two-class background, BEB under M8 can have slightly
more power.

.. code-block:: text

    site  p_positive  posterior_mean_omega  p_class_2a  p_class_2b  beb_grid_size
    28    0.997807    3.464894                                       10
    51    0.950097    3.345570                                       10
    66    0.999223    3.468424                                       10
    87    0.985611    3.434331                                       10

(``p_class_2a`` / ``p_class_2b`` are blank for site models -- they're populated
only for branch-site ModelA. ``beb_grid_size`` is the BEB integration grid;
default 10.)

Four sites pass the conventional 0.95 posterior-probability threshold. Sites
28, 66, and 87 are also recovered by M2a (tutorial 02); site 51 is new — it
just clears 0.95 under M8 but not under M2a. **Best practice:** report
candidate sites that clear 0.95 under *both* M2a and M8 as robust hits. Sites
that appear in only one analysis are worth flagging but interpreting cautiously.


Step 7: What if M7 vs M8 is significant but M8a vs M8 is not?
---------------------------------------------------------------

Suppose the M7 vs M8 LRT is significant (p < 0.05) but the M8a vs M8
boundary test is not. What does that mean?

M8 fits significantly better than M7, meaning there is a class of sites with
ω different from the beta background — but M8a (ω₂ = 1) cannot be ruled out.
In other words, the data support a separate class of sites, but that class
could simply be neutral (ω = 1) rather than positively selected. **Do not
report positive selection in this case.** The M8a vs M8 test exists precisely
to catch this scenario and avoid the false positive that a naive M7 vs M8
result would otherwise produce.

In the HIV V3 dataset both tests are strongly significant, so the conclusion
of positive selection is secure on two independent criteria.


Decision summary
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 40

   * - M1a vs M2a
     - M7 vs M8
     - M8a vs M8
     - Conclusion
   * - not significant
     - not significant
     - —
     - No positive selection detected.
   * - significant
     - not significant
     - —
     - Positive selection suggested by M2a only; M7/M8 result inconclusive.
   * - significant
     - significant
     - not significant
     - A separate ω class exists but may be ω = 1; do **not** claim positive selection.
   * - significant
     - significant
     - significant
     - Positive selection confirmed by all three tests (this HIV example).


What next?
----------

Running M7/M8/M8a by hand for a single gene is straightforward. Scaling the
same workflow to hundreds of genes in a loop — with automated LRT decisions,
BEB summary tables, and run provenance — is covered in:

:doc:`04_library_workflow`
