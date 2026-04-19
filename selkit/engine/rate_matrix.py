from __future__ import annotations

import numpy as np

from selkit.engine.genetic_code import NUCS, GeneticCode, PURINES, PYRIMIDINES


def build_q(
    gc: GeneticCode, *, omega: float, kappa: float, pi: np.ndarray
) -> np.ndarray:
    n = gc.n_sense
    if pi.shape != (n,):
        raise ValueError(f"pi has wrong shape: {pi.shape}, expected ({n},)")
    codons = gc.sense_codons
    Q = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ci = codons[i]
        for j in range(n):
            if i == j:
                continue
            cj = codons[j]
            diffs = [(k, ci[k], cj[k]) for k in range(3) if ci[k] != cj[k]]
            if len(diffs) != 1:
                continue
            _, x, y = diffs[0]
            syn = gc.is_synonymous(ci, cj)
            trans = (x in PURINES and y in PURINES) or (x in PYRIMIDINES and y in PYRIMIDINES)
            rate = pi[j]
            if trans:
                rate *= kappa
            if not syn:
                rate *= omega
            Q[i, j] = rate
    Q[np.diag_indices_from(Q)] = -Q.sum(axis=1)
    mean_rate = float(-(pi @ np.diag(Q)))
    if mean_rate <= 0:
        raise ValueError("non-positive mean substitution rate; check pi/params")
    Q /= mean_rate
    return Q


def prob_transition_matrix(Q: np.ndarray, t: float) -> np.ndarray:
    if t == 0.0:
        return np.eye(Q.shape[0])
    w, V = np.linalg.eig(Q)
    Vinv = np.linalg.inv(V)
    P = (V * np.exp(w * t)) @ Vinv
    return np.real(P)


def estimate_f3x4(
    codon_indices: np.ndarray, gc: GeneticCode, *, pseudocount: float = 0.0
) -> np.ndarray:
    """F3X4 codon equilibrium frequencies.

    Uses raw counts by default (matches PAML codeml). Pass `pseudocount > 0`
    for Laplace smoothing when the input doesn't observe every nucleotide at
    every codon position — useful only for degenerate test alignments; real
    runs should use raw counts.
    """
    n = gc.n_sense
    counts = np.full((3, 4), float(pseudocount))
    nuc_idx = {n_: i for i, n_ in enumerate(NUCS)}
    mask = codon_indices >= 0
    flat = codon_indices[mask]
    for idx in flat:
        codon = gc.index_to_codon(int(idx))
        for pos, nuc in enumerate(codon):
            counts[pos, nuc_idx[nuc]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    f = counts / totals
    pi = np.empty(n, dtype=np.float64)
    for i, codon in enumerate(gc.sense_codons):
        pi[i] = f[0, nuc_idx[codon[0]]] * f[1, nuc_idx[codon[1]]] * f[2, nuc_idx[codon[2]]]
    total = pi.sum()
    if total <= 0:
        raise ValueError(
            "F3X4 produced all-zero codon frequencies (likely all-gap or "
            "mutually-exclusive nucleotide observations per position); "
            "consider pseudocount>0 for degenerate inputs"
        )
    pi /= total
    return pi
