from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from selkit.io.results import BEBSite


def compute_neb(
    *,
    per_class_site_logL: np.ndarray,
    weights: list[float],
    omegas: list[float],
) -> list[BEBSite]:
    log_w = np.log(np.asarray(weights))[:, None]
    log_joint = per_class_site_logL + log_w
    log_norm = logsumexp(log_joint, axis=0)
    log_post = log_joint - log_norm
    post = np.exp(log_post)
    om = np.asarray(omegas)[:, None]
    mean_om = (post * om).sum(axis=0)
    positive_mask = (np.asarray(omegas) > 1.0)[:, None]
    p_pos = (post * positive_mask).sum(axis=0)
    out: list[BEBSite] = []
    for s in range(per_class_site_logL.shape[1]):
        out.append(BEBSite(
            site=s + 1,
            p_positive=float(p_pos[s]),
            mean_omega=float(mean_om[s]),
        ))
    return out
