from __future__ import annotations

import numpy as np

from selkit.engine.beb import BEBSite
from selkit.engine.beb.site import compute_neb


def test_neb_returns_one_entry_per_site() -> None:
    per_class_site_logL = np.log(np.array([
        [0.5, 0.2, 0.1],
        [0.1, 0.5, 0.2],
        [0.05, 0.1, 0.6],
    ]))
    weights = [0.5, 0.3, 0.2]
    omegas = [0.1, 1.0, 3.0]
    sites = compute_neb(
        per_class_site_logL=per_class_site_logL,
        weights=weights, omegas=omegas,
    )
    assert len(sites) == 3
    assert all(isinstance(s, BEBSite) for s in sites)
    assert sites[2].p_positive > 0.4


def test_neb_positive_posterior_is_zero_when_no_positive_class() -> None:
    per_class_site_logL = np.log(np.array([[0.5, 0.5]]))
    weights = [0.5, 0.5]
    omegas = [0.1, 1.0]
    sites = compute_neb(
        per_class_site_logL=per_class_site_logL,
        weights=weights, omegas=omegas,
    )
    assert sites[0].p_positive == 0.0
