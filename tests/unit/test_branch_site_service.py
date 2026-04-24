from __future__ import annotations


def test_branch_site_default_beb_models_is_model_a() -> None:
    from selkit.services.codeml import branch_site
    assert branch_site.DEFAULT_BEB_MODELS == ("ModelA",)
