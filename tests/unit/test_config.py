from __future__ import annotations

from pathlib import Path

from selkit.io.config import RunConfig, StrictFlags, dump_config, load_config


def test_run_config_round_trips_yaml(tmp_path: Path) -> None:
    cfg = RunConfig(
        alignment=Path("/data/x.fa"),
        alignment_dir=None,
        tree=Path("/data/x.nwk"),
        foreground=None,
        subcommand="codeml.site",
        models=("M0", "M1a", "M2a"),
        tests=("M1a-vs-M2a",),
        genetic_code="standard",
        output_dir=Path("/out"),
        threads=4,
        seed=42,
        n_starts=3,
        convergence_tol=0.5,
        strict=StrictFlags(
            strip_terminal_stop=True,
            strip_stop_codons=False,
            mask_stop_codons=False,
            trim_trailing=False,
        ),
        selkit_version="0.0.1",
        git_sha=None,
    )
    path = tmp_path / "run.yaml"
    dump_config(cfg, path)
    back = load_config(path)
    assert back == cfg


def test_run_yaml_roundtrips_family(tmp_path):
    from selkit.io.config import dump_config, load_config, RunConfig, StrictFlags
    from pathlib import Path
    cfg = RunConfig(
        alignment=Path("x.fa"), alignment_dir=None, tree=Path("y.nwk"),
        foreground=None, subcommand="codeml.branch-site",
        models=("ModelA", "ModelA_null"), tests=(),
        genetic_code="standard", output_dir=tmp_path,
        threads=1, seed=0, n_starts=3, convergence_tol=0.5,
        strict=StrictFlags(True, False, False, False),
        selkit_version="0.3.0", git_sha=None,
    )
    path = tmp_path / "run.yaml"
    dump_config(cfg, path)
    text = path.read_text()
    assert "family: branch-site" in text
    cfg2 = load_config(path)
    assert cfg2.subcommand == "codeml.branch-site"
