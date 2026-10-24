"""Gaussian route card: CADMIA names things the ORCA way and must translate."""

def _route(tmp_path, **kw) -> str:
    """First route line of a .gjf written with CalcParams(**kw)."""
    from cadmia.calculators.params import CalcParams
    from cadmia.calculators.gaussian import write_gjf
    defaults = dict(functional="PBE0", basis="def2-TZVP", dispersion="d3bj",
                    task="opt_tight+freq")
    defaults.update(kw)
    out = write_gjf(["O"], [[0.0, 0.0, 0.0]], CalcParams(**defaults), tmp_path / "j.gjf")
    return next(l for l in out.read_text().splitlines() if l.startswith("#"))


def test_route_card_has_a_space_after_the_hash(tmp_path):
    # "#PBE1PBE/..." is read as print level P + a stray "BE1PBE": l1.exe dies on the '/'.
    assert _route(tmp_path).startswith("# ")


def test_pbe0_is_written_as_gaussians_pbe1pbe(tmp_path):
    route = _route(tmp_path)
    assert "PBE1PBE" in route and "PBE0" not in route


def test_def2_basis_loses_the_hyphen(tmp_path):
    assert "Def2TZVP" in _route(tmp_path)
    assert "def2-TZVP" not in _route(tmp_path)
    assert "Def2SVP" in _route(tmp_path, basis="def2-SVP")


def test_unknown_names_pass_through_untouched(tmp_path):
    route = _route(tmp_path, functional="B3LYP", basis="6-31G(d)")
    assert route.startswith("# B3LYP/6-31G(d)")
