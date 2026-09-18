"""Runs the generated minimisation script with a real LAMMPS binary (skipped if none is found)."""
import numpy as np
import pytest

from data import cg_data, write_aa
from revdpd.core.cg_system import CGSystem
from revdpd.core.lammps_runner import find_lammps
from revdpd.core.mapping import auto_linear_mapping
from revdpd.core.pipeline import MinimizeSettings, OverlapSettings, run_backmapping
from revdpd.core.backmap import BackmapSettings
from revdpd.io.lammps_data import read_lammps_data
from revdpd.io.lammps_writer import OutputSettings
from revdpd.io.moltemplate import parse_molecule

LMP = find_lammps()


@pytest.mark.skipif(LMP is None, reason="LAMMPS executable not found")
def test_lammps_minimisation(tmp_path):
    aa = parse_molecule(write_aa(tmp_path, n_c=8))
    text, _ = cg_data("full", n_mol=12, box=10.0, seed=4)
    (tmp_path / "cg.data").write_text(text)
    cg = CGSystem(read_lammps_data(tmp_path / "cg.data"))
    mp = auto_linear_mapping(aa, 3)
    lines = []
    res = run_backmapping(cg, [(cg.species[0], aa, mp)], BackmapSettings(scale=4.5),
                          OverlapSettings(d_min=2.0), OutputSettings(cutoff=8.0, min_steps=200),
                          tmp_path / "out", MinimizeSettings(enabled=True, lammps_exe=LMP),
                          log=lines.append)
    assert res.lammps_exit == 0, "\n".join(lines[-30:])
    d = read_lammps_data(tmp_path / "out" / "system_min.data", "full")
    assert d.n_atoms == 12 * aa.n_atoms
    assert np.isfinite(d.pos).all()
