"""Runs the generated relaxation script with a real LAMMPS binary (skipped if none is found)."""
import numpy as np
import pytest

from data import cg_data, write_aa
from revdpd.core.backmap import BackmapSettings
from revdpd.core.cg_system import CGSystem
from revdpd.core.lammps_runner import find_lammps
from revdpd.core.mapping import BeadMapping, auto_linear_mapping
from revdpd.core.ions import IonSettings
from revdpd.core.pipeline import Job, MinimizeSettings, OverlapSettings, run_backmapping
from revdpd.io.lammps_data import read_lammps_data
from revdpd.io.lammps_writer import OutputSettings
from revdpd.io.moltemplate import parse_molecule, spc_water

LMP = find_lammps()


@pytest.mark.skipif(LMP is None, reason="LAMMPS executable not found")
@pytest.mark.parametrize("mode", ["rigid", "fragment"])
def test_restrained_relaxation_with_water(tmp_path, mode):
    aa = parse_molecule(write_aa(tmp_path, n_c=8))
    aa.ff.masses.update({"OW": 15.9994, "H": 1.008, "NA+": 22.99, "CL-": 35.453})
    aa.charges = aa.charges.copy()
    aa.charges[1] += 1.0                       # charged solute -> needs counter-ions
    for t in ("OW", "H", "NA+", "CL-"):
        for u in ("CX", "NX", "HX", "OW", "H", "NA+", "CL-"):
            aa.ff.pair.setdefault((t, u), ["0.1", "3.1"] if "O" in t + u and "H" not in (t, u) else ["0.0", "0.0"])
    water = spc_water(aa.ff)
    text, _ = cg_data("full", n_mol=12, box=10.0, seed=4)
    (tmp_path / "cg.data").write_text(text)
    wtext, _ = cg_data("full", n_mol=30, n_beads=1, box=10.0, seed=5)
    (tmp_path / "w.data").write_text(wtext)
    cg = CGSystem(read_lammps_data(tmp_path / "cg.data"))
    cgw = CGSystem(read_lammps_data(tmp_path / "w.data"))
    # put the water species into the same system object for the test
    cg.species.append(cgw.species[0])
    cgw_inst = cgw.instance_coords
    orig = cg.instance_coords
    cg.instance_coords = lambda sp, k: cgw_inst(sp, k) if sp is cgw.species[0] else orig(sp, k)
    lines = []
    out = OutputSettings(cutoff=8.0, min_steps=200, md_steps=50, restraint_k="500 50")
    res = run_backmapping(
        cg, [Job(cg.species[0], aa, auto_linear_mapping(aa, 3)),
             Job(cgw.species[0], water, BeadMapping(1, [[0]]), copies=3)],
        BackmapSettings(scale=4.5, mode=mode), OverlapSettings(d_min=2.0), out, tmp_path / "out",
        MinimizeSettings(enabled=True, lammps_exe=LMP), log=lines.append,
        ions=IonSettings(neutralize=True, add_salt=True, concentration=1.0,
                         min_dist_solute=2.0, min_dist_ion=2.0))
    assert res.lammps_exit == 0, "\n".join(lines[-40:])
    script = res.write.files["run"].read_text()
    assert "group solute molecule 1:12" in script and "fix posres posres spring/self v_kres" in script
    assert "pair_style zero" in script and "unfix posres" in script
    d = read_lammps_data(tmp_path / "out" / "system_min.data", "full")
    n_pairs = round(1.0 * 90 / 55.5)                      # 2 pairs
    n_ions = 12 + 2 * n_pairs
    assert d.n_atoms == 12 * aa.n_atoms + (90 - n_ions) * 3 + n_ions
    assert abs(d.charges.sum()) < 1e-4
    assert np.isfinite(d.pos).all()


def test_kill_process_tree_stops_child_immediately(tmp_path):
    import threading
    import time

    from revdpd.core.lammps_runner import kill_process_tree, run_lammps
    procs = []
    t0 = time.time()
    th = threading.Thread(target=lambda: procs.append(
        run_lammps(["sh", "-c", "sleep 60 & sleep 60; wait"], tmp_path, log=lambda *_: None,
                   on_start=procs.append)))
    th.start()
    while not procs:
        time.sleep(0.01)
    kill_process_tree(procs[0])
    th.join(10)
    assert not th.is_alive() and time.time() - t0 < 5
    assert procs[0].poll() is not None


def test_build_command():
    from revdpd.core.lammps_runner import build_command
    assert build_command("lmp", "s.in") == ["lmp", "-in", "s.in"]
    assert build_command("lmp", "s.in", "mpirun -np 4", "-sf omp -pk omp 2") == \
        ["mpirun", "-np", "4", "lmp", "-in", "s.in", "-sf", "omp", "-pk", "omp", "2"]
    assert build_command("lmp", "s.in", mpi=3)[:3] == ["mpirun", "-np", "3"]
