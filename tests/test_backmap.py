import numpy as np

from data import cg_data, write_aa
from revdpd.core.backmap import BackmapSettings, Fitter, estimate_scale, kabsch, random_rotation
from revdpd.core.cg_system import CGSystem
from revdpd.core.mapping import BeadMapping, auto_linear_mapping, bead_centers
from revdpd.core.overlap import count_overlaps, remove_overlaps
from revdpd.core.pipeline import (MinimizeSettings, OverlapSettings, Project, SpeciesAssignment,
                                  run_project)
from revdpd.io.lammps_data import Box, read_lammps_data
from revdpd.io.lammps_writer import OutputSettings
from revdpd.io.moltemplate import parse_molecule


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(6, 3))
    R = random_rotation(rng)
    t = np.array([1.0, -2.0, 3.0])
    R2, t2 = kabsch(P, P @ R.T + t)
    assert np.allclose(R2, R, atol=1e-8) and np.allclose(t2, t, atol=1e-8)


def test_auto_mapping_and_mapping_roundtrip(tmp_path):
    m = parse_molecule(write_aa(tmp_path, n_c=8))
    mp = auto_linear_mapping(m, 3)
    names = [[m.atom_names[a] for a in b] for b in mp.beads]
    assert names[0][0] == "N1"                     # starts at the hetero atom
    assert sum(len(b) for b in names) == 9          # every heavy atom once
    mp2 = BeadMapping.from_dict(mp.to_dict(m), m)
    assert mp2.beads == mp.beads
    mp.toggle(1, mp.beads[0][0])                    # move an atom between beads
    assert mp.bead_of(m.atom_names.index("N1")) == 1


def test_fit_preserves_orientation(tmp_path):
    m = parse_molecule(write_aa(tmp_path, n_c=8))
    mp = auto_linear_mapping(m, 3)
    B = bead_centers(m, mp)
    rng = np.random.default_rng(3)
    R = random_rotation(rng)
    target = (B @ R.T + [5.0, 6.0, 7.0]) / 4.0       # CG units with scale 4
    f = Fitter(m, mp, BackmapSettings(scale=4.0, random_spin=False))
    y = f.fit(target, rng)
    assert f.rmsd(target, y) < 1e-6
    # head-to-tail vector points the same way as in the CG molecule
    d_aa = y[m.atom_names.index("C8")] - y[m.atom_names.index("N1")]
    d_cg = target[-1] - target[0]
    assert np.dot(d_aa, d_cg) / np.linalg.norm(d_aa) / np.linalg.norm(d_cg) > 0.95
    # internal geometry is unchanged
    assert np.allclose(np.linalg.norm(y[1:] - y[:-1], axis=1), np.linalg.norm(m.pos[1:] - m.pos[:-1], axis=1))


def test_overlap_removal_separates_molecules():
    box = Box(np.zeros(3), np.full(3, 30.0))
    a = np.array([[10.0, 10, 10], [11.5, 10, 10], [13.0, 10, 10]])
    b = a + [0.3, 0.8, 0.0]
    masks = [np.ones(3, bool)] * 2
    assert count_overlaps([a, b], masks, box, 2.5) > 0
    out = remove_overlaps([a, b], masks, box, d_min=2.5, max_iter=100)
    assert count_overlaps(out, masks, box, 2.5) == 0
    assert np.allclose(out[0] - out[0][0], a - a[0])   # rigid


def test_full_pipeline_writes_valid_lammps(tmp_path):
    aa = write_aa(tmp_path, n_c=8)
    from data import cg_data
    text, _ = cg_data("angle", n_mol=10, box=12.0)
    cgp = tmp_path / "cg.data"
    cgp.write_text(text)
    cg = CGSystem(read_lammps_data(cgp))
    mol = parse_molecule(aa)
    mp = auto_linear_mapping(mol, 3)
    s = estimate_scale(cg, cg.species[0], mol, mp)
    assert 2 < s < 10
    proj = Project(cg_path=str(cgp), out_dir=str(tmp_path / "out"))
    proj.assignments = [SpeciesAssignment(species_bead_names=cg.species[0].bead_names, species_index=0,
                                          aa_path=str(aa), ff_path=None, mapping=mp.to_dict(mol))]
    proj.backmap.scale = s
    proj.overlap = OverlapSettings(enabled=True, d_min=2.0)
    proj.output = OutputSettings(cutoff=8.0)
    proj.minimize = MinimizeSettings(enabled=False)
    pj = tmp_path / "p.json"
    proj.save(pj)
    res = run_project(Project.load(pj), log=lambda *_: None)
    assert res.write.n_atoms == 10 * mol.n_atoms
    d = read_lammps_data(res.write.files["data"], "full")
    assert d.n_atoms == 100 and len(d.bonds()) == 10 * 9
    assert d.has_image_flags
    assert np.all(d.pos >= d.box.lo - 1e-6) and np.all(d.pos <= d.box.hi + 1e-6)
    pair = (tmp_path / "out" / "system.in.pair").read_text()
    assert "pair_coeff 1 1" in pair
    run = res.write.files["run"].read_text()
    assert "read_data system.data" in run and "pair_style lj/cut/coul/long 8" in run


def test_project_relative_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    pj = tmp_path / "sub" / "p.json"
    Project(cg_path="../cg.data", out_dir="out").save(pj)
    p = Project.load(pj)
    assert p.cg_path == str((tmp_path / "cg.data").resolve())
    assert p.out_dir == str((tmp_path / "sub" / "out").resolve())
