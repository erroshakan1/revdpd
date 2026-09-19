import numpy as np
import pytest

from data import write_aa
from revdpd.core.backmap import place_cluster, random_rotation
from revdpd.core.ions import IonSettings, add_ions, ion_counts, total_charge
from revdpd.io.lammps_data import Box
from revdpd.io.lammps_writer import MoleculeSet
from revdpd.io.moltemplate import parse_molecule, spc_water


def _system(tmp_path, n_solute=4, charge=1.0, box=40.0, n_water_beads=400):
    m = parse_molecule(write_aa(tmp_path))
    m.charges = m.charges.copy()
    m.charges[1] += charge                      # make the toy molecule charged
    ff = m.ff
    ff.masses.update({"OW": 15.9994, "H": 1.008, "NA+": 22.99, "CL-": 35.453})
    w = spc_water(ff)
    rng = np.random.default_rng(0)
    B = Box(np.zeros(3), np.full(3, box))
    sol = [(m.pos - m.pos.mean(0)) @ random_rotation(rng).T + rng.uniform(5, box - 5, 3)
           for _ in range(n_solute)]
    wat = []
    for _ in range(n_water_beads):
        wat += place_cluster(rng.uniform(0, box, 3), w, 3, rng)
    return [MoleculeSet(m, sol), MoleculeSet(w, wat, restrain=False)], B, ff


def test_ion_counts():
    st = IonSettings(neutralize=True)
    assert ion_counts(4.0, 1000, st) == (0, 4)
    assert ion_counts(-3.0, 1000, st) == (3, 0)
    st = IonSettings(neutralize=True, add_salt=True, concentration=0.15)
    n_pairs = round(0.15 * 55500 / 55.5)                         # 150 pairs for 55 500 waters
    assert ion_counts(2.0, 55500, st) == (n_pairs, n_pairs + 2)


def test_neutralize_and_salt(tmp_path):
    sets, box, ff = _system(tmp_path)
    q0 = total_charge(sets)
    assert np.isclose(q0, 4.0)
    n_w = len(sets[1].coords)
    st = IonSettings(neutralize=True, add_salt=True, concentration=0.5, min_dist_solute=3.0,
                     min_dist_ion=3.0)
    out = add_ions(sets, box, ff, st, log=lambda *_: None)
    assert np.isclose(total_charge(out), 0.0)
    names = {s.template.name: len(s.coords) for s in out}
    pairs = round(0.5 * n_w / 55.5)
    assert names["NA+"] == pairs and names["CL-"] == pairs + 4
    assert len(out[1].coords) == n_w - (2 * pairs + 4)            # water replaced, not added
    ions = np.array([c[0] for s in out if s.template.name in ("NA+", "CL-") for c in s.coords])
    d = ions[:, None] - ions[None]
    d -= box.lengths * np.round(d / box.lengths)
    r = np.linalg.norm(d, axis=-1)[np.triu_indices(len(ions), 1)]
    assert r.min() >= 3.0 - 1e-9
    heavy = np.concatenate([c[s.template.heavy_mask()] for s in out[:1] for c in s.coords])
    dd = ions[:, None] - heavy[None]
    dd -= box.lengths * np.round(dd / box.lengths)
    assert np.linalg.norm(dd, axis=-1).min() >= 3.0 - 1e-9


def test_no_water_is_an_error(tmp_path):
    sets, box, ff = _system(tmp_path, n_water_beads=0)
    with pytest.raises(ValueError, match="water"):
        add_ions(sets[:1], box, ff, IonSettings(neutralize=True), log=lambda *_: None)


def test_disabled_does_nothing(tmp_path):
    sets, box, ff = _system(tmp_path)
    assert add_ions(sets, box, ff, IonSettings(), log=lambda *_: None) is sets
