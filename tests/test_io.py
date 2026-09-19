import numpy as np
import pytest

from data import cg_data, write_aa
from revdpd.core.cg_system import CGSystem
from revdpd.io.lammps_data import DataFileError, detect_atom_style, read_lammps_data
from revdpd.io.moltemplate import parse_molecule


@pytest.mark.parametrize("style", ["angle", "full"])
@pytest.mark.parametrize("images", [False, True])
def test_read_styles_and_unwrap(tmp_path, style, images):
    text, coords = cg_data(style, images=images)
    p = tmp_path / "cg.data"
    p.write_text(text)
    assert detect_atom_style(p) == style
    d = read_lammps_data(p)            # auto
    assert d.atom_style == style
    assert d.n_atoms == coords.shape[0] * coords.shape[1]
    assert d.type_labels == {1: "HD", 2: "TL"}
    cg = CGSystem(d)
    assert len(cg.species) == 1
    sp = cg.species[0]
    assert sp.count == 20 and sp.bead_names == ["HD", "TL", "TL"]
    for k in range(sp.count):
        x = cg.instance_coords(sp, k)
        # molecule must be whole and identical to the original up to a lattice translation
        shift = x[0] - coords[k, 0]
        assert np.allclose(np.mod(shift + 1e-6, 12.0), 0, atol=1e-4) or np.allclose(np.mod(shift - 1e-6, 12.0), 12.0, atol=1e-4)
        assert np.allclose(x - x[0], coords[k] - coords[k, 0], atol=1e-5)


def test_wrong_style_is_reported(tmp_path):
    text, _ = cg_data("full")
    p = tmp_path / "cg.data"
    p.write_text(text)
    with pytest.raises(DataFileError, match="atom style"):
        read_lammps_data(p, "angle")


def test_split_by_bonds(tmp_path):
    text, _ = cg_data("angle")
    text = text.replace("Atoms # angle", "Atoms")
    # set all molecule IDs to 0 -> bonds must be used
    out = []
    in_atoms = False
    for ln in text.splitlines():
        if ln.startswith("Atoms"):
            in_atoms = True
        elif ln.startswith("Bonds"):
            in_atoms = False
        elif in_atoms and ln.strip():
            t = ln.split()
            t[1] = "0"
            ln = " ".join(t)
        out.append(ln)
    p = tmp_path / "cg.data"
    p.write_text("\n".join(out))
    cg = CGSystem(read_lammps_data(p, "angle"))
    assert cg.split_mode == "bonds"
    assert cg.species[0].count == 20


def test_moltemplate_with_import(tmp_path):
    m = parse_molecule(write_aa(tmp_path))
    assert m.name == "TOY" and m.n_atoms == 10
    assert m.ff is not None and m.ff.name == "TOYFF"
    assert m.elements[:3] == ["H", "N", "C"]
    assert m.heavy_mask().sum() == 9
    assert len(m.bonds) == 9
    assert m.ff.style("pair_style") == "lj/cut/coul/long ${cutoff}"
    assert m.ff.pair_coeff("CX", "HX") == ["0.0", "0.0"]


def test_moltemplate_object_name_starting_with_digit(tmp_path):
    # ATB residue codes such as "2WOD" may start with a digit
    p = write_aa(tmp_path)
    p.write_text(p.read_text().replace("TOY inherits", "2WOD inherits"))
    m = parse_molecule(p)
    assert m.name == "2WOD" and m.n_atoms == 10 and m.ff is not None


def test_missing_coeffs_are_reported(tmp_path):
    p = write_aa(tmp_path)
    p.write_text(p.read_text().replace("$bond:bn @bond:b1", "$bond:bn @bond:b99"))
    m = parse_molecule(p)
    assert m.missing_coeffs() == {"bond": ["b99"]}


def test_inconsistent_image_flags_are_ignored(tmp_path):
    # OVITO can write image flags that differ between bonded beads of a whole molecule
    text, coords = cg_data("angle", images=True, box=12.0)
    out, in_atoms = [], False
    for ln in text.splitlines():
        if ln.startswith("Atoms"):
            in_atoms = True
        elif ln.startswith("Bonds"):
            in_atoms = False
        elif in_atoms and ln.strip():
            t = ln.split()
            if int(t[0]) % 3 == 0:          # corrupt the flag of every third bead
                t[-1] = str(int(t[-1]) + 1)
            ln = " ".join(t)
        out.append(ln)
    p = tmp_path / "cg.data"
    p.write_text("\n".join(out))
    cg = CGSystem(read_lammps_data(p))
    sp = cg.species[0]
    for k in range(sp.count):
        x = cg.instance_coords(sp, k)
        assert np.allclose(np.linalg.norm(x[1:] - x[:-1], axis=1), 0.5, atol=1e-5)
