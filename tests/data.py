"""Small synthetic inputs shared by the tests."""
from pathlib import Path

import numpy as np

FF_LT = """
TOYFF {
  write_once("In Init") {
    units real
    atom_style full
    bond_style harmonic
    angle_style harmonic
    dihedral_style fourier
    improper_style harmonic
    pair_style lj/cut/coul/long \\$\\{cutoff\\}
    kspace_style pppm 0.0001
    special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 1.0
  }
  write_once("In Settings") {
    mass @atom:CX 14.027
    mass @atom:NX 14.0067
    mass @atom:HX 1.008
    pair_coeff @atom:CX @atom:CX 0.1 3.5
    pair_coeff @atom:CX @atom:NX 0.1 3.4
    pair_coeff @atom:NX @atom:NX 0.1 3.3
    pair_coeff @atom:HX @atom:CX 0.0 0.0
    pair_coeff @atom:HX @atom:NX 0.0 0.0
    pair_coeff @atom:HX @atom:HX 0.0 0.0
    bond_coeff @bond:b1 300.0 1.53
    bond_coeff @bond:b2 400.0 1.00
    angle_coeff @angle:a1 60.0 111.0
  }
}
"""


def chain_lt(n_c: int = 8) -> str:
    """A linear N-(C)n molecule with one H on N, along x with a zig-zag."""
    atoms = ["        $atom:H1 $mol:... @atom:HX 0.3 -2.4 0.8 0.0",
             "        $atom:N1 $mol:... @atom:NX -0.3 -1.5 0.3 0.0"]
    for i in range(n_c):
        atoms.append(f"        $atom:C{i + 1} $mol:... @atom:CX 0.0 {i * 1.26:.3f} {0.4 * (-1) ** i:.3f} 0.0")
    bonds = ["        $bond:bh @bond:b2 $atom:H1 $atom:N1",
             "        $bond:bn @bond:b1 $atom:N1 $atom:C1"]
    bonds += [f"        $bond:b{i} @bond:b1 $atom:C{i} $atom:C{i + 1}" for i in range(1, n_c)]
    angles = [f"        $angle:a{i} @angle:a1 $atom:C{i} $atom:C{i + 1} $atom:C{i + 2}" for i in range(1, n_c - 1)]
    return ("import \"toyff.lt\"\nTOY inherits TOYFF {\n"
            "    write(\"Data Atoms\"){\n" + "\n".join(atoms) + "\n    }\n"
            "    write(\"Data Bonds\"){\n" + "\n".join(bonds) + "\n    }\n"
            "    write(\"Data Angles\"){\n" + "\n".join(angles) + "\n    }\n}\n")


def write_aa(tmp: Path, n_c: int = 8) -> Path:
    (tmp / "toyff.lt").write_text(FF_LT)
    p = tmp / "toy.lt"
    p.write_text(chain_lt(n_c))
    return p


def cg_data(style: str, n_mol: int = 20, n_beads: int = 3, box: float = 12.0,
            seed: int = 0, images: bool = False) -> tuple[str, np.ndarray]:
    """Random rod-like CG molecules; returns (file text, unwrapped bead coords (n_mol, n_beads, 3))."""
    rng = np.random.default_rng(seed)
    coords = []
    for _ in range(n_mol):
        c = rng.uniform(0, box, 3)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        coords.append(c + np.outer(np.arange(n_beads) - (n_beads - 1) / 2, u) * 0.5)
    coords = np.array(coords)
    lines = ["# test", "", f"{n_mol * n_beads} atoms", f"{n_mol * (n_beads - 1)} bonds",
             "2 atom types", "1 bond types", "",
             f"0 {box} xlo xhi", f"0 {box} ylo yhi", f"0 {box} zlo zhi", "",
             "Masses", "", "1 1.0 # HD", "2 1.0 # TL", "", f"Atoms # {style}", ""]
    aid = 0
    for m in range(n_mol):
        for b in range(n_beads):
            aid += 1
            x = coords[m, b]
            w = np.mod(x, box)
            img = np.floor(x / box).astype(int)
            t = 1 if b == 0 else 2
            q = " 0.0" if style == "full" else ""
            extra = f" {img[0]} {img[1]} {img[2]}" if images else ""
            lines.append(f"{aid} {m + 1} {t}{q} {w[0]:.6f} {w[1]:.6f} {w[2]:.6f}{extra}")
    lines += ["", "Bonds", ""]
    bid = 0
    for m in range(n_mol):
        for b in range(n_beads - 1):
            bid += 1
            a = m * n_beads + b + 1
            lines.append(f"{bid} 1 {a} {a + 1}")
    return "\n".join(lines) + "\n", coords
