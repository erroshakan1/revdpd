"""Reader for LAMMPS data files.

Reference: https://docs.lammps.org/read_data.html

Supported atom styles (column layout of the ``Atoms`` section):

=========  ========================================
style      columns
=========  ========================================
angle      atom-ID molecule-ID atom-type x y z
bond       atom-ID molecule-ID atom-type x y z
molecular  atom-ID molecule-ID atom-type x y z
full       atom-ID molecule-ID atom-type q x y z
charge     atom-ID atom-type q x y z
atomic     atom-ID atom-type x y z
=========  ========================================

Each line may optionally end with three integer image flags ``nx ny nz``.
Type columns may be numeric or type labels (``Atom Type Labels`` section).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# name -> (has_mol, has_charge)
ATOM_STYLES: dict[str, tuple[bool, bool]] = {
    "angle": (True, False),
    "bond": (True, False),
    "molecular": (True, False),
    "full": (True, True),
    "charge": (False, True),
    "atomic": (False, False),
}

_SECTIONS = [
    "Atoms", "Velocities", "Masses", "Ellipsoids", "Lines", "Triangles", "Bodies",
    "Bonds", "Angles", "Dihedrals", "Impropers",
    "Pair Coeffs", "PairIJ Coeffs", "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs",
    "Improper Coeffs", "BondBond Coeffs", "BondAngle Coeffs", "MiddleBondTorsion Coeffs",
    "EndBondTorsion Coeffs", "AngleTorsion Coeffs", "AngleAngleTorsion Coeffs",
    "BondBond13 Coeffs", "AngleAngle Coeffs",
    "Atom Type Labels", "Bond Type Labels", "Angle Type Labels",
    "Dihedral Type Labels", "Improper Type Labels",
]
_TOPO = {"Bonds": 2, "Angles": 3, "Dihedrals": 4, "Impropers": 4}


class DataFileError(ValueError):
    pass


@dataclass
class Box:
    lo: np.ndarray                      # (3,)
    hi: np.ndarray                      # (3,)
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(3))  # xy xz yz

    @property
    def lengths(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def triclinic(self) -> bool:
        return bool(np.any(np.abs(self.tilt) > 1e-12))

    def h_matrix(self) -> np.ndarray:
        """Columns are the box vectors a, b, c."""
        lx, ly, lz = self.lengths
        xy, xz, yz = self.tilt
        return np.array([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]])

    def minimum_image(self, d: np.ndarray) -> np.ndarray:
        """Apply the minimum image convention to displacement vectors (..., 3)."""
        h = self.h_matrix()
        hinv = np.linalg.inv(h)
        f = d @ hinv.T
        f -= np.round(f)
        return f @ h.T

    def scaled(self, s: float) -> "Box":
        return Box(self.lo * s, self.hi * s, self.tilt * s)


@dataclass
class LammpsData:
    """Contents of a LAMMPS data file. Topology arrays hold atom *IDs*."""

    path: str
    atom_style: str
    box: Box
    ids: np.ndarray            # (N,) int
    mol: np.ndarray            # (N,) int (zeros if style has no molecule column)
    types: np.ndarray          # (N,) int
    charges: np.ndarray        # (N,) float
    pos: np.ndarray            # (N,3) float
    image: np.ndarray          # (N,3) int
    masses: dict[int, float] = field(default_factory=dict)
    type_labels: dict[int, str] = field(default_factory=dict)
    # section name -> (M, 1+k) int array: [type, id1, ..., idk]
    topology: dict[str, np.ndarray] = field(default_factory=dict)
    has_image_flags: bool = False

    @property
    def n_atoms(self) -> int:
        return len(self.ids)

    def type_name(self, t: int) -> str:
        return self.type_labels.get(int(t), str(int(t)))

    def bonds(self) -> np.ndarray:
        return self.topology.get("Bonds", np.zeros((0, 3), dtype=np.int64))


def _strip(line: str) -> tuple[str, str]:
    """Return (content, comment) of a line."""
    if "#" in line:
        i = line.index("#")
        return line[:i].strip(), line[i + 1:].strip()
    return line.strip(), ""


def _match_section(content: str) -> str | None:
    for s in _SECTIONS:
        if content == s:
            return s
    return None


def detect_atom_style(path: str | Path) -> str | None:
    """Return the atom style hinted in the ``Atoms # style`` comment, if any."""
    with open(path) as fh:
        for line in fh:
            content, comment = _strip(line)
            if content == "Atoms":
                style = comment.split()[0] if comment else None
                return style if style in ATOM_STYLES else None
    return None


def read_lammps_data(path: str | Path, atom_style: str = "auto") -> LammpsData:
    """Read a LAMMPS data file.

    ``atom_style`` may be one of :data:`ATOM_STYLES` or ``"auto"`` (use the
    ``Atoms # style`` comment, falling back to a guess from the column count).
    """
    path = str(path)
    with open(path) as fh:
        lines = fh.readlines()

    counts: dict[str, int] = {}
    lo = np.zeros(3)
    hi = np.ones(3)
    tilt = np.zeros(3)
    sections: dict[str, list[str]] = {}
    style_hint = None

    i = 1  # first line is always a comment
    n = len(lines)
    # ---- header
    while i < n:
        content, comment = _strip(lines[i])
        if not content:
            i += 1
            continue
        if _match_section(content):
            break
        tok = content.split()
        if len(tok) >= 4 and tok[2] == "xlo":
            lo[0], hi[0] = float(tok[0]), float(tok[1])
        elif len(tok) >= 4 and tok[2] == "ylo":
            lo[1], hi[1] = float(tok[0]), float(tok[1])
        elif len(tok) >= 4 and tok[2] == "zlo":
            lo[2], hi[2] = float(tok[0]), float(tok[1])
        elif len(tok) >= 6 and tok[3] == "xy":
            tilt[:] = [float(t) for t in tok[:3]]
        elif len(tok) >= 2:
            key = " ".join(tok[1:])
            try:
                counts[key] = int(tok[0])
            except ValueError:
                pass
        i += 1

    # ---- body
    current = None
    while i < n:
        content, comment = _strip(lines[i])
        i += 1
        if not content:
            continue
        sec = _match_section(content)
        if sec:
            current = sec
            sections[current] = []
            if sec == "Atoms" and comment:
                style_hint = comment.split()[0]
            continue
        if current is None:
            raise DataFileError(f"{path}:{i}: data outside of any section: {content!r}")
        sections[current].append(content)

    if "Atoms" not in sections:
        raise DataFileError(f"{path}: no Atoms section found")

    # ---- type labels
    labels: dict[str, dict[int, str]] = {}
    for kind in ("Atom", "Bond", "Angle", "Dihedral", "Improper"):
        d: dict[int, str] = {}
        for content in sections.get(f"{kind} Type Labels", []):
            tok = content.split()
            d[int(tok[0])] = tok[1]
        labels[kind] = d
    atom_label_to_int = {v: k for k, v in labels["Atom"].items()}

    def atom_type(tok: str) -> int:
        try:
            return int(tok)
        except ValueError:
            if tok in atom_label_to_int:
                return atom_label_to_int[tok]
            raise DataFileError(f"unknown atom type label {tok!r}")

    # ---- masses
    masses: dict[int, float] = {}
    for content in sections.get("Masses", []):
        tok = content.split()
        masses[atom_type(tok[0])] = float(tok[1])
    atom_rows = [c.split() for c in sections["Atoms"]]
    ncol = {len(r) for r in atom_rows}
    if len(ncol) != 1:
        raise DataFileError(f"{path}: inconsistent column count in Atoms section: {sorted(ncol)}")
    ncol_v = ncol.pop()

    style = atom_style
    if style == "auto":
        style = style_hint if style_hint in ATOM_STYLES else None
        if style is None:
            style = {6: "angle", 9: "angle", 7: "full", 10: "full"}.get(ncol_v)
            if style is None:
                raise DataFileError(f"{path}: cannot guess atom style from {ncol_v} columns; select it explicitly")
    if style not in ATOM_STYLES:
        raise DataFileError(f"unsupported atom style {style!r}")
    has_mol, has_q = ATOM_STYLES[style]
    base = 2 + int(has_mol) + int(has_q) + 3
    if ncol_v == base:
        has_img = False
    elif ncol_v == base + 3:
        has_img = True
    else:
        raise DataFileError(
            f"{path}: Atoms section has {ncol_v} columns, which does not match atom style "
            f"'{style}' ({base} or {base + 3} columns expected). Did you select the right format?"
        )

    N = len(atom_rows)
    ids = np.empty(N, dtype=np.int64)
    mol = np.zeros(N, dtype=np.int64)
    types = np.empty(N, dtype=np.int64)
    q = np.zeros(N)
    pos = np.empty((N, 3))
    img = np.zeros((N, 3), dtype=np.int64)
    for k, r in enumerate(atom_rows):
        ids[k] = int(r[0])
        c = 1
        if has_mol:
            mol[k] = int(r[1])
            c = 2
        types[k] = atom_type(r[c])
        c += 1
        if has_q:
            q[k] = float(r[c])
            c += 1
        pos[k] = float(r[c]), float(r[c + 1]), float(r[c + 2])
        if has_img:
            img[k] = int(r[c + 3]), int(r[c + 4]), int(r[c + 5])

    order = np.argsort(ids, kind="stable")
    ids, mol, types, q, pos, img = ids[order], mol[order], types[order], q[order], pos[order], img[order]

    if "atoms" in counts and counts["atoms"] != N:
        raise DataFileError(f"{path}: header says {counts['atoms']} atoms but {N} were read")

    topology: dict[str, np.ndarray] = {}
    for sec, k in _TOPO.items():
        rows = sections.get(sec)
        if not rows:
            continue
        kind = sec[:-1]
        lab = {v: kk for kk, v in labels[kind].items()}
        arr = np.empty((len(rows), k + 1), dtype=np.int64)
        for j, content in enumerate(rows):
            tok = content.split()
            t = tok[1]
            try:
                arr[j, 0] = int(t)
            except ValueError:
                arr[j, 0] = lab[t]
            arr[j, 1:] = [int(x) for x in tok[2:2 + k]]
        topology[sec] = arr

    # Atom type labels may also be given as comments in the Masses section.
    type_labels = dict(labels["Atom"])
    if not type_labels:
        in_masses = False
        for raw in lines:
            content, comment = _strip(raw)
            if content == "Masses":
                in_masses = True
                continue
            if in_masses:
                if not content:
                    continue
                if _match_section(content):
                    break
                if comment:
                    type_labels[atom_type(content.split()[0])] = comment.split()[0]

    return LammpsData(
        path=path, atom_style=style, box=Box(lo, hi, tilt), ids=ids, mol=mol, types=types,
        charges=q, pos=pos, image=img, masses=masses, type_labels=type_labels,
        topology=topology, has_image_flags=has_img,
    )
