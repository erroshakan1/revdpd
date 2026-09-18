"""Minimal moltemplate (.lt) reader for all-atom molecule templates and force fields.

Designed for files produced by the Automated Topology Builder (ATB) and similar
generators: one molecule object (``NAME inherits FF { ... }``) containing explicit
``write("Data Atoms")``, ``write("Data Bonds")``, ... blocks, plus a force-field
object with ``write_once("In Init")`` and ``write_once("In Settings")`` blocks
(``mass``, ``pair_coeff``, ``bond_coeff``, ...).

The force-field file is located automatically: explicit ``import`` statements are
followed, and otherwise every ``*.lt`` file in the molecule's directory is scanned
for a definition of the inherited object.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TOPO_SECTIONS = {"Data Bonds": ("bond", 2), "Data Angles": ("angle", 3),
                  "Data Dihedrals": ("dihedral", 4), "Data Impropers": ("improper", 4)}
_COEFF_CMDS = {"bond_coeff": "bond", "angle_coeff": "angle",
               "dihedral_coeff": "dihedral", "improper_coeff": "improper"}

_STD_MASSES = {
    "H": 1.008, "C": 12.011, "N": 14.0067, "O": 15.9994, "F": 18.9984, "P": 30.9738,
    "S": 32.06, "Cl": 35.453, "Br": 79.904, "I": 126.904, "Na": 22.9898, "K": 39.098,
    "Mg": 24.305, "Ca": 40.08, "Si": 28.086, "B": 10.81, "Zn": 65.37, "Fe": 55.847,
    "Cu": 63.546, "Ar": 39.948, "Li": 6.94,
}
_ELEMENTS = set(_STD_MASSES)


class LtError(ValueError):
    pass


@dataclass
class ForceField:
    name: str
    path: str
    init_lines: list[str] = field(default_factory=list)      # "In Init" commands
    masses: dict[str, float] = field(default_factory=dict)
    pair: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    coeffs: dict[str, dict[str, list[str]]] = field(
        default_factory=lambda: {"bond": {}, "angle": {}, "dihedral": {}, "improper": {}})
    extra_settings: list[str] = field(default_factory=list)

    def pair_coeff(self, a: str, b: str) -> list[str] | None:
        return self.pair.get((a, b)) or self.pair.get((b, a))

    def style(self, kind: str) -> str | None:
        """Return the value of e.g. ``pair_style`` from the init block."""
        for ln in self.init_lines:
            tok = ln.split()
            if tok and tok[0] == kind:
                return " ".join(tok[1:])
        return None


@dataclass
class AAMolecule:
    """An all-atom molecule template with force-field type names."""

    name: str
    path: str
    atom_names: list[str]
    atom_types: list[str]
    charges: np.ndarray
    pos: np.ndarray                                # (n,3) Angstrom
    topology: dict[str, list[tuple[str, tuple[int, ...]]]]   # kind -> [(type, atom indices)]
    ff: ForceField | None = None
    masses: np.ndarray = field(default=None)       # (n,)
    elements: list[str] = field(default=None)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_names)

    @property
    def bonds(self) -> list[tuple[int, int]]:
        return [idx for _, idx in self.topology.get("bond", [])]

    def heavy_mask(self) -> np.ndarray:
        return np.array([e != "H" for e in self.elements])

    def neighbors(self) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(self.n_atoms)]
        for a, b in self.bonds:
            adj[a].append(b)
            adj[b].append(a)
        return adj


# --------------------------------------------------------------------- lexing
def _strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        # '#' inside quotes is rare in lt files; handle the common case
        if "#" in line:
            line = line[: line.index("#")]
        out.append(line)
    return "\n".join(out)


def _find_block(text: str, start: int) -> tuple[int, int]:
    """Given index of an opening '{', return (start, end) of the enclosed text."""
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start + 1, i
    raise LtError("unbalanced braces")


_OBJ_RE = re.compile(r"(?m)^\s*([A-Za-z_][\w.\-]*)\s*(?:inherits\s+([\w.\-/ ,]+?))?\s*\{")
_WRITE_RE = re.compile(r"write(?:_once)?\s*\(\s*\"([^\"]+)\"\s*\)\s*\{")


def _objects(text: str) -> dict[str, tuple[list[str], str]]:
    """Top-level object definitions: name -> (inherited names, body)."""
    objs = {}
    pos = 0
    while True:
        m = _OBJ_RE.search(text, pos)
        if not m:
            break
        name = m.group(1)
        if name in ("write", "write_once"):
            pos = m.end()
            continue
        b0, b1 = _find_block(text, m.end() - 1)
        parents = [p.strip() for p in re.split(r"[ ,]+", m.group(2) or "") if p.strip()]
        objs[name] = (parents, text[b0:b1])
        pos = b1 + 1
    return objs


def _writes(body: str) -> list[tuple[str, list[str]]]:
    out = []
    pos = 0
    while True:
        m = _WRITE_RE.search(body, pos)
        if not m:
            break
        b0, b1 = _find_block(body, m.end() - 1)
        lines = [ln.strip() for ln in body[b0:b1].splitlines() if ln.strip()]
        out.append((m.group(1), lines))
        pos = b1 + 1
    return out


def _unescape(line: str) -> str:
    return line.replace("\\$", "$").replace("\\{", "{").replace("\\}", "}")


def _tail(tok: str) -> str:
    """'@atom:GROMOS/HS14' or '@atom:HS14' -> 'HS14'."""
    s = tok.split(":", 1)[1] if ":" in tok else tok
    return s.rsplit("/", 1)[-1]


# ---------------------------------------------------------------- force field
def parse_forcefield(path: str | Path, name: str | None = None) -> ForceField:
    text = _strip_comments(Path(path).read_text())
    objs = _objects(text)
    if not objs:
        raise LtError(f"{path}: no moltemplate objects found")
    if name is None:
        name = next(iter(objs))
    if name not in objs:
        raise LtError(f"{path}: object {name!r} not defined")
    ff = ForceField(name=name, path=str(path))
    for sec, lines in _writes(objs[name][1]):
        if sec == "In Init":
            ff.init_lines += [_unescape(ln) for ln in lines]
        elif sec in ("In Settings", "In Settings Coeffs"):
            for ln in lines:
                tok = ln.split()
                cmd = tok[0]
                if cmd == "mass":
                    ff.masses[_tail(tok[1])] = float(tok[2])
                elif cmd == "pair_coeff":
                    ff.pair[(_tail(tok[1]), _tail(tok[2]))] = tok[3:]
                elif cmd in _COEFF_CMDS:
                    ff.coeffs[_COEFF_CMDS[cmd]][_tail(tok[1])] = tok[2:]
                else:
                    ff.extra_settings.append(_unescape(ln))
        elif sec == "Data Masses":
            for ln in lines:
                tok = ln.split()
                ff.masses[_tail(tok[0])] = float(tok[1])
    return ff


def find_forcefield(mol_path: str | Path, ff_name: str) -> Path | None:
    """Locate the .lt file defining ``ff_name`` (via ``import`` or the same directory)."""
    mol_path = Path(mol_path)
    text = mol_path.read_text()
    cands: list[Path] = []
    for m in re.finditer(r"(?m)^\s*import\s+\"?([^\s\"]+)\"?", text):
        p = (mol_path.parent / m.group(1))
        if p.exists():
            cands.append(p)
    cands += sorted(p for p in mol_path.parent.glob("*.lt") if p != mol_path)
    pat = re.compile(rf"(?m)^\s*{re.escape(ff_name)}\s*\{{")
    for p in cands:
        try:
            if pat.search(_strip_comments(p.read_text())):
                return p
        except OSError:
            continue
    return None


# ------------------------------------------------------------------- molecule
def guess_element(atom_name: str, type_name: str, mass: float | None) -> str:
    if mass is not None:
        for el, m in _STD_MASSES.items():
            if abs(m - mass) < 0.01:
                return el
    for s in (atom_name, type_name):
        letters = re.match(r"[A-Za-z]+", s)
        if not letters:
            continue
        w = letters.group(0)
        if len(w) >= 2 and (w[0].upper() + w[1].lower()) in _ELEMENTS and w[:2].upper() not in ("CH", "HC", "HS", "NT", "OA", "OM", "OE"):
            return w[0].upper() + w[1].lower()
        if w[0].upper() in _ELEMENTS:
            return w[0].upper()
    if mass is not None:
        return min(_STD_MASSES, key=lambda e: abs(_STD_MASSES[e] - mass))
    return "C"


def parse_molecule(path: str | Path, ff_path: str | Path | None = None,
                   mol_name: str | None = None) -> AAMolecule:
    """Parse an all-atom moltemplate molecule, resolving its force field automatically."""
    path = Path(path)
    text = _strip_comments(path.read_text())
    objs = _objects(text)
    mols = {k: v for k, v in objs.items() if any(s == "Data Atoms" for s, _ in _writes(v[1]))}
    if not mols:
        raise LtError(f"{path}: no object with a 'Data Atoms' section found")
    if mol_name is None:
        mol_name = next(iter(mols))
    parents, body = mols[mol_name]

    names: list[str] = []
    types: list[str] = []
    q: list[float] = []
    xyz: list[list[float]] = []
    index: dict[str, int] = {}
    topo: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
    for sec, lines in _writes(body):
        if sec == "Data Atoms":
            for ln in lines:
                tok = ln.split()
                aid = next((t for t in tok if t.startswith("$atom:")), None)
                aty = next((t for t in tok if t.startswith("@atom:")), None)
                if aid is None or aty is None:
                    raise LtError(f"{path}: cannot parse atom line {ln!r}")
                nums = [t for t in tok if not t.startswith(("$", "@"))]
                if len(nums) < 4:
                    raise LtError(f"{path}: expected 'q x y z' in atom line {ln!r}")
                nm = _tail(aid)
                index[nm] = len(names)
                names.append(nm)
                types.append(_tail(aty))
                q.append(float(nums[-4]))
                xyz.append([float(v) for v in nums[-3:]])
        elif sec in _TOPO_SECTIONS:
            kind, k = _TOPO_SECTIONS[sec]
            lst = topo.setdefault(kind, [])
            for ln in lines:
                tok = ln.split()
                ty = next((t for t in tok if t.startswith(f"@{kind}:")), None)
                atoms = [t for t in tok if t.startswith("$atom:")]
                if ty is None:
                    raise LtError(f"{path}: {kind} without explicit type: {ln!r} "
                                  "(type-by-rule force fields are not supported yet)")
                lst.append((_tail(ty), tuple(index[_tail(a)] for a in atoms[:k])))

    ff = None
    if ff_path is None and parents:
        found = find_forcefield(path, parents[0])
        ff_path = found
    if ff_path is not None:
        ff = parse_forcefield(ff_path, parents[0] if parents else None)

    masses = np.array([ff.masses.get(t, np.nan) if ff else np.nan for t in types])
    elements = [guess_element(n, t, None if np.isnan(m) else m) for n, t, m in zip(names, types, masses)]
    for i, (e, m) in enumerate(zip(elements, masses)):
        if np.isnan(m):
            masses[i] = _STD_MASSES.get(e, 12.011)

    return AAMolecule(
        name=mol_name, path=str(path), atom_names=names, atom_types=types,
        charges=np.array(q), pos=np.array(xyz), topology=topo, ff=ff,
        masses=masses, elements=elements,
    )


# ------------------------------------------------------------ built-in water
SPC_WATER = {
    # flexible SPC geometry/charges; bonded constants for relaxation only
    "r_oh": 1.0, "theta": 109.47, "q_o": -0.82, "q_h": 0.41,
    "k_bond": 450.0, "k_angle": 55.0,
}


def spc_water(ff: ForceField, o_type: str = "OW", h_type: str = "H") -> AAMolecule:
    """An SPC water molecule using the atom types ``o_type``/``h_type`` of ``ff``.

    Bond and angle types ``SPC_OH``/``SPC_HOH`` (harmonic, flexible) are added to the
    force field. For production runs constrain water with SHAKE/RATTLE.
    """
    for t in (o_type, h_type):
        if t not in ff.masses:
            raise LtError(f"force field {ff.name} has no atom type {t!r} for water")
    w = SPC_WATER
    half = np.radians(w["theta"]) / 2
    pos = np.array([[0.0, 0.0, 0.0],
                    [w["r_oh"] * np.sin(half), w["r_oh"] * np.cos(half), 0.0],
                    [-w["r_oh"] * np.sin(half), w["r_oh"] * np.cos(half), 0.0]])
    ff.coeffs["bond"].setdefault("SPC_OH", [f"{w['k_bond']}", f"{w['r_oh']}"])
    ff.coeffs["angle"].setdefault("SPC_HOH", [f"{w['k_angle']}", f"{w['theta']}"])
    return AAMolecule(
        name="SPC", path="<built-in SPC water>", atom_names=["OW", "HW1", "HW2"],
        atom_types=[o_type, h_type, h_type], charges=np.array([w["q_o"], w["q_h"], w["q_h"]]),
        pos=pos, topology={"bond": [("SPC_OH", (0, 1)), ("SPC_OH", (0, 2))],
                           "angle": [("SPC_HOH", (1, 0, 2))]},
        ff=ff, masses=np.array([ff.masses[o_type], ff.masses[h_type], ff.masses[h_type]]),
        elements=["O", "H", "H"],
    )
