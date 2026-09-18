"""Write the back-mapped all-atom system as LAMMPS input (data + settings + run script)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from .lammps_data import Box
from .moltemplate import AAMolecule, ForceField

_KINDS = ("bond", "angle", "dihedral", "improper")
_SECTION = {"bond": "Bonds", "angle": "Angles", "dihedral": "Dihedrals", "improper": "Impropers"}


@dataclass
class MoleculeSet:
    """Many copies of one all-atom template."""

    template: AAMolecule
    coords: list[np.ndarray]


@dataclass
class OutputSettings:
    basename: str = "system"
    cutoff: float = 12.0
    long_range: bool = True       # keep coul/long + kspace from the force field
    soft_stage: bool = True       # soft-potential push-off before the real minimisation
    soft_a: float = 30.0          # kcal/mol, pair_style soft prefactor
    soft_rc: float = 2.5          # Angstrom
    min_steps: int = 5000
    etol: float = 1.0e-4
    ftol: float = 1.0e-6


@dataclass
class WriteResult:
    files: dict[str, Path] = field(default_factory=dict)
    n_atoms: int = 0
    n_molecules: int = 0
    total_charge: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _wrap_with_images(x: np.ndarray, box: Box) -> tuple[np.ndarray, np.ndarray]:
    h = box.h_matrix()
    hinv = np.linalg.inv(h)
    f = (x - box.lo) @ hinv.T
    img = np.floor(f).astype(np.int64)
    f -= img
    return f @ h.T + box.lo, img


def _merge_ff(sets: list[MoleculeSet]) -> ForceField:
    ffs = [s.template.ff for s in sets]
    if any(f is None for f in ffs):
        raise ValueError("every all-atom template needs a force field")
    base = ffs[0]
    for f in ffs[1:]:
        if f.path != base.path or f.name != base.name:
            raise ValueError("all templates must use the same force field file "
                             f"({base.path} vs {f.path})")
    return base


def write_lammps(out_dir: str | Path, sets: list[MoleculeSet], box: Box,
                 settings: OutputSettings | None = None) -> WriteResult:
    st = settings or OutputSettings()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ff = _merge_ff(sets)
    res = WriteResult()

    # ---- type numbering (order of first appearance)
    atypes: dict[str, int] = {}
    ttypes: dict[str, dict[str, int]] = {k: {} for k in _KINDS}
    for s in sets:
        for t in s.template.atom_types:
            atypes.setdefault(t, len(atypes) + 1)
        for k in _KINDS:
            for t, _ in s.template.topology.get(k, []):
                ttypes[k].setdefault(t, len(ttypes[k]) + 1)
    for k in _KINDS:
        missing = [t for t in ttypes[k] if t not in ff.coeffs[k]]
        if missing:
            raise ValueError(f"force field lacks {k}_coeff for types {missing}")

    # ---- atoms and topology
    atom_lines, topo_lines = [], {k: [] for k in _KINDS}
    aid = 0
    mol_id = 0
    qtot = 0.0
    for s in sets:
        m = s.template
        tid = np.array([atypes[t] for t in m.atom_types])
        for xyz in s.coords:
            mol_id += 1
            xw, img = _wrap_with_images(xyz, box)
            base = aid
            for i in range(m.n_atoms):
                aid += 1
                atom_lines.append(
                    f"{aid} {mol_id} {tid[i]} {m.charges[i]:.6f} "
                    f"{xw[i, 0]:.6f} {xw[i, 1]:.6f} {xw[i, 2]:.6f} {img[i, 0]} {img[i, 1]} {img[i, 2]}")
            qtot += float(m.charges.sum())
            for k in _KINDS:
                for t, idx in m.topology.get(k, []):
                    topo_lines[k].append((ttypes[k][t], [base + 1 + j for j in idx]))
    res.n_atoms, res.n_molecules, res.total_charge = aid, mol_id, qtot
    if abs(qtot) > 1e-3:
        res.warnings.append(f"system is not neutral (total charge {qtot:+.4f} e)")

    # ---- data file
    L = []
    L.append(f"LAMMPS data file written by revdpd {__version__} (DPD -> all-atom back-mapping)")
    L.append("")
    L.append(f"{aid} atoms")
    for k in _KINDS:
        L.append(f"{len(topo_lines[k])} {_SECTION[k].lower()}")
    L.append(f"{len(atypes)} atom types")
    for k in _KINDS:
        L.append(f"{max(1, len(ttypes[k])) if topo_lines[k] else 0} {k} types")
    L.append("")
    for d, ax in enumerate("xyz"):
        L.append(f"{box.lo[d]:.6f} {box.hi[d]:.6f} {ax}lo {ax}hi")
    if box.triclinic:
        L.append(f"{box.tilt[0]:.6f} {box.tilt[1]:.6f} {box.tilt[2]:.6f} xy xz yz")
    L.append("")
    L.append("Masses")
    L.append("")
    for t, i in atypes.items():
        L.append(f"{i} {ff.masses[t]:.6f}  # {t}")
    L.append("")
    L.append("Atoms  # full")
    L.append("")
    L += atom_lines
    for k in _KINDS:
        if not topo_lines[k]:
            continue
        L.append("")
        L.append(_SECTION[k])
        L.append("")
        for n, (t, idx) in enumerate(topo_lines[k], 1):
            L.append(f"{n} {t} " + " ".join(map(str, idx)))
    L.append("")
    b = st.basename
    data = out / f"{b}.data"
    data.write_text("\n".join(L))
    res.files["data"] = data

    # ---- settings
    pair_lines = [f"# pair coefficients ({ff.name}) - atom types: " +
                  ", ".join(f"{i}={t}" for t, i in atypes.items())]
    names = list(atypes)
    n_missing_pair = 0
    for a in range(len(names)):
        for c in range(a, len(names)):
            p = ff.pair_coeff(names[a], names[c])
            if p is None:
                n_missing_pair += 1
                continue
            pair_lines.append(f"pair_coeff {atypes[names[a]]} {atypes[names[c]]} {' '.join(p)}  # {names[a]}-{names[c]}")
    if n_missing_pair:
        res.warnings.append(f"{n_missing_pair} pair_coeff entries missing (LAMMPS mixing rules will be used)")
    bonded = [f"# bonded coefficients ({ff.name})"]
    for k in _KINDS:
        for t, i in ttypes[k].items():
            bonded.append(f"{k}_coeff {i} {' '.join(ff.coeffs[k][t])}  # {t}")
    (out / f"{b}.in.pair").write_text("\n".join(pair_lines) + "\n")
    (out / f"{b}.in.bonded").write_text("\n".join(bonded) + "\n")
    (out / f"{b}.in.settings").write_text(
        f"include {b}.in.bonded\ninclude {b}.in.pair\n" + "".join(x + "\n" for x in ff.extra_settings))

    init, pair_style, kspace = _init_block(ff, st)
    (out / f"{b}.in.init").write_text("\n".join(init + [pair_style] + ([kspace] if kspace else [])) + "\n")
    res.files.update(settings=out / f"{b}.in.settings", init=out / f"{b}.in.init")

    # ---- run script
    R = [f"# minimisation of back-mapped structure (revdpd {__version__})", ""]
    R += init
    R += ["", f"read_data {b}.data", ""]
    R += [f"include {b}.in.bonded", ""]
    R += ["thermo 100", "thermo_style custom step pe ebond eangle edihed evdwl ecoul elong press", ""]
    if st.soft_stage:
        R += ["# --- stage 1: soft push-off (removes overlaps without infinite forces)",
              f"pair_style soft {st.soft_rc}",
              f"pair_coeff * * {st.soft_a}",
              "comm_modify cutoff 8.0",
              f"minimize {st.etol} {st.ftol} {st.min_steps // 2} {st.min_steps * 5}",
              "reset_timestep 0", ""]
    R += ["# --- stage 2: full force field", pair_style]
    if kspace:
        R.append(kspace)
    R += [f"include {b}.in.pair", "neigh_modify delay 0 every 1 check yes one 5000",
          f"minimize {st.etol} {st.ftol} {st.min_steps} {st.min_steps * 10}",
          "", f"write_data {b}_min.data pair ij", ""]
    run = out / f"{b}.min.in"
    run.write_text("\n".join(R))
    res.files["run"] = run
    return res


def _init_block(ff: ForceField, st: OutputSettings) -> tuple[list[str], str, str | None]:
    init, pair_style, kspace = [], None, None
    for ln in ff.init_lines:
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "pair_style":
            pair_style = " ".join(tok).replace("${cutoff}", f"{st.cutoff:g}")
        elif tok[0] == "kspace_style":
            kspace = " ".join(tok)
        else:
            init.append(" ".join(tok))
    if pair_style is None:
        pair_style = f"pair_style lj/cut/coul/cut {st.cutoff:g}"
    if not st.long_range:
        pair_style = pair_style.replace("coul/long", "coul/cut")
        kspace = None
    return init, pair_style, kspace
