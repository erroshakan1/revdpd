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
    restrain: bool = True          # position-restrain heavy atoms during relaxation


@dataclass
class OutputSettings:
    basename: str = "system"
    cutoff: float = 14.0           # Angstrom (GROMOS was parametrised with 1.4 nm)
    long_range: bool = True       # keep coul/long + kspace from the force field
    soft_stage: bool = True       # soft-potential push-off before the real minimisation
    soft_a: float = 30.0          # kcal/mol, pair_style soft prefactor
    soft_rc: float = 2.5          # Angstrom
    min_steps: int = 5000
    etol: float = 1.0e-4
    ftol: float = 1.0e-6
    # restrained, gradual relaxation (Backward/initram-like)
    bonded_stage: bool = True     # bonded-only minimisation first (no non-bonded terms)
    restrained: bool = True       # restrain heavy atoms to their back-mapped positions
    restraint_k: str = "1000 100 10"   # kcal/mol/A^2, one full-force-field minimisation per value
    md_steps: int = 1000          # steps per MD stage (0 = no MD)
    md_timesteps: str = "0.2 0.5 1.0"  # fs, one restrained MD stage per value
    temperature: float = 300.0
    release: bool = True          # final minimisation without restraints
    seed: int = 4928459


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
    # templates may add their own bonded types (e.g. built-in water): merge them
    merged = ForceField(name=base.name, path=base.path, init_lines=base.init_lines,
                        masses=base.masses, pair=base.pair, extra_settings=base.extra_settings)
    for f in ffs:
        for k, d in f.coeffs.items():
            for t, v in d.items():
                merged.coeffs[k].setdefault(t, v)
    return merged


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
            raise ValueError(f"force field {ff.path} lacks {k}_coeff for types {missing}; use the "
                             "force-field file that belongs to the same ATB revision as the topology")

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
    heavy_types = [str(i) for t, i in atypes.items() if ff.masses[t] > 1.5]
    n_restr = 0
    for srt in sets:
        if not srt.restrain:
            break
        n_restr += len(srt.coords)
    restrain = st.restrained and n_restr > 0
    ks = [float(x) for x in st.restraint_k.split()] if restrain else []
    dts = [float(x) for x in st.md_timesteps.split()] if st.md_steps > 0 else []
    run = out / f"{b}.min.in"
    run.write_text("\n".join(relaxation_script(b, init, pair_style, kspace, st, heavy_types,
                                                 n_restr if restrain else 0, ks, dts)))
    res.files["run"] = run
    return res


def relaxation_script(b: str, init: list[str], pair_style: str, kspace: str | None,
                      st: OutputSettings, heavy_types: list[str], n_restr: int,
                      ks: list[float], dts: list[float]) -> list[str]:
    """LAMMPS input for a restrained, gradual relaxation of the back-mapped structure.

    1. bonded-only minimisation (no non-bonded terms)   - repairs stretched bonds/angles
    2. soft-core push-off                               - removes overlaps
    3. full force field, restraints k1 > k2 > ...        - one minimisation per value
    4. restrained MD with increasing time step           - relaxes locally at T
    5. minimisation without restraints                   - final structure
    Heavy atoms of the restrained molecules (molecule IDs 1..n_restr) are tied to their
    back-mapped positions with ``fix spring/self`` (reference fixed at step 0).
    """
    mn = f"minimize {st.etol} {st.ftol} {st.min_steps} {st.min_steps * 10}"
    R = [f"# restrained relaxation of the back-mapped structure (revdpd {__version__})", ""]
    R += init
    R += ["", f"read_data {b}.data", f"include {b}.in.bonded",
          "neigh_modify delay 0 every 1 check yes one 5000", "thermo 100", ""]
    if n_restr:
        R += ["# position restraints on heavy atoms of the back-mapped molecules",
              f"group solute molecule 1:{n_restr}",
              f"group heavy type {' '.join(heavy_types)}",
              "group posres intersect solute heavy",
              f"variable kres equal {ks[0] if ks else 1000.0:g}",
              "fix posres posres spring/self v_kres",
              "fix_modify posres energy yes"]
    R += ["thermo_style custom step temp pe ebond eangle edihed evdwl ecoul elong "
          + ("f_posres " if n_restr else "") + "press", ""]
    if st.bonded_stage:
        R += ["# --- stage 1: bonded terms only",
              f"pair_style zero {st.cutoff:g}", "pair_coeff * *",
              mn, "reset_timestep 0", ""]
    if st.soft_stage:
        R += ["# --- stage 2: soft push-off (removes overlaps without infinite forces)",
              f"pair_style soft {st.soft_rc}", f"pair_coeff * * {st.soft_a}",
              "comm_modify cutoff 8.0", mn, "reset_timestep 0", ""]
    R += ["# --- stage 3: full force field", pair_style]
    if kspace:
        R.append(kspace)
    R.append(f"include {b}.in.pair")
    if ks:
        for k in ks:
            R += [f"variable kres equal {k:g}", mn]
    else:
        R.append(mn)
    R.append("")
    if dts:
        R += ["# --- stage 4: restrained MD with increasing time step",
              f"velocity all create {st.temperature:g} {st.seed} dist gaussian",
              "fix mdint all nve/limit 0.1",
              f"fix mdtemp all langevin {st.temperature:g} {st.temperature:g} 100.0 {st.seed + 1}"]
        for dt in dts:
            R += [f"timestep {dt:g}", f"run {st.md_steps}"]
        R += ["unfix mdint", "unfix mdtemp", ""]
    if n_restr and st.release:
        R += ["# --- stage 5: release restraints", "unfix posres",
              "thermo_style custom step pe ebond eangle edihed evdwl ecoul elong press", mn, ""]
    R += [f"write_data {b}_min.data pair ij", ""]
    return R


def _init_block(ff: ForceField, st: OutputSettings) -> tuple[list[str], str, str | None]:
    """Split the force field's init commands into (general, pair block, kspace).

    The pair block is ``pair_style`` followed by any ``pair_modify`` lines (e.g.
    ``pair_modify mix geometric`` for OPLS-AA), which LAMMPS only accepts after the
    pair style has been defined and which must be repeated whenever it is redefined.
    """
    init, pair_style, kspace, modify = [], None, None, []
    for ln in ff.init_lines:
        tok = ln.split()
        if not tok:
            continue
        if tok[0] == "pair_style":
            pair_style = " ".join(tok).replace("${cutoff}", f"{st.cutoff:g}")
        elif tok[0] == "kspace_style":
            kspace = " ".join(tok)
        elif tok[0] == "pair_modify":
            modify.append(" ".join(tok))
        else:
            init.append(" ".join(tok))
    if pair_style is None:
        pair_style = f"pair_style lj/cut/coul/cut {st.cutoff:g}"
    if not st.long_range:
        pair_style = pair_style.replace("coul/long", "coul/cut")
        kspace = None
    return init, "\n".join([pair_style] + modify), kspace
