"""Neutralising counter-ions and salt, placed by replacing water molecules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..io.lammps_data import Box
from ..io.lammps_writer import MoleculeSet
from ..io.moltemplate import AAMolecule, ForceField, LtError

WATER_MOLARITY = 55.5      # mol/L of pure water


@dataclass
class IonSettings:
    neutralize: bool = False
    add_salt: bool = False
    concentration: float = 0.15      # mol/L of added salt (0.15 M ~ physiological NaCl)
    cation: str = "NA+"              # force-field atom types
    anion: str = "CL-"
    cation_charge: float = 1.0
    anion_charge: float = -1.0
    min_dist_solute: float = 5.0     # A, ion - solute heavy atom
    min_dist_ion: float = 5.0        # A, ion - ion
    seed: int = 7


def ion_template(ff: ForceField, atom_type: str, charge: float) -> AAMolecule:
    """A single-atom ion molecule using ``atom_type`` of ``ff``."""
    if atom_type not in ff.masses:
        raise LtError(f"force field {ff.name} has no atom type {atom_type!r}")
    name = atom_type.rstrip("+-0123456789") or atom_type
    elem = name[0].upper() + name[1:].lower()
    return AAMolecule(name=atom_type, path=f"<built-in ion {atom_type}>", atom_names=[name],
                      atom_types=[atom_type], charges=np.array([float(charge)]),
                      pos=np.zeros((1, 3)), topology={}, ff=ff,
                      masses=np.array([ff.masses[atom_type]]), elements=[elem])


def is_water(mol: AAMolecule) -> bool:
    return sorted(mol.elements) == ["H", "H", "O"] and mol.n_atoms == 3


def total_charge(sets: list[MoleculeSet]) -> float:
    return float(sum(s.template.charges.sum() * len(s.coords) for s in sets))


def ion_counts(net_charge: float, n_water: int, st: IonSettings) -> tuple[int, int]:
    """(cations, anions) needed for neutralisation plus the requested salt concentration."""
    n_cat = n_an = 0
    if st.neutralize:
        q = int(round(net_charge))
        if q > 0:
            n_an += int(np.ceil(q / abs(st.anion_charge)))
        elif q < 0:
            n_cat += int(np.ceil(-q / st.cation_charge))
    if st.add_salt and st.concentration > 0:
        pairs = int(round(st.concentration * n_water / WATER_MOLARITY))
        # for monovalent salt one pair = one cation + one anion
        n_cat += pairs
        n_an += pairs
    return n_cat, n_an


def _pick_sites(candidates: np.ndarray, n: int, box: Box, d_ion: float,
                rng: np.random.Generator) -> list[int]:
    """Greedy random choice of ``n`` candidate points at least ``d_ion`` apart (periodic)."""
    L = box.lengths
    order = rng.permutation(len(candidates))
    cell = max(d_ion, 1e-6)
    ncell = np.maximum((L // cell).astype(int), 1)
    grid: dict[tuple, list[int]] = {}
    chosen: list[int] = []
    wrapped = np.mod(candidates - box.lo, L)
    for i in order:
        p = wrapped[i]
        c = tuple((p // (L / ncell)).astype(int) % ncell)
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get(((c[0] + dx) % ncell[0], (c[1] + dy) % ncell[1],
                                       (c[2] + dz) % ncell[2]), ()):
                        d = p - wrapped[j]
                        d -= L * np.round(d / L)
                        if d @ d < d_ion * d_ion:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            chosen.append(int(i))
            grid.setdefault(c, []).append(int(i))
            if len(chosen) == n:
                break
    return chosen


def add_ions(sets: list[MoleculeSet], box: Box, ff: ForceField, st: IonSettings,
             log=print) -> list[MoleculeSet]:
    """Replace water molecules by counter-ions / salt. Returns the new list of sets."""
    if not (st.neutralize or (st.add_salt and st.concentration > 0)):
        return sets
    q = total_charge(sets)
    wi = [k for k, s in enumerate(sets) if is_water(s.template)]
    n_water = sum(len(sets[k].coords) for k in wi)
    n_cat, n_an = ion_counts(q, n_water, st)
    log(f"ions: net charge {q:+.3f} e, {n_water} water molecules -> "
        f"{n_cat} {st.cation} + {n_an} {st.anion}")
    if abs(q - round(q)) > 0.01:
        log(f"WARNING: net charge {q:+.3f} e is not an integer; the system cannot be neutralised exactly")
    if n_cat + n_an == 0:
        return sets
    if n_water < n_cat + n_an:
        raise ValueError(f"{n_cat + n_an} ions requested but only {n_water} water molecules are available "
                         "(ions replace water: back-map the solvent too)")
    if box.triclinic:
        raise ValueError("ion placement currently supports orthogonal boxes only")

    # water oxygens (or water centres) as candidate sites
    owners, sites = [], []
    for k in wi:
        o = sets[k].template.elements.index("O")
        for m, c in enumerate(sets[k].coords):
            owners.append((k, m))
            sites.append(c[o])
    sites = np.array(sites)
    owners = np.array(owners)

    # distance to solute heavy atoms (molecules that are neither water nor ions)
    solute = [s for s in sets if not is_water(s.template)]
    d_sol = st.min_dist_solute
    if solute and d_sol > 0:
        heavy = np.concatenate([c[s.template.heavy_mask()] for s in solute for c in s.coords])
        tree = cKDTree(np.mod(heavy - box.lo, box.lengths), boxsize=box.lengths)
        dist, _ = tree.query(np.mod(sites - box.lo, box.lengths), k=1)
    else:
        dist = np.full(len(sites), np.inf)

    rng = np.random.default_rng(st.seed)
    need = n_cat + n_an
    d_ion = st.min_dist_ion
    chosen: list[int] = []
    for attempt in range(6):
        cand = np.flatnonzero(dist >= d_sol)
        chosen = [int(cand[i]) for i in _pick_sites(sites[cand], need, box, d_ion, rng)]
        if len(chosen) == need:
            break
        d_sol *= 0.7
        d_ion *= 0.7
        log(f"  not enough free water sites; relaxing distances to {d_sol:.1f} / {d_ion:.1f} A")
    if len(chosen) < need:
        raise ValueError(f"could only place {len(chosen)} of {need} ions")

    rng.shuffle(chosen)
    cat_sites, an_sites = sites[chosen[:n_cat]], sites[chosen[n_cat:]]
    remove: dict[int, set[int]] = {}
    for i in chosen:
        k, m = owners[i]
        remove.setdefault(int(k), set()).add(int(m))
    out = []
    for k, s in enumerate(sets):
        if k in remove:
            s = MoleculeSet(s.template, [c for m, c in enumerate(s.coords) if m not in remove[k]],
                            restrain=s.restrain)
        out.append(s)
    if n_cat:
        out.append(MoleculeSet(ion_template(ff, st.cation, st.cation_charge),
                               [p[None, :] for p in cat_sites], restrain=False))
    if n_an:
        out.append(MoleculeSet(ion_template(ff, st.anion, st.anion_charge),
                               [p[None, :] for p in an_sites], restrain=False))
    q2 = total_charge(out)
    log(f"  replaced {need} water molecules; net charge now {q2:+.3f} e"
        + (f", salt ~{st.concentration:g} M" if st.add_salt else ""))
    return out
