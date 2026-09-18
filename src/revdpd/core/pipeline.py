"""End-to-end back-mapping pipeline and JSON project files."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..io.lammps_data import read_lammps_data
from ..io.lammps_writer import MoleculeSet, OutputSettings, WriteResult, write_lammps
from ..io.moltemplate import AAMolecule, parse_molecule
from .backmap import BackmapSettings, backmap_species
from .cg_system import CGSpecies, CGSystem
from .lammps_runner import find_lammps, run_lammps
from .mapping import BeadMapping
from .overlap import remove_overlaps


@dataclass
class SpeciesAssignment:
    """A CG species linked to an all-atom template and a mapping."""

    species_bead_names: list[str]
    species_index: int
    aa_path: str
    ff_path: str | None
    mapping: dict                    # BeadMapping.to_dict()
    enabled: bool = True


@dataclass
class OverlapSettings:
    enabled: bool = True
    d_min: float = 2.5
    max_iter: int = 60
    heavy_only: bool = True


@dataclass
class MinimizeSettings:
    enabled: bool = False
    lammps_exe: str = ""
    mpi: int = 1


@dataclass
class Project:
    cg_path: str = ""
    atom_style: str = "auto"
    split: str = "auto"
    assignments: list[SpeciesAssignment] = field(default_factory=list)
    backmap: BackmapSettings = field(default_factory=BackmapSettings)
    overlap: OverlapSettings = field(default_factory=OverlapSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    minimize: MinimizeSettings = field(default_factory=MinimizeSettings)
    out_dir: str = "backmapped"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        d = json.loads(Path(path).read_text())
        base = Path(path).resolve().parent

        def rel(x):
            # relative paths in a project file are relative to the project file itself
            if not x:
                return x
            q = Path(x).expanduser()
            return str(q if q.is_absolute() else (base / q).resolve())

        d["cg_path"] = rel(d["cg_path"])
        for a in d.get("assignments", []):
            a["aa_path"] = rel(a["aa_path"])
            a["ff_path"] = rel(a.get("ff_path"))
        if "out_dir" in d:
            d["out_dir"] = rel(d["out_dir"])
        p = cls(cg_path=d["cg_path"], atom_style=d.get("atom_style", "auto"),
                split=d.get("split", "auto"), out_dir=d.get("out_dir", "backmapped"))
        p.assignments = [SpeciesAssignment(**a) for a in d.get("assignments", [])]
        p.backmap = BackmapSettings(**d.get("backmap", {}))
        p.overlap = OverlapSettings(**d.get("overlap", {}))
        p.output = OutputSettings(**d.get("output", {}))
        p.minimize = MinimizeSettings(**d.get("minimize", {}))
        return p


def resolve_species(cg: CGSystem, a: SpeciesAssignment) -> CGSpecies:
    """Find the species of an assignment (by index, verified by bead names; else by names)."""
    if 0 <= a.species_index < len(cg.species) and cg.species[a.species_index].bead_names == a.species_bead_names:
        return cg.species[a.species_index]
    for sp in cg.species:
        if sp.bead_names == a.species_bead_names:
            return sp
    raise ValueError(f"no CG species with beads {a.species_bead_names}")


@dataclass
class BackmapResult:
    sets: list[MoleculeSet]
    rmsd: dict[str, float]
    write: WriteResult | None = None
    lammps_exit: int | None = None


def run_backmapping(cg: CGSystem, jobs: list[tuple[CGSpecies, AAMolecule, BeadMapping]],
                    bm: BackmapSettings, ov: OverlapSettings, outset: OutputSettings,
                    out_dir: str | Path, mini: MinimizeSettings | None = None,
                    log=print, progress=None, stop=None) -> BackmapResult:
    rng = np.random.default_rng(bm.seed)
    sets: list[MoleculeSet] = []
    rmsd: dict[str, float] = {}
    spin_flags: list[bool] = []
    for sp, mol, mapping in jobs:
        log(f"fitting {mol.name} onto {sp.count} x {sp.name}")
        coords, r, lin = backmap_species(cg, sp, mol, mapping, bm, rng,
                                    progress=(lambda i, n: progress(sp.name, i, n)) if progress else None)
        rmsd[sp.name] = float(r.mean())
        log(f"  bead-centre RMSD after fit: mean {r.mean():.2f} A, max {r.max():.2f} A")
        sets.append(MoleculeSet(mol, coords))
        spin_flags += lin.tolist()

    box = cg.box.scaled(bm.scale)
    log(f"box scaled by {bm.scale:g}: {box.lengths.round(2).tolist()} A")

    if ov.enabled and sets:
        allc = [c for s in sets for c in s.coords]
        masks = [s.template.heavy_mask() if ov.heavy_only else np.ones(s.template.n_atoms, bool)
                 for s in sets for _ in s.coords]
        fixed = remove_overlaps(allc, masks, box, d_min=ov.d_min, max_iter=ov.max_iter,
                                spin=spin_flags, rng=rng, log=log)
        k = 0
        for s in sets:
            s.coords = fixed[k:k + len(s.coords)]
            k += len(s.coords)

    wr = write_lammps(out_dir, sets, box, outset)
    log(f"wrote {wr.n_atoms} atoms in {wr.n_molecules} molecules to {Path(out_dir).resolve()}")
    for w in wr.warnings:
        log("WARNING: " + w)
    res = BackmapResult(sets=sets, rmsd=rmsd, write=wr)

    if mini and mini.enabled:
        exe = mini.lammps_exe or find_lammps()
        if not exe:
            log("ERROR: LAMMPS executable not found; skipping minimisation")
        else:
            res.lammps_exit = run_lammps(exe, wr.files["run"], Path(out_dir), log=log, mpi=mini.mpi, stop=stop)
            log(f"LAMMPS finished with exit code {res.lammps_exit}")
    return res


def run_project(project: Project, log=print) -> BackmapResult:
    data = read_lammps_data(project.cg_path, project.atom_style)
    cg = CGSystem(data, project.split)
    jobs = []
    for a in project.assignments:
        if not a.enabled:
            continue
        sp = resolve_species(cg, a)
        mol = parse_molecule(a.aa_path, a.ff_path)
        jobs.append((sp, mol, BeadMapping.from_dict(a.mapping, mol)))
    return run_backmapping(cg, jobs, project.backmap, project.overlap, project.output,
                           project.out_dir, project.minimize, log=log)
