"""End-to-end back-mapping pipeline and JSON project files."""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..io.lammps_data import read_lammps_data
from ..io.lammps_writer import MoleculeSet, OutputSettings, WriteResult, write_lammps
from ..io.moltemplate import AAMolecule, parse_forcefield, parse_molecule, spc_water

BUILTIN_SPC = "builtin:spc"


def load_template(aa_path: str, ff_path: str | None) -> AAMolecule:
    """Load an all-atom template (``builtin:spc`` = SPC water with the given force field)."""
    if aa_path == BUILTIN_SPC:
        if not ff_path:
            raise ValueError("built-in SPC water needs a force-field file")
        return spc_water(parse_forcefield(ff_path))
    return parse_molecule(aa_path, ff_path)
from .backmap import BackmapSettings, backmap_species
from .cg_system import CGSpecies, CGSystem
from .lammps_runner import Cancelled, build_command, find_lammps, run_lammps
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
    copies_per_bead: int = 1         # >1: e.g. N_m water molecules per DPD water bead


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
    mpi: int = 1                     # legacy; used only when prefix is empty
    prefix: str = ""                 # e.g. "mpirun -np 4"
    extra_args: str = ""             # e.g. "-sf omp -pk omp 8"

    def command(self, script: str) -> list[str]:
        return build_command(self.lammps_exe or find_lammps() or "lmp", script,
                             self.prefix, self.extra_args, self.mpi)


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
            if a["aa_path"] != BUILTIN_SPC:
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


@dataclass
class Job:
    species: CGSpecies
    mol: AAMolecule
    mapping: BeadMapping
    copies: int = 1


def run_backmapping(cg: CGSystem, jobs: list, bm: BackmapSettings, ov: OverlapSettings,
                    outset: OutputSettings, out_dir: str | Path, mini: MinimizeSettings | None = None,
                    log=print, progress=None, stop=None, on_process=None) -> BackmapResult:
    """``jobs``: :class:`Job` objects or (species, molecule, mapping[, copies]) tuples.

    ``stop()`` returning True aborts with :class:`Cancelled`; ``on_process(proc)`` receives
    the LAMMPS process so the caller can kill it.
    """
    jobs = [j if isinstance(j, Job) else Job(*j) for j in jobs]
    # molecules placed one per CG molecule are restrained and written first; solvent last
    jobs.sort(key=lambda j: j.copies > 1)
    rng = np.random.default_rng(bm.seed)
    sets: list[MoleculeSet] = []
    rmsd: dict[str, float] = {}
    spin_flags: list[bool] = []
    for j in jobs:
        sp, mol = j.species, j.mol
        if j.copies > 1:
            log(f"placing {j.copies} x {mol.name} per bead on {sp.count} x {sp.name}")
        else:
            log(f"fitting {mol.name} onto {sp.count} x {sp.name} ({bm.mode} fit)")
        coords, r, lin = backmap_species(cg, sp, mol, j.mapping, bm, rng, copies=j.copies, stop=stop,
                                         progress=(lambda i, n, name=sp.name: progress(name, i, n))
                                         if progress else None)
        if j.copies == 1:
            rmsd[sp.name] = float(r.mean())
            log(f"  bead-centre RMSD after fit: mean {r.mean():.2f} A, max {r.max():.2f} A")
        sets.append(MoleculeSet(mol, coords, restrain=j.copies == 1))
        spin_flags += lin.tolist()

    box = cg.box.scaled(bm.scale)
    log(f"box scaled by {bm.scale:g}: {box.lengths.round(2).tolist()} A")

    if ov.enabled and sets:
        allc = [c for s in sets for c in s.coords]
        masks = [s.template.heavy_mask() if ov.heavy_only else np.ones(s.template.n_atoms, bool)
                 for s in sets for _ in s.coords]
        fixed = remove_overlaps(allc, masks, box, d_min=ov.d_min, max_iter=ov.max_iter,
                                spin=spin_flags, rng=rng, log=log, stop=stop)
        k = 0
        for s in sets:
            s.coords = fixed[k:k + len(s.coords)]
            k += len(s.coords)

    if stop and stop():
        raise Cancelled("stopped by user")
    wr = write_lammps(out_dir, sets, box, outset)
    log(f"wrote {wr.n_atoms} atoms in {wr.n_molecules} molecules to {Path(out_dir).resolve()}")
    for w in wr.warnings:
        log("WARNING: " + w)
    mem = neighbor_memory_gb(wr.n_atoms, float(np.prod(box.lengths)), outset.cutoff, outset.skin)
    log(f"estimated LAMMPS neighbour-list memory: {mem:.1f} GB in total "
        f"(cutoff {outset.cutoff:g} A + skin {outset.skin:g} A; split across MPI ranks)")
    avail = _available_memory_gb()
    if avail and mem > 0.7 * avail:
        log(f"WARNING: this is close to or above the available memory ({avail:.0f} GB); "
            "reduce the pair cutoff for the relaxation or run on more nodes")
    mini = mini or MinimizeSettings()
    cmd = mini.command(wr.files["run"].name)
    sh = Path(out_dir) / "run_lammps.sh"
    sh.write_text("#!/bin/sh\n# run the relaxation (written by revdpd)\ncd \"$(dirname \"$0\")\"\n"
                  + " ".join(shlex.quote(c) for c in cmd) + " -log log.lammps\n")
    sh.chmod(0o755)
    wr.files["script"] = sh
    res = BackmapResult(sets=sets, rmsd=rmsd, write=wr)

    if mini.enabled:
        if not (mini.lammps_exe or find_lammps()):
            log("ERROR: LAMMPS executable not found; skipping the relaxation")
        else:
            res.lammps_exit = run_lammps(cmd + ["-log", "log.lammps"], Path(out_dir), log=log,
                                         on_start=on_process)
            if stop and stop():
                raise Cancelled("LAMMPS was stopped by user")
            log(f"LAMMPS finished with exit code {res.lammps_exit}")
    return res


def neighbor_memory_gb(n_atoms: int, volume: float, cutoff: float, skin: float) -> float:
    """Rough size of a LAMMPS half neighbour list (4 bytes per pair, 30 % page overhead)."""
    rho = n_atoms / volume
    pairs = n_atoms * 0.5 * rho * 4.0 / 3.0 * np.pi * (cutoff + skin) ** 3
    return float(pairs * 4 * 1.3 / 1e9)


def _available_memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def run_project(project: Project, log=print) -> BackmapResult:
    data = read_lammps_data(project.cg_path, project.atom_style)
    cg = CGSystem(data, project.split)
    jobs = []
    for a in project.assignments:
        if not a.enabled:
            continue
        sp = resolve_species(cg, a)
        mol = load_template(a.aa_path, a.ff_path)
        jobs.append(Job(sp, mol, BeadMapping.from_dict(a.mapping, mol), a.copies_per_bead))
    return run_backmapping(cg, jobs, project.backmap, project.overlap, project.output,
                           project.out_dir, project.minimize, log=log)
