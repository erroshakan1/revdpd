# revdpd — DPD → all-atom back-mapping GUI

**revdpd** takes a coarse-grained **DPD** configuration stored as a LAMMPS data file,
splits it into molecules, lets you define **which all-atom heavy atoms belong to which
CG bead** by clicking in two side-by-side 3D views, and then places an all-atom copy of
every molecule into the system **keeping the orientation of each CG molecule**.
Optionally it removes overlaps and minimises the result with LAMMPS.

![revdpd screenshot](docs/screenshot.png)

## Features

- Reads LAMMPS data files in atom style **`angle`** or **`full`** (also `molecular`,
  `bond`), selectable or auto-detected from the `Atoms # style` comment; type labels,
  image flags and triclinic boxes are handled
  ([LAMMPS `read_data` format](https://docs.lammps.org/read_data.html)).
- Splits the system into molecules (by molecule ID or bond connectivity), makes
  them whole across periodic boundaries and groups identical molecules into species.
- Reads all-atom templates with force field from **moltemplate** `.lt` files, e.g.
  as produced by the [Automated Topology Builder (ATB)](https://atb.uq.edu.au). The
  force-field file (`GROMOS_54A7_ATB.lt` …) is located automatically in the same
  folder or through `import` statements.
- Interactive mapping: pick a bead on the CG side, click (or Shift+drag box-select)
  heavy atoms on the all-atom side. Hydrogens follow their heavy atom. An *Auto (chain)*
  proposal is available for chain-like molecules.
- Live preview of the fitted molecule on top of any CG molecule of the species.
- Scale (Å per DPD length unit) can be estimated from the template and CG bond lengths.
- Placement by weighted **Kabsch** fit of bead centres (rigid; optionally with a
  per-bead shift). For (near-)linear molecules the undetermined spin about the long axis
  is randomised.
- Simple **overlap removal**: spin search about the long axis, then rigid-body pushes.
- Writes a LAMMPS `full` data file, `*.in.init`, `*.in.settings` (pair and bonded
  coefficients from the force field) and a minimisation script (soft push-off + full
  force field), and can run LAMMPS directly.
- Projects (CG file, templates, mappings, settings) are saved as JSON and can be re-run
  headless from the command line.

## Installation

```bash
git clone <this repository> revdpd && cd revdpd
python -m venv .venv && source .venv/bin/activate      # fish: source .venv/bin/activate.fish
pip install -e .            # or: pip install -e ".[dev]" to also get pytest
```

Requirements: Python ≥ 3.10, NumPy, SciPy, PySide6. LAMMPS is only needed for the
optional minimisation (with the `MOLECULE`, `KSPACE` and `EXTRA-MOLECULE` packages for
GROMOS/ATB force fields). The executable is searched as `lmp`, `lmp_serial`, `lmp_mpi` …
in `PATH` or taken from `$LAMMPS_EXE`, and can be chosen in the GUI.

## Usage

```bash
revdpd                        # start the GUI
revdpd gui system.data        # start the GUI with a CG data file (or a project .json)
revdpd info system.data       # list the molecule species of a CG data file
revdpd run project.json [--out DIR] [--minimize]     # run a saved project without GUI
```

### Workflow in the GUI

1. **CG system** – choose the data file and the atom style (`angle`/`full`/`auto`) and
   press *Load*. The species table lists every distinct molecule type with its count.
   Untick species you do not want in the output.
2. **All-atom template** – select a species and load its moltemplate `.lt` file.
3. **Mapping** – select a bead (click it in the CG panel, click its row, or press `1`–`9`)
   and click the heavy atoms it represents in the all-atom panel. Clicking an assigned
   atom again removes it; clicking an atom of another bead moves it.
   The bead centre can be the centre of mass (including hydrogens), the heavy-atom centre
   of mass or the heavy-atom centroid.
4. **Scale** – press *Estimate* (or enter the length of one DPD unit in Å). With
   *Overlay fit* the fitted template is drawn on top of the CG molecule; browse
   molecules with the spin box. The bead-fit RMSD is shown in the corner.
5. **Back-map system** – writes the files to the output folder and, if enabled, runs
   the LAMMPS minimisation.

Mouse: left-drag rotate, Ctrl+left-drag roll, right/middle-drag pan, wheel zoom,
double-click on empty space resets the view.

### Output files

| File | Content |
| --- | --- |
| `system.data` | all-atom LAMMPS data file (`atom_style full`, image flags) |
| `system.in.init` | units and styles taken from the force field |
| `system.in.settings` | includes `system.in.bonded` and `system.in.pair` |
| `system.min.in` | minimisation: soft push-off, then full force field → `system_min.data` |

Run it yourself with `lmp -in system.min.in` inside the output folder.

## Method

For every CG molecule *i* with bead positions **X**ᵢ (DPD units, made whole) the target
bead positions are **T**ᵢ = *s* **X**ᵢ, with *s* the scale in Å per DPD length unit. The
template bead centres **B** are computed from the mapped atoms. The rotation **R** and
translation **t** minimising Σₖ |**R** **B**ₖ + **t** − **T**ₖ|² (Kabsch) are applied to
all template atoms, so the head/tail direction and orientation of every CG molecule are
kept. When the beads are (nearly) collinear the rotation about the long axis is not
defined and is drawn at random (optional). In *rigid + per-bead shift* mode each atom is
additionally moved by the residual of its bead, which follows bent CG conformations at the
cost of distorted inter-bead bonds (fixed by the minimisation). The box is scaled by *s*.

Overlap removal looks for heavy atoms of different molecules closer than *d*ₘᵢₙ,
rotates linear molecules about their axis to the angle with the fewest contacts, and then
pushes the remaining pairs apart with rigid-body translations.

## Example: stearylamine

`examples/stearylamine_project.json` maps the 5-bead stearylamine of a
PC/cholesterol/CHEMS/SA DPD membrane (NC3–CA×4) onto the ATB GROMOS 54A7 all-atom
stearylamine (`N1 C18 C17 C16 | C15–C12 | C11–C9 | C8–C5 | C4–C1`). Paths in a project
file are relative to the project file; adjust `cg_path` and `aa_path` to your files.

## Roadmap

- more atom styles and all-atom input formats (PDB/GRO + ITP, LAMMPS data)
- fractional/shared atom mappings, per-bead rotations for flexible molecules
- solvent/ion insertion

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The LAMMPS test is skipped when no LAMMPS executable is found.

## License

MIT
