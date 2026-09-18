"""Command line interface.

    revdpd                       start the GUI
    revdpd gui [file]            start the GUI (optionally opening a CG data file or project .json)
    revdpd run project.json      run a saved project without the GUI
    revdpd info cg.data          list the molecule species of a CG data file
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="revdpd", description="DPD -> all-atom back-mapping")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="start the graphical interface")
    g.add_argument("file", nargs="?", help="CG data file or project .json")
    r = sub.add_parser("run", help="run a saved project headless")
    r.add_argument("project")
    r.add_argument("--out", help="override output folder")
    r.add_argument("--minimize", action="store_true", help="run the LAMMPS minimisation")
    i = sub.add_parser("info", help="list molecule species of a CG data file")
    i.add_argument("data")
    i.add_argument("--style", default="auto")
    args = ap.parse_args(argv)

    if args.cmd in (None, "gui"):
        from .gui.app import main as gui_main
        return gui_main([sys.argv[0]] + ([args.file] if getattr(args, "file", None) else []))
    if args.cmd == "run":
        from .core.pipeline import Project, run_project
        p = Project.load(args.project)
        if args.out:
            p.out_dir = args.out
        if args.minimize:
            p.minimize.enabled = True
        res = run_project(p)
        return 0 if (res.lammps_exit in (None, 0)) else 1
    if args.cmd == "info":
        from .core.cg_system import CGSystem
        from .io.lammps_data import read_lammps_data
        d = read_lammps_data(args.data, args.style)
        cg = CGSystem(d)
        print(f"{d.n_atoms} beads, atom style {d.atom_style}, box {d.box.lengths.tolist()}")
        for sp in cg.species:
            print(f"  {sp.name:60s} {sp.count:6d} molecules  bond length {cg.mean_bond_length(sp, 50):.3f}")
        return 0
    return 1


def entry() -> int:
    try:
        return main()
    except (ValueError, OSError) as exc:
        print(f"revdpd: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entry())
