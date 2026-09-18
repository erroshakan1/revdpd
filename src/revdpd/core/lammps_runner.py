"""Run LAMMPS on the generated minimisation script."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi", "lmp_omp", "lammps")


def find_lammps() -> str | None:
    env = os.environ.get("LAMMPS_EXE")
    if env and shutil.which(env):
        return shutil.which(env)
    for c in CANDIDATES:
        p = shutil.which(c)
        if p:
            return p
    for p in (Path.home() / "lammps-install/bin/lmp", Path.home() / ".local/bin/lmp"):
        if p.exists():
            return str(p)
    return None


def run_lammps(exe: str, script: Path, workdir: Path, log=None, mpi: int = 1,
               stop=None) -> int:
    """Run ``exe -in script`` in ``workdir``; stream output lines to ``log``. Returns exit code."""
    cmd = [exe, "-in", script.name, "-log", "log.lammps"]
    if mpi > 1:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun:
            cmd = [mpirun, "-np", str(mpi)] + cmd
    log = log or print
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
        if stop is not None and stop():
            proc.terminate()
            break
    return proc.wait()
