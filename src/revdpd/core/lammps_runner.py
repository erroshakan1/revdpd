"""Run LAMMPS on the generated relaxation script."""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
from pathlib import Path

CANDIDATES = ("lmp", "lmp_serial", "lmp_mpi", "lmp_omp", "lammps")


class Cancelled(RuntimeError):
    """Raised when the user stops a running back-mapping."""


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


def build_command(exe: str, script: str, prefix: str = "", extra_args: str = "",
                  mpi: int = 1) -> list[str]:
    """Command line ``[prefix] exe -in script [extra_args]``.

    ``prefix`` is e.g. ``mpirun -np 4``; ``extra_args`` e.g. ``-sf omp -pk omp 8``.
    For old projects without a prefix, ``mpi > 1`` adds ``mpirun -np <mpi>``.
    """
    pre = shlex.split(prefix)
    if not pre and mpi > 1:
        pre = ["mpirun", "-np", str(mpi)]
    return pre + [exe, "-in", script] + shlex.split(extra_args)


def kill_process_tree(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """Terminate a process started in its own session, including children (mpirun ranks)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def run_lammps(cmd: list[str], workdir: Path, log=None, on_start=None) -> int:
    """Run ``cmd`` in ``workdir``; stream output lines to ``log``. Returns the exit code.

    ``on_start(proc)`` receives the Popen object so that another thread can kill it
    with :func:`kill_process_tree` at any time.
    """
    log = log or print
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    if on_start:
        on_start(proc)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log(line.rstrip())
    finally:
        proc.stdout.close()
    return proc.wait()
