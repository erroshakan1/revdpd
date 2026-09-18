"""Simple rigid-body overlap removal between back-mapped molecules."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..io.lammps_data import Box
from .backmap import principal_axis, rotation_about


def _wrap(x: np.ndarray, box: Box) -> np.ndarray:
    L = box.lengths
    return np.mod(x - box.lo, L)


def count_overlaps(mols: list[np.ndarray], masks: list[np.ndarray], box: Box, d_min: float) -> int:
    X, mid = _flatten(mols, masks)
    return len(_pairs(X, mid, box, d_min))


def _flatten(mols, masks):
    X = np.concatenate([m[k] for m, k in zip(mols, masks)])
    mid = np.concatenate([np.full(int(k.sum()), i) for i, k in enumerate(masks)])
    return X, mid


def _pairs(X, mid, box, d_min):
    L = box.lengths
    tree = cKDTree(_wrap(X, box), boxsize=L)
    p = tree.query_pairs(d_min, output_type="ndarray")
    if len(p):
        p = p[mid[p[:, 0]] != mid[p[:, 1]]]
    return p


def remove_overlaps(mols: list[np.ndarray], masks: list[np.ndarray], box: Box,
                    d_min: float = 2.5, max_iter: int = 60, max_step: float = 1.0,
                    spin: list[bool] | None = None, n_spin: int = 12,
                    rng: np.random.Generator | None = None, log=None,
                    margin: float = 1.1) -> list[np.ndarray]:
    """Push overlapping molecules apart as rigid bodies.

    ``mols`` are per-molecule coordinates (Angstrom, unwrapped); ``masks`` select the
    atoms used for the overlap test (typically heavy atoms). Molecules flagged in
    ``spin`` (near-linear ones) are first rotated about their long axis to the angle
    with the fewest contacts. Only translations/rotations are applied, so internal
    geometry and orientation of the long axis are preserved. Orthogonal boxes only.
    """
    if box.triclinic:
        raise ValueError("overlap removal currently supports orthogonal boxes only")
    rng = rng or np.random.default_rng(0)
    mols = [m.copy() for m in mols]
    L = box.lengths
    log = log or (lambda *_: None)

    X, mid = _flatten(mols, masks)
    pairs = _pairs(X, mid, box, d_min)
    log(f"overlap check (d < {d_min:.2f} A): {len(pairs)} close contacts between molecules")
    if len(pairs) == 0:
        return mols

    # ---- stage 1: spin search for linear molecules
    if spin is not None and any(spin):
        involved = np.unique(mid[pairs.ravel()])
        tree = cKDTree(_wrap(X, box), boxsize=L)
        angles = np.linspace(0, 2 * np.pi, n_spin, endpoint=False)
        changed = 0
        for i in involved:
            if not spin[i]:
                continue
            m = mols[i]
            sel = masks[i]
            axis, _ = principal_axis(m[sel])
            c = m[sel].mean(0)
            best, best_n = None, None
            for a in angles:
                R = rotation_about(axis, a)
                y = (m[sel] - c) @ R.T + c
                hits = tree.query_ball_point(_wrap(y, box), d_min)
                n = sum(1 for h in hits for j in h if mid[j] != i)
                if best_n is None or n < best_n:
                    best, best_n = a, n
            if best:
                R = rotation_about(axis, best)
                mols[i] = (m - c) @ R.T + c
                changed += 1
        X, mid = _flatten(mols, masks)
        pairs = _pairs(X, mid, box, d_min)
        log(f"  after spin search ({changed} molecules rotated): {len(pairs)} contacts")

    # ---- stage 2: rigid translations
    for it in range(max_iter):
        if len(pairs) == 0:
            break
        d = X[pairs[:, 0]] - X[pairs[:, 1]]
        d -= L * np.round(d / L)
        r = np.linalg.norm(d, axis=1)
        r = np.where(r < 1e-6, 1e-6, r)
        u = d / r[:, None]
        tiny = r < 1e-5
        if np.any(tiny):
            u[tiny] = rng.normal(size=(int(tiny.sum()), 3))
            u[tiny] /= np.linalg.norm(u[tiny], axis=1)[:, None]
        push = 0.5 * (margin * d_min - r)[:, None] * u
        disp = np.zeros((len(mols), 3))
        np.add.at(disp, mid[pairs[:, 0]], push)
        np.add.at(disp, mid[pairs[:, 1]], -push)
        nrm = np.linalg.norm(disp, axis=1)
        scale = np.where(nrm > max_step, max_step / np.maximum(nrm, 1e-12), 1.0)
        disp *= scale[:, None]
        for i in np.flatnonzero(nrm > 0):
            mols[i] += disp[i]
        X, mid = _flatten(mols, masks)
        pairs = _pairs(X, mid, box, d_min)
        if it % 5 == 0 or len(pairs) == 0:
            log(f"  iteration {it + 1}: {len(pairs)} contacts")
    log(f"overlap removal finished: {len(pairs)} contacts remain")
    return mols
