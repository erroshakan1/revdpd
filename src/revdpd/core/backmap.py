"""Rigid (and optionally flexible) fitting of all-atom templates onto CG molecules."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.moltemplate import AAMolecule
from .cg_system import CGSpecies, CGSystem
from .mapping import BeadMapping, atom_owners, bead_centers


@dataclass
class BackmapSettings:
    scale: float = 10.0            # Angstrom per CG length unit
    mode: str = "rigid"            # "rigid" | "flex"
    flex_weight: float = 1.0       # fraction of the per-bead residual applied in flex mode
    random_spin: bool = True       # randomise rotation about the axis of (near-)linear molecules
    linear_threshold: float = 0.15  # s2/s1 of bead centres below which a molecule counts as linear
    seed: int = 2024


def kabsch(P: np.ndarray, Q: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R and translation t minimising sum w |R p + t - q|^2."""
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    pc = (P * w[:, None]).sum(0)
    qc = (Q * w[:, None]).sum(0)
    H = ((P - pc) * w[:, None]).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d if d != 0 else 1.0])
    R = Vt.T @ D @ U.T
    return R, qc - R @ pc


def rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def principal_axis(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(axis, singular values) of a point cloud."""
    Xc = X - X.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[0], s


def is_linear(points: np.ndarray, threshold: float) -> bool:
    if len(points) < 3:
        return True
    _, s = principal_axis(points)
    return s[0] == 0 or (s[1] / s[0]) < threshold


def estimate_scale(cg: CGSystem, species: CGSpecies, mol: AAMolecule, mapping: BeadMapping) -> float:
    """Angstrom per CG length unit from template bead distances vs. CG bond lengths."""
    B = bead_centers(mol, mapping)
    aa, cgl = [], []
    for i, j in species.bonds:
        if np.isnan(B[i, 0]) or np.isnan(B[j, 0]):
            continue
        aa.append(np.linalg.norm(B[i] - B[j]))
    if not aa:
        raise ValueError("need at least one CG bond between two mapped beads to estimate the scale")
    ib = np.array([(i, j) for i, j in species.bonds
                   if not (np.isnan(B[i, 0]) or np.isnan(B[j, 0]))])
    for k in range(min(300, species.count)):
        x = cg.instance_coords(species, k)
        cgl.append(np.linalg.norm(x[ib[:, 0]] - x[ib[:, 1]], axis=1))
    cg_all = np.concatenate(cgl)
    return float(np.mean(aa) / np.mean(cg_all))


class Fitter:
    """Places copies of an all-atom template onto CG molecule instances."""

    def __init__(self, mol: AAMolecule, mapping: BeadMapping, settings: BackmapSettings):
        self.mol = mol
        self.mapping = mapping
        self.s = settings
        self.mapped = np.array(mapping.mapped_beads(), dtype=np.int64)
        if len(self.mapped) == 0:
            raise ValueError("mapping is empty")
        self.B = bead_centers(mol, mapping)
        self.owner = atom_owners(mol, mapping)
        self.linear = is_linear(self.B[self.mapped], settings.linear_threshold)

    def instance_is_linear(self, cg_coords: np.ndarray) -> bool:
        """True when the spin about the long axis is undetermined for this instance."""
        return (self.linear or len(self.mapped) <= 2
                or is_linear(cg_coords[self.mapped], self.s.linear_threshold))

    def fit(self, cg_coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """All-atom coordinates (Angstrom) for one CG instance (CG units, unwrapped)."""
        T = cg_coords[self.mapped] * self.s.scale
        P = self.B[self.mapped]
        X = self.mol.pos
        if len(self.mapped) == 1:
            R = random_rotation(rng)
            t = T[0] - R @ P[0]
        elif len(self.mapped) == 2:
            R = _align_vectors(P[1] - P[0], T[1] - T[0])
            t = T.mean(0) - R @ P.mean(0)
        else:
            R, t = kabsch(P, T)
        Y = X @ R.T + t
        fitted_centers = self.B @ R.T + t
        if self.s.random_spin and self.instance_is_linear(cg_coords):
            axis, _ = principal_axis(T) if len(T) >= 2 else (np.array([0, 0, 1.0]), None)
            c = T.mean(0)
            Rs = rotation_about(axis, rng.uniform(0, 2 * np.pi))
            Y = (Y - c) @ Rs.T + c
            fitted_centers = (fitted_centers - c) @ Rs.T + c
        if self.s.mode == "flex":
            resid = np.zeros((self.mapping.n_beads, 3))
            resid[self.mapped] = T - fitted_centers[self.mapped]
            ok = self.owner >= 0
            Y[ok] += self.s.flex_weight * resid[self.owner[ok]]
        return Y

    def rmsd(self, cg_coords: np.ndarray, aa_coords: np.ndarray) -> float:
        T = cg_coords[self.mapped] * self.s.scale
        C = bead_centers(self.mol, self.mapping, aa_coords)[self.mapped]
        return float(np.sqrt(((C - T) ** 2).sum(1).mean()))


def _align_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0, 1, 0])
        return rotation_about(perp, np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def backmap_species(cg: CGSystem, species: CGSpecies, mol: AAMolecule, mapping: BeadMapping,
                    settings: BackmapSettings, rng: np.random.Generator | None = None,
                    progress=None) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Fit the template onto every instance.

    Returns (coords per molecule, bead RMSD per molecule, linear flag per molecule).
    """
    if mapping.n_beads != species.n_beads:
        raise ValueError(f"mapping has {mapping.n_beads} beads but species has {species.n_beads}")
    rng = rng or np.random.default_rng(settings.seed)
    fitter = Fitter(mol, mapping, settings)
    out, rmsd, lin = [], [], []
    n = species.count
    for k in range(n):
        x = cg.instance_coords(species, k)
        y = fitter.fit(x, rng)
        out.append(y)
        rmsd.append(fitter.rmsd(x, y))
        lin.append(fitter.instance_is_linear(x))
        if progress and (k % 50 == 0 or k == n - 1):
            progress(k + 1, n)
    return out, np.array(rmsd), np.array(lin)
