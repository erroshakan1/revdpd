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
    mode: str = "rigid"            # "rigid" | "flex" | "fragment"
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


FIT_MODES = ("rigid", "flex", "fragment")


class Fitter:
    """Places copies of an all-atom template onto CG molecule instances.

    Modes
    -----
    rigid
        one Kabsch fit of all bead centres; the template geometry is kept exactly.
    flex
        rigid fit, then every atom is shifted by the residual of its bead.
    fragment
        rigid fit, then every bead's atom fragment is rotated separately so that the
        directions to its bonded neighbour beads match the CG molecule, and centred on
        its bead (per-fragment alignment in the spirit of CG2AT). Follows bent CG
        conformations; bonds between fragments are repaired by the relaxation.
    """

    def __init__(self, mol: AAMolecule, mapping: BeadMapping, settings: BackmapSettings,
                 cg_bonds: list[tuple[int, int]] | None = None):
        self.mol = mol
        self.mapping = mapping
        self.s = settings
        self.mapped = np.array(mapping.mapped_beads(), dtype=np.int64)
        if len(self.mapped) == 0:
            raise ValueError("mapping is empty")
        self.B = bead_centers(mol, mapping)
        self.owner = atom_owners(mol, mapping)
        self.linear = is_linear(self.B[self.mapped], settings.linear_threshold)
        mapped = set(self.mapped.tolist())
        self.nbr: dict[int, list[int]] = {int(k): [] for k in self.mapped}
        for i, j in cg_bonds or []:
            if i in mapped and j in mapped:
                self.nbr[i].append(j)
                self.nbr[j].append(i)
        # second neighbours help when a bead has a single bonded neighbour
        self.nbr2: dict[int, list[int]] = {}
        for k, nb in self.nbr.items():
            second = {m for j in nb for m in self.nbr[j]} - set(nb) - {k}
            self.nbr2[k] = sorted(second)
        self.frag = {int(k): np.flatnonzero(self.owner == k) for k in self.mapped}

    def instance_is_linear(self, cg_coords: np.ndarray) -> bool:
        """True when the spin about the long axis is undetermined for this instance."""
        return (self.linear or len(self.mapped) <= 2
                or is_linear(cg_coords[self.mapped], self.s.linear_threshold))

    def global_fit(self, cg_coords: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Rotation and translation of the whole template (including the random spin)."""
        T = cg_coords[self.mapped] * self.s.scale
        P = self.B[self.mapped]
        if len(self.mapped) == 1:
            R = random_rotation(rng)
            t = T[0] - R @ P[0]
        elif len(self.mapped) == 2:
            R = _align_vectors(P[1] - P[0], T[1] - T[0])
            t = T.mean(0) - R @ P.mean(0)
        else:
            R, t = kabsch(P, T)
        if self.s.random_spin and self.instance_is_linear(cg_coords) and len(T) >= 2:
            axis, _ = principal_axis(T)
            c = T.mean(0)
            Rs = rotation_about(axis, rng.uniform(0, 2 * np.pi))
            R = Rs @ R
            t = Rs @ (t - c) + c
        return R, t

    def fit(self, cg_coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """All-atom coordinates (Angstrom) for one CG instance (CG units, unwrapped)."""
        R, t = self.global_fit(cg_coords, rng)
        Y = self.mol.pos @ R.T + t
        Tall = cg_coords * self.s.scale
        if self.s.mode == "flex":
            fitted = self.B @ R.T + t
            resid = np.zeros((self.mapping.n_beads, 3))
            resid[self.mapped] = Tall[self.mapped] - fitted[self.mapped]
            ok = self.owner >= 0
            Y[ok] += self.s.flex_weight * resid[self.owner[ok]]
        elif self.s.mode == "fragment":
            for k, atoms in self.frag.items():
                if len(atoms) == 0:
                    continue
                Q = self._local_rotation(k, R, Tall)
                Y[atoms] = (self.mol.pos[atoms] - self.B[k]) @ (Q @ R).T + Tall[k]
        return Y

    def _local_rotation(self, k: int, R: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Extra rotation for fragment ``k`` mapping template neighbour directions onto CG ones."""
        nb = self.nbr[k]
        if len(nb) < 2:
            nb = nb + self.nbr2[k][:2]
        if not nb:
            return np.eye(3)
        U = (self.B[nb] - self.B[k]) @ R.T          # template directions after the global fit
        V = T[nb] - T[k]                            # CG directions
        U /= np.linalg.norm(U, axis=1)[:, None]
        V /= np.linalg.norm(V, axis=1)[:, None]
        if len(nb) >= 2:
            _, s, _ = np.linalg.svd(U)
            if s[1] > 0.15 * s[0]:
                H = U.T @ V
                Us, _, Vt = np.linalg.svd(H)
                d = np.sign(np.linalg.det(Vt.T @ Us.T)) or 1.0
                return Vt.T @ np.diag([1.0, 1.0, d]) @ Us.T
        # (nearly) collinear neighbours: smallest rotation aligning the mean direction
        sign = np.sign(U @ U[0])[:, None]
        return _align_vectors((U * sign).sum(0), (V * sign).sum(0))

    def rmsd(self, cg_coords: np.ndarray, aa_coords: np.ndarray) -> float:
        T = cg_coords[self.mapped] * self.s.scale
        C = bead_centers(self.mol, self.mapping, aa_coords)[self.mapped]
        return float(np.sqrt(((C - T) ** 2).sum(1).mean()))


def molecular_volume(mol: AAMolecule) -> float:
    """Volume (A^3) of one molecule at a mass density of 1 g/cm^3 (29.9 A^3 for water)."""
    return float(mol.masses.sum() / 0.60221)


def place_cluster(center: np.ndarray, mol: AAMolecule, n: int, rng: np.random.Generator,
                  d_min: float = 2.6, max_tries: int = 2000) -> list[np.ndarray]:
    """``n`` randomly oriented copies of a small molecule packed around ``center`` (A).

    Used for CG beads that represent several molecules (e.g. a DPD water bead with
    N_m water molecules). Molecule centres are drawn uniformly in a sphere whose volume
    equals ``n`` molecular volumes, keeping centres at least ``d_min`` apart when possible.
    """
    com = (mol.pos * mol.masses[:, None]).sum(0) / mol.masses.sum()
    radius = (3 * n * molecular_volume(mol) / (4 * np.pi)) ** (1 / 3)
    centres: list[np.ndarray] = []
    tries = 0
    while len(centres) < n:
        p = rng.normal(size=3)
        p *= radius * rng.uniform() ** (1 / 3) / np.linalg.norm(p)
        tries += 1
        if tries < max_tries and any(np.linalg.norm(p - c) < d_min for c in centres):
            continue
        centres.append(p)
    # keep the cluster centred exactly on the bead
    cm = np.mean(centres, axis=0)
    out = []
    for c in centres:
        R = random_rotation(rng)
        out.append((mol.pos - com) @ R.T + center + c - cm)
    return out


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
                    progress=None, copies: int = 1) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Fit the template onto every instance.

    ``copies`` > 1 is allowed for single-bead species only: every bead is replaced by
    that many molecules (e.g. N_m water molecules per DPD water bead).

    Returns (coords per all-atom molecule, bead RMSD per CG molecule, linear flag per
    all-atom molecule).
    """
    if mapping.n_beads != species.n_beads:
        raise ValueError(f"mapping has {mapping.n_beads} beads but species has {species.n_beads}")
    if copies > 1 and species.n_beads != 1:
        raise ValueError("several molecules per bead are only supported for single-bead species")
    rng = rng or np.random.default_rng(settings.seed)
    n = species.count
    out, rmsd, lin = [], [], []
    if copies > 1:
        for k in range(n):
            x = cg.instance_coords(species, k)[0] * settings.scale
            out += place_cluster(x, mol, copies, rng)
            if progress and (k % 500 == 0 or k == n - 1):
                progress(k + 1, n)
        return out, np.zeros(n), np.zeros(len(out), dtype=bool)
    fitter = Fitter(mol, mapping, settings, species.bonds)
    for k in range(n):
        x = cg.instance_coords(species, k)
        y = fitter.fit(x, rng)
        out.append(y)
        rmsd.append(fitter.rmsd(x, y))
        lin.append(fitter.instance_is_linear(x))
        if progress and (k % 50 == 0 or k == n - 1):
            progress(k + 1, n)
    return out, np.array(rmsd), np.array(lin)
