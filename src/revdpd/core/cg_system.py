"""Coarse-grained system analysis: split into molecules and group identical molecules into species."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from ..io.lammps_data import LammpsData


@dataclass
class CGSpecies:
    """A set of topologically identical CG molecules."""

    key: tuple                          # (type sequence, local bond list)
    bead_types: list[int]
    bead_names: list[str]
    bonds: list[tuple[int, int]]        # local bead indices
    instances: list[np.ndarray] = field(default_factory=list)   # arrays of atom row indices
    mol_ids: list[int] = field(default_factory=list)
    name: str = ""

    @property
    def n_beads(self) -> int:
        return len(self.bead_types)

    @property
    def count(self) -> int:
        return len(self.instances)

    def summary(self) -> str:
        runs: list[list] = []
        for b in self.bead_names:
            if runs and runs[-1][0] == b:
                runs[-1][1] += 1
            else:
                runs.append([b, 1])
        return "-".join(f"{b}x{c}" if c > 1 else b for b, c in runs)


class CGSystem:
    """A LAMMPS CG configuration split into molecules and species."""

    def __init__(self, data: LammpsData, split: str = "auto"):
        self.data = data
        self.box = data.box
        self._row_of_id = {int(a): k for k, a in enumerate(data.ids)}
        bonds = data.bonds()
        self.bond_rows = np.array(
            [[self._row_of_id[int(a)], self._row_of_id[int(b)]] for a, b in bonds[:, 1:3]],
            dtype=np.int64,
        ).reshape(-1, 2)
        self.adj: list[list[int]] = [[] for _ in range(data.n_atoms)]
        for a, b in self.bond_rows:
            self.adj[a].append(b)
            self.adj[b].append(a)

        if split == "auto":
            split = "molid" if np.any(data.mol != 0) else "bonds"
        self.split_mode = split
        groups = self._groups_by_molid() if split == "molid" else self._groups_by_bonds()
        self.species: list[CGSpecies] = self._build_species(groups)

    # ------------------------------------------------------------------ split
    def _groups_by_molid(self) -> list[np.ndarray]:
        """Group by molecule ID; beads with ID 0 (often solvent) are split by bonds instead."""
        zero = self.data.mol == 0
        order = np.lexsort((self.data.ids, self.data.mol))
        order = order[~zero[order]]
        mols = self.data.mol[order]
        cut = np.flatnonzero(np.diff(mols)) + 1
        groups = [g for g in np.split(order, cut) if len(g)]
        if zero.any():
            groups += [g for g in self._groups_by_bonds() if zero[g[0]]]
        return groups

    def _groups_by_bonds(self) -> list[np.ndarray]:
        n = self.data.n_atoms
        seen = np.zeros(n, dtype=bool)
        out = []
        for s in range(n):
            if seen[s]:
                continue
            comp = []
            dq = deque([s])
            seen[s] = True
            while dq:
                a = dq.popleft()
                comp.append(a)
                for b in self.adj[a]:
                    if not seen[b]:
                        seen[b] = True
                        dq.append(b)
            comp = np.array(sorted(comp, key=lambda r: self.data.ids[r]), dtype=np.int64)
            out.append(comp)
        return out

    def _build_species(self, groups: list[np.ndarray]) -> list[CGSpecies]:
        by_key: dict[tuple, CGSpecies] = {}
        for g in groups:
            local = {int(r): k for k, r in enumerate(g)}
            types = tuple(int(self.data.types[r]) for r in g)
            lb = []
            for r in g:
                for b in self.adj[r]:
                    if b in local and local[int(r)] < local[b]:
                        lb.append((local[int(r)], local[b]))
            lb = tuple(sorted(lb))
            key = (types, lb)
            sp = by_key.get(key)
            if sp is None:
                sp = CGSpecies(
                    key=key,
                    bead_types=list(types),
                    bead_names=[self.data.type_name(t) for t in types],
                    bonds=list(lb),
                )
                by_key[key] = sp
            sp.instances.append(g)
            sp.mol_ids.append(int(self.data.mol[g[0]]))
        species = sorted(by_key.values(), key=lambda s: (-s.count, s.n_beads))
        for k, sp in enumerate(species, 1):
            sp.name = f"S{k}: {sp.summary()}"
        return species

    # ------------------------------------------------------------ coordinates
    def unwrapped(self, rows: np.ndarray) -> np.ndarray:
        """Coordinates of one molecule made whole across periodic boundaries.

        Uses image flags when present; otherwise walks the bond graph and applies the
        minimum image convention along every bond (atoms not reachable by bonds are
        placed at the minimum image of the first atom).
        """
        d = self.data
        if d.has_image_flags and np.any(d.image[rows] != 0):
            h = self.box.h_matrix()
            return d.pos[rows] + d.image[rows] @ h.T
        pos = d.pos[rows]
        local = {int(r): k for k, r in enumerate(rows)}
        out = np.full_like(pos, np.nan)
        for start in range(len(rows)):
            if not np.isnan(out[start, 0]):
                continue
            if start == 0:
                out[0] = pos[0]
            else:
                out[start] = out[0] + self.box.minimum_image(pos[start] - out[0])
            dq = deque([start])
            while dq:
                a = dq.popleft()
                for b in self.adj[int(rows[a])]:
                    kb = local.get(b)
                    if kb is None or not np.isnan(out[kb, 0]):
                        continue
                    out[kb] = out[a] + self.box.minimum_image(pos[kb] - pos[a])
                    dq.append(kb)
        return out

    def instance_coords(self, species: CGSpecies, k: int) -> np.ndarray:
        return self.unwrapped(species.instances[k])

    def mean_bond_length(self, species: CGSpecies, max_instances: int = 500) -> float:
        """Median CG bond length of a species (DPD length units)."""
        if not species.bonds:
            return float("nan")
        lens = []
        ib = np.array(species.bonds)
        for k in range(min(max_instances, species.count)):
            x = self.instance_coords(species, k)
            lens.append(np.linalg.norm(x[ib[:, 0]] - x[ib[:, 1]], axis=1))
        return float(np.median(np.concatenate(lens)))
