"""Atom-to-bead mapping between an all-atom template and a CG species."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..io.moltemplate import AAMolecule

CENTER_MODES = ("com", "heavy_com", "centroid")


@dataclass
class BeadMapping:
    """``beads[k]`` lists the all-atom (heavy) atom indices represented by CG bead ``k``.

    Hydrogens are never assigned explicitly; they follow the heavy atom they are
    bonded to (see :func:`atom_owners`).
    """

    n_beads: int
    beads: list[list[int]] = field(default_factory=list)
    center: str = "com"

    def __post_init__(self):
        if not self.beads:
            self.beads = [[] for _ in range(self.n_beads)]

    # -------------------------------------------------------------- editing
    def bead_of(self, atom: int) -> int:
        for k, lst in enumerate(self.beads):
            if atom in lst:
                return k
        return -1

    def toggle(self, bead: int, atom: int) -> None:
        """Assign ``atom`` to ``bead`` (moving it from any other bead) or unassign it."""
        cur = self.bead_of(atom)
        if cur >= 0:
            self.beads[cur].remove(atom)
        if cur != bead:
            self.beads[bead].append(atom)
            self.beads[bead].sort()

    def assign(self, bead: int, atoms) -> None:
        for a in atoms:
            cur = self.bead_of(a)
            if cur >= 0:
                self.beads[cur].remove(a)
            self.beads[bead].append(a)
        self.beads[bead] = sorted(set(self.beads[bead]))

    def clear(self, bead: int | None = None) -> None:
        if bead is None:
            self.beads = [[] for _ in range(self.n_beads)]
        else:
            self.beads[bead] = []

    def mapped_beads(self) -> list[int]:
        return [k for k, lst in enumerate(self.beads) if lst]

    def validate(self, mol: AAMolecule) -> list[str]:
        """Human-readable warnings (empty list = fine)."""
        msgs = []
        empty = [k for k, b in enumerate(self.beads) if not b]
        if len(empty) == self.n_beads:
            return ["No atoms are mapped yet."]
        if empty:
            msgs.append(f"Beads without atoms: {', '.join(str(k + 1) for k in empty)} (ignored in fitting).")
        heavy = set(np.flatnonzero(mol.heavy_mask()).tolist())
        assigned = {a for b in self.beads for a in b}
        missing = heavy - assigned
        if missing:
            names = ", ".join(mol.atom_names[a] for a in sorted(missing))
            msgs.append(f"Unassigned heavy atoms (follow their neighbours): {names}.")
        if len(self.mapped_beads()) < 3:
            msgs.append("Fewer than 3 mapped beads: rotation about the bead axis is undetermined.")
        return msgs

    # ------------------------------------------------------------ serialise
    def to_dict(self, mol: AAMolecule) -> dict:
        return {"n_beads": self.n_beads, "center": self.center,
                "beads": [[mol.atom_names[a] for a in b] for b in self.beads]}

    @classmethod
    def from_dict(cls, d: dict, mol: AAMolecule) -> "BeadMapping":
        idx = {n: i for i, n in enumerate(mol.atom_names)}
        beads = []
        for b in d["beads"]:
            unknown = [n for n in b if n not in idx]
            if unknown:
                raise ValueError(f"mapping refers to atoms not in {mol.name}: {unknown}")
            beads.append(sorted(idx[n] for n in b))
        return cls(n_beads=int(d["n_beads"]), beads=beads, center=d.get("center", "com"))


def atom_owners(mol: AAMolecule, mapping: BeadMapping) -> np.ndarray:
    """Bead index owning every atom (multi-source BFS from assigned atoms; -1 if unreachable)."""
    owner = np.full(mol.n_atoms, -1, dtype=np.int64)
    adj = mol.neighbors()
    dq: deque[int] = deque()
    for k, lst in enumerate(mapping.beads):
        for a in lst:
            owner[a] = k
            dq.append(a)
    while dq:
        a = dq.popleft()
        for b in adj[a]:
            if owner[b] < 0:
                owner[b] = owner[a]
                dq.append(b)
    return owner


def bead_centers(mol: AAMolecule, mapping: BeadMapping, coords: np.ndarray | None = None) -> np.ndarray:
    """Centers of every bead in the all-atom template (NaN rows for empty beads)."""
    x = mol.pos if coords is None else coords
    out = np.full((mapping.n_beads, 3), np.nan)
    owner = atom_owners(mol, mapping) if mapping.center == "com" else None
    for k, lst in enumerate(mapping.beads):
        if not lst:
            continue
        if mapping.center == "com":
            sel = np.flatnonzero(owner == k)
            w = mol.masses[sel]
        elif mapping.center == "heavy_com":
            sel = np.array(lst)
            w = mol.masses[sel]
        else:
            sel = np.array(lst)
            w = np.ones(len(sel))
        out[k] = (x[sel] * w[:, None]).sum(0) / w.sum()
    return out


def auto_linear_mapping(mol: AAMolecule, n_beads: int) -> BeadMapping:
    """Guess a mapping for chain-like molecules.

    The longest heavy-atom path is split into ``n_beads`` consecutive chunks of
    (almost) equal size; side heavy atoms join the chunk of the path atom they hang
    from. The path starts at the end closest to a hetero atom, so that e.g. a head
    group maps to bead 1. Use :func:`reverse_mapping` to flip it.
    """
    heavy = np.flatnonzero(mol.heavy_mask())
    hs = set(heavy.tolist())
    adj = [[b for b in nb if b in hs] for nb in mol.neighbors()]

    def bfs(src):
        dist = {src: 0}
        prev = {src: -1}
        dq = deque([src])
        while dq:
            a = dq.popleft()
            for b in adj[a]:
                if b not in dist:
                    dist[b] = dist[a] + 1
                    prev[b] = a
                    dq.append(b)
        far = max(dist, key=dist.get)
        return far, prev

    a, _ = bfs(int(heavy[0]))
    b, prev = bfs(a)
    path = [b]
    while prev[path[-1]] != -1:
        path.append(prev[path[-1]])

    def hetero_dist(end):
        dist = {end: 0}
        dq = deque([end])
        while dq:
            x = dq.popleft()
            if mol.elements[x] not in ("C", "H"):
                return dist[x]
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    dq.append(y)
        return 10**9

    if hetero_dist(path[-1]) < hetero_dist(path[0]):
        path.reverse()

    # attach side atoms to their path atom
    on_path = {p: i for i, p in enumerate(path)}
    anchor = dict(on_path)
    dq = deque(path)
    while dq:
        x = dq.popleft()
        for y in adj[x]:
            if y not in anchor:
                anchor[y] = anchor[x]
                dq.append(y)
    bounds = np.linspace(0, len(path), n_beads + 1)
    chunk_of_pos = np.searchsorted(bounds, np.arange(len(path)) + 0.5) - 1
    m = BeadMapping(n_beads=n_beads)
    for atom, p in anchor.items():
        m.beads[int(chunk_of_pos[p])].append(atom)
    m.beads = [sorted(b) for b in m.beads]
    return m


def reverse_mapping(m: BeadMapping) -> BeadMapping:
    return BeadMapping(n_beads=m.n_beads, beads=[list(b) for b in reversed(m.beads)], center=m.center)
