"""Vietoris–Rips building blocks.

Vertices (0-simplices) enter at scale 0 and are always present.  Every
subset of two, three or four vertices enters at the *maximum pairwise
squared Euclidean distance* among its members, provided that value is at
most ``cutoff``; otherwise it does not enter at all.  Distances are exact
:class:`fractions.Fraction` values, so ties and threshold comparisons are
exact — no floating-point rounding can flip them.

Only simplices of dimension at most three exist (edges, triangles,
tetrahedra), exactly as the input format prescribes.  Enumeration uses
adjacency lists, so the work is proportional to the number of simplices
that actually survive the cutoff rather than to ``C(n, 4)``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

from .schema import Point

# A simplex is identified by its ascending tuple of point indices, where
# indices refer to the id-sorted point order.
Simplex = Tuple[int, ...]


def squared_distance(a: Point, b: Point) -> Fraction:
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return dx * dx + dy * dy + dz * dz


def edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass
class Filtration:
    """The complex, ordered by ``(birth scale, vertex tuple)`` per dimension.

    Ordering simplices with equal birth by their vertex tuple gives a
    valid filtration order; all simplices sharing a birth scale are
    simultaneously present in the sublevel complexes, so the barcode does
    not depend on the tie-break.
    """

    points: List[Point]
    simplices: Dict[int, List[Simplex]]
    birth: Dict[Simplex, Fraction]
    distances: Dict[Tuple[int, int], Fraction] = field(default_factory=dict)

    def of_dim(self, dim: int) -> List[Simplex]:
        return self.simplices.get(dim, [])

    def edges(self) -> List[Tuple[int, int]]:
        return list(self.simplices.get(1, []))


def pairwise_distances(points: Sequence[Point]) -> Dict[Tuple[int, int], Fraction]:
    distances: Dict[Tuple[int, int], Fraction] = {}
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances[(i, j)] = squared_distance(points[i], points[j])
    return distances


def build(points: Sequence[Point], cutoff: Fraction) -> Filtration:
    """Enumerate every simplex whose birth scale is at most ``cutoff``."""
    n = len(points)
    distances = pairwise_distances(points)

    vertices: List[Simplex] = [(i,) for i in range(n)]
    edges: List[Simplex] = []
    triangles: List[Simplex] = []
    tetrahedra: List[Simplex] = []

    neighbours: List[set] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if distances[(i, j)] <= cutoff:
                edges.append((i, j))
                neighbours[i].add(j)
                neighbours[j].add(i)

    # A triangle needs all three edges; enumerate j < k over common
    # neighbours of i, which visits each triangle exactly once.
    for i in range(n):
        for j in sorted(k for k in neighbours[i] if k > i):
            for k in sorted(neighbours[i] & neighbours[j]):
                if k > j:
                    triangles.append((i, j, k))

    # A tetrahedron needs all six edges; enumerate l over the common
    # neighbours of i, j and k.
    for i, j, k in triangles:
        for l in sorted(neighbours[i] & neighbours[j] & neighbours[k]):
            if l > k:
                tetrahedra.append((i, j, k, l))

    birth: Dict[Simplex, Fraction] = {vertex: Fraction(0) for vertex in vertices}

    # Comparing :class:`Fraction` values in a sort is expensive, so each
    # distinct distance is mapped to its integer rank.  Sorting by rank is
    # exactly sorting by value, and the rank also makes ``max`` over the
    # edges of a simplex a plain integer maximum.
    distinct = sorted(set(distances.values()))
    rank_of = {value: rank for rank, value in enumerate(distinct)}

    def rank_of_simplex(subset: Simplex) -> int:
        return max(rank_of[distances[(a, b)]] for a, b in itertools.combinations(subset, 2))

    rank: Dict[Simplex, int] = {vertex: -1 for vertex in vertices}
    for simplex in edges:
        rank[simplex] = rank_of[distances[simplex]]
        birth[simplex] = distances[simplex]
    for simplex in triangles:
        rank[simplex] = rank_of_simplex(simplex)
        birth[simplex] = distinct[rank[simplex]]
    for simplex in tetrahedra:
        rank[simplex] = rank_of_simplex(simplex)
        birth[simplex] = distinct[rank[simplex]]

    simplices = {0: vertices, 1: edges, 2: triangles, 3: tetrahedra}
    for dim, items in simplices.items():
        items.sort(key=lambda simplex: (rank[simplex], simplex))

    return Filtration(list(points), simplices, birth, distances)
