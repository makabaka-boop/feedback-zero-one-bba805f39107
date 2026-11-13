"""Persistent homology of the Vietoris–Rips filtration over ``F_p``.

The complex holds simplices of dimension 0 to 3 (vertices, edges,
triangles, tetrahedra) and is ordered within each dimension by
``(birth scale, vertex tuple)``.  Boundary matrices are reduced
dimension by dimension: a boundary matrix is block upper triangular by
dimension, so column operations never mix dimensions and reducing the
three blocks separately is identical to reducing the whole matrix once.

Pairings ``(pivot row, column)`` of ``d_k`` kill the ``(k-1)``-simplex of
the row at the birth scale of the ``k``-simplex of the column.  For
``d_1`` the reduction also tracks the change of basis ``V``; every zero
column of the reduced ``d_1`` then gives an explicit cycle, and those
cycles form a basis of ``H_1`` at any query scale (restricted to the
classes whose bar is alive there).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from .complex import Filtration, Simplex, edge_key
from .linalg import Column, reduction

Bar = Tuple[int, Fraction, Optional[Fraction]]


@dataclass
class Interval:
    dimension: int
    birth: Fraction
    death: Optional[Fraction]


@dataclass
class Persistence:
    bars: List[Interval]
    # Positive edge index -> cycle as a sparse 1-chain over edge indices.
    edge_cycles: Dict[int, Column]
    # Positive edge index -> (birth, death) of its class.
    edge_intervals: Dict[int, Tuple[Fraction, Optional[Fraction]]]

    def cycles_alive_at(self, scale: Fraction) -> List[Column]:
        chosen = [
            (interval[0], edge_index)
            for edge_index, interval in self.edge_intervals.items()
            if interval[0] <= scale and (interval[1] is None or scale < interval[1])
        ]
        chosen.sort()
        return [self.edge_cycles[edge_index] for _birth, edge_index in chosen]


def _edge_boundary(edge: Tuple[int, int], prime: int) -> Column:
    """``[v1] - [v0]`` for the edge ``(v0, v1)`` with ``v0 < v1`` by id."""
    v0, v1 = edge
    column: Column = {v1: 1}
    minus_one = (-1) % prime
    if minus_one:
        column[v0] = (column.get(v0, 0) + minus_one) % prime
        if not column[v0]:
            del column[v0]
    return column


def _triangle_boundary(triangle: Simplex, edge_position: Dict[Tuple[int, int], int], prime: int) -> Column:
    """``[j,k] - [i,k] + [i,j]``."""
    i, j, k = triangle
    column: Column = {}
    for edge, sign in (((j, k), 1), ((i, k), -1), ((i, j), 1)):
        position = edge_position[edge_key(*edge)]
        value = (column.get(position, 0) + sign) % prime
        if value:
            column[position] = value
        else:
            column.pop(position, None)
    return column


def _tetrahedron_boundary(tetrahedron: Simplex, triangle_position: Dict[Simplex, int], prime: int) -> Column:
    """``[j,k,l] - [i,k,l] + [i,j,l] - [i,j,k]``."""
    i, j, k, l = tetrahedron
    column: Column = {}
    for triangle, sign in (
        ((j, k, l), 1),
        ((i, k, l), -1),
        ((i, j, l), 1),
        ((i, j, k), -1),
    ):
        position = triangle_position[triangle]
        value = (column.get(position, 0) + sign) % prime
        if value:
            column[position] = value
        else:
            column.pop(position, None)
    return column


def compute(
    filtration: Filtration,
    prime: int,
) -> Persistence:
    """Compute the reduced 0-, 1- and 2-dimensional barcode and H1 cycles."""
    vertices = filtration.of_dim(0)
    edges = filtration.of_dim(1)
    triangles = filtration.of_dim(2)
    tetrahedra = filtration.of_dim(3)

    edge_position = {edge: index for index, edge in enumerate(edges)}
    triangle_position = {triangle: index for index, triangle in enumerate(triangles)}

    # -- d_1: C_1 -> C_0, tracking V for cycle representatives ------------
    d1 = [_edge_boundary(edge, prime) for edge in edges]
    zero_columns1, pairs1, transform1 = reduction(d1, prime, track_transform=True)
    vertex_death: Dict[int, Fraction] = {
        vertex_row: filtration.birth[edges[column]] for vertex_row, column in pairs1
    }
    edge_cycles: Dict[int, Column] = {
        column: transform1[column] for column in zero_columns1
    }

    # -- d_2: C_2 -> C_1 ---------------------------------------------------
    d2 = [_triangle_boundary(triangle, edge_position, prime) for triangle in triangles]
    zero_columns2, pairs2, _ = reduction(d2, prime)
    edge_death: Dict[int, Fraction] = {
        edge_row: filtration.birth[triangles[column]] for edge_row, column in pairs2
    }
    positive_triangles = zero_columns2

    # -- d_3: C_3 -> C_2 ---------------------------------------------------
    d3 = [_tetrahedron_boundary(tetrahedron, triangle_position, prime) for tetrahedron in tetrahedra]
    _zero_columns3, pairs3, _ = reduction(d3, prime)
    triangle_death: Dict[int, Fraction] = {
        triangle_row: filtration.birth[tetrahedra[column]] for triangle_row, column in pairs3
    }

    # -- assemble intervals ------------------------------------------------
    bars: List[Interval] = []
    for vertex_row, vertex in enumerate(vertices):
        bars.append(Interval(0, filtration.birth[vertex], vertex_death.get(vertex_row)))

    edge_intervals: Dict[int, Tuple[Fraction, Optional[Fraction]]] = {}
    for edge_index in sorted(edge_cycles):
        interval = (
            filtration.birth[edges[edge_index]],
            edge_death.get(edge_index),
        )
        edge_intervals[edge_index] = interval
        bars.append(Interval(1, interval[0], interval[1]))

    for triangle_index in sorted(positive_triangles):
        triangle = triangles[triangle_index]
        bars.append(
            Interval(2, filtration.birth[triangle], triangle_death.get(triangle_index))
        )

    return Persistence(bars, edge_cycles, edge_intervals)
