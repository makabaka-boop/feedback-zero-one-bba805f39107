"""Exact sparse linear algebra over ``F_p``.

Boundary matrices are stored as lists of sparse columns, each a ``dict``
mapping a row index to a nonzero coefficient in ``0 .. p-1``.  The
persistent reduction is the standard left-to-right column reduction.
Because column operations never touch rows above the pivot, a boundary
matrix that is block upper triangular by dimension can be reduced
dimension by dimension; the caller does exactly that.

Pivot columns are stored normalised so their pivot entry equals ``1``,
which removes a modular inverse from the inner loop.

``rank_mod_p`` and ``nullspace_basis`` use a dense row-echelon algorithm
rather than the sparse column reduction, so the acceptance checker
verifies results along a code path independent of the main computation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

Column = Dict[int, int]


def inverse(value: int, prime: int) -> int:
    """Multiplicative inverse modulo a prime."""
    value %= prime
    if value == 0:
        raise ZeroDivisionError("zero has no inverse")
    return pow(value, prime - 2, prime)


def subtract(target: Column, factor: int, source: Column, prime: int) -> None:
    """``target -= factor * source`` in place, pruning zero entries."""
    factor %= prime
    if not factor:
        return
    for row, coeff in source.items():
        new = (target.get(row, 0) - factor * coeff) % prime
        if new:
            target[row] = new
        else:
            target.pop(row, None)


def scale_column(column: Column, factor: int, prime: int) -> Column:
    """``factor * column`` over ``F_p`` with zero entries pruned."""
    factor %= prime
    if factor == 1:
        return column
    return {row: (coeff * factor) % prime for row, coeff in column.items()}


def reduction(
    columns: Sequence[Column],
    prime: int,
    *,
    track_transform: bool = False,
) -> Tuple[List[int], List[Tuple[int, int]], Dict[int, Column]]:
    """Reduce ``columns`` left to right over ``F_p``.

    Returns ``(zero_column_indices, pivot_pairs, transform)``:

    * ``zero_column_indices`` — columns that reduced to zero, ascending;
    * ``pivot_pairs`` — ``(pivot_row, column_index)`` in column order;
    * ``transform`` — for zero columns only, the expression of that zero
      column in the original columns (``R = D V``); empty unless
      ``track_transform``.
    """
    pivot_column: Dict[int, Column] = {}
    pivot_transform: Dict[int, Column] = {}
    zero_columns: List[int] = []
    pivot_pairs: List[Tuple[int, int]] = []
    transform: Dict[int, Column] = {}

    for j, source in enumerate(columns):
        column: Column = dict(source)
        expression: Dict[int, int] = {j: 1} if track_transform else {}

        while column:
            low = max(column)
            owner_column = pivot_column.get(low)
            if owner_column is None:
                break
            factor = column[low]  # pivot entry of the stored column is 1
            subtract(column, factor, owner_column, prime)
            if track_transform:
                subtract(expression, factor, pivot_transform[low], prime)

        if column:
            low = max(column)
            # Normalising the reduced column by ``factor`` requires scaling
            # the tracked expression by exactly the same factor, otherwise
            # ``R = D V`` no longer holds.
            factor = inverse(column[low], prime)
            pivot_column[low] = scale_column(column, factor, prime)
            if track_transform:
                pivot_transform[low] = scale_column(expression, factor, prime)
            pivot_pairs.append((low, j))
        else:
            zero_columns.append(j)
            if track_transform:
                transform[j] = expression

    if not track_transform:
        return zero_columns, pivot_pairs, {}
    return zero_columns, pivot_pairs, transform


def rank_mod_p(rows: Iterable[Sequence[int]], prime: int, width: int) -> int:
    """Rank of a dense matrix over ``F_p`` by row echelon (independent path)."""
    basis: Dict[int, List[int]] = {}
    rank = 0
    for raw in rows:
        row = list(raw) + [0] * (width - len(raw))
        for col in range(width):
            value = row[col] % prime
            if not value:
                continue
            pivot_row = basis.get(col)
            if pivot_row is not None:
                factor = value  # pivot entry of the basis row is 1
                for k in range(col, width):
                    row[k] = (row[k] - factor * pivot_row[k]) % prime
            else:
                inv = inverse(value, prime)
                row = [(entry * inv) % prime for entry in row]
                basis[col] = row
                rank += 1
                break
    return rank


def nullspace_basis(matrix: Sequence[Column], n_columns: int, prime: int) -> List[Column]:
    """Basis of ``{x : matrix @ x = 0}`` for sparse columns (independent path)."""
    equations: Dict[int, Dict[int, int]] = {}
    for j, column in enumerate(matrix):
        for row, coeff in column.items():
            value = coeff % prime
            if value:
                equations.setdefault(row, {})[j] = value

    row_list = [dict(row) for row in equations.values()]
    pivots: Dict[int, int] = {}
    pivot_count = 0
    for col in range(n_columns):
        selected = None
        for i in range(pivot_count, len(row_list)):
            if row_list[i].get(col, 0) % prime:
                selected = i
                break
        if selected is None:
            continue
        row_list[pivot_count], row_list[selected] = row_list[selected], row_list[pivot_count]
        pivot_row = row_list[pivot_count]
        inv = inverse(pivot_row[col], prime)
        for key in list(pivot_row):
            pivot_row[key] = (pivot_row[key] * inv) % prime
        for i, other in enumerate(row_list):
            if i == pivot_count:
                continue
            factor = other.get(col, 0) % prime
            if factor:
                subtract(other, factor, pivot_row, prime)
        pivots[col] = pivot_count
        pivot_count += 1
        if pivot_count == len(row_list):
            break

    basis: List[Column] = []
    for free in (c for c in range(n_columns) if c not in pivots):
        vector: Column = {free: 1}
        for pivot_col, row_index in pivots.items():
            value = row_list[row_index].get(free, 0) % prime
            if value:
                vector[pivot_col] = (-value) % prime
        basis.append(vector)
    return basis
