#!/usr/bin/env python3
"""Independent acceptance check for the persistent-homology backend.

This script deliberately shares **no** computation code with the
``persist`` package.  It reads the input document and the produced output
document and re-derives everything with dense linear algebra over
:class:`fractions.Fraction` and over ``F_p``:

* it builds the Vietoris–Rips complex from scratch and computes Betti
  numbers at each query by dense rank computations of boundary matrices
  (``b_k = dim ker d_k - rank d_{k+1}``);
* it checks that every reported bar is consistent with those Betti
  numbers (the barcode determines the Betti function, and vice versa the
  reported bars must reproduce every queried Betti number);
* it checks that every reported H1 cycle is a cycle with the exact
  coefficients by parsing it back into a vector and verifying
  ``d_1 x = 0``;
* it checks that the reported cycles are linearly independent modulo
  boundaries, and that their count equals the reported ``b_1`` (so they
  span ``H_1``);
* it checks sorting, reduced rationals, omitted zero-length bars, and the
  ``null``-last ordering.

Usage::

    python3 tools/verify.py INPUT.json OUTPUT.json            # pass/fail
    python3 tools/verify.py INPUT.json OUTPUT.json --ranks    # + rank table

Exit code 0 means every check passed; otherwise a message is printed and
the exit code is 1.
"""

from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

Simplex = Tuple[int, ...]


# --------------------------------------------------------------------------
# input parsing (independent of persist.schema: plain json + manual checks)
# --------------------------------------------------------------------------
def parse_ratio(text: object) -> Fraction:
    if not isinstance(text, str) or text.count("/") != 1:
        raise ValueError(f"not a p/q string: {text!r}")
    numer, denom = text.split("/")
    if int(denom) <= 0:
        raise ValueError(f"non-positive denominator: {text!r}")
    return Fraction(int(numer), int(denom))


def load_input(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sorted_points(document: dict):
    points = sorted(document["points"], key=lambda p: p["id"])
    ids = [p["id"] for p in points]
    coords = [tuple(parse_ratio(v) for v in p["xyz"]) for p in points]
    return ids, coords


def sqdist(a: Sequence[Fraction], b: Sequence[Fraction]) -> Fraction:
    return sum((x - y) ** 2 for x, y in zip(a, b))


# --------------------------------------------------------------------------
# dense linear algebra over F_p
# --------------------------------------------------------------------------
def rank(rows: List[List[int]], prime: int, width: int) -> int:
    matrix = [row[:] + [0] * (width - len(row)) for row in rows]
    r = 0
    for col in range(width):
        pivot = None
        for i in range(r, len(matrix)):
            if matrix[i][col] % prime:
                pivot = i
                break
        if pivot is None:
            continue
        matrix[r], matrix[pivot] = matrix[pivot], matrix[r]
        inv = pow(matrix[r][col] % prime, prime - 2, prime)
        matrix[r] = [(v * inv) % prime for v in matrix[r]]
        for i in range(len(matrix)):
            if i != r and matrix[i][col] % prime:
                factor = matrix[i][col] % prime
                matrix[i] = [
                    (a - factor * b) % prime for a, b in zip(matrix[i], matrix[r])
                ]
        r += 1
        if r == len(matrix):
            break
    return r


# --------------------------------------------------------------------------
# complex construction, independent of persist.complex
# --------------------------------------------------------------------------
def build_complex(ids, coords, cutoff: Fraction):
    n = len(ids)
    dist: Dict[Tuple[int, int], Fraction] = {}
    for i, j in itertools.combinations(range(n), 2):
        dist[(i, j)] = sqdist(coords[i], coords[j])

    present = {0: [(i,) for i in range(n)], 1: [], 2: [], 3: []}
    for i, j in itertools.combinations(range(n), 2):
        if dist[(i, j)] <= cutoff:
            present[1].append((i, j))
    for subset in itertools.combinations(range(n), 3):
        value = max(dist[(subset[a], subset[b])] for a, b in itertools.combinations(range(3), 2))
        if value <= cutoff:
            present[2].append(subset)
    for subset in itertools.combinations(range(n), 4):
        value = max(dist[(subset[a], subset[b])] for a, b in itertools.combinations(range(4), 2))
        if value <= cutoff:
            present[3].append(subset)
    return present


def boundary_rows(simplices, faces, prime: int) -> List[List[int]]:
    """Rows = faces, columns = simplices, matrix of the boundary operator."""
    position = {face: index for index, face in enumerate(faces)}
    rows = [[0] * len(simplices) for _ in range(len(faces))]
    for column, simplex in enumerate(simplices):
        for k in range(len(simplex)):
            face = simplex[:k] + simplex[k + 1 :]
            rows[position[face]][column] = (rows[position[face]][column] + (-1) ** k) % prime
    return rows


def betti_at(ids, coords, cutoff: Fraction, scale: Fraction, prime: int) -> List[int]:
    """Betti numbers at ``scale`` from the sublevel complex, via dense ranks."""
    present = build_complex(ids, coords, cutoff)
    # Only simplices whose birth scale is <= the query scale.
    scales = {}
    for dim in (1, 2, 3):
        for simplex in present[dim]:
            scales[simplex] = max(
                sqdist(coords[a], coords[b]) for a, b in itertools.combinations(simplex, 2)
            )
    levels = {
        dim: [s for s in present[dim] if dim == 0 or scales[s] <= scale]
        for dim in (0, 1, 2, 3)
    }
    d1 = boundary_rows(levels[1], levels[0], prime) if levels[1] else []
    d2 = boundary_rows(levels[2], levels[1], prime) if levels[2] else []
    d3 = boundary_rows(levels[3], levels[2], prime) if levels[3] else []

    r1 = rank(d1, prime, len(levels[1]))
    r2 = rank(d2, prime, len(levels[2]))
    r3 = rank(d3, prime, len(levels[3]))
    b0 = len(levels[0]) - r1
    b1 = (len(levels[1]) - r1) - r2
    b2 = (len(levels[2]) - r2) - r3
    return [b0, b1, b2]


def ranks_and_euler(present, cutoff: Fraction, prime: int) -> dict:
    """Independent boundary ranks of the full (cutoff) complex.

    Verifies ``d1 d2 = 0`` and ``d2 d3 = 0`` explicitly and returns the
    ranks, so the caller can compare ``sum (-1)^k n_k`` with the ranked
    Euler characteristic.  Nothing here depends on reported values.
    """
    d1 = boundary_rows(present[1], present[0], prime) if present[1] else []
    d2 = boundary_rows(present[2], present[1], prime) if present[2] else []
    d3 = boundary_rows(present[3], present[2], prime) if present[3] else []

    def multiply(left, right):
        if not left or not right:
            return []
        inner = len(right)
        out = []
        for row in left:
            out.append(
                [
                    sum(row[k] * right[k][j] for k in range(inner)) % prime
                    for j in range(len(right[0]))
                ]
            )
        return out

    for product, name in ((multiply(d1, d2), "d1 d2"), (multiply(d2, d3), "d2 d3")):
        assert all(entry == 0 for row in product for entry in row), f"{name} != 0"

    return {
        "r1": rank(d1, prime, len(present[1])),
        "r2": rank(d2, prime, len(present[2])),
        "r3": rank(d3, prime, len(present[3])),
        "counts": [len(present[k]) for k in (0, 1, 2, 3)],
    }


def cycle_vector(cycle, ids, edges, prime: int) -> List[int]:
    """Turn a reported cycle into a dense vector over the *full* edge list.

    Also enforces the presentation rules: endpoint ids sorted ascending,
    distinct edges, coefficients in ``1 .. prime-1`` (booleans rejected),
    and items sorted lexicographically by the endpoint pair.
    """
    if not isinstance(cycle, list):
        raise AssertionError("a cycle must be an array")
    position = {edge: index for index, edge in enumerate(edges)}
    vector = [0] * len(edges)
    previous = None
    for entry in cycle:
        if not (isinstance(entry, list) and len(entry) == 2):
            raise AssertionError(f"cycle item must be [[start, end], coefficient]: {entry!r}")
        pair, coefficient = entry
        if not (isinstance(pair, list) and len(pair) == 2):
            raise AssertionError(f"cycle endpoints must be a pair: {entry!r}")
        start, end = pair
        if not (isinstance(start, str) and isinstance(end, str)):
            raise AssertionError("cycle endpoints must be strings")
        if isinstance(coefficient, bool) or not isinstance(coefficient, int):
            raise AssertionError(f"coefficient must be an integer: {coefficient!r}")
        if start not in ids or end not in ids:
            raise AssertionError(f"unknown endpoint id: {start},{end}")
        if not (start < end):
            raise AssertionError(f"endpoints not ascending: {start},{end}")
        if not (1 <= coefficient <= prime - 1):
            raise AssertionError(f"coefficient out of range: {coefficient}")
        if (start, end) in position and vector[position[(start, end)]]:
            raise AssertionError(f"edge repeated inside one cycle: {start},{end}")
        if previous is not None and previous >= (start, end):
            raise AssertionError(f"cycle items not in lexicographic order at {start},{end}")
        previous = (start, end)
        i, j = ids.index(start), ids.index(end)
        vector[position[(i, j)]] = coefficient
    return vector


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check(input_path: str, output_path: str) -> dict:
    """Run every acceptance check and return an independent rank summary."""
    document = load_input(input_path)
    ids, coords = sorted_points(document)
    prime = document["prime"]
    cutoff = parse_ratio(document["cutoff"])
    queries = [parse_ratio(q) for q in document["queries"]]

    with open(output_path, "r", encoding="utf-8") as handle:
        output = json.load(handle)

    bars = output["bars"]
    answers = output["answers"]
    assert len(answers) == len(queries), "answers must follow the query order"

    # ---- bars: shape, reduced rationals, ordering ------------------------
    def ratio(text):
        value = parse_ratio(text)
        if f"{value.numerator}/{value.denominator}" != text:
            raise AssertionError(f"rational not reduced: {text}")
        return value

    previous = None
    for bar in bars:
        assert len(bar) == 3
        dim, birth_text, death_text = bar
        assert dim in (0, 1, 2)
        birth = ratio(birth_text)
        death = None if death_text is None else ratio(death_text)
        if death is not None:
            assert birth < death, f"zero-length or reversed bar: {bar}"
        key = (dim, birth, (1, Fraction(0)) if death is None else (0, death))
        if previous is not None:
            assert previous <= key, f"bars out of order at {bar}"
        previous = key

    # ---- complex-level checks: d d = 0, ranks, essential classes --------
    present = build_complex(ids, coords, cutoff)
    info = ranks_and_euler(present, cutoff, prime)

    # Ranks over the full cutoff complex give the Betti numbers of that
    # complex (the essential classes, i.e. bars with death null).
    independent_betti = [
        info["counts"][0] - info["r1"],
        info["counts"][1] - info["r1"] - info["r2"],
        info["counts"][2] - info["r2"] - info["r3"],
    ]
    # A scale strictly beyond every birth in the complex.
    beyond = max(
        [Fraction(0)]
        + [
            max(sqdist(coords[a], coords[b]) for a, b in itertools.combinations(s, 2))
            for dim in (1, 2, 3)
            for s in present[dim]
        ]
    )

    def implied_betti(scale: Fraction) -> List[int]:
        counts = [0, 0, 0]
        for dim, birth_text, death_text in bars:
            birth = parse_ratio(birth_text)
            death = None if death_text is None else parse_ratio(death_text)
            if birth <= scale and (death is None or scale < death):
                counts[dim] += 1
        return counts

    assert implied_betti(beyond) == independent_betti, (
        f"bars with null death imply {implied_betti(beyond)} essential classes, "
        f"geometry gives {independent_betti}"
    )

    # A bar must not be born after the cutoff, nor die before its birth,
    # and every bar's birth must be an actual simplex scale.
    scales = set()
    for dim in (1, 2, 3):
        for simplex in present[dim]:
            scales.add(
                max(sqdist(coords[a], coords[b]) for a, b in itertools.combinations(simplex, 2))
            )
    scales.add(Fraction(0))
    for dim, birth_text, death_text in bars:
        birth = parse_ratio(birth_text)
        assert birth in scales, f"bar birth {birth_text} is not a simplex scale"
        assert birth <= cutoff, f"bar born after cutoff: {birth_text}"
        if death_text is not None:
            assert parse_ratio(death_text) in scales, f"bar death {death_text} is not a simplex scale"

    # ---- each query: betti and cycle basis ------------------------------
    edges = present[1]
    vertices = present[0]
    triangles = present[2]
    per_query = []

    for index, (scale, answer) in enumerate(zip(queries, answers)):
        expected_betti = betti_at(ids, coords, cutoff, scale, prime)
        assert answer["betti"] == expected_betti, (
            f"query {index}: betti {answer['betti']} != independently computed {expected_betti}"
        )

        # Betti numbers implied by the reported bars.
        implied = implied_betti(scale)
        assert implied == expected_betti, (
            f"query {index}: bars imply {implied}, geometry implies {expected_betti}"
        )

        cycles = answer["cycles"]
        assert len(cycles) == expected_betti[1], (
            f"query {index}: {len(cycles)} H1 cycles reported, betti_1 is {expected_betti[1]}"
        )

        # Restrict to the edges present at this scale.
        at_scale = [
            e
            for e in edges
            if sqdist(coords[e[0]], coords[e[1]]) <= scale
        ]
        position = {edge: k for k, edge in enumerate(at_scale)}
        d1_local = boundary_rows(at_scale, vertices, prime)
        d2_local = boundary_rows(
            [t for t in triangles if max(sqdist(coords[a], coords[b]) for a, b in itertools.combinations(t, 2)) <= scale],
            at_scale,
            prime,
        )

        vectors = []
        for cycle in cycles:
            dense = cycle_vector(cycle, ids, edges, prime)
            # Every edge used must exist at this scale.
            for k, coefficient in enumerate(dense):
                if coefficient:
                    assert edges[k] in position, (
                        f"query {index}: cycle uses edge {edges[k]} not present at the query scale"
                    )
            local = [0] * len(at_scale)
            for k, coefficient in enumerate(dense):
                if coefficient:
                    local[position[edges[k]]] = coefficient
            # d1 * x == 0
            for row in d1_local:
                total = sum(a * b for a, b in zip(row, local)) % prime
                assert total == 0, f"query {index}: reported H1 chain is not a cycle"
            vectors.append(local)

        # Independence modulo boundaries: dim span(vectors + im d2) == rank(d2) + len(vectors)
        combined = d2_local + vectors if d2_local else vectors
        r_d2 = rank(d2_local, prime, len(at_scale)) if d2_local else 0
        r_all = rank(combined, prime, len(at_scale)) if combined else 0
        assert r_all == r_d2 + len(vectors), (
            f"query {index}: H1 cycles are not independent modulo boundaries"
        )
        assert len(vectors) == expected_betti[1], (
            f"query {index}: H1 cycles do not span H1"
        )
        per_query.append(
            {
                "query": index,
                "scale": f"{scale.numerator}/{scale.denominator}",
                "rank_d1": rank(d1_local, prime, len(at_scale)) if at_scale else 0,
                "rank_d2": r_d2,
                "rank_d2_plus_cycles": r_all,
                "cycles": len(vectors),
                "betti": expected_betti,
            }
        )

    return {
        "prime": prime,
        "counts": {
            "vertices": info["counts"][0],
            "edges": info["counts"][1],
            "triangles": info["counts"][2],
            "tetrahedra": info["counts"][3],
        },
        "rank_d1": info["r1"],
        "rank_d2": info["r2"],
        "rank_d3": info["r3"],
        "betti_full": independent_betti,
        "queries": per_query,
    }


def main(argv: Sequence[str]) -> int:
    if len(argv) not in (2, 3):
        print("usage: python3 tools/verify.py INPUT.json OUTPUT.json [--ranks]", file=sys.stderr)
        return 2
    show_ranks = len(argv) == 3 and argv[2] == "--ranks"
    if len(argv) == 3 and not show_ranks:
        print(f"unknown option: {argv[2]}", file=sys.stderr)
        return 2
    try:
        summary = check(argv[0], argv[1])
    except AssertionError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as error:
        print(f"FAIL: malformed document ({error})", file=sys.stderr)
        return 1
    if show_ranks:
        print(json.dumps(summary, indent=2))
    print("OK: all acceptance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
