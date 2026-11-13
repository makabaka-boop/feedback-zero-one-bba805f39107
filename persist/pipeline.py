"""End-to-end pipeline: JSON text -> validation -> persistence -> JSON.

Output contract::

    {
      "bars": [[dim, "p/q", "p/q" | null], ...],
      "answers": [
        {"betti": [b0, b1, b2], "cycles": [[[[u, v], c], ...], ...]},
        ...
      ]
    }

* ``bars`` is sorted by ``(dimension, birth, death)`` with ``null`` deaths
  last.  Zero-length bars are omitted; duplicates are preserved.
* ``answers`` keeps the original query order.  Each entry carries the
  Betti triple at that scale and a basis of ``H_1`` as explicit 1-chains.
* A cycle is a list of ``[[start_id, end_id], coefficient]`` items with
  ``start_id < end_id`` (ASCII order), items sorted lexicographically by
  endpoint pair, zero coefficients omitted, coefficients in
  ``1 .. prime - 1``.

Every rational is emitted as a reduced ``p/q`` with a positive denominator.
"""

from __future__ import annotations

import json
import os
import tempfile
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple

from . import persistence
from .complex import build
from .persistence import Interval, Persistence
from .rational import format_rational
from .schema import Document, Point, loads_strict, validate


def _bar_key(bar: Interval) -> Tuple[int, Fraction, int, Fraction]:
    """Sort bars by dimension, birth, death, with ``None`` (still alive) last."""
    if bar.death is None:
        return (bar.dimension, bar.birth, 1, Fraction(0))
    return (bar.dimension, bar.birth, 0, bar.death)


def visible_bars(result: Persistence) -> List[Interval]:
    """Bars with strictly positive length, sorted for output (duplicates kept)."""
    kept = [
        bar
        for bar in result.bars
        if bar.death is None or bar.death != bar.birth
    ]
    kept.sort(key=_bar_key)
    return kept


def betti_numbers(bars: Sequence[Interval], scale: Fraction) -> List[int]:
    counts = [0, 0, 0]
    for bar in bars:
        if bar.dimension > 2:
            continue
        if bar.birth <= scale and (bar.death is None or scale < bar.death):
            counts[bar.dimension] += 1
    return counts


def format_cycle(
    column: Dict[int, int],
    edges: Sequence[Tuple[int, int]],
    point_ids: Sequence[str],
    prime: int,
) -> List[list]:
    items: List[list] = []
    for edge_index in sorted(column):
        coefficient = column[edge_index] % prime
        if not coefficient:
            continue
        start, end = edges[edge_index]
        items.append([[point_ids[start], point_ids[end]], coefficient])
    items.sort(key=lambda item: (item[0][0], item[0][1]))
    return items


def build_output(document: Document) -> dict:
    """Compute the output structure for a validated document."""
    # Points in ascending id order: this fixes boundary orientations.
    points: List[Point] = sorted(document.points, key=lambda point: point.identifier)
    point_ids = [point.identifier for point in points]

    filtration = build(points, document.cutoff)
    result = persistence.compute(filtration, document.prime)
    bars = visible_bars(result)
    edges = filtration.edges()

    answers = []
    for scale in document.queries:
        cycles = [
            format_cycle(column, edges, point_ids, document.prime)
            for column in result.cycles_alive_at(scale)
        ]
        answers.append({"betti": betti_numbers(bars, scale), "cycles": cycles})

    return {
        "bars": [
            [
                bar.dimension,
                format_rational(bar.birth),
                None if bar.death is None else format_rational(bar.death),
            ]
            for bar in bars
        ],
        "answers": answers,
    }


def run(document_text: str) -> dict:
    """Validate raw JSON text and return the output document."""
    raw = loads_strict(document_text)
    document = validate(raw)
    return build_output(document)


def dumps(output: dict) -> str:
    """Serialise an output document deterministically."""
    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def run_text(document_text: str) -> str:
    return dumps(run(document_text))


def run_file(input_path: str, output_path: str) -> None:
    """Compute and atomically replace ``output_path``.

    ``run_file`` writes to a temporary file in the destination directory
    and renames it into place only after the whole computation succeeded,
    so a failure (invalid input, I/O error, internal error) leaves any
    existing output file byte-for-byte untouched.
    """
    output_path = os.path.abspath(output_path)
    with open(input_path, "rb") as handle:
        raw_bytes = handle.read()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        from .errors import InputError

        raise InputError("$", f"invalid UTF-8: {exc}") from None

    payload = run_text(text)

    directory = os.path.dirname(output_path) or "."
    descriptor, temporary = tempfile.mkstemp(prefix=".persist-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
