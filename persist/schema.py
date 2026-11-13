"""Strict payload validation.

Everything accepted by the backend is defined here, so the rules live in
one place:

* the document must be a JSON object with exactly the members
  ``points``, ``prime``, ``cutoff`` and ``queries`` (each required, no
  extras);
* every point has exactly ``id`` and ``xyz``;
* ``id`` is a non-empty ASCII alphanumeric string, unique across points;
* coordinates and the cutoff are ``p/q`` strings with an optionally
  negative integer numerator and a positive integer denominator
  (``"0/1"``, ``"6/4"``, ``"-6/4"`` and ``"12/8"`` are accepted; the
  backend normalises them); queries are ``p/q`` strings lying in
  ``[0, cutoff]``;
* ``prime`` is a JSON integer with value 2, 3, 5 or 7 (booleans and
  non-integer numbers rejected);
* ``points`` and ``queries`` may be empty.

Any violation raises :class:`persist.errors.InputError` carrying the JSON
path and, for failures attached to an array element, its zero-based index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Tuple

from .errors import InputError
from .jsonparse import parse_json
from .rational import RATIONAL_RE, parse_rational

POINT_KEYS = frozenset({"id", "xyz"})
TOP_KEYS = frozenset({"points", "prime", "cutoff", "queries"})
VALID_PRIMES = (2, 3, 5, 7)
ID_RE = re.compile(r"\A[A-Za-z0-9]+\Z")


@dataclass(frozen=True)
class Point:
    identifier: str
    x: Fraction
    y: Fraction
    z: Fraction

    @property
    def xyz(self) -> Tuple[Fraction, Fraction, Fraction]:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class Document:
    points: Tuple[Point, ...]
    prime: int
    cutoff: Fraction
    queries: Tuple[Fraction, ...]


def _fail(path: str, message: str, index: int | None = None) -> None:
    raise InputError(path, message, index)


def loads_strict(text: str) -> Any:
    """Parse strict JSON; syntax errors (and duplicate members) are reported."""
    return parse_json(text)


def _check_rational(value: Any, path: str, index: int | None, what: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, str):
        _fail(path, f"{what} must be a p/q string", index)
    if not RATIONAL_RE.match(value):
        _fail(path, f"{what} must match integer/integer, got {value!r}", index)
    try:
        return parse_rational(value)
    except ZeroDivisionError:
        _fail(path, f"{what} denominator must be positive, got {value!r}", index)


def _require_object(value: Any, path: str, index: int | None, keys: frozenset) -> dict:
    if not isinstance(value, dict):
        _fail(path, "must be an object", index)
    extra = sorted(set(value) - keys)
    if extra:
        _fail(f"{path}.{extra[0]}", "unexpected member", index)
    for key in keys:
        if key not in value:
            _fail(f"{path}.{key}", "missing required member", index)
    return value


def validate(document: Any) -> Document:
    if not isinstance(document, dict):
        _fail("$", "top-level value must be an object")

    extra = sorted(set(document) - TOP_KEYS)
    if extra:
        _fail(f"$.{extra[0]}", "unexpected member")
    for key in ("points", "prime", "cutoff", "queries"):
        if key not in document:
            _fail(f"$.{key}", "missing required member")

    raw_points = document["points"]
    if not isinstance(raw_points, list):
        _fail("$.points", "must be an array")

    points: List[Point] = []
    seen: Dict[str, int] = {}
    for i, raw in enumerate(raw_points):
        base = f"$.points[{i}]"
        _require_object(raw, base, i, POINT_KEYS)

        identifier = raw["id"]
        if isinstance(identifier, bool) or not isinstance(identifier, str):
            _fail(f"{base}.id", "id must be a string", i)
        if not ID_RE.match(identifier):
            _fail(f"{base}.id", "id must be non-empty ASCII alphanumeric", i)
        if identifier in seen:
            _fail(
                f"{base}.id",
                f"duplicate id {identifier!r} (first seen at index {seen[identifier]})",
                i,
            )
        seen[identifier] = i

        xyz = raw["xyz"]
        if not isinstance(xyz, list):
            _fail(f"{base}.xyz", "xyz must be an array of 3 p/q strings", i)
        if len(xyz) != 3:
            _fail(f"{base}.xyz", "xyz must have exactly 3 entries", i)
        coordinates = [
            _check_rational(xyz[axis], f"{base}.xyz[{axis}]", axis, f"xyz[{axis}]")
            for axis in range(3)
        ]
        points.append(Point(identifier, coordinates[0], coordinates[1], coordinates[2]))

    prime = document["prime"]
    if isinstance(prime, bool) or not isinstance(prime, int):
        _fail("$.prime", "prime must be an integer")
    if prime not in VALID_PRIMES:
        _fail("$.prime", "prime must be one of 2, 3, 5, 7")

    cutoff = _check_rational(document["cutoff"], "$.cutoff", None, "cutoff")
    if cutoff < 0:
        _fail("$.cutoff", "cutoff must be non-negative")

    raw_queries = document["queries"]
    if not isinstance(raw_queries, list):
        _fail("$.queries", "must be an array")
    queries: List[Fraction] = []
    for i, raw_query in enumerate(raw_queries):
        value = _check_rational(raw_query, f"$.queries[{i}]", i, "query")
        if value < 0 or value > cutoff:
            _fail(f"$.queries[{i}]", "query must lie in [0, cutoff]", i)
        queries.append(value)

    return Document(tuple(points), prime, cutoff, tuple(queries))
