"""Exact rational arithmetic helpers built on :class:`fractions.Fraction`.

Coordinates, the cutoff and queries are all given as ``p/q`` text with an
optionally negative integer numerator and a positive integer denominator
(ASCII digits only).  Values are normalised through :class:`Fraction`, so
``"6/4"`` and ``"3/2"`` denote the same scale and ``"-0/7"`` is zero.
Output always uses the normal form: positive denominator and
``gcd(p, q) == 1``.
"""

from __future__ import annotations

import re
from fractions import Fraction

__all__ = ["parse_rational", "format_rational", "is_rational", "RATIONAL_RE"]

# Anchored with \Z (not $) so a trailing newline cannot slip through, and
# with [0-9] rather than \d so non-ASCII digits are rejected.
RATIONAL_RE = re.compile(r"-?[0-9]+/[0-9]+\Z")


def is_rational(value: object) -> bool:
    """True if ``value`` matches the ``p/q`` grammar."""
    return isinstance(value, str) and RATIONAL_RE.match(value) is not None


def parse_rational(text: str) -> Fraction:
    """Parse validated ``p/q`` text into a reduced :class:`Fraction`."""
    numerator, denominator = text.split("/", 1)
    denom = int(denominator)
    if denom <= 0:
        raise ZeroDivisionError(f"denominator must be positive: {text!r}")
    return Fraction(int(numerator), denom)


def format_rational(value: Fraction) -> str:
    """Render a rational in reduced form with a positive denominator."""
    return f"{value.numerator}/{value.denominator}"
