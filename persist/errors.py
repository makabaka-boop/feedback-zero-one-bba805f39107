"""JSON input errors.

A failure is reported as a single message whose first component is the
JSON path of the offending member (``$`` for syntax errors) and whose
second component is a zero-based index into a JSON array when the failure
is attached to an array element, otherwise omitted.

Examples::

    {"path": "$.points[3].xyz", "index": 3, "message": "..."}
    {"path": "$", "message": "..."}
"""

from __future__ import annotations


class InputError(Exception):
    """Raised when the input document is rejected.

    Parameters
    ----------
    path:
        JSONPath-like location of the offending member.  ``$`` for the
        document root (used by syntax errors).
    message:
        Human readable explanation.
    index:
        Zero-based array index the failure is attached to, if any.
    """

    def __init__(self, path: str, message: str, index: int | None = None) -> None:
        self.path = path
        self.message = message
        self.index = index
        super().__init__(self.render())

    def render(self) -> str:
        if self.index is None:
            return f"{self.path}: {self.message}"
        return f"{self.path}: [index {self.index}] {self.message}"

    def to_dict(self) -> dict:
        out = {"path": self.path, "message": self.message}
        if self.index is not None:
            out["index"] = self.index
        return out

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.render()


# Kept for a friendlier import surface; both names denote the same failure.
ValidationError = InputError
