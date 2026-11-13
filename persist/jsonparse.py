"""A small, strict JSON parser.

Using ``json.loads`` would accept duplicate object members (keeping the
last one) and ``NaN``/``Infinity``, and would report syntax errors without
any location.  The input contract for this project is strict, so the
parser is written here from scratch:

* only the JSON grammar of RFC 8259 is accepted (UTF-8 text, no
  comments, no trailing commas, no leading zeros, no ``NaN``); a single
  leading byte order mark is tolerated and skipped;
* duplicate members inside one object are rejected with the path of the
  offending member and, when the object sits in an array, its zero-based
  index;
* every syntax error is reported at ``$``.

Parsed values are plain Python objects: ``dict``, ``list``, ``str``,
``int`` (integer literals), ``float`` (anything with a fraction or an
exponent), ``bool`` and ``None``.
"""

from __future__ import annotations

from typing import Any, List

from .errors import InputError

_MAX_DEPTH = 200


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.index = 0
        self.length = len(text)

    # -- helpers ----------------------------------------------------------
    def fail(self, message: str) -> None:
        raise InputError("$", f"invalid JSON: {message} (at offset {self.index})")

    def peek(self) -> str:
        if self.index >= self.length:
            return ""
        return self.text[self.index]

    def skip_whitespace(self) -> None:
        while self.index < self.length and self.text[self.index] in " \t\n\r":
            self.index += 1

    def expect(self, char: str, what: str) -> None:
        if self.peek() != char:
            self.fail(f"expected {what}")
        self.index += 1

    # -- grammar ----------------------------------------------------------
    def parse(self) -> Any:
        # RFC 8259 allows implementations to ignore a leading BOM; do so.
        if self.text.startswith("﻿"):
            self.index = 1
        self.skip_whitespace()
        value = self.value("$", None, 0)
        self.skip_whitespace()
        if self.index != self.length:
            self.fail("trailing content after the top-level value")
        return value

    def value(self, path: str, array_index: int | None, depth: int) -> Any:
        if depth > _MAX_DEPTH:
            self.fail("maximum nesting depth exceeded")
        char = self.peek()
        if char == "{":
            return self.object(path, array_index, depth)
        if char == "[":
            return self.array(path, array_index, depth)
        if char == '"':
            return self.string()
        if char in "-0123456789":
            return self.number()
        if self.text.startswith("true", self.index):
            self.index += 4
            return True
        if self.text.startswith("false", self.index):
            self.index += 5
            return False
        if self.text.startswith("null", self.index):
            self.index += 4
            return None
        self.fail("expected a JSON value")

    def object(self, path: str, array_index: int | None, depth: int) -> dict:
        self.expect("{", "'{'")
        result: dict = {}
        self.skip_whitespace()
        if self.peek() == "}":
            self.index += 1
            return result
        while True:
            self.skip_whitespace()
            if self.peek() != '"':
                self.fail("expected an object member name")
            key = self.string()
            member_path = f"{path}.{key}"
            if key in result:
                raise InputError(member_path, "duplicate member", array_index)
            self.skip_whitespace()
            self.expect(":", "':'")
            self.skip_whitespace()
            result[key] = self.value(member_path, None, depth + 1)
            self.skip_whitespace()
            char = self.peek()
            if char == ",":
                self.index += 1
                continue
            if char == "}":
                self.index += 1
                return result
            self.fail("expected ',' or '}'")

    def array(self, path: str, array_index: int | None, depth: int) -> list:
        self.expect("[", "'['")
        result: list = []
        self.skip_whitespace()
        if self.peek() == "]":
            self.index += 1
            return result
        while True:
            self.skip_whitespace()
            element_index = len(result)
            result.append(self.value(f"{path}[{element_index}]", element_index, depth + 1))
            self.skip_whitespace()
            char = self.peek()
            if char == ",":
                self.index += 1
                continue
            if char == "]":
                self.index += 1
                return result
            self.fail("expected ',' or ']'")

    def string(self) -> str:
        self.expect('"', "a string")
        chunks: List[str] = []
        while True:
            if self.index >= self.length:
                self.fail("unterminated string")
            char = self.text[self.index]
            if char == '"':
                self.index += 1
                return "".join(chunks)
            if char == "\\":
                self.index += 1
                escape = self.peek()
                simple = {
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                    "b": "\b",
                    "f": "\f",
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                }
                if escape in simple:
                    chunks.append(simple[escape])
                    self.index += 1
                    continue
                if escape == "u":
                    chunks.append(self.hex4())
                    continue
                self.fail("invalid escape sequence")
            if ord(char) < 0x20:
                self.fail("unescaped control character in string")
            chunks.append(char)
            self.index += 1

    def hex4(self) -> str:
        # self.index points at 'u'
        unit = self._hex_unit()
        if 0xD800 <= unit <= 0xDBFF:
            if not self.text.startswith("\\u", self.index):
                self.fail("unpaired high surrogate in \\u escape")
            self.index += 2
            low = self._hex_unit()
            if not 0xDC00 <= low <= 0xDFFF:
                self.fail("invalid low surrogate in \\u escape")
            return chr(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00))
        if 0xDC00 <= unit <= 0xDFFF:
            self.fail("unpaired low surrogate in \\u escape")
        return chr(unit)

    def _hex_unit(self) -> int:
        # self.index points at 'u'
        start = self.index + 1
        end = start + 4
        if end > self.length:
            self.fail("truncated \\u escape")
        digits = self.text[start:end]
        if any(ch not in "0123456789abcdefABCDEF" for ch in digits):
            self.fail("invalid \\u escape")
        self.index = end
        return int(digits, 16)

    def number(self) -> Any:
        start = self.index
        if self.peek() == "-":
            self.index += 1
        if self.peek() == "0":
            self.index += 1
        elif self.peek() in "123456789":
            while self.peek() and self.peek() in "0123456789":
                self.index += 1
        else:
            self.fail("invalid number")
        is_integer = True
        if self.peek() == ".":
            is_integer = False
            self.index += 1
            if not self.peek().isdigit():
                self.fail("invalid number")
            while self.peek() and self.peek() in "0123456789":
                self.index += 1
        if self.peek() in "eE":
            is_integer = False
            self.index += 1
            if self.peek() in "+-":
                self.index += 1
            if not self.peek().isdigit():
                self.fail("invalid number")
            while self.peek() and self.peek() in "0123456789":
                self.index += 1
        literal = self.text[start : self.index]
        if is_integer:
            return int(literal)
        return float(literal)


def parse_json(text: str) -> Any:
    """Parse ``text`` as strict JSON, raising :class:`InputError` on failure."""
    return _Parser(text).parse()
