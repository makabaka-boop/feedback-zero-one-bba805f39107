"""Command-line entry point.

Usage::

    python -m persist INPUT.json OUTPUT.json

The input file must be UTF-8 JSON.  On success the output file is
replaced atomically; on failure the previous output file is left intact
and the error is printed to stderr as a single JSON object, for example::

    {"path": "$.points[2].xyz[1]", "index": 1, "error": "..."}

Exit status is 0 on success and 1 on any rejected input or I/O failure.
"""

from __future__ import annotations

import json
import sys

from .errors import InputError
from .pipeline import run_file


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        sys.stderr.write("usage: python -m persist INPUT.json OUTPUT.json\n")
        return 2
    input_path, output_path = arguments
    try:
        run_file(input_path, output_path)
    except InputError as error:
        payload = error.to_dict()
        payload["error"] = error.message
        sys.stderr.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return 1
    except OSError as error:
        sys.stderr.write(json.dumps({"path": "$", "error": str(error)}) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
