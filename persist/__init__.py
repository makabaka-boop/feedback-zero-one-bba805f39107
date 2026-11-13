"""Self-contained 0/1/2-dimensional persistent homology backend.

Public API:

* ``persist.run(json_text) -> dict`` — validate and compute, raising
  :class:`persist.errors.InputError` on rejected input;
* ``persist.run_file(input_path, output_path)`` — same, atomically
  replacing the output file;
* ``persist.main`` — the command-line entry point.

Run as a module with ``python3 -m persist INPUT.json OUTPUT.json`` or via
the ``persist.py`` wrapper in the project root.
"""

from .errors import InputError, ValidationError
from .pipeline import run, run_file


def main(argv=None):
    """Command-line entry point; imported lazily to keep ``-m persist`` clean."""
    from .__main__ import main as _main

    return _main(argv)


__all__ = ["run", "run_file", "main", "InputError", "ValidationError"]
__version__ = "1.0.0"
