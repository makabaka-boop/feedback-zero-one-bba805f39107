#!/usr/bin/env python3
"""Standalone entry point: ``python3 persist.py INPUT.json OUTPUT.json``."""

import sys

from persist.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
