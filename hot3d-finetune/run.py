#!/usr/bin/env python3
"""Keep the hyphenated project directory usable without installing a package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hot3d_ft.cli import main

if __name__ == '__main__':
    raise SystemExit(main())
