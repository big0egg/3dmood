#!/usr/bin/env python3
"""Direct entry point; no package installation or shell activation required."""
import sys
from pathlib import Path

project = Path(__file__).resolve().parent
sys.path[:0] = [str(project), str(project.parent)]
from finefune.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
