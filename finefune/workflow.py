#!/usr/bin/env python3
"""Compatibility spelling for the structured suite command."""
import sys
from pathlib import Path
project = Path(__file__).resolve().parent
sys.path[:0] = [str(project), str(project.parent)]
from finefune.cli import main
if __name__ == "__main__":
    args = sys.argv[1:]
    if "--config" not in args:
        args = ["--config", str(project / "configs/suites/model_data_compare.yaml"), *args]
    raise SystemExit(main(["suite", *args]))
