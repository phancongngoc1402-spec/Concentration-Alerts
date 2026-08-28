#!/usr/bin/env python3
"""Compatibility entry point for the manuscript reproducibility package.

The maintained implementation is modular and lives in ``analysis/`` with
``run_all.py`` as the recommended command-line entry point. This wrapper keeps
the historical filename used by earlier versions of the public repository.
"""

from run_all import main


if __name__ == "__main__":
    main()
