#!/usr/bin/env python3
"""
Normalize Jupyter notebooks by adding missing cell ids to avoid nbformat warnings.
"""
from pathlib import Path
import sys

import nbformat
from nbformat.validator import normalize


def normalize_ipynb(path: Path) -> bool:
    try:
        nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
        normalize(nb)  # adds missing ids and normalizes structure
        nbformat.write(nb, path)
        return True
    except Exception as e:
        print(f"Failed to normalize {path}: {e}", file=sys.stderr)
        return False


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    docs_dir = root / "docs"
    changed = 0
    failed = 0
    for ipynb in docs_dir.rglob("*.ipynb"):
        ok = normalize_ipynb(ipynb)
        if ok:
            changed += 1
        else:
            failed += 1
    print(f"Normalized notebooks: {changed}, failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
