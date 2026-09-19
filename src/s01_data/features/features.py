"""Compatibility notice: retired feature CLI; use scripts/02_data/run_features.py.

The historical implementation is preserved under legacy/s01_data/features/.
It used requested coordinates and is never a formal research entry point.
"""
from __future__ import annotations


def main() -> None:
    raise RuntimeError(
        "retired feature entry point used request coordinates; "
        "run scripts/02_data/run_features.py with a validated clean parquet instead"
    )


if __name__ == "__main__":
    main()
