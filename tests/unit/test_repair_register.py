from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_repair_register_never_claims_an_unverified_pass() -> None:
    with (ROOT / "migration" / "final_repair_register.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 635
    for row in rows:
        assert row["evidence"] != "System.Object[]"
        if row["status"] == "PASS":
            assert row["verification"] not in {"", "PENDING", "NOT_VERIFIED"}
            assert row["evidence"].strip()
# End of repair-ledger guards.

def test_final_closure_register_covers_every_instruction_without_false_passes() -> None:
    with (ROOT / "migration" / "final_closure_register.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["item"]) for row in rows] == list(range(220))
    for row in rows:
        if row["status"] == "PASS":
            assert row["verification"] not in {"", "PENDING", "NOT_VERIFIED"}
            assert row["evidence"].strip()
