"""Fail-closed write-once lifecycle for persisted runs."""
from __future__ import annotations

import json
from pathlib import Path

from .schema import ContractError

STATES = frozenset({"CREATED", "RUNNING", "BLOCKED", "FAILED", "COMPLETE", "FROZEN", "IMPORTED"})
WRITABLE_STATES = frozenset({"CREATED", "RUNNING"})
TRANSITIONS = {
    "CREATED": frozenset({"RUNNING", "BLOCKED", "FAILED"}),
    "RUNNING": frozenset({"BLOCKED", "FAILED", "COMPLETE", "FROZEN"}),
    "BLOCKED": frozenset(), "FAILED": frozenset(), "COMPLETE": frozenset(),
    "FROZEN": frozenset(), "IMPORTED": frozenset(),
}


def read_state(path: Path) -> str:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))["state"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractError(f"invalid run lifecycle file: {path}") from exc
    if state not in STATES:
        raise ContractError(f"invalid run lifecycle state: {state!r}")
    return state


def write_initial_state(path: Path, state: str = "CREATED") -> None:
    if path.exists() or state not in STATES:
        raise ContractError("run lifecycle may only be initialized once with a valid state")
    path.write_text(json.dumps({"state": state}, indent=2) + "\n", encoding="utf-8")


def transition(path: Path, target: str) -> None:
    current = read_state(path)
    if target not in TRANSITIONS[current]:
        raise ContractError(f"illegal lifecycle transition: {current} -> {target}")
    path.write_text(json.dumps({"state": target}, indent=2) + "\n", encoding="utf-8")


def require_writable(path: Path) -> str:
    state = read_state(path)
    if state not in WRITABLE_STATES:
        raise ContractError(f"run state {state} is read-only; create a child run")
    return state
