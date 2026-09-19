"""Shared, side-effect-free protocol contracts for the active research workflow."""

from .config_loader import load_manifest, project_root

__all__ = ["load_manifest", "project_root"]
