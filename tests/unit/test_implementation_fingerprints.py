from __future__ import annotations

import shutil

from nwp.core.artifacts import environment_packages
from nwp.core.config import project_root, resolve_config
from nwp.core.fingerprints import (
    model_implementation_sources, source_files_fingerprint)


def _fingerprints(root, records):
    common = ("src/nwp/models/base.py", "src/nwp/models/factory.py",
              "src/nwp/features/preprocessing.py")
    return {
        model_id: source_files_fingerprint(
            root, (*common, *model_implementation_sources(record)))
        for model_id, record in records.items()}


def test_implementation_file_changes_invalidate_only_true_consumers(tmp_path):
    shutil.copytree(project_root() / "src", tmp_path / "src")
    records = resolve_config("nanjing_cpu_diagnostic").models["registry"]
    selected = {model_id: records[model_id] for model_id in
                ("raw_gfs", "bias_correction", "linear_mos",
                 "ridge_mos", "lgbm", "xgboost")}
    baseline = _fingerprints(tmp_path, selected)
    expectations = (
        ("src/nwp/models/baselines.py",
         {"raw_gfs", "bias_correction", "linear_mos"}),
        ("src/nwp/models/statistical.py", {"ridge_mos"}),
        ("src/nwp/models/trees.py", {"lgbm", "xgboost"}),
    )
    for relative, expected in expectations:
        path = tmp_path / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\n# bounded fingerprint change\n")
        changed = _fingerprints(tmp_path, selected)
        assert {key for key in baseline if baseline[key] != changed[key]} == expected
        path.write_bytes(original)


def test_deep_artifact_environment_includes_torch():
    for implementation in ("mlp", "transformer", "timesnet", "pinn"):
        packages = environment_packages("model", implementation)
        assert "torch" in packages
        assert packages != environment_packages("model", "ridge")
