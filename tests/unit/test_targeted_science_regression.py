from __future__ import annotations

import subprocess

import yaml

from nwp.core.config import load_bundle, project_root, to_plain


BASE = "71729cfb67e9d0248e7243a4707f28bfaa474ad1"


def _baseline(path: str):
    text = subprocess.check_output(
        ["git", "show", f"{BASE}:{path}"], cwd=project_root(), text=True,
        encoding="utf-8")
    return yaml.safe_load(text)


def test_targeted_repair_preserves_frozen_science_contract():
    current = to_plain(load_bundle())
    protocol = _baseline("config/protocol.yaml")
    data = _baseline("config/data.yaml")
    sites = _baseline("config/sites.yaml")
    models = _baseline("config/models.yaml")
    experiments = _baseline("config/experiments.yaml")

    for key in ("development_period", "test_period"):
        assert current["protocol"][key] == protocol[key]
    validation = current["protocol"]["validation"]
    baseline_validation = protocol["validation"]
    assert validation["outer_folds"] == baseline_validation["outer_folds"]
    assert len(validation["outer_folds"]) == 5
    assert validation["inner_folds"] == baseline_validation["inner_folds"] == 3
    assert validation["purge_hours"] == baseline_validation["purge_hours"]
    probability = current["protocol"]["probability"]
    assert probability["quantiles"] == protocol["probability"]["quantiles"]
    assert len(probability["quantiles"]) == 7
    assert probability["selection_metric"] == "mean_pinball"
    assert current["data"]["forecast"]["variables"] == data["forecast"]["variables"]
    assert len(current["data"]["forecast"]["variables"]) == 15
    assert current["sites"]["sets"]["training_20"] == sites["sets"]["training_20"]
    assert current["protocol"]["spatial_design"] == protocol["spatial_design"]
    assert set(current["models"]["registry"]) == set(models["registry"])
    assert set(current["experiments"]["profiles"]) == set(experiments["profiles"])
    assert len(current["experiments"]["profiles"]) == 4
