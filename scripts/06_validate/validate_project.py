"""Validate the active v2 scientific contract."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())
REPORT_BASE = ROOT / "reports" / "01_data_audit" / "project_validation"


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, condition, name, detail=""):
        self.checks.append({"name": name, "passed": bool(condition), "detail": str(detail)})

    @property
    def failures(self):
        return [row for row in self.checks if not row["passed"]]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_source(audit):
    manifest = yaml.safe_load((ROOT / "project_manifest.yaml").read_text(encoding="utf-8"))
    models = manifest["models"]
    deep = [row for row in models if row.get("internal_version")]
    audit.check(manifest["primary_target"] == "ghi", "GHI is the primary target")
    audit.check(manifest["model_count"] == len(models) == 24, "model registry count", len(models))
    audit.check(manifest["deep_model_count"] == len(deep) == 14, "deep-model count", len(deep))
    audit.check(manifest["official_result_set"] is None, "no unsupported official result set")
    audit.check(manifest["validation_protocol"]["purge_hours"] == 240,
                "purge equals 168-hour lookback plus 72-hour lead")
    audit.check(len(manifest["validation_protocol"]["outer_folds"]) == 5,
                "five outer rolling-origin folds")
    audit.check(manifest["tuning_budget"]["trials_per_model"] == 6,
                "six-trial equal-budget policy")

    variables = yaml.safe_load((ROOT / "config" / "02_variables.yaml").read_text(encoding="utf-8"))
    forecast = variables["forecast_variables"]
    audit.check(len(forecast) == 18, "forecast-variable contract count", len(forecast))
    for field in ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
        audit.check(field in forecast, f"forecast-variable contract includes {field}")

    required_docs = {"01_research.md", "02_project_status.md", "03_model_config.md",
                     "04_project_structure.md", "05_literature.md", "issues_open.md"}
    audit.check(all((ROOT / "docs" / name).is_file() for name in required_docs),
                "active v2 documentation set")



def write_report(audit):
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not audit.failures else "fail",
        "checks": len(audit.checks),
        "passed": len(audit.checks) - len(audit.failures),
        "failed": len(audit.failures),
        "failures": audit.failures,
    }
    REPORT_BASE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_BASE.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Project validation", "", f"- Status: **{result['status']}**",
             f"- Checks: {result['passed']}/{result['checks']} passed", ""]
    if audit.failures:
        lines += ["## Failures", ""] + [f"- {row['name']}: {row['detail']}" for row in audit.failures]
    else:
        lines.append("All active scientific-contract checks passed.")
    REPORT_BASE.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    audit = Audit()
    validate_source(audit)
    result = write_report(audit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(1 if audit.failures else 0)


if __name__ == "__main__":
    main()
