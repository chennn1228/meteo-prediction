"""Current repository-shape regression gates."""
from pathlib import Path


ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "project_manifest.yaml").is_file())


def test_forbidden_root_directories_are_absent():
    forbidden = {"archive", "legacy", "figs", "reports", "scripts"}
    assert not {name for name in forbidden if (ROOT / name).exists()}


def test_active_docs_are_exactly_the_seven_named_documents():
    expected = {
        "01_protocol.md", "02_status.md", "03_data_contract.md",
        "04_repository_guide.md", "05_literature.md", "06_issues.md",
        "07_history.md",
    }
    assert {path.name for path in (ROOT / "docs").iterdir() if path.is_file()} == expected


def test_scientific_configs_are_exactly_six_and_local_example_is_machine_only():
    expected = {
        "protocol.yaml", "data.yaml", "features.yaml", "sites.yaml",
        "models.yaml", "experiments.yaml",
    }
    actual = {path.name for path in (ROOT / "config").glob("*.yaml")
              if path.name not in {"local.yaml", "local.example.yaml"}}
    assert actual == expected
    assert (ROOT / "config" / "local.example.yaml").is_file()


def test_temporary_workflow_is_absent():
    assert not (ROOT / "src" / "nwp" / "workflow" / "stages" / "implementation.py").exists()
    assert not (ROOT / "migration").exists()
