from __future__ import annotations

from pathlib import Path

from nwp.core.config import project_root, resolve_config
from nwp.core.context import RunContext
from nwp.core.lifecycle import transition
import nwp.workflow.stages.analysis as analysis_stage


def _prime_indexes(context: RunContext) -> Path:
    source = context.paths.aggregate_metrics_dir() / "source.csv"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("value\n1\n", encoding="utf-8")
    indexes = {}
    for stage, name in (("evaluation", "metrics_index"),
                        ("analysis", "analysis_index"),
                        ("prediction", "prediction_index")):
        path = context.paths.stage_dir(stage) / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
        context.stage_results[stage] = {
            "status": "success", "artifact_outputs": {name: str(path)},
            "metadata_outputs": {}, "input_hashes": {stage: stage * 8}}
    return source


def test_compatible_figures_are_resolved_before_render(tmp_path, monkeypatch):
    config = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    render_calls = {"count": 0}

    def planner(_metrics, _analysis, output, **_kwargs):
        source = planner.source

        def render():
            render_calls["count"] += 1
            files = [output / "bounded.svg", output / "bounded.png"]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"bounded-figure")
            return {
                "figure_id": "bounded", "source_table": str(source),
                "generation_function": "bounded.render",
                "caption": "bounded", "files": [str(path) for path in files]}

        return [{
            "figure_id": "bounded", "source_table": str(source),
            "generation_function": "bounded.render", "caption": "bounded",
            "files": [str(output / "bounded.svg"),
                      str(output / "bounded.png")],
            "render": render}]

    monkeypatch.setattr(analysis_stage, "plan_run_figures", planner)
    monkeypatch.setattr(analysis_stage, "apply_publication_style", lambda: None)

    parent = RunContext.create(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="figure-parent")
    planner.source = _prime_indexes(parent)
    parent.active_stage = "figures"
    result = analysis_stage.figures(parent)
    parent.active_stage = None
    assert result.status == "success"
    assert render_calls["count"] == 1
    transition(parent.paths.meta_dir / "status.json", "FROZEN")

    child = RunContext.create(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="figure-child",
        parent_run_id=parent.run_id, change_reason="same figure dependencies",
        changed_dependencies=["unrelated"])
    planner.source = _prime_indexes(child)
    child.active_stage = "figures"
    result = analysis_stage.figures(child)
    child.active_stage = None
    assert result.status == "success"
    assert render_calls["count"] == 1
    assert (child.paths.figures_dir / "bounded.svg").read_bytes() == b"bounded-figure"
    assert (child.paths.figures_dir / "bounded.png").read_bytes() == b"bounded-figure"
