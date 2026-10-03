"""The only command-line entry point: python -m nwp."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from nwp.core.config import (
    ConfigError, assert_official_ready, load_bundle, load_local_paths,
    project_root, resolve_config)
from nwp.core.context import RunContext
from nwp.core.validation import readiness_result
from nwp.data.audit import inventory_data, probe_service_round
from nwp.workflow.pipeline import ALIASES, STAGES, run_pipeline


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile")
    parser.add_argument("--models")
    parser.add_argument("--sites")
    parser.add_argument("--site-set")
    parser.add_argument("--n-sites", type=int)
    parser.add_argument("--gap-days", type=int)
    parser.add_argument("--seed", type=int)


def _resolve(args: argparse.Namespace):
    return resolve_config(args.profile, root=project_root(), models=args.models, sites=args.sites, site_set=args.site_set, n_sites=args.n_sites, gap_days=args.gap_days, seed=args.seed)


def _resume_config(run_id: str):
    root = project_root()
    outputs_root = load_local_paths(root)["outputs_root"]
    matches = [outputs_root / execution / run_id
               for execution in ("development", "official")
               if (outputs_root / execution / run_id).is_dir()]
    if len(matches) != 1:
        raise ConfigError(
            f"run_id must resolve to exactly one development/official run: {run_id}")
    saved = yaml.safe_load(
        (matches[0] / "00_meta" / "resolved_config.yaml").read_text(
            encoding="utf-8"))
    if not isinstance(saved, dict) or not isinstance(saved.get("overrides"), dict):
        raise ConfigError("saved resolved configuration is malformed")
    overrides = saved["overrides"]
    config = resolve_config(
        saved.get("profile"), root=root,
        models=overrides.get("models"), sites=overrides.get("sites"),
        site_set=overrides.get("site_set"), n_sites=overrides.get("n_sites"),
        gap_days=overrides.get("gap_days"), seed=overrides.get("seed"))
    if config.as_dict() != saved:
        raise ConfigError(
            "saved run configuration no longer resolves identically; refusing resume")
    return config


def _run(args: argparse.Namespace) -> int:
    if args.command in {"evaluate", "figures"} and not getattr(args, "run_id", None):
        raise ConfigError(f"nwp {args.command} requires --run-id")
    config = (_resume_config(args.run_id)
              if getattr(args, "run_id", None) else _resolve(args))
    assert_official_ready(config)
    context = (RunContext.resume(
        project_root(), config, run_id=args.run_id,
        allow_model_execution=args.execute_model_stages)
        if getattr(args, "run_id", None)
        else RunContext.create(
            project_root(), config,
            allow_model_execution=args.execute_model_stages))
    statuses = run_pipeline(context, from_stage=args.from_stage, to_stage=args.to_stage)
    print(json.dumps({"run_id": context.run_id, "run_path": str(context.paths.run_root), "stages": {key: value["status"] for key, value in statuses.items()}}, ensure_ascii=False, indent=2))
    acceptable = {"success", "reused", "skipped"}
    start = ALIASES.get(args.from_stage, args.from_stage)
    end = ALIASES.get(args.to_stage, args.to_stage)
    selected = STAGES[STAGES.index(start):STAGES.index(end) + 1]
    return 0 if all(statuses.get(stage, {}).get("status") in acceptable
                    for stage in selected) else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nwp")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="resolve and validate a run configuration")
    _add_config_arguments(validate)
    validate.add_argument(
        "--mode", choices=("config", "structural", "data_ready", "cpu_ready",
                           "spatial_ready", "deep_ready", "official_full",
                           "official"), default="config")
    data = sub.add_parser("data", help="data catalog and audit operations")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    inventory = data_sub.add_parser("inventory", help="hash and classify every data file")
    inventory.add_argument("--quarantine-unknown", action="store_true")
    probe = data_sub.add_parser(
        "probe-service", help="run a budgeted returned-service-point probe")
    probe.add_argument("--step", type=float, required=True)
    probe.add_argument("--max-new-batches", type=int, default=0)
    probe.add_argument("--batch-size", type=int, default=40)
    probe.add_argument("--boundary", type=Path)
    run = sub.add_parser("run", help="run the dependency-aware workflow")
    _add_config_arguments(run)
    run.add_argument("--from-stage", default="validate")
    run.add_argument("--to-stage", default="report")
    run.add_argument("--run-id", help="resume an existing run with its saved configuration")
    run.add_argument("--execute-model-stages", action="store_true")
    evaluate = sub.add_parser("evaluate", help="run the evaluation stage for a resolved development run")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--execute-model-stages", action="store_true")
    figures = sub.add_parser("figures", help="run the figure stage for a resolved development run")
    figures.add_argument("--run-id", required=True)
    figures.add_argument("--execute-model-stages", action="store_true")
    status = sub.add_parser("status", help="show registered profiles and output root")
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            if args.mode != "config":
                report = readiness_result(args.mode)
                print(json.dumps(report, ensure_ascii=False, indent=2))
                return 0 if report["status"] == "pass" else 2
            config = _resolve(args)
            print(json.dumps({"profile": config.profile, "config_hash": config.config_hash, "execution": config.execution, "sites": list(config.selected_sites), "models": list(config.selected_models)}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "data":
            root = project_root()
            if args.data_command == "inventory":
                payload = inventory_data(
                    root / "data",
                    quarantine_unknown=args.quarantine_unknown)
            else:
                bundle = load_bundle(str(root))
                payload = probe_service_round(
                    args.step, batch_size=args.batch_size,
                    max_new_batches=args.max_new_batches,
                    data_root=root / "data",
                    boundary_path=(
                        args.boundary or root / "data" / "00_geo" /
                        "jiangsu.geojson"),
                    model=bundle["data"]["forecast"]["model"])
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            return _run(args)
        if args.command in {"evaluate", "figures"}:
            args.from_stage = args.command
            args.to_stage = args.command
            return _run(args)
        bundle = load_bundle(str(project_root()))
        print(json.dumps({"profiles": sorted(bundle["experiments"]["profiles"]), "outputs_root": str(project_root() / "outputs")}, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, ValueError) as exc:
        parser.error(str(exc))
    return 2
