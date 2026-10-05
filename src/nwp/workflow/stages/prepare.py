"""Stage handlers split by workflow responsibility."""
from nwp.core.config import ConfigError, assert_official_ready, to_plain
from nwp.core.context import RunContext
from nwp.core.dependencies import (
    data_dependency_fingerprint, feature_dependency_fingerprint,
    split_dependency_fingerprint)
from nwp.core.fingerprints import source_files_fingerprint
from nwp.core.hashing import content_hash, file_sha256
from nwp.core.provenance import read_receipt
from nwp.core.schema import ContractError, StageResult
from nwp.features.build import build_feature_month
from nwp.splits.diagnostics import split_specs
from nwp.splits.rolling import inner_folds, outer_folds

from .common import (
    _load_feature_frame, _manifest_records, _months, _now, _stage_result,
    _stage_outputs, _time_range, _verified_record, _write_json,
)

def validate_config(context: RunContext) -> StageResult:
    started = _now()
    try:
        assert_official_ready(
            context.config,
            readiness_passed=(context.config.execution != "official" or
                              context.lifecycle_state == "RUNNING"))
    except ConfigError as exc:
        return _stage_result(context, "blocked", inputs={"profile": context.config.profile},
                       message=str(exc), started_at=started)
    return _stage_result(
        context, "success", inputs={"profile": context.config.profile},
        outputs={"config_hash": context.config.config_hash}, started_at=started)


def resolve_sites(context: RunContext) -> StageResult:
    started = _now()
    path = context.paths.stage_dir("selection") / "sites.json"
    registry = context.config.sites["registry"]
    records = [{"site_id": site_id, **to_plain(registry[site_id])}
               for site_id in context.selected_sites]
    payload = {
        "site_set": context.config.scope.get("site_set"),
        "selector": context.config.scope.get("selector"),
        "selector_parameters": to_plain(context.config.scope.get("selector_parameters", {})),
        "selected_site_ids": list(context.selected_sites),
        "selected_site_records": records,
        "selector_implementation_fingerprint": source_files_fingerprint(
            context.paths.root, ("src/nwp/core/config.py",)),
    }
    payload["selection_hash"] = content_hash(payload, length=64)
    _write_json(path, payload)
    return _stage_result(
        context, "success", outputs={"sites": str(path), "count": len(records)},
        input_hashes={"site_selection": content_hash(payload)}, started_at=started)


def resolve_data(context: RunContext) -> StageResult:
    """Resolve one verified clean catalog partition for every site and month."""
    started = _now()
    months = _months(context.config.protocol)
    records = context.catalog.records()
    chosen, missing, ambiguous = [], [], []
    for site_id in context.selected_sites:
        for month in months:
            matches = [record for record in records
                       if record.stage == "clean" and record.status == "ready"
                       and record.sites == (site_id,)
                       and record.time_range == _time_range(month)]
            valid = []
            for record in matches:
                try:
                    record = _verified_record(context, record)
                    receipt = read_receipt(context.catalog.receipt_path(record))
                    if receipt.get("data_version") in {
                            None, context.config.data["version"]}:
                        valid.append(record)
                except (ContractError, OSError, ValueError):
                    continue
            if len(valid) == 1:
                chosen.append(valid[0])
            elif not valid:
                missing.append(f"{site_id}/{month}")
            else:
                ambiguous.append(f"{site_id}/{month}")
    path = context.paths.meta_dir / "clean_manifest.json"
    payload = {
        "stage": "clean", "sites": list(context.selected_sites),
        "months": list(months), "records": [record.as_dict() for record in chosen],
        "missing": missing, "ambiguous": ambiguous}
    _write_json(path, payload)
    dependency = data_dependency_fingerprint(
        context.config.data, [record.source_hashes for record in chosen])
    if missing or ambiguous:
        return _stage_result(
            context, "blocked", inputs={"missing": missing, "ambiguous": ambiguous},
            outputs={"clean_manifest": str(path)},
            input_hashes={"data": dependency},
            message="clean catalog coverage is incomplete or ambiguous",
            started_at=started)
    return _stage_result(
        context, "success", outputs={"clean_manifest": str(path),
                                     "partitions": len(chosen)},
        input_hashes={"data": dependency}, started_at=started)


def build_features(context: RunContext) -> StageResult:
    """Build or reuse every clean-to-feature monthly catalog partition."""
    started = _now()
    data_stage = context.stage_results.get("data", {})
    manifest = _stage_outputs(data_stage).get("clean_manifest")
    if data_stage.get("status") != "success" or not manifest:
        return _stage_result(context, "blocked",
                       message="features require complete clean catalog coverage",
                       started_at=started)
    records = []
    for item in _manifest_records(manifest):
        clean = context.catalog.record_from_dict(item)
        month = clean.time_range[:7]
        record = build_feature_month(
            site_id=clean.sites[0], month=month,
            clean_config_hash=clean.config_hash,
            data_config=context.config.data,
            feature_config=context.config.features,
            paths=context.paths, catalog=context.catalog,
            execution_level=context.execution_level)
        records.append(record)
    path = context.paths.meta_dir / "feature_manifest.json"
    _write_json(path, {"stage": "features",
                       "records": [record.as_dict() for record in records]})
    hashes = {record.dataset_id: file_sha256(
        context.catalog.dataset_path(record)) for record in records}
    dependency = feature_dependency_fingerprint(
        data_stage["input_hashes"]["data"], context.config.features, hashes)
    return _stage_result(
        context, "success", outputs={"feature_manifest": str(path),
                                     "partitions": len(records)},
        input_hashes={"features": dependency}, started_at=started)


def build_splits(context: RunContext) -> StageResult:
    """Materialize real outer/inner fold counts for the resolved feature data."""
    started = _now()
    if context.stage_results.get("features", {}).get("status") != "success":
        return _stage_result(context, "blocked",
                       message="splits require materialized feature partitions",
                       started_at=started)
    frame = _load_feature_frame(context)
    validation = context.config.protocol["validation"]
    development = context.config.protocol["development_period"]
    gap = int(validation["gap_days"])
    outer = outer_folds(frame, validation, development, gap_days=gap)
    payload = {"specification": split_specs(validation), "outer_folds": []}
    for fold in outer:
        inner = inner_folds(fold, validation, development, gap_days=gap)
        payload["outer_folds"].append({
            "fold_id": fold.fold_id, "fit_rows": len(fold.fit),
            "score_rows": len(fold.score),
            "inner": [{"fold_id": item.fold_id, "fit_rows": len(item.fit),
                       "early_stop_rows": len(item.early_stop),
                       "score_rows": len(item.score)} for item in inner]})
    path = context.paths.stage_dir("splits") / "splits.json"
    _write_json(path, payload)
    dependency = split_dependency_fingerprint(
        context.stage_results["features"]["input_hashes"]["features"],
        context.selected_sites, validation)
    return _stage_result(context, "success", outputs={"splits": str(path)},
                   input_hashes={"splits": dependency}, started_at=started)
