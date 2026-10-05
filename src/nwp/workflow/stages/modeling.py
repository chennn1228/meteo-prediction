"""Stage handlers split by workflow responsibility."""
from .common import *  # noqa: F401,F403
from .common import (
    _artifact_scope, _eligible, _final_blocks, _find_reusable_model_artifact,
    _load_feature_frame, _model_implementation_fingerprint, _months, _now,
    _record_model_artifact, _stage_result, _stage_outputs, _time_range, _write_json,
    _ridge_fit_predict,
)

def tune_models(context: RunContext) -> StageResult:
    """Run the registered equal-budget selector on every outer prefix."""
    started = _now()
    if context.stage_results.get("splits", {}).get("status") != "success":
        return _stage_result(context, "blocked", message="tuning requires real splits",
                       started_at=started)
    if not context.allow_model_execution:
        return _stage_result(
            context, "blocked",
            message="tuning is gated; pass --execute-model-stages explicitly",
            started_at=started)
    frame = _load_feature_frame(context)
    protocol, models = context.config.protocol, context.config.models
    validation, development = protocol["validation"], protocol["development_period"]
    gap = int(validation["gap_days"])
    selected: dict[str, Any] = {}
    try:
        for outer in outer_folds(frame, validation, development, gap_days=gap):
            folds = inner_folds(outer, validation, development, gap_days=gap)
            eligible = [type(fold)(
                fold.fold_id, _eligible(context, fold.fit),
                _eligible(context, fold.early_stop), _eligible(context, fold.score),
                fold.fit_end, fold.early_stop_start, fold.early_stop_end,
                fold.score_start) for fold in folds]
            for model_id in context.selected_models:
                entry = models["registry"][model_id]
                if not entry["tuning"]["enabled"]:
                    continue
                adapter_name = entry.get("quantile_adapter")
                if adapter_name == "ridge":
                    def adapter(parameters, fit, early, score, seed, levels):
                        return _ridge_fit_predict(
                            parameters, fit, early, score, seed, levels,
                            feature_config=context.config.features)
                    tree_receipts = []
                elif adapter_name == "tree":
                    adapter = tree_fit_predict(
                        str(entry["algorithm"]),
                        feature_config=context.config.features)
                    tree_receipts = adapter.receipts
                else:
                    raise ContractError(
                        f"workflow tuning adapter is not registered for {model_id}")
                dependency = model_dependency_fingerprint(
                    "tuning", model_id,
                    upstream={
                        "features": context.stage_results["features"]["input_hashes"]["features"],
                        "splits": context.stage_results["splits"]["input_hashes"]["splits"],
                        "fold": stable_object_hash(outer.fold_id)},
                    model_record=entry,
                    search_space=context.config.selected_search_spaces[model_id],
                    stage_contract={"gap_days": gap,
                                    "metric": protocol["probability"]["selection_metric"]})
                implementation = _model_implementation_fingerprint(context, model_id)
                local_dir = context.paths.tuning_fold(model_id, outer.fold_id)
                local_selection = local_dir / "selection.json"
                reusable = _find_reusable_model_artifact(
                    context, artifact_type="tuning", model_id=model_id,
                    fold_id=outer.fold_id, dependency=dependency,
                    implementation=implementation, time_scope="outer_validation")
                reuse_meta = None
                if reusable is not None:
                    source_path, source_record = reusable
                    source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
                    reuse_meta = context.artifact_resolver.materialize(
                        source_path, local_selection, source_record.sha256,
                        reused_from_run=source_run,
                        source_artifact_id=source_record.artifact_id,
                        source_dependency_fingerprint=source_record.dependency_fingerprint,
                        source_execution_level=source_record.execution_level)
                    detail = json.loads(local_selection.read_text(encoding="utf-8"))
                    _write_json(local_dir / "trials.json", detail["trials"])
                    _write_json(local_dir / "receipt.json", {
                        "dependency_fingerprint": dependency,
                        "implementation_fingerprint": implementation,
                        "selection_sha256": file_sha256(local_selection),
                        **reuse_meta})
                else:
                    chosen, ledger = run_trials(
                        model_id, outer.fold_id, eligible, adapter,
                        model_config=models, protocol_config=protocol,
                        device=str(entry["device"]), gap_days=gap)
                    detail = {
                        "selected_candidate": chosen,
                        "parameters": candidates(model_id, models)[chosen],
                        "trials": [asdict(trial) for trial in ledger],
                        "tree_round_receipts": list(tree_receipts)}
                    _write_json(local_dir / "trials.json", detail["trials"])
                    _write_json(local_selection, detail)
                    _write_json(local_dir / "receipt.json", {
                        "dependency_fingerprint": dependency,
                        "implementation_fingerprint": implementation,
                        "selection_sha256": file_sha256(local_selection)})
                selected.setdefault(outer.fold_id, {})[model_id] = detail
                _record_model_artifact(
                    context, local_selection, artifact_type="tuning",
                    stage="tuning", model_id=model_id, fold_id=outer.fold_id,
                    dependency=dependency, implementation=implementation,
                    time_scope="outer_validation", reuse=reuse_meta)
    except IncompleteTrialsError as exc:
        return _stage_result(
            context, "failed", inputs={"ledger": [asdict(row) for row in exc.ledger]},
            message=str(exc), started_at=started)
    final_selection: dict[str, Any] = {}
    for model_id in context.selected_models:
        if not models["registry"][model_id]["tuning"]["enabled"]:
            continue
        candidate_scores: dict[int, list[float]] = {
            index: [] for index in range(len(candidates(model_id, models)))}
        for outer_models in selected.values():
            for trial in outer_models[model_id]["trials"]:
                if trial["status"] == "ok" and trial["mean_pinball"] is not None:
                    candidate_scores[int(trial["candidate"])].append(
                        float(trial["mean_pinball"]))
        expected = (len(selected)
                    * int(protocol["validation"]["inner_folds"]))
        if any(len(values) != expected for values in candidate_scores.values()):
            raise ContractError(
                f"{model_id}: final selection lacks the common outer-inner score grid")
        means = {candidate: float(np.mean(values))
                 for candidate, values in candidate_scores.items()}
        chosen = min(means, key=lambda candidate: (means[candidate], candidate))
        final_selection[model_id] = {
            "selected_candidate": chosen,
            "parameters": candidates(model_id, models)[chosen],
            "mean_pinball_by_candidate": means,
            "score_count_per_candidate": expected}
    path = context.paths.tuning_dir / "selection.json"
    _write_json(path, {"outer": selected, "final": final_selection})
    dependency = content_hash({
        "features": context.stage_results["features"]["input_hashes"]["features"],
        "splits": context.stage_results["splits"]["input_hashes"]["splits"],
        "models": models, "selected_models": context.selected_models})
    actual_trials = sum(len(detail["trials"]) for outer in selected.values()
                        for detail in outer.values())
    return _stage_result(context, "success", outputs={
                       "selection": str(path),
                       "trial_budget": int(models["search"]["budget"]["trials_per_model"]),
                       "actual_trials": actual_trials,
                       "selection_metric": protocol["probability"]["selection_metric"]},
                   input_hashes={"tuning": dependency}, started_at=started)


def fit_models(context: RunContext) -> StageResult:
    """Fit and serialize every selected outer-fold model artifact."""
    started = _now()
    tuning = context.stage_results.get("tuning", {})
    if tuning.get("status") != "success":
        return _stage_result(context, "blocked",
                       message="fitting requires successful tuning",
                       started_at=started)
    frame = _load_feature_frame(context)
    protocol = context.config.protocol
    validation, development = protocol["validation"], protocol["development_period"]
    gap = int(validation["gap_days"])
    selection = json.loads(Path(
        _stage_outputs(tuning)["selection"]).read_text(encoding="utf-8"))
    outer_selection = selection["outer"]
    artifacts = []
    for outer in outer_folds(frame, validation, development, gap_days=gap):
        fit = _eligible(context, outer.fit)
        folds = inner_folds(outer, validation, development, gap_days=gap)
        eligible = [type(fold)(
            fold.fold_id, _eligible(context, fold.fit),
            _eligible(context, fold.early_stop), _eligible(context, fold.score),
            fold.fit_end, fold.early_stop_start, fold.early_stop_end,
            fold.score_start) for fold in folds]
        for model_id in context.selected_models:
            entry = context.config.models["registry"][model_id]
            path = context.paths.model_fold(model_id, outer.fold_id) / "model.pkl"
            dependency = model_dependency_fingerprint(
                "fitting", model_id,
                upstream={
                    "features": context.stage_results["features"]["input_hashes"]["features"],
                    "tuning": tuning["input_hashes"]["tuning"],
                    "fold": stable_object_hash(outer.fold_id)},
                model_record=entry,
                stage_contract=(outer_selection[outer.fold_id].get(model_id)
                                if entry["tuning"]["enabled"] else {"fixed": True}))
            implementation = _model_implementation_fingerprint(context, model_id)
            reusable = _find_reusable_model_artifact(
                context, artifact_type="model", model_id=model_id,
                fold_id=outer.fold_id, dependency=dependency,
                implementation=implementation, time_scope="outer_validation")
            reuse_meta = None
            if reusable is not None:
                source_path, source_record = reusable
                source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
                reuse_meta = context.artifact_resolver.materialize(
                    source_path, path, source_record.sha256,
                    reused_from_run=source_run,
                    source_artifact_id=source_record.artifact_id,
                    source_dependency_fingerprint=source_record.dependency_fingerprint,
                    source_execution_level=source_record.execution_level)
            else:
                if entry["tuning"]["enabled"]:
                    choice = outer_selection[outer.fold_id][model_id]
                    model = fit_outer_quantile_model(
                        context, model_id, fit, eligible, choice["parameters"],
                        tree_receipts=choice["tree_round_receipts"])
                else:
                    model = fit_model(context, model_id, fit, fit["y"].to_numpy())
                model.save(path)
            _record_model_artifact(
                context, path, artifact_type="model", stage="fitting",
                model_id=model_id, fold_id=outer.fold_id,
                dependency=dependency, implementation=implementation,
                time_scope="outer_validation", reuse=reuse_meta)
            artifacts.append({
                "model_id": model_id, "outer_fold": outer.fold_id,
                "fit_scope": "outer", "path": str(path),
                "sha256": file_sha256(path)})
    final_fit, final_early, final_calibration, _ = _final_blocks(
        context, frame)
    final_history = pd.concat(
        [final_early, final_calibration], ignore_index=True)
    for model_id in context.selected_models:
        entry = context.config.models["registry"][model_id]
        path = context.paths.model_fold(model_id, "final") / "model.pkl"
        dependency = model_dependency_fingerprint(
            "fitting", model_id,
            upstream={
                "features": context.stage_results["features"]["input_hashes"]["features"],
                "tuning": tuning["input_hashes"]["tuning"],
                "fold": stable_object_hash("final")},
            model_record=entry,
            stage_contract=(selection["final"].get(model_id)
                            if entry["tuning"]["enabled"] else {"fixed": True}))
        implementation = _model_implementation_fingerprint(context, model_id)
        reusable = _find_reusable_model_artifact(
            context, artifact_type="model", model_id=model_id,
            fold_id="final", dependency=dependency,
            implementation=implementation, time_scope="calibration")
        reuse_meta = None
        if reusable is not None:
            source_path, source_record = reusable
            source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
            reuse_meta = context.artifact_resolver.materialize(
                source_path, path, source_record.sha256,
                reused_from_run=source_run,
                source_artifact_id=source_record.artifact_id,
                source_dependency_fingerprint=source_record.dependency_fingerprint,
                source_execution_level=source_record.execution_level)
        else:
            if entry["tuning"]["enabled"]:
                choice = selection["final"][model_id]
                model = fit_final_quantile_model(
                    context, model_id, final_fit, final_early,
                    choice["parameters"])
            else:
                model = fit_model(
                    context, model_id,
                    {"fit": final_fit, "history": final_history})
            model.save(path)
        _record_model_artifact(
            context, path, artifact_type="model", stage="fitting",
            model_id=model_id, fold_id="final", dependency=dependency,
            implementation=implementation, time_scope="calibration",
            reuse=reuse_meta)
        artifacts.append({
            "model_id": model_id, "outer_fold": "final",
            "fit_scope": "final", "path": str(path),
            "sha256": file_sha256(path)})
    path = context.paths.models_dir / "model_index.json"
    _write_json(path, {"models": artifacts})
    dependency = content_hash({
        "features": context.stage_results["features"]["input_hashes"]["features"],
        "tuning": tuning["input_hashes"]["tuning"],
        "model_config": context.config.models})
    return _stage_result(context, "success", outputs={"model_index": str(path)},
                   input_hashes={"fitting": dependency}, started_at=started)


def _prediction_rows(context: RunContext, score: pd.DataFrame, model_id: str,
                     fold_id: str, values: np.ndarray,
                     is_quantile: bool, *, inner_id: str = "outer_refit") -> pd.DataFrame:
    columns = quantile_columns(context.config.protocol)
    median_index = quantiles(context.config.protocol).index(0.5)
    output = pd.DataFrame({
        "model_id": model_id,
        "implementation_level": (
            "validated" if context.config.models["registry"][model_id]
            ["implementation_status"] == "validated" else "prototype"),
        "execution_level": context.execution_level,
        "prediction_type": "quantile" if is_quantile else "point",
        "location_id": score["location_id"].to_numpy(),
        "requested_coordinates": list(zip(
            score.requested_latitude, score.requested_longitude)),
        "gfs_service_coordinates": list(zip(
            score.gfs_service_latitude, score.gfs_service_longitude)),
        "truth_service_coordinates": list(zip(
            score.himawari_service_latitude, score.himawari_service_longitude)),
        "target_time_utc": score.target_time_utc.to_numpy(),
        "forecast_issue_time_utc": score.forecast_issue_time_utc.to_numpy(),
        "lead_time": score.lead_time.to_numpy(), "outer_fold": fold_id,
        "inner_fold": inner_id, "seed": int(
            context.config.protocol["seed_policy"]["tuning_seed"]),
        "y": score.y.to_numpy(dtype=float),
        "point_prediction": values[:, median_index] if is_quantile else values,
        "data_version": context.config.data["version"],
        "feature_version": context.config.features["version"],
        "protocol_revision": context.config.protocol_version,
        "experiment_id": context.run_id,
        "result_status": (
            "official" if context.execution_level == "official" else "diagnostic"),
    })
    for index, column in enumerate(columns):
        output[column] = values[:, index] if is_quantile else np.nan
    return output


def predict(context: RunContext) -> StageResult:
    """Load fitted artifacts and write canonical outer-fold predictions."""
    started = _now()
    fitting = context.stage_results.get("fitting", {})
    if fitting.get("status") != "success":
        return _stage_result(context, "blocked",
                       message="prediction requires fitted model artifacts",
                       started_at=started)
    index = json.loads(Path(
        _stage_outputs(fitting)["model_index"]).read_text(encoding="utf-8"))
    by_key = {(item["outer_fold"], item["model_id"]): item
              for item in index["models"]}
    frame = _load_feature_frame(context)
    protocol = context.config.protocol
    validation, development = protocol["validation"], protocol["development_period"]
    gap = int(validation["gap_days"])
    outer_outputs, final_calibration_outputs, final_test_outputs = [], [], []
    levels = quantiles(protocol)
    for outer in outer_folds(frame, validation, development, gap_days=gap):
        score = _eligible(context, outer.score)
        for model_id in context.selected_models:
            item = by_key[(outer.fold_id, model_id)]
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ContractError("serialized model hash changed before prediction")
            is_quantile = bool(context.config.models["registry"][model_id]["tuning"]["enabled"])
            path = context.paths.prediction_file(model_id, outer.fold_id)
            dependency = model_dependency_fingerprint(
                "prediction", model_id,
                upstream={"model": item["sha256"],
                          "score": stable_object_hash(outer.fold_id)},
                model_record=context.config.models["registry"][model_id],
                stage_contract={"scope": "outer_validation"})
            implementation = _model_implementation_fingerprint(context, model_id)
            reusable = _find_reusable_model_artifact(
                context, artifact_type="prediction", model_id=model_id,
                fold_id=outer.fold_id, dependency=dependency,
                implementation=implementation, time_scope="outer_validation")
            reuse_meta = None
            if reusable is not None:
                source_path, source_record = reusable
                source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
                reuse_meta = context.artifact_resolver.materialize(
                    source_path, path, source_record.sha256,
                    reused_from_run=source_run,
                    source_artifact_id=source_record.artifact_id,
                    source_dependency_fingerprint=source_record.dependency_fingerprint,
                    source_execution_level=source_record.execution_level)
            else:
                model = BaseModel.load(Path(item["path"]))
                values = (np.asarray(model.predict_quantiles(score, levels), dtype=float)
                          if is_quantile else np.asarray(model.predict(score), dtype=float))
                rows = _prediction_rows(
                    context, score, model_id, outer.fold_id, values, is_quantile)
                write_predictions(rows, path, protocol)
            _record_model_artifact(
                context, path, artifact_type="prediction", stage="prediction",
                model_id=model_id, fold_id=outer.fold_id,
                dependency=dependency, implementation=implementation,
                time_scope="outer_validation", reuse=reuse_meta)
            outer_outputs.append({
                "model_id": model_id, "outer_fold": outer.fold_id,
                "path": str(path), "sha256": file_sha256(path)})
    _, _, final_calibration, final_test = _final_blocks(context, frame)
    for model_id in context.selected_models:
        item = by_key[("final", model_id)]
        model_path = Path(item["path"])
        if file_sha256(model_path) != item["sha256"]:
            raise ContractError("serialized final model hash changed before prediction")
        is_quantile = bool(
            context.config.models["registry"][model_id]["tuning"]["enabled"])
        implementation = _model_implementation_fingerprint(context, model_id)
        model = None
        if is_quantile:
            calibration_path = context.paths.prediction_file(
                model_id, "final_calibration")
            calibration_dependency = model_dependency_fingerprint(
                "prediction", model_id, upstream={"model": item["sha256"]},
                model_record=context.config.models["registry"][model_id],
                stage_contract={"scope": "final_calibration"})
            reusable = _find_reusable_model_artifact(
                context, artifact_type="prediction", model_id=model_id,
                fold_id="final_calibration", dependency=calibration_dependency,
                implementation=implementation, time_scope="calibration")
            calibration_reuse = None
            if reusable is not None:
                source_path, source_record = reusable
                source_run = source_path.relative_to(
                    context.paths.outputs_root).parts[1]
                calibration_reuse = context.artifact_resolver.materialize(
                    source_path, calibration_path, source_record.sha256,
                    reused_from_run=source_run,
                    source_artifact_id=source_record.artifact_id,
                    source_dependency_fingerprint=(
                        source_record.dependency_fingerprint),
                    source_execution_level=source_record.execution_level)
            else:
                model = BaseModel.load(model_path)
                calibration_values = np.asarray(
                    model.predict_quantiles(final_calibration, levels), dtype=float)
                calibration_rows = _prediction_rows(
                    context, final_calibration, model_id, "final_calibration",
                    calibration_values, True, inner_id="final_fit")
                write_predictions(
                    calibration_rows, calibration_path, protocol,
                    require_truth=True)
            _record_model_artifact(
                context, calibration_path, artifact_type="prediction",
                stage="prediction", model_id=model_id,
                fold_id="final_calibration", dependency=calibration_dependency,
                implementation=implementation, time_scope="calibration",
                reuse=calibration_reuse)
            final_calibration_outputs.append({
                "model_id": model_id, "path": str(calibration_path),
                "sha256": file_sha256(calibration_path)})
        test_path = context.paths.prediction_file(
            model_id, "final_test_uncalibrated")
        test_dependency = model_dependency_fingerprint(
            "prediction", model_id, upstream={"model": item["sha256"]},
            model_record=context.config.models["registry"][model_id],
            stage_contract={"scope": "final_test_uncalibrated"})
        reusable = _find_reusable_model_artifact(
            context, artifact_type="prediction", model_id=model_id,
            fold_id="final_test_uncalibrated", dependency=test_dependency,
            implementation=implementation, time_scope="final_test")
        test_reuse = None
        if reusable is not None:
            source_path, source_record = reusable
            source_run = source_path.relative_to(
                context.paths.outputs_root).parts[1]
            test_reuse = context.artifact_resolver.materialize(
                source_path, test_path, source_record.sha256,
                reused_from_run=source_run,
                source_artifact_id=source_record.artifact_id,
                source_dependency_fingerprint=source_record.dependency_fingerprint,
                source_execution_level=source_record.execution_level)
        else:
            if model is None:
                model = BaseModel.load(model_path)
            test_values = (
                np.asarray(model.predict_quantiles(final_test, levels), dtype=float)
                if is_quantile else
                np.asarray(model.predict(final_test), dtype=float))
            test_rows = _prediction_rows(
                context, final_test, model_id, "final_test", test_values,
                is_quantile, inner_id="final_fit")
            write_predictions(
                test_rows, test_path, protocol, require_truth=False)
        _record_model_artifact(
            context, test_path, artifact_type="prediction", stage="prediction",
            model_id=model_id, fold_id="final_test_uncalibrated",
            dependency=test_dependency,
            implementation=implementation, time_scope="final_test",
            reuse=test_reuse)
        final_test_outputs.append({
            "model_id": model_id, "path": str(test_path),
            "sha256": file_sha256(test_path)})
    index_path = context.paths.predictions_dir / "prediction_index.json"
    payload = {
        "outer_predictions": outer_outputs,
        "final_calibration_predictions": final_calibration_outputs,
        "final_test_uncalibrated": final_test_outputs}
    _write_json(index_path, payload)
    dependency = content_hash({
        "fitting": fitting["input_hashes"]["fitting"], "artifacts": payload})
    return _stage_result(context, "success",
                   outputs={"prediction_index": str(index_path)},
                   input_hashes={"prediction": dependency}, started_at=started)


def calibrate(context: RunContext) -> StageResult:
    started = _now()
    prediction = context.stage_results.get("prediction", {})
    if prediction.get("status") != "success":
        return _stage_result(context, "blocked",
                       message="calibration requires canonical predictions",
                       started_at=started)
    prediction_index = Path(_stage_outputs(prediction)["prediction_index"])
    source = json.loads(prediction_index.read_text(encoding="utf-8"))
    calibration_by_model = {
        item["model_id"]: item
        for item in source.get("final_calibration_predictions", [])}
    test_by_model = {
        item["model_id"]: item
        for item in source.get("final_test_uncalibrated", [])}
    schedule = context.config.protocol["validation"]["final_fit"]
    fit_end = pd.Timestamp(schedule["fit_end"], tz="UTC") + pd.Timedelta(hours=23)
    early_end = (pd.Timestamp(schedule["early_stop_end"], tz="UTC")
                 + pd.Timedelta(hours=23))
    artifacts = []
    for model_id, calibration_item in sorted(calibration_by_model.items()):
        test_item = test_by_model.get(model_id)
        if test_item is None:
            raise ContractError(
                f"missing final-test predictions for calibration model {model_id}")
        calibration_path, test_path = (
            Path(calibration_item["path"]), Path(test_item["path"]))
        if (file_sha256(calibration_path) != calibration_item["sha256"]
                or file_sha256(test_path) != test_item["sha256"]):
            raise ContractError(
                "final prediction hash changed before calibration")
        path = context.paths.calibration_file(model_id, "final_test")
        dependency = model_dependency_fingerprint(
            "calibration", model_id,
            upstream={"calibration_predictions": calibration_item["sha256"],
                      "test_predictions": test_item["sha256"]},
            model_record=context.config.models["registry"][model_id],
            stage_contract=context.config.protocol["probability"]["calibration"])
        implementation = _model_implementation_fingerprint(context, model_id)
        reusable = _find_reusable_model_artifact(
            context, artifact_type="calibration", model_id=model_id,
            fold_id="final_test", dependency=dependency,
            implementation=implementation, time_scope="final_test")
        reuse_meta = None
        if reusable is not None:
            source_path, source_record = reusable
            source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
            reuse_meta = context.artifact_resolver.materialize(
                source_path, path, source_record.sha256,
                reused_from_run=source_run,
                source_artifact_id=source_record.artifact_id,
                source_dependency_fingerprint=source_record.dependency_fingerprint,
                source_execution_level=source_record.execution_level)
        else:
            calibration_rows = read_predictions(
                calibration_path, context.config.protocol)
            test_rows = read_predictions(
                test_path, context.config.protocol, require_truth=False)
            calibrated = CausalIssueQuantileCalibrator(
                context.config.protocol).fit(
                    calibration_rows, fit_end=fit_end,
                    early_stop_end=early_end).apply(test_rows)
            write_predictions(
                calibrated, path, context.config.protocol, require_truth=False)
        _record_model_artifact(
            context, path, artifact_type="calibration", stage="calibration",
            model_id=model_id, fold_id="final_test", dependency=dependency,
            implementation=implementation,
            time_scope="final_test", reuse=reuse_meta)
        artifacts.append({
            "model_id": model_id, "path": str(path),
            "sha256": file_sha256(path)})
    path = context.paths.calibration_dir / "calibration_index.json"
    payload = {
        "method": context.config.protocol["probability"]["calibration"]["primary"],
        "chronology": context.config.protocol["probability"]["calibration"]["chronology"],
        "prediction_index": str(prediction_index),
        "calibrated_final_test": artifacts,
        "outer_prediction_policy": "uncalibrated_development_evaluation"}
    _write_json(path, payload)
    dependency = content_hash({
        "prediction": prediction["input_hashes"]["prediction"],
        "calibration": context.config.protocol["probability"]["calibration"]})
    status = "success" if artifacts else "skipped"
    return _stage_result(context, status, outputs={"calibration_index": str(path)},
                   input_hashes={"calibration": dependency}, started_at=started)
