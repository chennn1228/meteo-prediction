from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from nwp.core.schema import ContractError
from nwp.visualization.figures import generate_run_figures, write_figure_index
from nwp.visualization.style import FigureContract, apply_publication_style, relationship


class VisualizationContractTests(unittest.TestCase):
    def test_formal_claim_gate(self):
        contract = FigureContract("test claim", (), "source.csv", "provisional",
                                  "n=20 sites", "median and bootstrap CI")
        with self.assertRaises(ContractError):
            contract.validate()

    def test_editable_svg_and_hexbin_default(self):
        apply_publication_style()
        self.assertEqual(matplotlib.rcParams["svg.fonttype"], "none")
        fig, ax = plt.subplots()
        chart_type, _ = relationship(ax, np.arange(250), np.arange(250),
                                     xlabel="observed", ylabel="predicted")
        self.assertEqual(chart_type, "hexbin")
        plt.close(fig)

    def test_run_figures_reference_source_tables_without_copying_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics, analysis, figures = root / "06_metrics", root / "07_analysis", root / "08_figures"
            metrics.mkdir(); analysis.mkdir(); figures.mkdir()
            probability = pd.DataFrame({
                "model_id": ["ridge_mos"], "prediction_type": ["quantile"],
                "mean_pinball": [12.0], "n": [10]})
            point = pd.DataFrame({
                "model_id": ["ridge_mos"], "prediction_type": ["quantile"],
                "rmse": [30.0], "n": [10]})
            reliability = pd.DataFrame({
                "model_id": ["ridge_mos", "ridge_mos"],
                "nominal_probability": [.05, .95],
                "empirical_probability": [.06, .94]})
            paths = {}
            for name, table in (("probability_primary", probability),
                                ("point_secondary", point),
                                ("reliability", reliability)):
                path = metrics / f"{name}.csv"; table.to_csv(path, index=False); paths[name] = str(path)
            metrics_index = metrics / "metrics_index.json"
            metrics_index.write_text(json.dumps({"overview": paths, "grouped": {}}), encoding="utf-8")
            summary = analysis / "model_summary.csv"; point.to_csv(summary, index=False)
            analysis_index = analysis / "analysis_index.json"
            analysis_index.write_text(json.dumps({"model_summary": str(summary), "tuning_trials": None}), encoding="utf-8")
            entries = generate_run_figures(
                metrics_index, analysis_index, figures, result_status="diagnostic")
            write_figure_index(figures / "figure_index.json", entries)
            self.assertEqual(len(entries), 3)
            self.assertFalse(list(figures.glob("*.csv")))
            self.assertTrue(all(Path(path).is_file() for item in entries for path in item["files"]))


if __name__ == "__main__":
    unittest.main()
