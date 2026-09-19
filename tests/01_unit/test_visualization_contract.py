from __future__ import annotations

import sys
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import ProtocolError
from s13_visualization.s01_style.publication import FigureContract, apply_publication_style
from s13_visualization.s02_relationships.density import relationship


class VisualizationContractTests(unittest.TestCase):
    def test_formal_claim_gate(self):
        contract = FigureContract("test claim", ("panel a supports claim",),
                                  "quantitative_grid", "source.csv", "provisional",
                                  "n=20 sites", "median and bootstrap CI")
        with self.assertRaises(ProtocolError):
            contract.validate()

    def test_editable_svg_and_hexbin_default(self):
        apply_publication_style()
        self.assertEqual(matplotlib.rcParams["svg.fonttype"], "none")
        fig, ax = plt.subplots()
        chart_type, _ = relationship(ax, np.arange(250), np.arange(250),
                                     xlabel="observed", ylabel="predicted")
        self.assertEqual(chart_type, "hexbin")
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
