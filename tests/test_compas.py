from __future__ import annotations

import unittest
from pathlib import Path

from src.features.build_compas import build_compas_2yr


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class TestCompasBenchmark(unittest.TestCase):
    def test_compas_2yr_builder_contract(self) -> None:
        raw_csv = _repo_root() / "data" / "raw" / "compas" / "compas-scores-two-years.csv"
        if not raw_csv.exists():
            self.skipTest("COMPAS raw CSV not present; run src/data/download_compas.py first")

        frame = build_compas_2yr()
        self.assertEqual(len(frame), 6172)
        self.assertIn("y", frame.columns)
        self.assertNotIn("target", frame.columns)

        values = set(frame["y"].dropna().astype(int).unique().tolist())
        self.assertTrue(values.issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()

