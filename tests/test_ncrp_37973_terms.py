import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipelines.run_ncrp_37973_benchmark import _sanitize_frame
from src.pipelines.run_ncrp_37973_terms import SampleSpec, write_outputs


class TestNcrp37973Terms(unittest.TestCase):
    def test_write_outputs_excludes_next_admit_year_from_model_tables(self):
        frame = pd.DataFrame(
            {
                "ABT_INMATE_ID": ["A1", "A2", "A3"],
                "state": ["1", "1", "2"],
                "sex": ["1", "2", "1"],
                "race": ["1", "9", "1"],
                "RELEASEYR": [2010, 2011, 2012],
                "NEXT_ADMITYR": [2011, None, 2014],
                "y1": [1, 0, 0],
                "y2": [0, 1, 0],
                "y3": [0, 0, 1],
            }
        )

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            write_outputs(frame, spec=SampleSpec(id_mod=1, id_rem=0), out_dir=out_dir)

            y1 = pd.read_parquet(out_dir / "ncrp_icpsr_37973_terms_mod1_r0_y1.parquet")
            y2 = pd.read_parquet(out_dir / "ncrp_icpsr_37973_terms_mod1_r0_y2.parquet")
            y3 = pd.read_parquet(out_dir / "ncrp_icpsr_37973_terms_mod1_r0_y3.parquet")

            for df in (y1, y2, y3):
                self.assertIn("y", df.columns)
                self.assertNotIn("NEXT_ADMITYR", df.columns)

    def test_sanitize_frame_masks_year_sentinels(self):
        frame = pd.DataFrame(
            {
                "MAND_PRISREL_YEAR": [9999, 2012],
                "PROJ_PRISREL_YEAR": [9999, 9999],
                "PARELIG_YEAR": [2011, 9999],
            }
        )
        out = _sanitize_frame(frame)
        self.assertTrue(pd.isna(out.loc[0, "MAND_PRISREL_YEAR"]))
        self.assertTrue(pd.isna(out.loc[0, "PROJ_PRISREL_YEAR"]))
        self.assertTrue(pd.isna(out.loc[1, "PROJ_PRISREL_YEAR"]))
        self.assertTrue(pd.isna(out.loc[1, "PARELIG_YEAR"]))


if __name__ == "__main__":
    unittest.main()

