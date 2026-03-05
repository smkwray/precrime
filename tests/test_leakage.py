import unittest

from src.data.load_nij import _load_schema
from src.features.build_nij_dynamic import get_dynamic_feature_columns
from src.features.build_nij_static import HORIZON_FILTERS, HORIZON_TARGETS, get_static_feature_columns, row_passes_horizon


class TestLeakageGuardrails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fields = _load_schema()
        cls.target_columns = [str(field["name"]) for field in fields if str(field["family"]) == "target"]
        cls.dynamic_columns = [str(field["name"]) for field in fields if str(field["family"]) == "dynamic_supervision"]

    def test_static_excludes_dynamic_supervision(self):
        static_features = set(get_static_feature_columns())
        self.assertTrue(static_features)
        self.assertFalse(static_features.intersection(self.dynamic_columns))

    def test_dynamic_track_y1_excludes_dynamic_supervision(self):
        y1_features = set(get_dynamic_feature_columns("y1"))
        self.assertFalse(y1_features.intersection(self.dynamic_columns))

    def test_dynamic_track_y2_y3_include_dynamic_supervision(self):
        y2_features = set(get_dynamic_feature_columns("y2"))
        y3_features = set(get_dynamic_feature_columns("y3"))
        self.assertTrue(self.dynamic_columns)
        self.assertTrue(set(self.dynamic_columns).issubset(y2_features))
        self.assertTrue(set(self.dynamic_columns).issubset(y3_features))

    def test_feature_sets_exclude_target_columns(self):
        static_features = set(get_static_feature_columns())
        dynamic_y1 = set(get_dynamic_feature_columns("y1"))
        dynamic_y2 = set(get_dynamic_feature_columns("y2"))
        dynamic_y3 = set(get_dynamic_feature_columns("y3"))

        for feature_set in (static_features, dynamic_y1, dynamic_y2, dynamic_y3):
            self.assertFalse(feature_set.intersection(self.target_columns))

    def test_horizon_filter_definitions(self):
        self.assertEqual(HORIZON_FILTERS["y1"], tuple())
        self.assertEqual(HORIZON_FILTERS["y2"], ("Recidivism_Arrest_Year1",))
        self.assertEqual(HORIZON_FILTERS["y3"], ("Recidivism_Arrest_Year1", "Recidivism_Arrest_Year2"))
        self.assertEqual(set(HORIZON_TARGETS.keys()), {"y1", "y2", "y3"})

    def test_row_filter_logic_matches_horizon_rules(self):
        baseline = {
            "Recidivism_Arrest_Year1": 0,
            "Recidivism_Arrest_Year2": 0,
            "Recidivism_Arrest_Year3": 1,
        }
        self.assertTrue(row_passes_horizon(baseline, "y1"))
        self.assertTrue(row_passes_horizon(baseline, "y2"))
        self.assertTrue(row_passes_horizon(baseline, "y3"))

        year1_recid = dict(baseline)
        year1_recid["Recidivism_Arrest_Year1"] = 1
        self.assertTrue(row_passes_horizon(year1_recid, "y1"))
        self.assertFalse(row_passes_horizon(year1_recid, "y2"))
        self.assertFalse(row_passes_horizon(year1_recid, "y3"))

        year2_recid = dict(baseline)
        year2_recid["Recidivism_Arrest_Year2"] = 1
        self.assertTrue(row_passes_horizon(year2_recid, "y2"))
        self.assertFalse(row_passes_horizon(year2_recid, "y3"))


if __name__ == "__main__":
    unittest.main()
