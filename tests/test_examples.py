import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import report_blind_spot  # noqa: E402


class ReportBlindSpotTest(unittest.TestCase):
    def test_wrong_or_missing_reports_pass_unless_none_is_filed(self):
        result = report_blind_spot.check()
        self.assertEqual(result["gold"], 1.0)
        self.assertEqual(result["wrong_report_numbers"], 1.0)
        self.assertEqual(result["db_tables_that_differ"], [])
        self.assertEqual(result["first_report_only"], 1.0)
        self.assertEqual(result["no_report_at_all"], 0.0)


if __name__ == "__main__":
    unittest.main()
