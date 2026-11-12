import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_site  # noqa: E402


class SiteTest(unittest.TestCase):
    def test_the_page_matches_the_results_files(self):
        self.assertEqual(
            (REPO / "docs" / "index.html").read_text(),
            build_site.render(),
            "docs/index.html is stale: run python3 scripts/build_site.py",
        )


if __name__ == "__main__":
    unittest.main()
