from __future__ import annotations

import unittest
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


class TestEnvPolicy(unittest.TestCase):
    def test_no_repo_local_virtualenvs(self) -> None:
        root = _repo_root()
        forbidden = sorted(
            [
                path.name
                for path in root.glob(".venv*")
                if path.is_dir()
            ]
        )
        self.assertEqual(
            forbidden,
            [],
            msg=f"Forbidden repo-local venv(s) present: {forbidden}. Create your venv outside the repo (e.g., ~/venvs/precrime)",
        )


if __name__ == "__main__":
    unittest.main()

