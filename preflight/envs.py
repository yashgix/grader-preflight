"""The two pinned tau2 interpreters: TAU2_OLD_PYTHON / TAU2_NEW_PYTHON, else the venvs scripts/setup-tau2.sh builds."""

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OLD = os.environ.get("TAU2_OLD_PYTHON", str(_ROOT / ".tau2" / "pre-v1.0.1" / "venv" / "bin" / "python"))
DEFAULT_NEW = os.environ.get("TAU2_NEW_PYTHON", str(_ROOT / ".tau2" / "v1.0.1" / "venv" / "bin" / "python"))
