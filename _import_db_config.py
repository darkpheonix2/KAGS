"""Add RAGs root to sys.path so dataset scripts can import db_config."""

import sys
from pathlib import Path

_RAGS_ROOT = Path(__file__).resolve().parent
if str(_RAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAGS_ROOT))
