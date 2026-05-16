"""Path setup for tutorial scripts (import plot_utils from examples root)."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent
TUTORIAL_DIR = Path(__file__).resolve().parent

if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))
