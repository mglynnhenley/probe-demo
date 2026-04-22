"""Put train/ on sys.path so its flat internal imports (``from models import ...``) resolve."""
from __future__ import annotations

import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
