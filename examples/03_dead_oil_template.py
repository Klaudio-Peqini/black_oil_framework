"""Compatibility wrapper for the dead-oil example.

The dead-oil stage is now implemented in ``03_dead_oil_compressible.py``.
"""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "examples" / "03_dead_oil_compressible.py"), run_name="__main__")
