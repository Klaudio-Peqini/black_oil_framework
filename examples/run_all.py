"""Run the currently implemented stable examples.

The Step 4B/4C/5A/5B/5C/5D/6A/6B/6C/7 phase-switching and 3D examples are more experimental and can be
included with ``--include-experimental``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_EXAMPLES = [
    ROOT / "examples" / "01_single_phase_pressure_diffusion.py",
    ROOT / "examples" / "02_two_phase_water_oil_fully_implicit.py",
    ROOT / "examples" / "03_dead_oil_compressible.py",
    ROOT / "examples" / "04_live_oil_saturated_black_oil.py",
]

EXPERIMENTAL_EXAMPLES = [
    ROOT / "examples" / "05_live_oil_phase_switching_black_oil.py",
    ROOT / "examples" / "06_live_oil_conservative_ad_black_oil.py",
    ROOT / "examples" / "07_live_oil_gravity_capillary_controls.py",
    ROOT / "examples" / "08_live_oil_sparse_newton_krylov.py",
    ROOT / "examples" / "09_2d_heterogeneous_black_oil.py",
    ROOT / "examples" / "10_scheduled_field_controls_restart.py",
    ROOT / "examples" / "11_3d_mesh_property_visualization.py",
    ROOT / "examples" / "12_3d_property_fault_well_model.py",
    ROOT / "examples" / "13_3d_visualization_export_diagnostics.py",
    ROOT / "examples" / "14_full_3d_black_oil_integrated.py",
]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--include-experimental",
    action="store_true",
    help="also run the Step 4B/4C/5A/5B/5C/5D/6A/6B/6C/7 experimental examples",
)
args = parser.parse_args()

examples = CORE_EXAMPLES + (EXPERIMENTAL_EXAMPLES if args.include_experimental else [])
for example in examples:
    print(f"\n=== Running {example.name} ===", flush=True)
    subprocess.run([sys.executable, str(example)], check=True)
