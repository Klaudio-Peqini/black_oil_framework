"""Example 04 template: live-oil black-oil extension.

This is the next major physics step after the compressible dead-oil model. The
recommended first implementation should assume a globally saturated live-oil
case so that the primary variables are p_o, S_w and S_g. Later we can add local
variable switching between undersaturated and saturated cells.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    print("Live-oil black-oil extension: proposed implementation sequence")
    print()
    print("Primary variables for the first saturated implementation:")
    print("  p_o, S_w, S_g")
    print("  S_o = 1 - S_w - S_g")
    print()
    print("Required PVT table columns:")
    print("  Bo(p), Bw(p), Bg(p), muo(p), muw(p), mug(p), Rs(p)")
    print()
    print("Component equations:")
    print("  water: d(phi Sw/Bw)/dt + div(uw/Bw) = qw")
    print("  oil:   d(phi So/Bo)/dt + div(uo/Bo) = qo")
    print("  gas:   d(phi(Rs So/Bo + Sg/Bg))/dt + div(Rs uo/Bo + ug/Bg) = qg")
    print()
    print("Recommended first benchmark:")
    print("  pressure depletion below bubble point with free-gas appearance already enabled globally")
    print()
    print("Detailed plan written in docs/live_oil_next_step.md")


if __name__ == "__main__":
    main()
