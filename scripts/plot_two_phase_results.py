"""Utility script to replot outputs from example 02."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="Directory containing two_phase_results.npz")
    args = parser.parse_args()

    data = np.load(args.output_dir / "two_phase_results.npz")
    report = pd.read_csv(args.output_dir / "two_phase_timestep_report.csv")

    time_days = data["time"] / 86400.0
    sw = data["sw"]
    nx = sw.shape[1]
    x = np.arange(nx)

    plt.figure(figsize=(7.5, 4.8))
    for idx in [0, len(time_days) // 4, len(time_days) // 2, -1]:
        plt.plot(x, sw[idx], label=f"t={time_days[idx]:.0f} d")
    plt.xlabel("Cell index")
    plt.ylabel("Water saturation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "replot_sw_profiles.png", dpi=180)

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(report["time_days"], report["recovery_factor"], marker="o")
    plt.xlabel("Time [days]")
    plt.ylabel("Recovery factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output_dir / "replot_recovery_factor.png", dpi=180)


if __name__ == "__main__":
    main()
