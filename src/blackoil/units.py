"""Unit-conversion helpers.

The framework uses SI units internally. These constants make examples easier to read.
"""

day = 24.0 * 3600.0
year = 365.0 * day
cp = 1.0e-3  # Pa s
bar = 1.0e5  # Pa
psi = 6894.757293168  # Pa
md = 9.869233e-16  # m^2


def md_to_m2(value_md: float) -> float:
    """Convert millidarcy to square metres."""
    return float(value_md) * md


def m2_to_md(value_m2: float) -> float:
    """Convert square metres to millidarcy."""
    return float(value_m2) / md
