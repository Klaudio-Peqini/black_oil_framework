from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    np.savez(path, **arrays)


def save_csv(path: str | Path, data: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    pd.DataFrame(data).to_csv(path, index=False)
