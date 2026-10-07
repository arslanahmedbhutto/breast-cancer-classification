"""Loading and cleaning the Wisconsin Diagnostic Breast Cancer (WDBC) dataset."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data.csv"

TARGET = "diagnosis"
LABELS = {0: "benign", 1: "malignant"}

# 10 cell-nucleus measurements x 3 statistics (mean, standard error, worst) = 30 features.
_MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
FEATURES = [f"{m}_{stat}" for stat in ("mean", "se", "worst") for m in _MEASUREMENTS]


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with snake_case feature names and y encoded as B=0, M=1."""
    df = pd.read_csv(path)
    # The raw CSV has an id column and a trailing empty column ("Unnamed: 32").
    df = df.drop(columns=["id"]).loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.replace(" ", "_")
    y = df.pop(TARGET).map({"B": 0, "M": 1}).astype(int)
    return df[FEATURES], y
