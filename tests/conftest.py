from pathlib import Path

import pandas as pd
import pytest

from breast_cancer.data import load_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def source():
    return pd.read_csv(ROOT / "data/Breast_Cancer.csv")


@pytest.fixture(scope="session")
def dataset():
    return load_data(ROOT / "data/Breast_Cancer.csv")


@pytest.fixture
def small(dataset):
    frame = dataset[0]
    return frame.groupby("Status", group_keys=False).head(60).reset_index(drop=True)
