from __future__ import annotations
import pandas as pd
from pathlib import Path




def load_train(path: str | Path) -> pd.DataFrame:
path = Path(path)
assert path.exists(), f"Missing train file: {path}"
df = pd.read_csv(path)
assert "SalePrice" in df.columns, "train.csv must contain 'SalePrice'"
return df




def load_test(path: str | Path) -> pd.DataFrame:
path = Path(path)
assert path.exists(), f"Missing test file: {path}"
df = pd.read_csv(path)
return df
