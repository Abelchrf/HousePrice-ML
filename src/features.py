from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder




def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
# Sépare types
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if "SalePrice" in numeric_features:
numeric_features.remove("SalePrice")
categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()


numeric_transformer = SimpleImputer(strategy="median")
categorical_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="most_frequent")),
("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])


preprocessor = ColumnTransformer(
transformers=[
("num", numeric_transformer, numeric_features),
("cat", categorical_transformer, categorical_features),
],
remainder="drop",
)
return preprocessor
