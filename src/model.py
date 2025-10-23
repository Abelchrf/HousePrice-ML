from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from .features import build_preprocessor




def build_pipeline(df_sample):
pre = build_preprocessor(df_sample)
model = RandomForestRegressor(
n_estimators=400,
max_depth=None,
random_state=42,
n_jobs=-1,
)
pipe = Pipeline([
("pre", pre),
("rf", model),
])
return pipe
