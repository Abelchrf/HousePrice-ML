import argparse
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import joblib
from pathlib import Path
from .data import load_train
from .model import build_pipeline




def rmse_scorer(y_true, y_pred):
return np.sqrt(mean_squared_error(y_true, y_pred))




def main():
p = argparse.ArgumentParser()
p.add_argument("--train_path", default="data/raw/train.csv")
p.add_argument("--model_dir", default="models/random_forest")
args = p.parse_args()


df = load_train(args.train_path)
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]


pipe = build_pipeline(df)
# CV 5 folds sur RMSE (négatif par scikit → on inverse le signe)
neg_rmse = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)


print(f"CV RMSE: {(-neg_rmse).mean():.2f} ± {(-neg_rmse).std():.2f}")
print(f"CV R2 : {r2.mean():.3f} ± {r2.std():.3f}")


# Entraîne sur tout et sauvegarde
pipe.fit(X, y)
out = Path(args.model_dir)
out.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, out / "model.joblib")
print(f"✅ Modèle sauvegardé -> {out / 'model.joblib'}")




if __name__ == "__main__":
main()
