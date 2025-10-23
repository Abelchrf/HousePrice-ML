import argparse
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


from .data import load_train
from .model import build_pipeline




def main():
p = argparse.ArgumentParser()
p.add_argument("--train_path", default="data/raw/train.csv")
p.add_argument("--model_dir", default="models/random_forest")
args = p.parse_args()


df = load_train(args.train_path)
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


pipe = build_pipeline(df)
pipe.fit(X_train, y_train)


preds = pipe.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, preds))
print(f"Validation RMSE: {rmse:.2f}")


out = Path(args.model_dir)
out.mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, out / "model.joblib")
print(f"✅ Modèle sauvegardé -> {out / 'model.joblib'}")




if __name__ == "__main__":
main()
