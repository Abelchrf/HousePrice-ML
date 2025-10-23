import argparse
import pandas as pd
import joblib
from pathlib import Path




def main():
p = argparse.ArgumentParser()
p.add_argument("--test_path", default="data/raw/test.csv")
p.add_argument("--model_dir", default="models/random_forest")
p.add_argument("--out_path", default="submission.csv")
args = p.parse_args()


model_path = Path(args.model_dir) / "model.joblib"
assert model_path.exists(), f"Model not found: {model_path}"
pipe = joblib.load(model_path)


test_df = pd.read_csv(args.test_path)
preds = pipe.predict(test_df)


sub = pd.DataFrame({"Id": test_df["Id"], "SalePrice": preds})
sub.to_csv(args.out_path, index=False)
print(f"✅ Submission saved to {args.out_path}")




if __name__ == "__main__":
main()
