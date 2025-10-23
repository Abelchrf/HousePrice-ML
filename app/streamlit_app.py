import argparse
from pathlib import Path
import joblib
import streamlit as st
import numpy as np


DEFAULTS = {
"OverallQual": 6,
"GrLivArea": 1500,
"GarageCars": 2,
"TotalBsmtSF": 800,
"YearBuilt": 1975,
}




def load_model(model_dir: str):
path = Path(model_dir) / "model.joblib"
return joblib.load(path)




def main(model_dir: str):
st.set_page_config(page_title="HousePrice-ML")
st.title("🏠 HousePrice-ML — Prédiction de prix")
st.caption("Mini démo : quelques features → prix prédit (modèle Random Forest)")


with st.sidebar:
st.header("Caractéristiques (simplifiées)")
overallqual = st.slider("OverallQual (1-10)", 1, 10, DEFAULTS["OverallQual"])
grlivarea = st.number_input("GrLivArea (surface habitable)", value=DEFAULTS["GrLivArea"], step=10)
garagecars = st.slider("GarageCars", 0, 4, DEFAULTS["GarageCars"])
totalbsmtsf = st.number_input("TotalBsmtSF", value=DEFAULTS["TotalBsmtSF"], step=10)
yearbuilt = st.number_input("YearBuilt", value=DEFAULTS["YearBuilt"], step=1)


model = load_model(model_dir)


# Construit une ligne minimale —
# NB: Le pipeline gère colonnes manquantes via imputation.
sample = {
"OverallQual": overallqual,
"GrLivArea": grlivarea,
"GarageCars": garagecars,
"TotalBsmtSF": totalbsmtsf,
"YearBuilt": yearbuilt,
}


if st.button("Prédire le prix"):
import pandas as pd
X = pd.DataFrame([sample])
pred = model.predict(X)[0]
st.subheader(f"Prix estimé : ${pred:,.0f}")




if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", default="models/random_forest")
args, _ = parser.parse_known_args()
main(args.model_dir)
