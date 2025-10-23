# HousePrice-ML — Prédire le prix de maisons (Kaggle)


Petit projet ML **facilement explicable** : pipeline `scikit-learn` + `RandomForestRegressor` sur le dataset Kaggle *House Prices*.


## ⚙️ Stack
- Python 3.10+
- pandas, numpy, scikit-learn, joblib
- (option) streamlit pour une démo légère


## 📦 Données
1. Télécharge les fichiers depuis Kaggle : **House Prices — Advanced Regression Techniques**.
2. Place `train.csv` et `test.csv` dans `data/raw/` (créé si besoin).


## 🚀 Démarrage rapide
```bash
# Créer/activer l'environnement (Win PowerShell)
python -m venv .venv
.venv\Scripts\activate


# Installer les dépendances
pip install -r requirements.txt


# Entraîner le modèle (sauvegarde dans models/random_forest)
python -m src.train --train_path data/raw/train.csv --model_dir models/random_forest


# Évaluer (CV RMSE / R2)
python -m src.evaluate --train_path data/raw/train.csv --model_dir models/random_forest


# Générer un fichier de soumission Kaggle (Id,SalePrice)
python -m src.predict --test_path data/raw/test.csv --model_dir models/random_forest --out_path submission.csv


# (Option) Lancer la démo Streamlit
streamlit run app/streamlit_app.py -- --model_dir models/random_forest
