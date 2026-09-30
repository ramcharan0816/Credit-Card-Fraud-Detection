# FraudGuard – Credit Card Fraud Detection web app

Flask API + light-themed single-page frontend around the Decision Tree model from `creditcardfraud.ipynb`.

## Run locally
    pip install -r requirements.txt
    python app.py            # http://localhost:5000

## Use your REAL model (important)
The zip has no dataset or trained model, so the app builds a **synthetic demo model** on first start
(the site shows a banner while this is the case). To use the real one:

    pip install -r requirements-train.txt
    python train_model.py --csv creditcard.csv     # Kaggle "Credit Card Fraud Detection" CSV
    git add model/fraud_model.joblib && git commit -m "Add trained model" && git push

Train with the pinned versions in `requirements.txt` so the pickle loads on Render.

## Deploy on Render
1. Push this folder to GitHub (add `model/fraud_model.joblib` if you trained it).
2. Render → New → Blueprint (uses `render.yaml`), or New Web Service with
   build `pip install -r requirements.txt && python train_model.py --demo-if-missing`,
   start `gunicorn app:app`.
