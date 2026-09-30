"""Train the fraud model exactly as in creditcardfraud.ipynb and save it for the web app.
Usage:  python train_model.py --csv creditcard.csv      (real model)
        python train_model.py --demo-if-missing         (used by Render build; demo fallback)"""
import argparse, os, joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "fraud_model.joblib")
FRAUD_MEAN = [-4.77,3.62,-7.03,4.54,-3.15,-1.40,-5.57,0.57,-2.58,-5.68,3.80,-6.26,-0.11,-6.97,
              -0.09,-4.14,-6.67,-2.24,0.68,0.37,0.71,0.01,-0.04,-0.10,0.04,0.05,0.17,0.08]

def save(model, scaler, source, path=OUT):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "source": source}, path)

def train_real(csv):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    df = pd.read_csv(csv, engine="python", on_bad_lines="skip").apply(pd.to_numeric, errors="coerce").dropna()
    sc = StandardScaler(); df["Amount"] = sc.fit_transform(df["Amount"].values.reshape(-1, 1))
    df = df.drop("Time", axis=1)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = DecisionTreeClassifier(random_state=0, criterion="entropy").fit(Xtr, ytr)
    print(classification_report(yte, clf.predict(Xte)))
    save(clf, sc, "trained")

def build_demo(path=OUT):
    """Synthetic stand-in so the site runs without the dataset. NOT the real model."""
    rng = np.random.default_rng(0)
    Xl = rng.normal(0, 1.2, (20000, 28)); Xf = rng.normal(FRAUD_MEAN, 1.6, (1500, 28))
    amt = np.r_[rng.gamma(2, 45, 20000), rng.gamma(1.5, 60, 1500)]
    sc = StandardScaler().fit(amt.reshape(-1, 1))
    X = np.c_[np.r_[Xl, Xf], sc.transform(amt.reshape(-1, 1))]
    y = np.r_[np.zeros(20000), np.ones(1500)]
    save(DecisionTreeClassifier(random_state=0, criterion="entropy", min_samples_leaf=5).fit(X, y), sc, "demo", path)

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--csv"); a.add_argument("--demo-if-missing", action="store_true")
    a = a.parse_args()
    if a.csv: train_real(a.csv)
    elif a.demo_if_missing and not os.path.exists(OUT): build_demo()
