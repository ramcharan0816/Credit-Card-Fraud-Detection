import os, math, joblib, numpy as np
from flask import Flask, jsonify, request, send_from_directory
import train_model

app = Flask(__name__, static_folder="static")
if not os.path.exists(train_model.OUT):
    train_model.build_demo()
B = joblib.load(train_model.OUT)

@app.get("/")
def index(): return send_from_directory("static", "index.html")

@app.get("/api/health")
def health(): return jsonify(status="ok", model_source=B["source"])

@app.post("/api/predict")
def predict():
    d = request.get_json(silent=True) or {}
    try:
        f = [float(v) for v in d.get("features", [])]; amt = float(d.get("amount", 0))
    except (TypeError, ValueError):
        return jsonify(error="All inputs must be numbers."), 400
    if len(f) != 28 or not all(map(math.isfinite, f + [amt])) or amt < 0:
        return jsonify(error="Provide amount >= 0 and exactly 28 finite features (V1-V28)."), 400
    x = np.array([f + [B["scaler"].transform([[amt]])[0][0]]])
    p = float(B["model"].predict_proba(x)[0][1])
    return jsonify(fraud=bool(B["model"].predict(x)[0]), probability=p, model_source=B["source"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
