from pathlib import Path
import joblib
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "iris_pipeline.joblib"
META_PATH = BASE_DIR / "model" / "meta.joblib"

app = Flask(__name__)
CORS(app)

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run `python train.py` first to generate model files."
    )

model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    features = [[
        float(payload["sepal_length"]),
        float(payload["sepal_width"]),
        float(payload["petal_length"]),
        float(payload["petal_width"]),
    ]]

    pred_idx = int(model.predict(features)[0])
    pred_name = meta["target_names"][pred_idx]
    probs = model.predict_proba(features)[0].tolist()

    return jsonify(
        {
            "prediction_index": pred_idx,
            "prediction_label": pred_name,
            "probabilities": {
                name: round(prob, 4)
                for name, prob in zip(meta["target_names"], probs)
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)