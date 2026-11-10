from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "Models" / "linear_regression_model.pkl"
RAW_DATA_DIR = PROJECT_DIR / "Data" / "Raw"

FEATURES = ["Location", "Area", "BHK", "Floor", "Balconies", "Property Age",
            "Parking", "Near Market", "Main Road", "Furnishing"]
NUMERIC = ["Area", "BHK", "Floor", "Balconies", "Property Age"]
CATEGORICAL = ["Location", "Parking", "Near Market", "Main Road", "Furnishing"]

app = Flask(__name__)
model = None

def get_categories():
    result = {c: [] for c in CATEGORICAL}
    csvs = list(RAW_DATA_DIR.glob("*.csv")) if RAW_DATA_DIR.exists() else []
    if csvs:
        try:
            df = pd.read_csv(csvs[0])
            for c in CATEGORICAL:
                if c in df.columns:
                    result[c] = sorted(df[c].dropna().astype(str).unique().tolist())
        except Exception as exc:
            app.logger.warning("Could not read CSV categories: %s", exc)
    return result

@app.route("/")
def home():
    return render_template("index.html", categories=get_categories())

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error="The ML model is not loaded. Check the server terminal."), 500
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error="Invalid form data."), 400
    row = {}
    try:
        for col in NUMERIC:
            value = data.get(col)
            if value is None or str(value).strip() == "":
                return jsonify(error=f"Please enter {col}."), 400
            value = float(value)
            if col == "Area" and value <= 1:
                return jsonify(error="Area must be greater than 1."), 400
            if value < 0:
                return jsonify(error=f"{col} cannot be negative."), 400
            row[col] = value
        for col in CATEGORICAL:
            value = data.get(col)
            if value is None or not str(value).strip():
                return jsonify(error=f"Please select {col}."), 400
            row[col] = str(value).strip()

        input_df = pd.DataFrame([row], columns=FEATURES)
        prediction = float(model.predict(input_df)[0])*1e5
        return jsonify(predicted_price=round(prediction, 2))
    except ValueError:
        return jsonify(error="Please enter valid numeric values."), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed. Check that the saved Pipeline was trained with all 10 features and that categories match the dataset."), 500

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model file not found: {MODEL_PATH}\nSave your model to Models/linear_regression_model.pkl")
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")
    app.run(debug=True)
