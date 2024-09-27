from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

try:
    model = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return "API de Predição de Cogumelos com Machine Learning"


@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Modelo não carregado corretamente."}), 500

    try:
        data = request.get_json().get("data", None)

        if data is None:
            return jsonify({"error": "Dados ausentes na solicitação."}), 400

        feature_names = [
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ]

        # Gera o DataFrame a partir dos dados
        df = pd.DataFrame([data], columns=feature_names)

        # Realiza a predição usando o modelo carregado
        prediction = model.predict(df)

        # Retorna o resultado como JSON
        return jsonify({"prediction": prediction.tolist()})

    except ValueError as ve:
        return jsonify({"error": f"Value error: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error occurred: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
