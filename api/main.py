from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from adaptive.adaptive_router import (
    predict_ticket,
    confirm_new_category,
)

TRAIN_CSV = PROJECT_ROOT / "data" / "splits" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_svm_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "best_tfidf_vectorizer.pkl"

app = Flask(__name__)


# =========================================================
# FRONTEND
# =========================================================

@app.route("/")
def home():
    return send_from_directory(PROJECT_ROOT / "frontend", "index.html")


@app.route("/<path:filename>")
def frontend_files(filename):
    return send_from_directory(PROJECT_ROOT / "frontend", filename)


# =========================================================
# PREDICT TICKET
# =========================================================

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Invalid JSON request."}), 400

        ticket = str(data.get("ticket", "")).strip()

        if not ticket:
            return jsonify({"error": "Ticket description cannot be empty."}), 400

        return jsonify(predict_ticket(ticket))

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# CREATE NEW CATEGORY + RETRAIN
# =========================================================

@app.route("/api/create-category", methods=["POST"])
def create_category():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({"error": "Invalid JSON request."}), 400

        ticket = str(data.get("ticket", "")).strip()
        category = str(data.get("category", "")).strip()

        if not ticket:
            return jsonify({"error": "Ticket description cannot be empty."}), 400

        if not category:
            return jsonify({"error": "Category name cannot be empty."}), 400

        return jsonify(confirm_new_category(ticket, category))

    except Exception as e:
        print("Category creation error:", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# CATEGORIES (read live from data/splits/train.csv)
# =========================================================

@app.route("/api/categories", methods=["GET"])
def categories():
    try:
        if not TRAIN_CSV.exists():
            return jsonify({"error": f"Training file not found: {TRAIN_CSV}"}), 404

        df = pd.read_csv(TRAIN_CSV)

        if "queue" not in df.columns:
            return jsonify({"error": "train.csv has no 'queue' column."}), 500

        counts = df["queue"].dropna().astype(str).str.strip().value_counts()

        items = [
            {"name": name, "count": int(count)}
            for name, count in counts.items()
            if name
        ]
        items.sort(key=lambda c: c["name"].lower())

        return jsonify({
            "categories": items,
            "total_examples": int(sum(c["count"] for c in items)),
        })

    except Exception as e:
        print("Categories error:", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# MODEL STATUS (file facts only, no invented metrics)
# =========================================================

def file_info(path):
    if not path.exists():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "exists": False}

    stat = path.stat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "size_kb": round(stat.st_size / 1024, 1),
        "last_modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.route("/api/model-status", methods=["GET"])
def model_status():
    try:
        return jsonify({
            "status": "online",
            "model": file_info(MODEL_PATH),
            "vectorizer": file_info(VECTORIZER_PATH),
            "training_data": file_info(TRAIN_CSV),
        })
    except Exception as e:
        print("Model status error:", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)