from flask import Flask, jsonify, render_template, request
from models.predictor import FEATURE_ORDER, predict_from_dict

app = Flask(__name__)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = predict_from_dict(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Error during prediction: {exc}"}), 500


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    if request.method == "POST":
        try:
            result = predict_from_dict(request.form)
        except Exception as exc:
            error = str(exc)
    return render_template("index.html", feature_order=FEATURE_ORDER, result=result, error=error)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
