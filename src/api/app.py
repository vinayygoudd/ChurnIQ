from flask import Flask,request,jsonify
from src.api.schemas import validate_payload
from src.models.predict import predict
from src.explainability.shap_explainer import explain_row
from src.database.database import init_db,save_prediction,stats
from src.utils.config import settings
import json

def create_app():
    app=Flask(__name__); init_db()
    @app.get("/health")
    def health(): return jsonify({"status":"ok","model_version":settings.model_version})
    @app.post("/predict")
    def prediction():
        try:
            payload=validate_payload(request.get_json(force=True)); result=predict(payload); save_prediction(payload.get("customer_id"),result["prediction"],result["probability"],result["risk_level"],result["model_version"]); return jsonify(result)
        except ValueError as e: return jsonify({"error":str(e)}),400
        except FileNotFoundError: return jsonify({"error":"Model artifact not found. Run training first."}),503
        except Exception: return jsonify({"error":"Prediction failed."}),500
    @app.get("/model-info")
    def model_info():
        try: return jsonify(json.loads((settings.model_dir/"metadata.json").read_text()))
        except FileNotFoundError: return jsonify({"error":"Model metadata not found."}),503
    @app.post("/explain")
    def explain():
        try: return jsonify(explain_row(validate_payload(request.get_json(force=True))))
        except ValueError as e: return jsonify({"error":str(e)}),400
        except Exception: return jsonify({"error":"Explanation failed."}),500
    @app.get("/analytics")
    def analytics(): return jsonify(stats())
    return app

app=create_app()
if __name__=="__main__": app.run(host="0.0.0.0",port=8000,debug=False)
