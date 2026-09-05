from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib, json, numpy as np, logging, time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("heart-disease-api")

app = FastAPI(title="Heart Disease Classifier API")

model = joblib.load("artifacts/model.joblib")
FEATURE_COLS = json.load(open("artifacts/feature_cols.json"))
MEDIANS = json.load(open("artifacts/impute_medians.json"))

GENDER_MAP = {"male": 0, "female": 1}

class HeartInput(BaseModel):
    age: float
    gender: str          # "male" / "female"
    cp: float
    trestbps: float | None = None
    chol: float | None = None
    fbs: float
    restecg: float
    thalach: float | None = None
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

@app.get("/")
def root():
    return {"message": "Heart Disease Classifier API is running"}

@app.get("/live_check")
def live_check():
    return {"status": "alive"}

@app.get("/ready_check")
def ready_check():
    return {"status": "ready"}

@app.post("/predict")
def predict(data: HeartInput):
    start = time.time()
    try:
        d = data.dict()
        for col in ["trestbps", "chol", "thalach"]:
            if d[col] is None:
                d[col] = MEDIANS[col]

        gender_enc = GENDER_MAP[d["gender"]]

        row = [d["age"], gender_enc, d["cp"], d["trestbps"], d["chol"], d["fbs"],
               d["restecg"], d["thalach"], d["exang"], d["oldpeak"], d["slope"],
               d["ca"], d["thal"]]

        pred = model.predict(np.array([row]))[0]
        result = {"prediction": "yes" if pred == 1 else "no"}

        logger.info(json.dumps({
            "event": "prediction", "input": d, "result": result,
            "latency_ms": round((time.time()-start)*1000, 2)
        }))
        return result
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
