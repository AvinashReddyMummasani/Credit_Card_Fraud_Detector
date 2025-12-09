from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import uvicorn

from source.predict import predict

app = FastAPI(
    title="Fraud Detection API",
    version="1.0"
)

class SingleTransaction(BaseModel):
    features: List[float]
    threshold: Optional[float] = None

class BatchRequest(BaseModel):
    transactions: List[SingleTransaction]

class PredictResponse(BaseModel):
    success: bool
    result: Any


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_single(req: SingleTransaction):
    try:
        output = predict(req.features)
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
        return {"success": True, "result": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(req: BatchRequest):
    try:
        batch = [t.features for t in req.transactions]
        output = predict(batch)
        return {"success": True, "result": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)