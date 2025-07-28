from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from src.inference import infer

app = FastAPI(title="Hallucination Detector")

class InferRequest(BaseModel):
    article: str
    summary: str

class InferResponse(BaseModel):
    # raw features
    raw_p_true: float
    raw_p_contrad: float
    raw_inv_ppl: float
    raw_fact_score: float
    raw_entity_precision: float
    raw_triple_precision: float
    raw_sem_entail: float
    raw_topic_drift: float

    # calibrated features
    p_true_cal: float
    p_contrad_cal: float
    inv_ppl_cal: float
    fact_score_cal: float
    entity_precision_cal: float
    triple_precision_cal: float
    sem_entail_cal: float
    topic_drift_cal: float

    # output
    probability: float
    predicted: int

@app.post("/infer", response_model=InferResponse)
async def infer_endpoint(req: InferRequest):
    try:
        out: Dict = infer(req.article, req.summary)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
