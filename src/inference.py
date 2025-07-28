import sys
sys.path.append('./src')
from LLM_based import run_self_questioning , prompt_paring_prob
from factScore_utils import compute_fact_score
from componentBase_utils import compute_semantic_entailment, compute_topic_drift
from pathlib import Path
import joblib, numpy as np, pandas as pd
from calculate_uncertainty import compute_uncertainty
from scipy.special import logit as sp_logit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
# ART       = Path("../configs")
CONFIG       = PROJECT_ROOT / "configs"
CALIBRORS = joblib.load(CONFIG/"calibrators.pkl")      # dict: feat → IsotonicRegression
META      = joblib.load(CONFIG/"meta.pkl")             # LogisticRegression
# with open(ART/"threshold.txt") as f:
#     THR = float(f.read().strip())

FEAT_RAW  = list(CALIBRORS.keys())                  # ['p_true', ..., 'topic_drift']

def infer(article: str, summary: str , THR=0.45):
    """
    For a single (article, summary), run:
     - Monte-Carlo Q1/Q2 and inverse-PPL → p_true, p_contrad, inv_ppl
     - FactScore → fact_score, entity_precision, triple_precision
     - Semantic entailment & topic drift → sem_entail, topic_drift
     - Isotonic calibration of each feature
     - Logistic meta-model → probability + binary prediction

    Returns a dict with keys:
     raw_<feat>, cal_<feat>, probability, predicted
    """
    # MC Q1/Q2 + inverse-PPL
    inv_ppl = compute_uncertainty(article, summary)
    p_true , p_contrad = prompt_paring_prob(article, summary)
    # FactScore
    fs     = compute_fact_score(summary, article)

    # Semantic entailment & Topic drift
    sem    = compute_semantic_entailment(article, summary)
    drift  = compute_topic_drift(article, summary)

    # Collect raw features
    raw = {
        "p_true":           p_true,
        "p_contrad":        p_contrad,
        "inv_ppl":          inv_ppl,
        "fact_score":       fs["fact_score"],
        "entity_precision": fs["entity_precision"],
        "triple_precision": fs["triple_precision"],
        "sem_entail":       sem,
        "topic_drift":      drift
    }

    # Calibrate each
    cal = {}
    for feat in FEAT_RAW:
        cal_val = CALIBRORS[feat].predict([ raw[feat] ])[0]
        cal[feat+"_cal"] = float(np.clip(cal_val, 1e-4, 1-1e-4))

    # Meta-model probability & prediction
    p_vec = np.array([cal[f"{feat}_cal"] for feat in FEAT_RAW])
    X = sp_logit(p_vec.reshape(1, -1))
    prob = float(META.predict_proba(X)[0, 1])
    pred = int(prob > THR)

    #Merge outputs
    out = {}
    # raw
    for k,v in raw.items():
        out[f"raw_{k}"] = v

    # calibrated
    out.update(cal)
    # meta
    out["probability"] = prob
    out["predicted"]   = pred

    return out


# A_summ ="Alireza lives in Tehran"
# B_summ ="Alireza hates learning."
# C_summ ="Alireza lives in Newyork."
# D_summ ="Alireza is unemployed."
# E_summ ="Alireza is engineer."
# A = infer(article="Alireza Delavari is an AI engineer. he lives in Tehran and loves to learn new things." , summary=A_summ)
# B = infer(article="Alireza Delavari is an AI engineer. he lives in Tehran and loves to learn new things." , summary=B_summ)
# C = infer(article="Alireza Delavari is an AI engineer. he lives in Tehran and loves to learn new things." , summary=C_summ)
# D = infer(article="Alireza Delavari is an AI engineer. he lives in Tehran and loves to learn new things." , summary=D_summ)
# E = infer(article="Alireza Delavari is an AI engineer. he lives in Tehran and loves to learn new things." , summary=E_summ)
# print(f"A : {A_summ}\n")
# print(A)
# print(f"B : {B_summ}\n")
# print(B)
# print(f"C : {C_summ} \n")
# print(C)
# print(f"D : {D_summ} \n")
# print(D)
# print(f"D : {E_summ} \n")
# print(E)