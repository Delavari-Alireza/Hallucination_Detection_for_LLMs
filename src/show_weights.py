import joblib, pandas as pd
from pathlib import Path

CONFIG = Path("../configs")


meta        = joblib.load(CONFIG / "meta.pkl")
calibrators = joblib.load(CONFIG / "calibrators.pkl")
feat_raw    = list(calibrators.keys())

coef  = meta.coef_.flatten()
inter = meta.intercept_[0]

print("\n===  Logistic-Regression Weights  ===")
print(f"{'feature':<18s} weight")
print("-"*30)
for f, w in zip(feat_raw, coef):
    print(f"{f:<18s} {w: .4f}")
print(f"{'bias (intercept)':<18s} {inter: .4f}")
