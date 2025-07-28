"""
200 → isotonic curves       (independent calibration)
500 → logistic regression   (learns weights + bias)
200 → final evaluation      (held-out test)

Outputs
  artifacts/calibrators.pkl   (dict: feature → IsotonicRegression)
  artifacts/meta.pkl          (sklearn LogisticRegression)
"""

import joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.isotonic   import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics    import f1_score, roc_auc_score, brier_score_loss
from scipy.special      import logit

RAW  = Path(".")
CONFIG  = Path("../configs"); CONFIG.mkdir(exist_ok=True)

train = pd.read_parquet(RAW/"xsum_train_raw.pq")        # 500
calib = pd.read_parquet(RAW/"xsum_calibration_raw.pq")  # 200
test  = pd.read_parquet(RAW/"xsum_test_raw.pq")         # 200

# list of raw feature columns (everything except keys & label)
feat_raw = [c for c in calib.columns if c not in {"bbcid","system","label"}]
print("Features:", feat_raw)

# per-feature isotonic calibration on 200-calib
calibrators = {}
for col in feat_raw:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(calib[col].values, calib.label.values)
    calibrators[col] = ir
joblib.dump(calibrators, CONFIG/"calibrators.pkl")
print("Saved calibrators.pkl")

# helper: return calibrated matrix for any DF
def calibrated_matrix(df):
    return np.vstack([
        calibrators[c].predict(df[c].values).clip(1e-4, 1-1e-4)
        for c in feat_raw
    ]).T        # shape (N, n_features)

# logistic regression on 500-train
y_train = train.label.values

# X_train = (X_train - 0.5) * 2           # centre to 0, range≈[–1,1]


from scipy.special import logit as sp_logit

X_train = calibrated_matrix(train)
X_train = sp_logit(X_train.clip(1e-4, 1-1e-4))   # ← apply true logit
meta    = LogisticRegression(max_iter=500, class_weight="balanced") \
              .fit(X_train, y_train)

joblib.dump(meta, CONFIG/"meta.pkl")
print("Saved meta.pkl")

# evaluation on 200-test
X_test = sp_logit(calibrated_matrix(test).clip(1e-4, 1-1e-4))
p_test = meta.predict_proba(X_test)[:,1]

y_test = test.label.values


print("\n=== Held-out 200-test ===")
print(f"F1  @0.5 : {f1_score(y_test, p_test>0.5):.3f}")
print(f"AUROC    : {roc_auc_score(y_test, p_test):.3f}")
print(f"Brier    : {brier_score_loss(y_test, p_test):.3f}")

