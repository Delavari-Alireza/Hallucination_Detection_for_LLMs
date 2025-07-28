# #!/usr/bin/env python
# import numpy as np
# import pandas as pd
# import joblib
# from tqdm.auto import tqdm
#
# # corrected import
# from sklearn.calibration import calibration_curve
# from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
# from scipy.special import logit
#
# from inference import infer, FEAT_RAW
# from utils     import extract_docs, build_summary_table, stratified_split
# import warnings
# warnings.filterwarnings("ignore")
#
# # ─── 1) Helper to compute ECE ──────────────────────────────────────────
# def compute_ece(y_true, y_prob, n_bins=10):
#     # calibration_curve expects inputs in [0,1]
#     # it will drop empty bins and return arrays of equal length
#     prob_true, prob_pred = calibration_curve(
#         y_true, y_prob,
#         n_bins=n_bins,
#         strategy="uniform"
#     )
#     # assign preds to bins
#     bins  = np.linspace(0.0, 1.0, n_bins + 1)
#     binids = np.digitize(y_prob, bins) - 1
#
#     ece = 0.0
#     total = len(y_prob)
#     # iterate only over the returned bins
#     for i in range(len(prob_true)):
#         mask = (binids == i)
#         if mask.sum() == 0:
#             continue
#         ece += (mask.sum() / total) * abs(prob_true[i] - prob_pred[i])
#     return ece
#
# # ─── 2) Load calibration data & calibrators ──────────────────────────
# calib_df = pd.read_parquet("xsum_calibration_raw.pq")
# y_calib  = calib_df["label"].values
#
# raw_feats = [
#     c for c in calib_df.columns
#     if c not in {"bbcid","system","label"}
# ]
#
# calibrators = joblib.load("../configs/calibrators.pkl")
#
# # ─── 3) Compute ECE for raw vs calibrated ────────────────────────────
# print("\n=== Calibration ECE (lower is better) ===")
# print(f"{'Feature':20s} {'ECE_raw':>8s} {'ECE_cal':>8s} {'Δ':>8s}")
# print("-"*46)
#
# for feat in raw_feats:
#     # clip raw into [0,1]
#     p_raw = np.clip(calib_df[feat].values, 0.0, 1.0)
#     e_raw = compute_ece(y_calib, p_raw, n_bins=10)
#
#     # calibrated predictions are already in (0,1)
#     p_cal = calibrators[feat].predict(p_raw).clip(1e-4, 1-1e-4)
#     e_cal = compute_ece(y_calib, p_cal, n_bins=10)
#
#     print(f"{feat:20s} {e_raw:8.3f} {e_cal:8.3f} {e_raw-e_cal:8.3f}")
#
# # ─── 4) Meta‐model evaluation on XSum test split ──────────────────────
# test_df = pd.read_parquet("xsum_test_raw.pq")
# y_test  = test_df["label"].values
#
# # build calibrated matrix
# X_cal = np.vstack([
#     calibrators[f].predict(
#         np.clip(test_df[f].values, 0.0, 1.0)
#     ).clip(1e-4,1-1e-4)
#     for f in raw_feats
# ]).T
#
# X_logit = logit(X_cal)
# meta    = joblib.load("../configs/meta.pkl")
#
# probs_test = meta.predict_proba(X_logit)[:,1]
# preds_test = (probs_test > 0.50).astype(int)
#
# print("\n=== Meta‐Model on XSum Test (200 samples) ===")
# print(f"F1    = {f1_score(y_test, preds_test):.3f}")
# print(f"AUROC = {roc_auc_score(y_test, probs_test):.3f}")
# print(f"Brier = {brier_score_loss(y_test, probs_test):.3f}")
#
# # ─── 5) CNN/DailyMail‐Faith evaluation via frank_results.csv ────────
# frank = pd.read_csv("frank_results.csv")
# y_f   = frank["label"].values
# p_f   = frank["probability"].values
# pred_f= frank["predicted"].values
#
# print("\n=== Meta‐Model on CNN/DM‐Faith (150 samples) ===")
# print(f"F1    = {f1_score(y_f, pred_f):.3f}")
# print(f"AUROC = {roc_auc_score(y_f, p_f):.3f}")
# print(f"Brier = {brier_score_loss(y_f, p_f):.3f}")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve
# import joblib
# from pathlib import Path
#
# # Ensure output directory exists
# fig_dir = Path("figures")
# fig_dir.mkdir(exist_ok=True)
#
# # Load calibration and test data
# calib = pd.read_parquet("xsum_calibration_raw.pq")
# test  = pd.read_parquet("xsum_test_raw.pq")
#
# # 1) Raw Signal Histograms (unchanged)
# features = [
#     "p_true", "p_contrad", "inv_ppl",
#     "fact_score", "entity_precision", "triple_precision",
#     "sem_entail", "topic_drift"
# ]
# fig, axes = plt.subplots(4, 2, figsize=(12, 16))
# axes = axes.flatten()
# for ax, feat in zip(axes, features):
#     ax.hist(test[feat][test.label == 1], bins=20, alpha=0.6, label="faithful")
#     ax.hist(test[feat][test.label == 0], bins=20, alpha=0.6, label="hallucinated")
#     ax.set_title(feat)
#     ax.set_xlabel("Raw value")
#     ax.set_ylabel("Count")
# axes[0].legend(loc="upper right")
# plt.tight_layout()
# plt.savefig(fig_dir / "raw_signal_histograms.pdf")
# plt.close(fig)
#
# # 2) Reliability Diagrams for Selected Features ∈ [0,1]
# calibrators = joblib.load("../configs/calibrators.pkl")
# selected = ["p_true", "p_contrad", "inv_ppl", "fact_score"]
#
# fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# axes = axes.flatten()
#
# for ax, feat in zip(axes, selected):
#     # raw curve
#     prob_true_raw, prob_pred_raw = calibration_curve(
#         calib.label.values,
#         calib[feat].values,
#         n_bins=10,
#         strategy="uniform"
#     )
#     # calibrated curve
#     cal_vals = calibrators[feat].predict(calib[feat].values)
#     cal_vals = np.clip(cal_vals, 1e-4, 1 - 1e-4)
#     prob_true_cal, prob_pred_cal = calibration_curve(
#         calib.label.values,
#         cal_vals,
#         n_bins=10,
#         strategy="uniform"
#     )
#
#     # plot them
#     ax.plot(prob_pred_raw, prob_true_raw, "s--", label="raw")
#     ax.plot(prob_pred_cal, prob_true_cal, "s-",  label="calibrated")
#     ax.plot([0,1],[0,1],"k:", label="ideal")
#     ax.set_title(feat)
#     ax.set_xlabel("Predicted probability")
#     ax.set_ylabel("Observed frequency")
#     ax.legend(loc="lower right")
#
# plt.tight_layout()
# plt.savefig(fig_dir / "reliability_diagrams.pdf")
# plt.close(fig)
#
# print("✅ Saved figures to:")
# print("  - figures/raw_signal_histograms.pdf")
# print("  - figures/reliability_diagrams.pdf")

# src/compute_ablation_and_baselines.py
#
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
# from scipy.special import logit
# import joblib
#
# # --- load data ---
# calib = pd.read_parquet("xsum_calibration_raw.pq")
# train = pd.read_parquet("xsum_train_raw.pq")
# test  = pd.read_parquet("xsum_test_raw.pq")
#
# # features in order
# FEATURES = ["p_true","p_contrad","inv_ppl",
#             "fact_score","entity_precision","triple_precision",
#             "sem_entail","topic_drift"]
#
# # load calibrators & meta-model
# calibrators = joblib.load("../configs/calibrators.pkl")
# meta        = joblib.load("../configs/meta.pkl")
# THR         = float(open("../configs/threshold.txt").read())
#
# # helper to get raw X, calibrated X, and logit-X
# def get_X(df, calibrated=False):
#     raw = df[FEATURES].values
#     if not calibrated:
#         return raw
#     # calibrate each feature
#     cal = np.vstack([
#         calibrators[f].predict(raw[:,i])
#         for i,f in enumerate(FEATURES)
#     ]).T
#     cal = np.clip(cal, 1e-4, 1-1e-4)
#     return cal
# def get_logit_X(df):
#     cal = get_X(df, calibrated=True)
#     return logit(cal)
#
# # 1) Ablation: train meta-model on raw features
# X_calib_raw = get_X(calib, calibrated=False)
# y_calib     = calib.label.values
# X_train_raw = get_X(train, calibrated=False)
# y_train     = train.label.values
# # retrain logistic (balanced, same hyperparams)
# ablation_meta = LogisticRegression(max_iter=500, class_weight="balanced")
# ablation_meta.fit(X_train_raw, y_train)
#
# # eval on test splits
# X_test_raw = get_X(test, calibrated=False)
# y_test     = test.label.values
# p_test_raw = ablation_meta.predict_proba(X_test_raw)[:,1]
# pred_test_raw = (p_test_raw > THR).astype(int)
#
# f1_raw = f1_score(y_test, pred_test_raw)
# auroc_raw = roc_auc_score(y_test, p_test_raw)
# brier_raw = brier_score_loss(y_test, p_test_raw)
#
# print("Ablation (raw features) on XSum test:")
# print(f" F1_raw = {f1_raw:.3f}, AUROC_raw = {auroc_raw:.3f}, Brier_raw = {brier_raw:.3f}")
#
# # 2) Baselines on CNN/DM-Faith (use test here or load cnn_daily_mail_data.txt if needed)
# # For simplicity, we treat `test` as CNN/DM; if you have a separate file, load it instead.
#
# # Baseline A: FactScore-only
# fact_idx = FEATURES.index("fact_score")
# p_fact = test.fact_score.values
# pred_fact = (p_fact > 0.5).astype(int)  # you might choose threshold 0.5
# f1_fact    = f1_score(y_test, pred_fact)
# auroc_fact = roc_auc_score(y_test, p_fact)
#
# # Baseline B: NLI-only (sem_entail)
# nli_idx = FEATURES.index("sem_entail")
# p_nli = (test.sem_entail + 1) / 2  # remap from [-1,1] to [0,1]
# pred_nli = (p_nli > 0.5).astype(int)
# f1_nli    = f1_score(y_test, pred_nli)
# auroc_nli = roc_auc_score(y_test, p_nli)
#
# # Baseline C: Topic-drift-only (invert: low drift → faithful)
# p_drift = 1 - test.topic_drift.values  # high = faithful
# pred_drift = (p_drift > 0.5).astype(int)
# f1_drift    = f1_score(y_test, pred_drift)
# auroc_drift = roc_auc_score(y_test, p_drift)
#
# print("\nSingle-signal baselines on CNN/DM-Faith (test):")
# print(f" FactScore only  —  F1 = {f1_fact:.3f},  AUROC = {auroc_fact:.3f}")
# print(f" NLI only        —  F1 = {f1_nli:.3f},  AUROC = {auroc_nli:.3f}")
# print(f" Topic-drift only—  F1 = {f1_drift:.3f},  AUROC = {auroc_drift:.3f}")
