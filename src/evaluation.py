import joblib, numpy as np, pandas as pd
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
from tqdm import tqdm
from inference import infer
import warnings
warnings.filterwarnings("ignore")

# load and sample 150 system summaries
ds       = load_dataset("mtc/frank-test-set-with-faithfulness-annotation", split="test")
df_full  = ds.to_pandas()
df_sys150 = (
    df_full
    .groupby("hash", group_keys=False)
    .apply(lambda g: g.sample(n=1, random_state=42))
    .reset_index(drop=True)
)
print("Sampled 150 system summaries:", len(df_sys150))

# run inference on each rows
records = []
for _, row in tqdm(df_sys150.iterrows(), total=len(df_sys150), desc="Infer"):
    art    = row["article"]
    summ   = row["summary"]
    label  = int(row["Factual"])

    out    = infer(art, summ)
    rec = {
        "article": art,
        "summary": summ,
        "label":   label,
        **out
    }
    records.append(rec)

df_res = pd.DataFrame(records)
print(f"\nBuilt result table: {df_res.shape[0]} rows × {df_res.shape[1]} cols")

# compute metrics
y     = df_res["label"].values
prob  = df_res["probability"].values
pred  = df_res["predicted"].values



PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
# CONFIG       = Path("../configs")
CONFIG       = PROJECT_ROOT / "configs"
with open(CONFIG/"threshold.txt") as f:
    THR = float(f.read().strip())

print("\n=== CNN/DM-Faith (150 system summaries) ===")
print(f"F1  @τ={THR:.2f} : {f1_score(y, pred):.3f}")
print(f"AUROC          : {roc_auc_score(y, prob):.3f}")
print(f"Brier          : {brier_score_loss(y, prob):.3f}")

# save CSV with every column
OUT = Path("frank_results.csv")
df_res.to_csv(OUT, index=False)
print(f"\n✓ Saved detailed results to {OUT}")
