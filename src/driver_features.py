import json, pandas as pd
from pathlib import Path
from tqdm import tqdm

from LLM_based             import run_self_questioning
from factScore_utils       import compute_fact_score
from componentBase_utils   import compute_semantic_entailment, compute_topic_drift
from utils                 import extract_docs, build_summary_table, stratified_split

DETAILS = open("factscore_details.jsonl", "w")

print("Loading documents …")
docs        = extract_docs()                # {bbcid: article}
summary_tbl = build_summary_table()         # (bbcid, system, label)

train_ids, calib_ids, test_ids = stratified_split(summary_tbl)

df_factuality = pd.read_csv("../Data/factuality_annotations_xsum_summaries.csv")

splits = {
    "train":        df_factuality.merge(train_ids[["bbcid","system"]]),
    "calibration":  df_factuality.merge(calib_ids[["bbcid","system"]]),
    "test":         df_factuality.merge(test_ids [["bbcid","system"]]),
}

for tag, df_w in splits.items():
    print(f"\n===== {tag.upper()}  ({len(df_w)} worker rows) =====")

    # Approach a  (N_mc=7)
    df = run_self_questioning(df_w, docs, N_mc=7, temp=1.0)

    # unify dtypes for merge
    df["bbcid"] = df["bbcid"].astype(str)
    df["system"]= df["system"].astype(str)
    summary_tbl["bbcid"] = summary_tbl["bbcid"].astype(str)
    summary_tbl["system"]= summary_tbl["system"].astype(str)

    # Approach b  (FactScore)
    fs_rows = []
    for r in tqdm(df_w.itertuples(), total=len(df_w), desc="FactScore"):
        src = docs[str(r.bbcid)]
        fs  = compute_fact_score(r.summary, src)
        fs_rows.append(fs)

        # write JSON error detail
        DETAILS.write(json.dumps({
            "bbcid": r.bbcid, "system": r.system,
            "unsupported_entities": fs["unsupported_entities"],
            "unsupported_triples" : fs["unsupported_triples"]
        })+"\n")

    df = pd.concat([df, pd.DataFrame(fs_rows)], axis=1)

    # Approach c  (Semantic Entail + Topic Drift)
    entails, drifts = [], []
    for r in tqdm(df_w.itertuples(), total=len(df_w), desc="Entail+Drift"):
        src = docs[str(r.bbcid)]
        entails.append( compute_semantic_entailment(src, r.summary) )
        drifts .append( compute_topic_drift(src,  r.summary) )
    df["sem_entail"]  = entails
    df["topic_drift"] = drifts

    # merge gold label  → 1 summary
    df = df.merge(summary_tbl[["bbcid","system","label"]],
                  on=["bbcid","system"], how="left")

    feature_cols = ["p_true","p_contrad","inv_ppl",
                    "fact_score","entity_precision","triple_precision",
                    "sem_entail","topic_drift"]

    agg = {c:"mean" for c in feature_cols};  agg["label"]="first"
    df_sum = (df.groupby(["bbcid","system"])
                .agg(agg).reset_index())

    assert not df_sum.label.isna().any()
    df_sum.to_parquet( f"xsum_{tag}_raw.pq", index=False)
    print(f"✓ {tag}: {len(df_sum)} summaries")

DETAILS.close()

