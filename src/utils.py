from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd



def split_dataset(df):
    df = df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    train = df.iloc[:500]
    calib = df.iloc[500:700]
    test = df.iloc[700:900]

    return {
        "train": train,
        "calibration": calib,
        "test": test,
    }



def build_summary_table(csv_path="../Data/factuality_annotations_xsum_summaries.csv"):
    df = pd.read_csv(csv_path, usecols=["bbcid","system","is_factual"])
    # majority vote → label
    labels = (
        df.groupby(["bbcid","system"])["is_factual"]
          .apply(lambda x: (x=="yes").mean() >= 0.5)
          .astype(int)
          .reset_index()
          .rename(columns={"is_factual":"label"})
    )
    return labels     # 1 row per summary

def stratified_split(summary_df, train=500, calib=200, test=200, seed=42):
    # first stratify by (label, system) combined
    summary_df["strata"] = summary_df["label"].astype(str) + "_" + summary_df["system"]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=calib+test, random_state=seed)
    idx_train, idx_temp = next(sss.split(summary_df, summary_df["strata"]))

    temp_df = summary_df.iloc[idx_temp].reset_index(drop=True)
    temp_df["strata2"] = temp_df["strata"]         # keep same strata
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test, random_state=seed)
    idx_calib, idx_test = next(sss2.split(temp_df, temp_df["strata2"]))

    return (
        summary_df.iloc[idx_train].sample(n=train, random_state=seed),
        temp_df.iloc[idx_calib],
        temp_df.iloc[idx_test],
    )




def extract_docs():
    from datasets import load_dataset
    xsum = load_dataset("EdinburghNLP/xsum")
    docs = {ex["id"]: ex["document"] for split in xsum.values()
            for ex in split}
    return docs
