
import pandas as pd
calib = pd.read_parquet("xsum_calibration_raw.pq")
# print("NaNs in label:", calib["label"].isna().sum())
print(calib[calib['label']!=0])


df = pd.read_parquet(f"xsum_train_raw.pq")
print(df.describe())
print(df.info(()))

print(df)
