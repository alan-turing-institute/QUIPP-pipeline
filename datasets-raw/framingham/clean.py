import pandas as pd

raw_df = pd.read_csv("framingham.csv")

df = raw_df.fillna(raw_df.mean())
df[["cigsPerDay", "age", "education", "BPMeds"]] = df[["cigsPerDay", "age", "education", "BPMeds"]].astype(int)

df.to_csv("../../datasets/framinghamframingham_cleaned.csv", index=False)