"Framingham Heart Study" dataset by found in Kaggle (https://www.kaggle.com/amanajmera1/framingham-heart-study-dataset), 
used under [CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/) modified by filling NA values with the column means

The dataset is not used for commercial purposes.

Note: Might want to replace this with the details from https://biolincc.nhlbi.nih.gov/studies/framcohort/ , due to time constraints I am using the version available freely on Kaggle rather than waiting for approval, but I believe this is the true original source of the data.

Instructions to clean (also included in `datasets-raw/framingham/clean.py`):

```{python}
import pandas as pd

raw_df = pd.read_csv("framingham.csv")

df = raw_df.fillna(raw_df.mean())
df[["cigsPerDay", "age", "education", "BPMeds"]] = df[["cigsPerDay", "age", "education", "BPMeds"]].astype(int)

df.to_csv("../../datasets/framinghamframingham_cleaned.csv", index=False)
```