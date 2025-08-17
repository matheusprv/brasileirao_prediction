import pandas as pd

df = pd.read_csv("BRA-pre-processed-lstm.csv")
test = df[df["Season"] == 2025]
print(test["Res"].value_counts(normalize=True).values)
