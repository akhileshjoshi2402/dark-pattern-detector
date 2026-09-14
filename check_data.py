import pandas as pd

# Load the tab-separated dataset
df = pd.read_csv("data/raw/dataset.tsv", sep="\t")

print("Dataset Shape:", df.shape)
print("\nColumn Names:", df.columns.tolist())
print("\nClass Distribution:\n", df["label"].value_counts(dropna=False))
print("\nSample Rows:")
print(df.head(5))