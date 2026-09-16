import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_path: str, output_dir: str, test_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(input_path, sep="\t")

    # Stratify by label ensures identical positive/negative ratios in train and test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"]
    )

    train_path = f"{output_dir}/train.tsv"
    test_path = f"{output_dir}/test.tsv"

    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)

    print(f"Train split saved to {train_path} | Shape: {train_df.shape}")
    print(f"Test split saved to {test_path}   | Shape: {test_df.shape}")
    print("\nTrain Label Balance:\n", train_df["label"].value_counts(normalize=True))
    print("\nTest Label Balance:\n", test_df["label"].value_counts(normalize=True))

if __name__ == "__main__":
    split_dataset("data/processed/clean_dataset.tsv", "data/processed")