import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text: str) -> str:
    """Normalize raw web text: strip HTML, fix spaces, keep punctuation."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace non-breaking spaces and escaped characters
    text = text.replace("\xa0", " ").replace("&amp;", "&")
    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def run_eda_and_preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path, sep="\t")

    # 1. Inspect Sub-Categories
    print("=== Dark Pattern Category Distribution ===")
    category_counts = df[df["label"] == 1]["Pattern Category"].value_counts()
    print(category_counts)

    # 2. Text Cleaning
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Drop empty rows or duplicates
    initial_len = len(df)
    df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)
    print(f"\nRemoved {initial_len - len(df)} duplicate text rows.")

    # 3. Feature Extraction for EDA
    df["char_count"] = df["clean_text"].apply(len)
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    print("\n=== Word Count Statistics by Label ===")
    print(df.groupby("label")[["char_count", "word_count"]].describe())

    # 4. Save Processed Dataset
    df.to_csv(output_path, sep="\t", index=False)
    print(f"\nSaved processed dataset to {output_path} (Shape: {df.shape})")

    # 5. Quick Distribution Plot
    plt.figure(figsize=(10, 4))
    sns.kdeplot(data=df, x="word_count", hue="label", common_norm=False, fill=True)
    plt.title("Word Count Distribution: Dark Pattern (1) vs Normal (0)")
    plt.xlim(0, 50)
    plt.savefig("data/word_count_distribution.png")
    print("Saved plot to data/word_count_distribution.png")

if __name__ == "__main__":
    run_eda_and_preprocess("data/raw/dataset.tsv", "data/processed/clean_dataset.tsv")