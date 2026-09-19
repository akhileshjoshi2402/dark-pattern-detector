import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

class DarkPatternDataset(Dataset):
    def __init__(self, tsv_path: str, tokenizer_name: str = "distilroberta-base", max_len: int = 64):
        self.df = pd.read_csv(tsv_path, sep="\t")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        # Drop any null clean_text entries if present
        self.texts = self.df["clean_text"].fillna("").tolist()
        self.labels = self.df["label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize with max_len=64 (covers >98% of UI snippets identified in Day 2 EDA)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),         # Tensor shape: [max_len]
            "attention_mask": encoding["attention_mask"].squeeze(0), # Tensor shape: [max_len]
            "labels": torch.tensor(label, dtype=torch.long)
        }

if __name__ == "__main__":
    # Sanity check on the training dataset
    train_dataset = DarkPatternDataset("data/processed/train.tsv")
    sample = train_dataset[0]
    print(f"Dataset Size: {len(train_dataset)}")
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    print(f"Sample label: {sample['labels']} (Type: {type(sample['labels'])})")