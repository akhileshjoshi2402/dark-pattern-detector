import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from src.dataset import DarkPatternDataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "macro_f1": f1
    }

def train_transformer():
    model_name = "distilroberta-base"
    output_dir = "checkpoints"
    final_model_dir = "models/distilroberta_dark_pattern"

    print("=" * 65)
    print("DAY 6: FINE-TUNING DISTILROBERTA CLASSIFIER")
    print("=" * 65)

    # 1. Load Datasets
    print("Loading tokenized train and test datasets...")
    train_dataset = DarkPatternDataset("data/processed/train.tsv", tokenizer_name=model_name)
    eval_dataset = DarkPatternDataset("data/processed/test.tsv", tokenizer_name=model_name)

    # 2. Initialize Model Architecture
    print(f"Loading pre-trained model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 3. Configure Training Arguments
    # Note: eval_strategy replaces the deprecated evaluation_strategy
    # Calculate exact warmup steps: 10% of total training steps
    # (1879 samples / batch size 16) ≈ 118 steps per epoch * 3 epochs ≈ 354 total steps
    warmup_steps = 35

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",            # or evaluation_strategy if your version requires it
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_steps=warmup_steps,        # Replaces warmup_ratio
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        use_cpu=True
    )

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # 5. Train
    print("\nStarting training loop (3 epochs)...")
    train_result = trainer.train()

    # 6. Evaluate final performance
    print("\nEvaluating best checkpoint on test set...")
    eval_results = trainer.evaluate()
    print("\n--- Final Test Metrics ---")
    for key in ["eval_loss", "eval_accuracy", "eval_precision", "eval_recall", "eval_macro_f1"]:
        print(f"{key}: {eval_results.get(key, 'N/A'):.4f}")

    # 7. Persist Best Model and Tokenizer
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.save_model(final_model_dir)
    train_dataset.tokenizer.save_pretrained(final_model_dir)
    print(f"\nBest model and tokenizer successfully saved to: {final_model_dir}")

if __name__ == "__main__":
    train_transformer()