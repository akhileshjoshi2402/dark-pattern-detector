import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer(model_dir: str = "models/distilroberta_dark_pattern"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict_texts(texts: list, tokenizer, model, max_len: int = 64):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

def run_day7_evaluation():
    model_dir = "models/distilroberta_dark_pattern"
    test_path = "data/processed/test.tsv"
    
    print("=" * 65)
    print("DAY 7: DISTILROBERTA ERROR ANALYSIS & ADVERSARIAL RE-EVALUATION")
    print("=" * 65)
    
    tokenizer, model = load_model_and_tokenizer(model_dir)
    test_df = pd.read_csv(test_path, sep="\t")
    
    texts = test_df["clean_text"].fillna("").tolist()
    labels = test_df["label"].tolist()
    
    # 1. Inference across the test split
    print(f"Evaluating {len(texts)} test samples...")
    preds, probs = predict_texts(texts, tokenizer, model)
    test_df["pred"] = preds
    test_df["prob_dark"] = probs[:, 1]
    
    # 2. Identify Remaining Errors
    fp = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)]
    fn = test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)]
    
    print(f"\n--- Residual Errors (Total: {len(fp) + len(fn)} / {len(test_df)}) ---")
    print(f"False Positives: {len(fp)} | False Negatives: {len(fn)}")
    
    if len(fp) > 0:
        print("\nTop False Positives (Benign marked as Dark Pattern):")
        for _, row in fp.sort_values(by="prob_dark", ascending=False).head(5).iterrows():
            print(f"  • [Conf: {row['prob_dark']:.4f}] \"{row['clean_text']}\"")
            
    if len(fn) > 0:
        print("\nTop False Negatives (Dark Pattern missed):")
        for _, row in fn.sort_values(by="prob_dark", ascending=True).head(5).iterrows():
            print(f"  • [Cat: {row.get('Pattern Category', 'N/A')} | Conf: {row['prob_dark']:.4f}] \"{row['clean_text']}\"")
            
    # Save transformer error audit
    os.makedirs("data/processed", exist_ok=True)
    test_df[test_df["label"] != test_df["pred"]].to_csv(
        "data/processed/transformer_errors.tsv", sep="\t", index=False
    )
    print("\nSaved residual transformer errors to data/processed/transformer_errors.tsv")

    # 3. Direct Day 4 Adversarial Re-evaluation
    print("\n" + "=" * 65)
    print("DAY 4 ADVERSARIAL BENCHMARK COMPARISON")
    print("=" * 65)
    
    adversarial_cases = [
        ("Factual Neutral", "Item is currently out of stock. Please check back next week."),
        ("Deceptive Scarcity", "Hurry! Only 2 items left in stock. Order now!"),
        ("Benign Navigation", "Please turn left at the counter to find your order pickup zone."),
        ("Confirmshaming", "No thanks, I prefer paying full price for items."),
        ("Formatted Countdown", "Sale ends in: 02 hours 14 minutes 10 seconds"),
        ("Raw Scraped Timer", "02Hours14Minutes10Seconds remaining")
    ]
    
    adv_texts = [text for _, text in adversarial_cases]
    adv_preds, adv_probs = predict_texts(adv_texts, tokenizer, model)
    
    for (category, text), pred, prob in zip(adversarial_cases, adv_preds, adv_probs):
        verdict = "DARK PATTERN (1)" if pred == 1 else "BENIGN (0)"
        confidence = prob[1] if pred == 1 else prob[0]
        print(f"[{category}]")
        print(f"  Input      : \"{text}\"")
        print(f"  Prediction : {verdict} (Score: {prob[1]:.4f})")
        print(f"  Confidence : {confidence * 100:.2f}%\n")

if __name__ == "__main__":
    run_day7_evaluation()