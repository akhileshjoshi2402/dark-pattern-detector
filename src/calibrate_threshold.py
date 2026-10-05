import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_test_probabilities(model_dir: str = "models/distilroberta_dark_pattern", test_path: str = "data/processed/test.tsv"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    df = pd.read_csv(test_path, sep="\t")
    texts = df["clean_text"].fillna("").tolist()
    labels = df["label"].to_numpy()

    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()

    prob_dark = probs[:, 1]
    return labels, prob_dark, df

def sweep_thresholds(labels: np.ndarray, prob_dark: np.ndarray):
    thresholds = np.arange(0.10, 0.95, 0.05)
    records = []

    for t in thresholds:
        preds = (prob_dark >= t).astype(int)
        p = precision_score(labels, preds, zero_division=0)
        r = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        
        # False Positives and False Negatives
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))

        records.append({
            "threshold": round(t, 2),
            "precision_dark": round(p, 4),
            "recall_dark": round(r, 4),
            "macro_f1": round(f1, 4),
            "false_positives": int(fp),
            "false_negatives": int(fn)
        })

    return pd.DataFrame(records)

def evaluate_tri_state(prob_dark: np.ndarray, labels: np.ndarray, tau_low: float = 0.30, tau_high: float = 0.70):
    """
    Tier 0: Benign (p < tau_low)
    Tier 1: Review / Ambiguous (tau_low <= p < tau_high)
    Tier 2: Flagged Deceptive (p >= tau_high)
    """
    safe_mask = prob_dark < tau_low
    flagged_mask = prob_dark >= tau_high
    review_mask = (prob_dark >= tau_low) & (prob_dark < tau_high)

    print("\n" + "=" * 65)
    print(f"TRI-STATE POLICY AUDIT (Tau Low: {tau_low} | Tau High: {tau_high})")
    print("=" * 65)
    print(f"Definitively Safe   (p < {tau_low})  : {np.sum(safe_mask)} items")
    print(f"Definitively Flagged(p >= {tau_high}) : {np.sum(flagged_mask)} items")
    print(f"Uncertain / Review  (mid-band)     : {np.sum(review_mask)} items")

    # Accuracy inside the confident bands
    confident_idx = np.where(~review_mask)[0]
    confident_preds = (prob_dark[confident_idx] >= 0.5).astype(int)
    confident_labels = labels[confident_idx]
    confident_acc = np.mean(confident_preds == confident_labels)
    print(f"Accuracy on confident subset: {confident_acc * 100:.2f}% (covers {len(confident_idx)} / {len(labels)} samples)")

def run_day8_calibration():
    print("=" * 65)
    print("DAY 8: DECISION THRESHOLD CALIBRATION & UNCERTAINTY ANALYSIS")
    print("=" * 65)

    labels, prob_dark, df = get_test_probabilities()
    results_df = sweep_thresholds(labels, prob_dark)

    print("\n--- Threshold Sweep Results ---")
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["macro_f1"].idxmax()]
    print(f"\nOptimal Macro-F1 Threshold: {best_row['threshold']}")
    print(f"  Macro-F1       : {best_row['macro_f1']}")
    print(f"  Precision (1)  : {best_row['precision_dark']}")
    print(f"  Recall (1)     : {best_row['recall_dark']}")
    print(f"  False Positives: {best_row['false_positives']} | False Negatives: {best_row['false_negatives']}")

    evaluate_tri_state(prob_dark, labels, tau_low=0.30, tau_high=0.70)

if __name__ == "__main__":
    run_day8_calibration()