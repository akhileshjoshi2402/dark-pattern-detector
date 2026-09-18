import joblib
import pandas as pd

def run_error_analysis():
    # Load test split and saved artifacts
    test_df = pd.read_csv("data/processed/test.tsv", sep="\t")
    model = joblib.load("models/best_baseline_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

    X_test = test_df["clean_text"]
    y_test = test_df["label"]

    # Generate predictions and prediction probabilities
    X_test_vec = vectorizer.transform(X_test)
    preds = model.predict(X_test_vec)
    probs = model.predict_proba(X_test_vec)[:, 1]

    test_df["pred"] = preds
    test_df["prob_dark_pattern"] = probs

    # 1. Isolate False Positives & False Negatives
    false_positives = test_df[(test_df["label"] == 0) & (test_df["pred"] == 1)]
    false_negatives = test_df[(test_df["label"] == 1) & (test_df["pred"] == 0)]

    print("=" * 65)
    print(f"DAY 4: ERROR ANALYSIS (Total Samples: {len(test_df)})")
    print("=" * 65)
    print(f"False Positives: {len(false_positives)} | False Negatives: {len(false_negatives)}\n")

    # 2. Inspect False Positives (Benign text marked as deceptive)
    print("--- TOP FALSE POSITIVES (Model was overly sensitive) ---")
    fp_sorted = false_positives.sort_values(by="prob_dark_pattern", ascending=False)
    for idx, row in fp_sorted.head(7).iterrows():
        print(f"• [Conf: {row['prob_dark_pattern']:.2f}] \"{row['clean_text']}\"")

    # 3. Inspect False Negatives (Deceptive patterns missed)
    print("\n--- TOP FALSE NEGATIVES (Model failed to catch) ---")
    fn_sorted = false_negatives.sort_values(by="prob_dark_pattern", ascending=True)
    for idx, row in fn_sorted.head(10).iterrows():
        print(f"• [Category: {row['Pattern Category']} | Conf: {row['prob_dark_pattern']:.2f}] \"{row['clean_text']}\"")

    # 4. Pattern Category Breakdown for False Negatives
    print("\n--- FALSE NEGATIVES BY PATTERN CATEGORY ---")
    print(false_negatives["Pattern Category"].value_counts())

    # 5. Export errors to CSV for portfolio documentation
    test_df[test_df["label"] != test_df["pred"]].to_csv(
        "data/processed/baseline_errors.tsv", sep="\t", index=False
    )
    print("\nSaved full error audit to data/processed/baseline_errors.tsv")

if __name__ == "__main__":
    run_error_analysis()