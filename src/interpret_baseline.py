import joblib
import numpy as np
import pandas as pd

def inspect_features(top_n=15):
    clf = joblib.load("models/best_baseline_model.joblib")
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = clf.coef_[0]

    # Top features for Dark Patterns (positive coefficients)
    top_dark_idx = np.argsort(coefficients)[-top_n:][::-1]
    top_dark_terms = pd.DataFrame({
        "Dark Pattern Token": feature_names[top_dark_idx],
        "Weight": coefficients[top_dark_idx]
    })

    # Top features for Normal UI (negative coefficients)
    top_normal_idx = np.argsort(coefficients)[:top_n]
    top_normal_terms = pd.DataFrame({
        "Normal UI Token": feature_names[top_normal_idx],
        "Weight": coefficients[top_normal_idx]
    })

    print("=" * 50)
    print(f"TOP {top_n} TOKENS PREDICTING DARK PATTERNS (1)")
    print("=" * 50)
    print(top_dark_terms.to_string(index=False))

    print("\n" + "=" * 50)
    print(f"TOP {top_n} TOKENS PREDICTING BENIGN UI (0)")
    print("=" * 50)
    print(top_normal_terms.to_string(index=False))

if __name__ == "__main__":
    inspect_features()