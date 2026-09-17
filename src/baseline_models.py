import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def run_baseline_training():
    train_df = pd.read_csv("data/processed/train.tsv", sep="\t")
    test_df = pd.read_csv("data/processed/test.tsv", sep="\t")

    X_train, y_train = train_df["clean_text"], train_df["label"]
    X_test, y_test = test_df["clean_text"], test_df["label"]

    # 1. Feature Extraction: TF-IDF with Unigrams and Bigrams
    # max_features=5000 prevents overfitting on rare typos
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 2. Define Candidate Models
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=42, C=1.0),
        "Linear SVM": LinearSVC(random_state=42, dual=False)
    }

    best_model_name = None
    best_f1 = 0.0
    best_model_obj = None

    print("=" * 60)
    print("DAY 3: CLASSICAL BASELINE MODEL EVALUATION")
    print("=" * 60)

    # 3. Train & Evaluate
    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)
        macro_f1 = f1_score(y_test, preds, average="macro")

        print(f"\n--- Model: {name} ---")
        print(f"Macro F1-Score: {macro_f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, digits=4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, preds))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_model_name = name
            best_model_obj = clf

    # 4. Persist the Best Classical Model & Vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model_obj, "models/best_baseline_model.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    
    print("\n" + "=" * 60)
    print(f"Top Performer: {best_model_name} (F1: {best_f1:.4f})")
    print("Saved baseline model and vectorizer to models/")
    print("=" * 60)

if __name__ == "__main__":
    run_baseline_training()