import joblib

def test_robustness():
    model = joblib.load("models/best_baseline_model.joblib")
    vec = joblib.load("models/tfidf_vectorizer.joblib")

    # Stress test cases: Factual UI, Paraphrased Deception, and Keyword Hijacks
    cases = [
        # Factual inventory vs Artificial Scarcity
        ("Factual Neutral", "Item is currently out of stock. Please check back next week."),
        ("Deceptive Scarcity", "Hurry! Only 2 items left in stock. Order now!"),
        
        # Spatial direction vs Scarcity trigger word ("left")
        ("Benign Navigation", "Please turn left at the counter to find your order pickup zone."),
        
        # Subtle Confirmshaming / Forced Action
        ("Confirmshaming", "No thanks, I prefer paying full price for items."),
        
        # Obfuscated Urgency (Concatenated timer text)
        ("Formatted Countdown", "Sale ends in: 02 hours 14 minutes 10 seconds"),
        ("Raw Scraped Timer", "02Hours14Minutes10Seconds remaining")
    ]

    print("=" * 65)
    print("DAY 4: BASELINE ROBUSTNESS & ADVERSARIAL STRESS TEST")
    print("=" * 65)

    for category, text in cases:
        features = vec.transform([text])
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]
        verdict = "DARK PATTERN (1)" if pred == 1 else "BENIGN (0)"
        print(f"[{category}]")
        print(f"  Input : \"{text}\"")
        print(f"  Output: {verdict} | Confidence: {prob:.4f}\n")

if __name__ == "__main__":
    test_robustness()