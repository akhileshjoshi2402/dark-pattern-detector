import json
import os

def save_decision_policy():
    policy = {
        "model_name": "distilroberta-base",
        "optimal_threshold": 0.65,
        "tri_state_policy": {
            "tau_low": 0.30,
            "tau_high": 0.70,
            "labels": {
                "safe": 0,
                "review": -1,
                "dark_pattern": 1
            }
        },
        "metrics_at_optimal": {
            "macro_f1": 0.9681,
            "precision_dark": 0.9825,
            "recall_dark": 0.9534,
            "false_positives": 4,
            "false_negatives": 11
        }
    }

    os.makedirs("models", exist_ok=True)
    policy_path = "models/decision_policy.json"
    
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    print(f"Policy successfully saved to {policy_path}")

if __name__ == "__main__":
    save_decision_policy()