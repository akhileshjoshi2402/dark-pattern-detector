import json
import time
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any
from transformers import AutoTokenizer

def load_policy_and_session(
    model_path: str = "models/model_quantized.onnx",
    policy_path: str = "models/decision_policy.json",
    tokenizer_dir: str = "models/distilroberta_dark_pattern"
):
    """
    Initializes the INT8 ONNX inference session, tokenizer, and calibrated policy.
    """
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)
        
    return session, tokenizer, policy

def softmax(logits: np.ndarray) -> np.ndarray:
    """Computes softmax probabilities across class logits."""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def score_candidates_batch(
    candidates: List[Dict[str, Any]],
    session: ort.InferenceSession,
    tokenizer: AutoTokenizer,
    policy: Dict[str, Any],
    batch_size: int = 16,
    max_len: int = 64
) -> List[Dict[str, Any]]:
    """
    Runs batched INT8 ONNX inference across pruned DOM text candidates.
    """
    if not candidates:
        return []

    tau_low = policy["tri_state_policy"]["tau_low"]
    tau_high = policy["tri_state_policy"]["tau_high"]
    results = []

    start_time = time.perf_counter()

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        texts = [c["text"] for c in batch]

        # Dynamic subword padding & truncation
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="np"
        )

        ort_inputs = {
            "input_ids": encodings["input_ids"].astype(np.int64),
            "attention_mask": encodings["attention_mask"].astype(np.int64)
        }

        ort_outputs = session.run(None, ort_inputs)
        logits = ort_outputs[0]
        probs = softmax(logits)

        for candidate, prob in zip(batch, probs):
            p_dark = float(prob[1])
            
            # Calibrated Tri-State Policy Evaluation
            if p_dark >= tau_high:
                verdict = "DARK_PATTERN"
            elif p_dark >= tau_low:
                verdict = "UNCERTAIN_REVIEW"
            else:
                verdict = "BENIGN"

            enriched = dict(candidate)
            enriched["prob_dark"] = round(p_dark, 4)
            enriched["confidence"] = round(max(p_dark, 1 - p_dark), 4)
            enriched["verdict"] = verdict
            results.append(enriched)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    throughput = (len(candidates) / (elapsed_ms / 1000.0)) if elapsed_ms > 0 else 0

    print(f"\n[Batch Engine] Scored {len(results)} elements in {elapsed_ms:.2f} ms ({throughput:.1f} elem/sec)")
    return results

if __name__ == "__main__":
    import asyncio
    from src.dom_scraper import extract_ui_elements_from_url
    from src.heuristic_filter import filter_dom_candidates

    test_url = "http://books.toscrape.com/"
    print("=" * 65)
    print("DAY 13: BATCHED ONNX INT8 INFERENCE TEST")
    print("=" * 65)

    # 1. Scrape DOM nodes
    raw_nodes = asyncio.run(extract_ui_elements_from_url(test_url, trigger_events=False))
    
    # 2. Prune via Heuristic Filter (Day 12)
    candidates = filter_dom_candidates(raw_nodes)

    # Inject simulated adversarial/deceptive candidates to verify live detection
    candidates.append({
        "node_id": 9991, "tag": "div", "selector": "div#urgency-banner",
        "class_name": "flash-timer", "text": "Hurry! Only 2 items left in stock. Order now!",
        "priority": 6, "word_count": 8, "is_overlay": False
    })
    candidates.append({
        "node_id": 9992, "tag": "button", "selector": "button.reject",
        "class_name": "btn-link", "text": "No thanks, I prefer paying full price",
        "priority": 6, "word_count": 7, "is_overlay": True
    })

    # 3. Load ONNX engine and run batch inference
    session, tokenizer, policy = load_policy_and_session()
    scored_elements = score_candidates_batch(candidates, session, tokenizer, policy, batch_size=16)

    # 4. Display high-confidence detections
    flagged = [e for e in scored_elements if e["verdict"] in ("DARK_PATTERN", "UNCERTAIN_REVIEW")]
    
    print("\n--- Summary Detections ---")
    print(f"Total Elements Scored : {len(scored_elements)}")
    print(f"Flagged Deceptive UI  : {len(flagged)}")
    
    for item in flagged:
        print(f"\n • [{item['verdict']}] P(dark): {item['prob_dark']} | <{item['tag']}> {item['selector']}")
        print(f"   Snippet: \"{item['text']}\"")