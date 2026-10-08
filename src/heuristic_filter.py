import re
from typing import List, Dict, Any, Tuple

# Boilerplate patterns to suppress immediately
BOILERPLATE_RE = re.compile(
    r"^(home|menu|all products|search|cart|checkout|contact us|about us|"
    r"privacy policy|terms of service|terms & conditions|cookie policy|"
    r"all rights reserved|copyright ©.*|\d{4} ©.*)$",
    re.IGNORECASE
)

# High-priority token triggers mapped from Day 3 & Day 4 audits
TRIGGER_PATTERNS = [
    r"\bonly \d+ left\b",
    r"\b\d+ (people|others|users|viewing|bought|purchased)\b",
    r"\b(hurry|limited time|flash sale|ends in|offer expires|countdown)\b",
    r"\b(no thanks|i prefer|i hate|don't leave|not now)\b",
    r"\b(selling fast|almost gone|in high demand|reserved for)\b",
    r"\b(auto-?renew|remove insurance|protection plan)\b",
]

COMPILED_TRIGGERS = [re.compile(p, re.IGNORECASE) for p in TRIGGER_PATTERNS]

def evaluate_snippet(text: str, is_overlay: bool = False) -> Tuple[bool, int, str]:
    """
    Evaluates whether a text node qualifies as a candidate for ONNX evaluation.
    Returns: (keep_candidate, priority_score, rejection_reason)
    """
    clean = text.strip()
    words = clean.split()
    word_count = len(words)

    # 1. Length boundaries (derived from Day 2 EDA)
    if word_count < 2:
        return False, 0, "too_short"
    if word_count > 35:
        return False, 0, "too_long"

    # 2. Boilerplate suppression
    if BOILERPLATE_RE.match(clean):
        return False, 0, "boilerplate"

    # Pure numeric or single-word punctuation checks
    if clean.isdigit() or re.match(r"^[\W_]+$", clean):
        return False, 0, "non_semantic"

    # 3. Priority scoring (overlay elements and trigger matches rank highest)
    priority = 1  # Base candidate priority

    if is_overlay:
        priority += 3

    for pattern in COMPILED_TRIGGERS:
        if pattern.search(clean):
            priority += 5
            break

    return True, priority, "candidate"

def filter_dom_candidates(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prunes extracted DOM nodes down to actionable, high-priority candidates.
    """
    filtered = []
    rejection_stats = {
        "too_short": 0,
        "too_long": 0,
        "boilerplate": 0,
        "non_semantic": 0,
        "candidate": 0
    }

    for node in nodes:
        keep, priority, reason = evaluate_snippet(
            node.get("text", ""),
            node.get("is_overlay", False)
        )
        rejection_stats[reason] += 1

        if keep:
            enriched_node = dict(node)
            enriched_node["priority"] = priority
            enriched_node["word_count"] = len(node.get("text", "").split())
            filtered.append(enriched_node)

    # Sort descending by priority so high-probability deceptive elements score first
    filtered.sort(key=lambda x: x["priority"], reverse=True)

    print(f"\n[Heuristic Filter] Total Input Nodes : {len(nodes)}")
    print(f"[Heuristic Filter] Pruning Breakdown   : {rejection_stats}")
    print(f"[Heuristic Filter] Qualified Candidates: {len(filtered)} (Pruned {100 - (len(filtered)/max(len(nodes),1)*100):.1f}%)")

    return filtered

if __name__ == "__main__":
    import asyncio
    from src.dom_scraper import extract_ui_elements_from_url

    test_url = "http://books.toscrape.com/"
    print("=" * 65)
    print("DAY 12: HEURISTIC FILTERING & PRUNING VERIFICATION")
    print("=" * 65)

    raw_nodes = asyncio.run(extract_ui_elements_from_url(test_url, trigger_events=False))
    candidates = filter_dom_candidates(raw_nodes)

    print("\nTop Filtered UI Candidates:")
    for item in candidates[:8]:
        sample = item["text"][:50] + ("..." if len(item["text"]) > 50 else "")
        print(f" • [Prio: {item['priority']} | {item['word_count']} words | <{item['tag']}>] \"{sample}\"")