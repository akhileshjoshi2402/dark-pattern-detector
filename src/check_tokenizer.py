from transformers import AutoTokenizer

def test_tokenizer():
    model_name = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample_texts = [
        "Please turn left at the counter.",
        "Only 2 items left in stock!",
        "02Hours14Minutes10Seconds remaining"
    ]

    print("=" * 65)
    print(f"DAY 5: TOKENIZATION INSPECTION ({model_name})")
    print("=" * 65)

    for text in sample_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        print(f"\nOriginal: {text}")
        print(f"Subwords: {tokens}")
        print(f"Token IDs: {token_ids} (Count: {len(token_ids)})")

if __name__ == "__main__":
    test_tokenizer()