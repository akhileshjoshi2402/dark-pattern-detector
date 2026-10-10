import time
import requests

url = "http://127.0.0.1:8000/analyze-snippets"
payload = {
    "snippets": [
        "Hurry! Only 1 item left in stock!",
        "In stock. Delivered by tomorrow.",
        "No thanks, I prefer paying full price",
        "Add to shopping basket",
        "Sale ends in 00:04:12"
    ]
}

# Warmup
requests.post(url, json=payload)

# Benchmark 10 consecutive cached requests
latencies = []
for _ in range(10):
    start = time.perf_counter()
    res = requests.post(url, json=payload)
    latencies.append((time.perf_counter() - start) * 1000)

print(f"Mean In-Memory Cached Latency: {sum(latencies)/len(latencies):.2f} ms")
print(f"Min Latency: {min(latencies):.2f} ms | Max Latency: {max(latencies):.2f} ms")