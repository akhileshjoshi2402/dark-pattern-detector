import os
import time
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_to_onnx(
    model_dir: str = "models/distilroberta_dark_pattern",
    output_onnx_path: str = "models/model.onnx"
):
    print("=" * 65)
    print("1. EXPORTING PYTORCH DISTILROBERTA TO ONNX GRAPH")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy_text = "Hurry! Limited stock available."
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        max_length=64,
        padding="max_length",
        truncation=True
    )

    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    }

    # Ensure clean export by forcing legacy TorchScript engine (dynamo=False)
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        output_onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )

    raw_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"Standard ONNX model saved to: {output_onnx_path}")
    print(f"ONNX Model File Size: {raw_size_mb:.2f} MB")
    return tokenizer, model

def quantize_onnx_model(
    input_onnx_path: str = "models/model.onnx",
    quantized_path: str = "models/model_quantized.onnx"
):
    print("\n" + "=" * 65)
    print("2. APPLYING DYNAMIC 8-BIT (INT8) QUANTIZATION")
    print("=" * 65)

    quantize_dynamic(
        model_input=input_onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8
    )

    quant_size_mb = os.path.getsize(quantized_path) / (1024 * 1024)
    raw_size_mb = os.path.getsize(input_onnx_path) / (1024 * 1024)
    reduction = ((raw_size_mb - quant_size_mb) / raw_size_mb) * 100

    print(f"Quantized ONNX model saved to: {quantized_path}")
    print(f"Quantized Model Size: {quant_size_mb:.2f} MB")
    print(f"Storage Reduction   : {reduction:.1f}% smaller")

def benchmark_inference(
    tokenizer,
    torch_model,
    onnx_path: str,
    quant_path: str,
    iterations: int = 100
):
    print("\n" + "=" * 65)
    print(f"3. LATENCY BENCHMARK ({iterations} INFERENCE RUNS ON CPU)")
    print("=" * 65)

    sample_text = "Only 2 items left in stock - order soon!"
    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=64,
        padding="max_length",
        truncation=True
    )

    input_ids_pt = encoded["input_ids"]
    attention_mask_pt = encoded["attention_mask"]

    ort_inputs = {
        "input_ids": input_ids_pt.numpy(),
        "attention_mask": attention_mask_pt.numpy()
    }

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    raw_session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    quant_session = ort.InferenceSession(quant_path, opts, providers=["CPUExecutionProvider"])

    # Warmup runs
    for _ in range(5):
        with torch.no_grad():
            _ = torch_model(input_ids_pt, attention_mask=attention_mask_pt)
        _ = raw_session.run(None, ort_inputs)
        _ = quant_session.run(None, ort_inputs)

    # 1. Benchmark PyTorch CPU
    pt_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = torch_model(input_ids_pt, attention_mask=attention_mask_pt)
        pt_times.append((time.perf_counter() - t0) * 1000)

    # 2. Benchmark Standard ONNX CPU
    onnx_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = raw_session.run(None, ort_inputs)
        onnx_times.append((time.perf_counter() - t0) * 1000)

    # 3. Benchmark Quantized ONNX CPU
    quant_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = quant_session.run(None, ort_inputs)
        quant_times.append((time.perf_counter() - t0) * 1000)

    def print_stats(name: str, times: list):
        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        print(f"{name:<22} | Mean: {avg:6.2f} ms | P50: {p50:6.2f} ms | P95: {p95:6.2f} ms")

    print_stats("PyTorch (FP32)", pt_times)
    print_stats("ONNX Runtime (FP32)", onnx_times)
    print_stats("ONNX Quantized (INT8)", quant_times)

def run_day9_pipeline():
    raw_onnx = "models/model.onnx"
    quant_onnx = "models/model_quantized.onnx"

    tokenizer, model = export_to_onnx(output_onnx_path=raw_onnx)
    quantize_onnx_model(input_onnx_path=raw_onnx, quantized_path=quant_onnx)
    benchmark_inference(tokenizer, model, raw_onnx, quant_onnx)

if __name__ == "__main__":
    run_day9_pipeline()