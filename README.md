# Multimodal & Text-Based Dark Pattern Detector

An end-to-end machine learning project to identify and classify deceptive UI/UX patterns (e.g., Urgency, Scarcity, Misdirection) in e-commerce interfaces.

## Current Progress
- **Day 1**: Ingestion and verification of the benchmark dataset.
- **Day 2**: Text cleaning, duplicate removal, word-count EDA, and stratified train/test splits.

## Setup
```bash
uv sync
python -m src.preprocess
python -m src.split_data