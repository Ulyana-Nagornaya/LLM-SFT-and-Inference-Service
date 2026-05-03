# LLM SFT & Inference Service

Production-ready pipeline for Supervised Fine-Tuning (LoRA/PEFT) and real-time model serving.

## Overview
A modular framework for efficiently adapting large language models to domain-specific tasks using Parameter-Efficient Fine-Tuning. The system integrates a complete data preparation, training, and evaluation pipeline with a FastAPI-based inference service. Supports seamless, runtime switching between base and fine-tuned models without service restarts.

## Features
• **Efficient Fine-Tuning**: LoRA/PEFT adaptation (`r=24`, `lora_alpha=36`) for minimal VRAM usage and fast convergence.  
• **Task-Agnostic Architecture**: Configurable for various NLP tasks.  
• **Optimized Data Pipeline**: Custom `TokenizedTaskDataset` with batched loading, max-length truncation, and HuggingFace `Trainer` integration.  
• **Production Inference**: FastAPI service with live model routing, structured logging, and automatic prediction export.  
• **Eval-Driven Workflow**: Built-in metric calculation and `dist/predictions.csv` export for quality tracking.

## Installation
```bash
git clone <repo-url> && cd <project-dir>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt  
```

## Project Structure
```bash
src/
├── main.py            # Core logic: tokenization, TokenizedTaskDataset, SFTPipeline
├── start.py           # Pipeline orchestrator & entry point
├── service.py         # FastAPI endpoints, routing logic & web UI
└── settings.json      # Runtime & training configuration
```
## Usage

### 1. Training Pipeline
Executes dataset preparation, LoRA fine-tuning, weight merging, evaluation, and saves predictions.
```bash
python src/start.py
```
*Output:* Fine-tuned weights saved to the configured path, evaluation metrics logged to console, predictions exported to `dist/predictions.csv`.

### 2. Inference Service
```bash
uvicorn src.service:app --host 0.0.0.0 --port 8000
```
- Web UI: `http://localhost:8000`  
- OpenAPI Docs: `http://localhost:8000/docs`

## Service Architecture & API
The service routes requests to either the base pretrained model or the LoRA-fine-tuned variant based on a runtime flag. Model switching occurs without reloading or downtime.


**Summarization Task**
```json
{
  "parameters": {
    "model": "mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization",
    "dataset": "cnn_dailymail",
    "metrics": ["bleu", "rouge"]
    }
}
```

**Request Example:**

```json
{
  "text": "WASHINGTON — The Biden administration announced Tuesday a new $10 billion investment package aimed at accelerating domestic semiconductor manufacturing, citing national security and supply chain resilience amid rising global competition.",
  "use_base_model": false,
}
```

**Response Example:**
```json
{
  "model_used": "fine-tuned",
  "summary": "Biden administration unveils $10B package to boost U.S. chip manufacturing.",
  "metrics": {
    "bleu": 0.549,
    "rougeL": 0.588
  },
}
```
