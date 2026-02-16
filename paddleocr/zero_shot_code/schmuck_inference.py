#!/usr/bin/env python3
"""
PaddleOCR-VL inference on Schmuck dataset (test split).

- Model: PaddlePaddle/PaddleOCR-VL
- Test JSONL: /home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl
- Images dir: /home/woody/iwi5/iwi5298h/schmuck_images

The model outputs free-form OCR text (no JSON). We:
- Serialize the GT JSON (without file_name/image_name) to a canonical string.
- Compare OCR string vs GT string using CER.
"""

import os
import json
import unicodedata
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from tqdm import tqdm
import jiwer
import editdistance
import glob
import numpy as np


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

TEST_JSONL_PATH = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
IMAGES_DIR = "/home/woody/iwi5/iwi5298h/schmuck_images"

HF_MODEL_NAME = "PaddlePaddle/PaddleOCR-VL"
LOCAL_MODEL_DIR = "/home/vault/iwi5/iwi5298h/models/PaddleOCR-VL"

BASE_OUTPUT_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/paddleocr/schmuck"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "ocr": "OCR:",
}

TASK = "ocr"
MAX_NEW_TOKENS = 1024


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Unicode NFC + collapse whitespace."""
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.split())
    return text


def dict_without_image_key(obj: dict) -> dict:
    """Remove file_name / image_name keys for GT string."""
    return {
        k: v
        for k, v in obj.items()
        if k not in ["file_name", "image_name"]
    }


def canonical_json_string(obj: dict) -> str:
    """Canonical compact JSON string (no sorted keys -> keep original order)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def find_image_path(image_name: str, images_dir: str) -> str:
    """
    Try to resolve image path.
    - First try exact match.
    - Then try glob "*basename*".
    """
    exact = os.path.join(images_dir, image_name)
    if os.path.exists(exact):
        return exact

    base = os.path.splitext(image_name)[0]
    pattern = os.path.join(images_dir, f"*{base}*")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]

    return exact  # may be non-existent; caller should check


def download_and_cache_model(model_name: str, local_dir: str):
    """
    Load PaddleOCR-VL from local cache if present, otherwise download + save.
    """
    os.makedirs(local_dir, exist_ok=True)
    config_path = os.path.join(local_dir, "config.json")

    if os.path.exists(config_path):
        print(f"Loading model from local cache: {local_dir}")
        processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        print(f"Downloading model {model_name} to {local_dir}...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        print("Saving model & processor to local cache...")
        processor.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print("✅ Model cached.")

    model.to(DEVICE)
    model.eval()
    return processor, model


def compute_cer_metrics(samples):
    """
    samples: list of dicts with keys:
        - gt_str
        - pred_str
    Returns dict of metrics.
    """
    if not samples:
        return {
            "avg_cer_jiwer": 1.0,
            "avg_cer_jiwer_percent": 100.0,
            "editdistance_cer": 1.0,
            "editdistance_cer_percent": 100.0,
            "num_samples": 0,
        }

    # Average of jiwer.cer per sample
    cer_list = []
    total_errors = 0
    total_chars = 0

    for s in samples:
        tgt = s["gt_str_norm"]
        pred = s["pred_str_norm"]

        if len(tgt) > 0:
            cer = jiwer.cer(tgt, pred)
            cer_list.append(cer)

            total_errors += editdistance.eval(pred, tgt)
            total_chars += len(tgt)

    avg_cer = float(np.mean(cer_list)) if cer_list else 1.0
    editdistance_cer = (total_errors / total_chars) if total_chars > 0 else 1.0

    return {
        "avg_cer_jiwer": avg_cer,
        "avg_cer_jiwer_percent": avg_cer * 100.0,
        "editdistance_cer": editdistance_cer,
        "editdistance_cer_percent": editdistance_cer * 100.0,
        "num_samples": len(samples),
    }


# -------------------------------------------------------------------
# Main inference
# -------------------------------------------------------------------

def run_inference():
    # Prepare output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"run_inference_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("PaddleOCR-VL INFERENCE - SCHMUCK DATASET (TEST)")
    print("=" * 60)
    print(f"Model HF name:   {HF_MODEL_NAME}")
    print(f"Local model dir: {LOCAL_MODEL_DIR}")
    print(f"Test JSONL:      {TEST_JSONL_PATH}")
    print(f"Images dir:      {IMAGES_DIR}")
    print(f"Output dir:      {run_dir}")
    print(f"Device:          {DEVICE}")
    print("=" * 60)

    # Load model & processor
    processor, model = download_and_cache_model(HF_MODEL_NAME, LOCAL_MODEL_DIR)

    # Load test data
    test_data = load_jsonl(TEST_JSONL_PATH)
    print(f"Loaded {len(test_data)} test samples.")

    predictions = []

    for idx, item in enumerate(tqdm(test_data, desc="Running OCR")):
        # Determine image key (file_name for Schmuck)
        if "file_name" in item:
            image_name = item["file_name"]
        elif "image_name" in item:
            image_name = item["image_name"]
        else:
            print(f"[{idx}] No file_name/image_name key in item, skipping.")
            continue

        image_path = find_image_path(image_name, IMAGES_DIR)
        if not os.path.exists(image_path):
            print(f"[{idx}] Image not found: {image_path}, skipping.")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}] Error opening image {image_path}: {e}, skipping.")
            continue

        # Build messages as in the official PaddleOCR-VL example
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPTS[TASK]},
                ],
            }
        ]

        with torch.no_grad():
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(DEVICE)

            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                pad_token_id=processor.tokenizer.eos_token_id
                if hasattr(processor, "tokenizer") and processor.tokenizer.eos_token_id is not None
                else None,
            )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded = decoded.strip()

        # Often model might echo "OCR:" etc. Try to strip leading prompt.
        if PROMPTS[TASK] in decoded:
            pred_text = decoded.split(PROMPTS[TASK], 1)[1].strip()
        else:
            pred_text = decoded

        # Ground truth JSON → canonical string (minus file_name/image_name)
        gt_json = dict_without_image_key(item)
        gt_str = canonical_json_string(gt_json)

        # Normalize both
        gt_str_norm = normalize_text(gt_str)
        pred_str_norm = normalize_text(pred_text)

        sample_record = {
            "index": idx,
            "image_name": image_name,
            "image_path": image_path,
            "prediction_raw": decoded,
            "prediction_text": pred_text,
            "gt_json": gt_json,
            "gt_string": gt_str,
            "gt_str_norm": gt_str_norm,
            "pred_str_norm": pred_str_norm,
        }
        predictions.append(sample_record)

    # Compute CER metrics
    metrics = compute_cer_metrics(predictions)

    # Save predictions JSONL
    pred_path = os.path.join(run_dir, "schmuck_test_predictions_paddleocr_vl.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Save metrics
    metrics_path = os.path.join(run_dir, "cer_results_paddleocr_vl.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("PaddleOCR-VL – Schmuck dataset (test split)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Num samples evaluated: {metrics['num_samples']}\n\n")
        f.write("CER (jiwer.cer, averaged per sample):\n")
        f.write(f"  Avg CER: {metrics['avg_cer_jiwer']:.4f} "
                f"({metrics['avg_cer_jiwer_percent']:.2f}%)\n\n")
        f.write("CER (editdistance, total_errors / total_chars):\n")
        f.write(f"  CER: {metrics['editdistance_cer']:.4f} "
                f"({metrics['editdistance_cer_percent']:.2f}%)\n")

    print("\nInference finished.")
    print(f"Predictions saved to: {pred_path}")
    print(f"CER results saved to: {metrics_path}")
    print(f"All outputs in: {run_dir}")


if __name__ == "__main__":
    run_inference()
