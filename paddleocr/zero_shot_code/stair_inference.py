#!/usr/bin/env python3
# paddleocr_vl_stair_infer.py
#
# Simple inference + CER evaluation of PaddleOCR-VL on Staircase dataset.
# - Uses HF model "PaddlePaddle/PaddleOCR-VL"
# - Loads from local cache dir if available, else downloads + caches.
# - Compares raw OCR text to JSON ground truth (stringified).
# - Computes character error rate (CER) over the whole split.

import os
import json
import argparse
import unicodedata
from typing import List, Dict, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor

try:
    import editdistance
except ImportError:
    editdistance = None
    print("⚠️ editdistance not installed; CER(levenshtein) will be skipped.")

try:
    import jiwer
except ImportError:
    jiwer = None
    print("⚠️ jiwer not installed; CER(jiwer) will be skipped.")


PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


# -----------------------
# Helper functions
# -----------------------

def normalize_text(text: str) -> str:
    """NFC normalize + collapse whitespace (same style as other scripts)."""
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def dict_without_image_name(obj: Dict) -> Dict:
    return {k: v for k, v in obj.items() if k != "image_name"}


def canonical_json_string(obj: Dict) -> str:
    """Stringify JSON in a stable way (no sort_keys to preserve structure feel)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def compute_cer_levenshtein(preds: List[str], gts: List[str]) -> float:
    """Character error rate using editdistance."""
    if editdistance is None:
        return float("nan")

    total_chars = 0
    total_errors = 0
    for p, t in zip(preds, gts):
        t = t or ""
        p = p or ""
        total_chars += len(t)
        total_errors += editdistance.eval(p, t)

    if total_chars == 0:
        return float("nan")
    return total_errors / total_chars


def compute_cer_jiwer(preds: List[str], gts: List[str]) -> float:
    """Character error rate using jiwer.cer."""
    if jiwer is None:
        return float("nan")

    total_cer = 0.0
    valid_pairs = 0
    for p, t in zip(preds, gts):
        t = t or ""
        p = p or ""
        if len(t) == 0:
            continue
        total_cer += jiwer.cer(t, p)
        valid_pairs += 1

    if valid_pairs == 0:
        return float("nan")
    return total_cer / valid_pairs


def load_model_and_processor(model_name: str, local_dir: str, device: str):
    """
    Load PaddleOCR-VL from local dir if present, otherwise download + cache.
    """
    os.makedirs(local_dir, exist_ok=True)
    config_path = os.path.join(local_dir, "config.json")

    if os.path.exists(config_path):
        print(f"✅ Found local model at {local_dir}, loading from disk...")
        processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
    else:
        print(f"⬇️  Downloading model {model_name} to {local_dir}...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(device)
        print(f"💾 Saving model + processor to {local_dir}...")
        processor.save_pretrained(local_dir)
        model.save_pretrained(local_dir)

    model.eval()
    return processor, model


# -----------------------
# Inference loop
# -----------------------

def run_inference_on_split(
    jsonl_path: str,
    images_dir: str,
    processor,
    model,
    device: str,
    task: str = "ocr",
    max_new_tokens: int = 512,
    max_samples: int = -1,
    verbose_every: int = 20,
) -> Tuple[List[Dict], Dict[str, float]]:
    """
    - Reads JSONL (staircase format with 'image_name' + fields)
    - Runs PaddleOCR-VL (OCR) on each image
    - Compares RAW OCR text vs JSON ground truth string
    - Returns predictions list + metrics dict
    """
    data = load_jsonl(jsonl_path)
    if max_samples > 0:
        data = data[:max_samples]

    print(f"Loaded {len(data)} samples from {jsonl_path}")

    all_preds = []
    norm_preds = []
    norm_gts = []

    for idx, item in enumerate(tqdm(data, desc="Running inference")):
        image_name = item["image_name"]
        image_path = os.path.join(images_dir, image_name)

        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}, skipping.")
            continue

        # Ground truth JSON (without image_name)
        gt_json = dict_without_image_name(item)
        gt_str = canonical_json_string(gt_json)
        gt_str_norm = normalize_text(gt_str)

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPTS[task]},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,     # deterministic for evaluation
                temperature=0.0,
                use_cache=True,
            )

        decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        decoded = decoded.strip()

        # Optional: remove "OCR:" prefix if present
        if PROMPTS["ocr"] in decoded:
            # Everything after the first occurrence of OCR:
            pred_text = decoded.split(PROMPTS["ocr"], 1)[1].strip()
        else:
            pred_text = decoded

        pred_text_norm = normalize_text(pred_text)

        all_preds.append(
            {
                "index": idx,
                "image_name": image_name,
                "image_path": image_path,
                "prediction_raw": pred_text,
                "ground_truth_json": gt_json,
                "ground_truth_str": gt_str,
            }
        )

        norm_preds.append(pred_text_norm)
        norm_gts.append(gt_str_norm)

        if verbose_every > 0 and (idx + 1) % verbose_every == 0:
            print("\n----------- SAMPLE [{}] -----------".format(idx + 1))
            print(f"Image: {image_name}")
            print("PRED:", pred_text[:300])
            print("GT  :", gt_str[:300])
            print("---------------------------------\n")

    # Compute metrics
    metrics = {}
    cer_lev = compute_cer_levenshtein(norm_preds, norm_gts)
    cer_jwr = compute_cer_jiwer(norm_preds, norm_gts)

    if cer_lev == cer_lev:  # not NaN
        metrics["cer_levenshtein"] = cer_lev
    if cer_jwr == cer_jwr:
        metrics["cer_jiwer"] = cer_jwr

    print("\n===== FINAL METRICS =====")
    if "cer_levenshtein" in metrics:
        print(f"CER (editdistance): {metrics['cer_levenshtein']:.4f} "
              f"({metrics['cer_levenshtein'] * 100:.2f}%)")
    if "cer_jiwer" in metrics:
        print(f"CER (jiwer)       : {metrics['cer_jiwer']:.4f} "
              f"({metrics['cer_jiwer'] * 100:.2f}%)")
    print("=========================\n")

    return all_preds, metrics


# -----------------------
# main
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simple inference + CER evaluation of PaddleOCR-VL on Staircase dataset"
    )
    parser.add_argument(
        "--data_dir",
        default="/home/woody/iwi5/iwi5298h/json_staircase",
        help="Directory with train/val/test JSONL files",
    )
    parser.add_argument(
        "--images_dir",
        default="/home/woody/iwi5/iwi5298h/staircase_images",
        help="Directory containing images",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--model_name",
        default="PaddlePaddle/PaddleOCR-VL",
        help="HF model id",
    )
    parser.add_argument(
        "--local_model_dir",
        default="/home/vault/iwi5/iwi5298h/models/PaddleOCR-VL",
        help="Local directory to cache PaddleOCR-VL model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit number of samples for quick tests (-1 = all)",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/vault/iwi5/iwi5298h/models_image_text/paddleocr/stair/inference_runs",
        help="Where to save predictions + metrics",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor, model = load_model_and_processor(
        args.model_name,
        args.local_model_dir,
        device,
    )

    jsonl_path = os.path.join(args.data_dir, f"{args.split}.jsonl")
    print(f"Evaluating split: {args.split} ({jsonl_path})")

    preds, metrics = run_inference_on_split(
        jsonl_path=jsonl_path,
        images_dir=args.images_dir,
        processor=processor,
        model=model,
        device=device,
        task="ocr",
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
    )

    # Save predictions and metrics
    split_out_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(split_out_dir, exist_ok=True)

    preds_path = os.path.join(split_out_dir, "predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    metrics_path = os.path.join(split_out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Predictions saved to: {preds_path}")
    print(f"Metrics saved to    : {metrics_path}")


if __name__ == "__main__":
    main()
