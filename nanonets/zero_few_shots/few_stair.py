#!/usr/bin/env python3
# nanonets_staircase_fewshot_schema_prompt.py
# Few-shot OCR on the Staircase dataset using Nanonets-OCR-s
# - Uses 2 few-shot examples from train.jsonl:
#     "FMIS_FormblätterMielke_gefüllt (8).jpg"
#     "FMIS_FormblätterMielke_gefüllt (149).jpg"
# - Uses the SAME big schema prompt as the zero-shot script
# - Few-shot JSONs from train.jsonl define the exact schema
# - Robust image search (exact / "(n).jpg" / fuzzy)
# - Resizes images to prevent OOM
# - CER computed on JSON strings excluding image_name/image_path
# - Outputs saved to .../nanonets/shots_staircase/few_shot_<timestamp>/

import os
# CRITICAL: Set this BEFORE importing PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from huggingface_hub import snapshot_download
import torch
import json
from typing import List, Dict, Tuple
from datetime import datetime
import re
import jiwer
from PIL import Image
import gc
import glob
import unicodedata

# --------------------------
# Paths
# --------------------------
TRAIN_JSONL = "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl"
TEST_JSONL  = "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl"
IMAGES_DIR  = "/home/woody/iwi5/iwi5298h/staircase_images"
OUT_BASE    = "/home/vault/iwi5/iwi5298h/models_image_text/nanonets/shots_staircase"
MODEL_PATH  = "/home/vault/iwi5/iwi5298h/models/Nanonets-OCR-s"

# --------------------------
# Helpers
# --------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def json_to_string(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))

def safe_json(obj: Dict, drop=("image_name", "image_path")) -> Dict:
    return {k: v for k, v in obj.items() if k not in drop}

def extract_json_from_response(response: str) -> Dict:
    """
    Extract FIRST JSON object from the model response.
    (slightly more robust than bare regex)
    """
    response = str(response).strip()

    # Strip fenced blocks if present
    if response.startswith("```"):
        parts = response.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.lstrip().lower().startswith("json"):
                body = body[4:].strip()
            response = body.strip()

    if "{" not in response or "}" not in response:
        # No obvious JSON, just give up
        try:
            return json.loads(response)
        except Exception:
            return {}

    # Take substring from first "{" to last "}" and then try to find
    # the first *balanced* JSON object inside it.
    start = response.find("{")
    end = response.rfind("}") + 1
    candidate = response[start:end]

    # Simple brace-balancing to cut at the first full JSON object
    brace_count = 0
    cut_idx = -1
    for i, ch in enumerate(candidate):
        if ch == "{":
            brace_count += 1
        elif ch == "}":
            brace_count -= 1
            if brace_count == 0:
                cut_idx = i + 1
                break

    if cut_idx > 0:
        first_obj = candidate[:cut_idx]
        try:
            return json.loads(first_obj)
        except Exception:
            pass

    # Fallback: try regex (can still work sometimes)
    matches = re.findall(r"\{.*\}", candidate, flags=re.DOTALL)
    if matches:
        for m in matches:
            try:
                return json.loads(m)
            except Exception:
                continue

    # Final fallback
    try:
        return json.loads(candidate)
    except Exception:
        return {}

def strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

def norm_filename_for_match(p: str) -> str:
    p = strip_accents(p)
    p = p.replace("\\", "/").lower()
    return p

def find_image_path(image_name: str, images_dir: str) -> str:
    """Try exact, numbered suffix, then fuzzy substring match."""
    exact = os.path.join(images_dir, image_name)
    if os.path.exists(exact):
        return exact

    # Try patterns like "(123).jpg"
    m = re.search(r"\((\d+)\)\.(jpg|jpeg|png|tif|tiff)$", image_name, re.IGNORECASE)
    if m:
        suffix = f"({m.group(1)}).{m.group(2)}"
        hits = glob.glob(os.path.join(images_dir, f"*{suffix}"))
        if hits:
            return hits[0]

    base = os.path.splitext(os.path.basename(image_name))[0]
    hits = glob.glob(os.path.join(images_dir, f"*{base}*"))
    if hits:
        return hits[0]

    return exact  # may be non-existent; caller will check

def resize_image_if_needed(image_path: str, max_size: int = 1024) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    if w >= h:
        new_w, new_h = max_size, int(max_size * h / w)
    else:
        new_h, new_w = max_size, int(max_size * w / h)
    return image.resize((new_w, new_h), Image.LANCZOS)

# --------------------------
# Use SAME schema prompt as zero-shot
# --------------------------
ZERO_SHOT_SCHEMA_PROMPT = """You are an OCR model for German historical staircase survey forms.
Read all printed and handwritten text from the form image and extract the information
into exactly ONE JSON object with the following structure:

{
  "stair_type": "Type of stair (e.g. INNENTREPPE, AUSSENTREPPE)",
  "Name des Hauses": "Name of the building",
  "Adresse": "Address",
  "Bauherr": "Building owner/client",
  "Baumeister": "Master builder / architect",
  "Bauzeit": "Construction period",
  "MATERIAL": "Material description (e.g. Holz, Stein)",
  "TREPPENTYP (Laufform)": "Stair type/form text",
  "LÄUFE": {
    "EG": {
      "Stufen": ["4 values for steps"],
      "Steig.": ["4 values for rise/step height"],
      "Breite": ["4 values for width"],
      "Plateaus": ["3 values for landings/plateaus"]
    },
    "1.OG": {
      "Stufen": ["4 values for steps"],
      "Steig.": ["4 values for rise/step height"],
      "Breite": ["4 values for width"],
      "Plateaus": ["3 values for landings/plateaus"]
    },
    "2.OG": {
      "Stufen": ["4 values for steps"],
      "Steig.": ["4 values for rise/step height"],
      "Breite": ["4 values for width"],
      "Plateaus": ["3 values for landings/plateaus"]
    },
    "3.OG": {
      "Stufen": ["4 values for steps"],
      "Steig.": ["4 values for rise/step height"],
      "Breite": ["4 values for width"],
      "Plateaus": ["3 values for landings/plateaus"]
    }
  },
  "Notes": "Any additional notes written on the form",
  "STUFENPROFIL": "Text describing the step profile",
  "Schweifung": boolean,  // true if curved / schweifung is indicated, else false
  "Untertritt": "Undertread value (cm)",
  "WANGE / HOLM": {
    "eingestemmte Stufen": boolean,
    "eingespannte Stufen": boolean,
    "aufgesattelte Stufen": boolean
  },
  "GELÄNDER": {
    "Stabwerk": {"Holz": boolean, "Stein": boolean, "Eisen": boolean},
    "Traljen": {"Holz": boolean, "Stein": boolean, "Eisen": boolean},
    "Docken": {"Holz": boolean, "Stein": boolean, "Eisen": boolean},
    "Baluster": {"Holz": boolean, "Stein": boolean, "Eisen": boolean},
    "Bal.-Brett": {"Holz": boolean, "Stein": boolean, "Eisen": boolean},
    "Ornament": {"Holz": boolean, "Stein": boolean, "Eisen": boolean},
    "Paneel": {"Holz": boolean, "Stein": boolean, "Eisen": boolean}
  },
  "ANFÄNGER": "Description / dimensions of starting post",
  "DEKORATION": "Decoration description",
  "HANDLAUFPROFIL": {
    "Hohe": "Handrail height",
    "Breite": "Handrail width"
  },
  "Datum": "Date written on the form",
  "Signature": "Signature text (if present)",
  "ORT": "Location (Ort) on the form"
}

Rules:
- Use exactly these JSON keys.
- If a field is not filled on the form, still include it with an empty string "" (or false / empty arrays).
- Use strings for all numeric values and free text.
- Use true/false for checkboxes and yes/no fields.
- Return ONLY one valid JSON object, with no extra text before or after it.
"""

# We'll use this as the "user" text for examples + test
USER_TEXT = ZERO_SHOT_SCHEMA_PROMPT

# --------------------------
# Select fixed few-shot examples
# --------------------------
def select_fixed_few_shot_examples(train_data: List[Dict]) -> List[Dict]:
    """
    Use exactly these two images from train.jsonl as few-shot examples:
    - "FMIS_FormblätterMielke_gefüllt (8).jpg"
    - "FMIS_FormblätterMielke_gefüllt (149).jpg"
    """
    print("\nSelecting fixed few-shot examples (8) and (149)...")

    target_images = [
        "FMIS_FormblätterMielke_gefüllt (8).jpg",
        "FMIS_FormblätterMielke_gefüllt (149).jpg",
    ]

    few_shot_examples: List[Dict] = []

    for target in target_images:
        target_norm = norm_filename_for_match(target)
        found = False
        for item in train_data:
            img_name = item.get("image_name", "")
            if norm_filename_for_match(img_name) == target_norm:
                few_shot_examples.append(item)
                print(f"  ✓ Found in train.jsonl: {img_name}")
                found = True
                break
        if not found:
            print(f"  ✗ NOT FOUND in train.jsonl: {target}")

    if len(few_shot_examples) < 2:
        raise RuntimeError("Could not find both requested few-shot images (8) and (149) in train.jsonl.")

    return few_shot_examples

# --------------------------
# Few-shot prompt builder (messages + images list)
# --------------------------
def build_fewshot(images_dir: str, few_shot_examples: List[Dict]) -> Tuple[List[Dict], List[Image.Image]]:
    """
    Create few-shot conversation and matching images list (order matters),
    using records from train.jsonl, plus a single system message.
    We now use the SAME schema prompt as in zero-shot, as user text.
    """
    messages: List[Dict] = []
    images: List[Image.Image] = []

    # System message: short, generic
    messages.append({
        "role": "system",
        "content": (
            "You extract structured JSON data from German architectural staircase forms. "
            "Always follow the user instructions and output exactly one JSON object."
        ),
    })

    # For each few-shot example: user (image + schema prompt) -> assistant (JSON)
    for ex in few_shot_examples:
        img_name = ex.get("image_name", "")
        img_path = find_image_path(img_name, images_dir)
        print(f"  Few-shot example: {img_name} -> {img_path}")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Few-shot image not found on disk: {img_path}")
        img = resize_image_if_needed(img_path, max_size=1024)

        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": USER_TEXT},
            ],
        })
        images.append(img)

        messages.append({
            "role": "assistant",
            "content": json_to_string(safe_json(ex)),
        })

    return messages, images

# --------------------------
# Model wrapper
# --------------------------
class NanonetsStairFewShot:
    def __init__(self, model_path: str = MODEL_PATH):
        print("Loading Nanonets-OCR-s...")
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Model not found at {model_path}. Downloading...")
            ensure_dir(model_path)
            snapshot_download(
                repo_id="nanonets/Nanonets-OCR-s",
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,    # V100/FP16-friendly
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        print("Model loaded.")

    def run_with_fewshots(self, test_img_path: str, few_messages: List[Dict], few_images: List[Image.Image]) -> str:
        """Append test image/user turn to few-shot convo; run generation."""
        torch.cuda.empty_cache()
        gc.collect()

        test_img = resize_image_if_needed(test_img_path, max_size=1024)

        messages = list(few_messages) + [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_TEXT},
                ],
            }
        ]
        images = list(few_images) + [test_img]

        # Build inputs
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,   # should be enough for the full JSON
                temperature=0.0,
                do_sample=False,
                use_cache=True,
            )

        # Remove input tokens to get only generated tokens
        generated_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        out_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        # Cleanup
        del inputs, output_ids, generated_ids, text, images, messages, test_img
        torch.cuda.empty_cache()
        gc.collect()

        return out_text

# --------------------------
# Main
# --------------------------
def main():
    # Timestamped output dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_BASE, f"few_shot_schema_{ts}")
    ensure_dir(out_dir)
    pred_path = os.path.join(out_dir, "predictions_fewshot.jsonl")
    summary_path = os.path.join(out_dir, "evaluation_summary_fewshot.json")

    print("=" * 60)
    print("NANONETS-OCR-S FEW-SHOT STAIRCASE OCR (SCHEMA PROMPT + EXAMPLES)")
    print("=" * 60)

    # Load train + select few-shot examples
    print("Loading train.jsonl ...")
    train_rows = load_jsonl(TRAIN_JSONL)
    print(f"Loaded {len(train_rows)} train samples.")

    few_shot_examples = select_fixed_few_shot_examples(train_rows)
    print("\nFew-shot examples selected:")
    for ex in few_shot_examples:
        print("  -", ex.get("image_name", "NO_NAME"))

    # Load test set
    print("\nLoading test.jsonl ...")
    test_rows = load_jsonl(TEST_JSONL)
    print(f"Loaded {len(test_rows)} test samples.")

    # Init model
    runner = NanonetsStairFewShot(MODEL_PATH)

    # Build few-shot conversation & images list
    print("\nBuilding few-shot messages ...")
    few_msgs, few_imgs = build_fewshot(IMAGES_DIR, few_shot_examples)
    print(
        f"Few-shot conversation length: {len(few_msgs)} messages "
        f"({len(few_imgs)} example images)."
    )

    predictions = []
    cer_scores = []
    total_chars = 0
    total_errors = 0

    for i, row in enumerate(test_rows, 1):
        image_name = row.get("image_name", "")
        img_path = find_image_path(image_name, IMAGES_DIR)
        print(f"[{i}/{len(test_rows)}] {image_name} -> {img_path}")

        gt_json = safe_json(row)
        gt_str = json_to_string(gt_json)

        if not os.path.exists(img_path):
            print("  WARNING: image not found; CER=1.0")
            pred_json = {}
            raw = "Error: Image not found"
            pred_str = ""
            cer = 1.0
        else:
            try:
                raw = runner.run_with_fewshots(img_path, few_msgs, few_imgs)
                pred_json = extract_json_from_response(raw)
                pred_str = json_to_string(pred_json) if pred_json else ""
                cer = jiwer.cer(gt_str, pred_str) if pred_str else 1.0
            except Exception as e:
                raw = f"Error: {e}"
                pred_json = {}
                pred_str = ""
                cer = 1.0

        predictions.append({
            "image_name": image_name,
            "matched_image_path": img_path,
            "predicted_json": pred_json,
            "ground_truth": gt_json,
            "raw_response": raw,
            "cer_score": cer,
        })
        cer_scores.append(cer)

        total_chars += len(gt_str)
        total_errors += int(round(cer * len(gt_str)))

        print(f"  CER: {cer:.4f}")

    # Save predictions
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nPredictions saved -> {pred_path}")

    # Summary
    if cer_scores:
        avg_cer = sum(cer_scores) / len(cer_scores)
        min_cer = min(cer_scores)
        max_cer = max(cer_scores)
        weighted_cer = (total_errors / total_chars) if total_chars > 0 else 0.0
        perfect = sum(1 for x in cer_scores if x == 0.0)

        summary = {
            "timestamp": ts,
            "model": "Nanonets-OCR-s",
            "dataset": "staircase_documentation",
            "method": "few-shot",
            "total_images": len(cer_scores),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "few_shot_examples": [
                {
                    "image_name": ex.get("image_name", "NO_NAME"),
                    "matched_path": find_image_path(ex.get("image_name", ""), IMAGES_DIR),
                }
                for ex in few_shot_examples
            ],
            "predictions_path": pred_path,
            "prompt_notes": "Same schema prompt as zero-shot + 2 few-shot examples.",
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 60)
        print("NANONETS-OCR-S FEW-SHOT CER EVALUATION RESULTS (SCHEMA PROMPT + EXAMPLES)")
        print("STAIRCASE DOCUMENTATION DATASET")
        print("=" * 60)
        print(f"Average CER:  {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Minimum CER:  {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER:  {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"Weighted CER: {weighted_cer:.4f} ({weighted_cer*100:.2f}%)")
        print(f"Perfect:      {perfect}/{len(cer_scores)}")
        print(f"Summary saved -> {summary_path}")
    else:
        print("No samples processed — no summary written.")

    print("\nDONE. Outputs in:", out_dir)

if __name__ == "__main__":
    main()
