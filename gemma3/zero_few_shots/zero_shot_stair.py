#!/usr/bin/env python3
# gemma3_staircase_zeroshot_chat_fixed.py
# Zero-shot OCR on the Staircase dataset with Gemma-3 (chat-style, transformers)
# - Uses a single English, schema-based JSON prompt (no 1/2a/2b/3b types)
# - Dual-view image (original + autocontrast) for faint marks
# - Strips trailing EOS from inputs to avoid zero-length generations
# - Greedy decoding with sampling fallback
# - Robust JSON extraction and CER logging

import os
import json
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import jiwer
import gc
import warnings

# --------------------------
# Paths
# --------------------------
TEST_JSONL  = "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl"
IMAGES_DIR  = "/home/woody/iwi5/iwi5298h/staircase_images"
OUT_DIR     = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/shots_stair"

# Local Gemma-3 snapshot (4-bit Unsloth). Adjust if you use a different path/model.
MODEL_PATH  = "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"

# --------------------------
# Utilities
# --------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def json_to_string(obj: Dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))

def extract_json_from_response(response: str) -> Dict:
    """Extract the largest JSON object from a free-form LLM response."""
    if not isinstance(response, str):
        return {}
    s = response.strip()

    # Remove markdown fences if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.lstrip().startswith("json"):
                body = body[4:].strip()
            s = body.strip()

    matches = re.findall(r"\{.*\}", s, flags=re.DOTALL)
    if matches:
        for m in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(m)
            except Exception:
                continue

    try:
        return json.loads(s)
    except Exception:
        return {}

def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))

def norm_path(s: str) -> str:
    return strip_accents(s).replace("\\", "/").lower()

def iter_all_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)

def find_image_path(images_root: str, image_name_from_json: str) -> Optional[str]:
    """Robust image lookup (exact, accent-insensitive, fallback substring)."""
    if not image_name_from_json:
        return None

    direct = os.path.join(images_root, image_name_from_json)
    if os.path.exists(direct):
        return direct

    target_norm = norm_path(image_name_from_json)
    for p in iter_all_files(images_root):
        if norm_path(os.path.basename(p)) == target_norm:
            return p

    m = re.search(r"\(\s*\d+\s*\)\.(jpg|jpeg|png|tif|tiff)$", image_name_from_json, re.IGNORECASE)
    if m:
        suffix = image_name_from_json[image_name_from_json.rfind("("):]
        suffix_norm = norm_path(suffix)
        for p in iter_all_files(images_root):
            if norm_path(os.path.basename(p)).endswith(suffix_norm):
                return p

    for p in iter_all_files(images_root):
        if target_norm in norm_path(p):
            return p
    return None

# --------------------------
# Single, English, schema-based prompt
# --------------------------
def staircase_schema_prompt() -> str:
    """
    Single generic schema for the main staircase form layout (Innentreppe etc).
    English instructions, but JSON keys are German to match your ground truth.
    """
    return """You are an OCR model for German historical staircase survey forms.
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

# --------------------------
# Gemma-3 Zero-shot runner
# --------------------------
class Gemma3StairZeroShot:
    def __init__(self, model_path: str):
        print("Loading Gemma-3 (transformers) ...")
        # BF16 if GPU >= Ampere; else FP16 (V100-safe)
        use_bf16 = False
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            use_bf16 = major >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        # Ensure generation_config has EOS/PAD
        tok = self.processor.tokenizer
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = tok.pad_token_id
        if getattr(self.model.generation_config, "eos_token_id", None) is None:
            self.model.generation_config.eos_token_id = tok.eos_token_id

        warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")
        warnings.filterwarnings("ignore", message="Using a slow image processor.*")
        self._token_candidates = ["<image>", "<start_of_image>"]

        print(f"Loaded Gemma-3 from {model_path}, dtype={dtype}")

    @staticmethod
    def _build_messages_dual(image: Image.Image) -> List[Dict]:
        """System + user messages with original and autocontrast images."""
        img2 = ImageOps.autocontrast(image, cutoff=1)
        return [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "You extract structured OCR data from German architectural staircase forms "
                        "and respond with JSON only."
                    ),
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": staircase_schema_prompt()},
                ],
            },
        ]

    @staticmethod
    def _gather_images(messages: List[Dict]) -> List[Image.Image]:
        imgs = []
        for m in messages:
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "image":
                    imgs.append(c.get("image"))
        return imgs

    def _prepare_inputs(self, messages: List[Dict]):
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Check number of <image> markers vs images
        token_count = 0
        for t in self._token_candidates:
            token_count = max(token_count, prompt_text.count(t))
        images = self._gather_images(messages)
        print(f"    [DEBUG] prompt image markers: {token_count}, collected images: {len(images)}")
        if token_count != len(images):
            raise ValueError(
                f"Image token mismatch: prompt has {token_count} markers but {len(images)} images."
            )

        inputs = self.processor(
            text=[prompt_text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Strip trailing EOS from input_ids to avoid immediate stop
        eos_id = self.model.generation_config.eos_token_id
        if eos_id is not None:
            input_ids = inputs["input_ids"]
            attn = inputs["attention_mask"]
            last = input_ids[0, -1].item()
            eos_scalar = eos_id if isinstance(eos_id, int) else eos_id[0]
            if last == eos_scalar:
                inputs["input_ids"] = input_ids[:, :-1]
                inputs["attention_mask"] = attn[:, :-1]
                print("    [DEBUG] stripped trailing EOS from input_ids")

        return inputs

    def generate(self, messages: List[Dict], max_new_tokens: int = 1400) -> Tuple[str, bool]:
        """Greedy decode with sampling fallback. Returns (text, used_sampling)."""
        inputs = self._prepare_inputs(messages)
        input_len = inputs["input_ids"].shape[-1]

        def _decode(gen_ids):
            out = gen_ids[0][input_len:]
            return self.processor.decode(out, skip_special_tokens=True)

        used_sampling = False
        with torch.inference_mode():
            # Stage 1: greedy; force at least some output
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
            )
            text = _decode(out).strip()

            # Stage 2: fallback if empty or obviously not JSON-like
            if (not text) or ("{" not in text):
                used_sampling = True
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=64,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    use_cache=True,
                )
                text = _decode(out).strip()

        # Clean up
        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return text, used_sampling

# --------------------------
# Save & summary
# --------------------------
def save_run(predictions, test_data, cer_list, base_dir: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"run_{ts}")
    ensure_dir(out_dir)

    pred_path = os.path.join(out_dir, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in predictions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_cer = None
    if cer_list:
        avg_cer = sum(cer_list) / len(cer_list)
        min_cer = min(cer_list)
        max_cer = max(cer_list)

        total_chars = 0
        total_errors = 0
        for i, test_item in enumerate(test_data[:len(predictions)]):
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_s = json_to_string(gt_json)
            n = len(gt_s)
            cer = cer_list[i]
            total_chars += n
            total_errors += int(round(cer * n))
        weighted = (total_errors / total_chars) if total_chars > 0 else 0.0
        perfect = sum(1 for c in cer_list if c == 0.0)

        summary = {
            "timestamp": ts,
            "total_images": len(predictions),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted),
            "perfect_matches": int(perfect),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "predictions_path": pred_path,
            "mode": "zero-shot",
        }
        with open(os.path.join(out_dir, "cer_evaluation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return out_dir, avg_cer

# --------------------------
# Main
# --------------------------
def main():
    ensure_dir(OUT_DIR)
    print("="*70)
    print("GEMMA-3 ZERO-SHOT OCR (staircase forms, English schema prompt)")
    print("="*70)

    print("Loading test JSONL ...")
    test_rows = load_jsonl(TEST_JSONL)
    print(f" Test samples: {len(test_rows)}")

    runner = Gemma3StairZeroShot(MODEL_PATH)
    predictions = []
    cer_scores = []

    print("\nProcessing test images (zero-shot) ...")
    for i, row in enumerate(test_rows, 1):
        img_name = row.get("image_name", "")
        print(f"[{i}/{len(test_rows)}] {img_name}")

        img_path = find_image_path(IMAGES_DIR, img_name)
        print(f"    [DEBUG] resolved path: {img_path}")
        if not img_path or not os.path.exists(img_path):
            print("    WARNING: image not found; CER=1.0")
            predictions.append({
                "image_name": img_name,
                "matched_image_path": img_path or "NOT_FOUND",
                "predicted_json": {},
                "ground_truth": {k: v for k, v in row.items() if k != "image_name"},
                "raw_response": "Error: Image not found",
                "cer_score": 1.0
            })
            cer_scores.append(1.0)
            continue

        test_img = Image.open(img_path).convert("RGB")
        messages = runner._build_messages_dual(test_img)

        try:
            raw, used_sampling = runner.generate(messages, max_new_tokens=1400)
            print(f"    [DEBUG] used_sampling_fallback: {used_sampling}")
            print(f"    [DEBUG] raw output length: {len(raw)}")
            preview = raw[:200].replace("\n", " ")
            print(f"    [DEBUG] raw preview: {preview!r}")

            pred_json = extract_json_from_response(raw)
            gt_json   = {k: v for k, v in row.items() if k != "image_name"}

            gt_s  = json_to_string(gt_json)
            pr_s  = json_to_string(pred_json) if pred_json else ""
            cer   = jiwer.cer(gt_s, pr_s) if pr_s else 1.0

            print(f"    CER: {cer:.4f} ({cer*100:.2f}%)")

            predictions.append({
                "image_name": img_name,
                "matched_image_path": img_path,
                "predicted_json": pred_json,
                "ground_truth": gt_json,
                "raw_response": raw,
                "cer_score": cer,
            })
            cer_scores.append(cer)

        except Exception as e:
            print(f"    ERROR: {e}")
            predictions.append({
                "image_name": img_name,
                "matched_image_path": img_path,
                "predicted_json": {},
                "ground_truth": {k: v for k, v in row.items() if k != "image_name"},
                "raw_response": f"Error: {e}",
                "cer_score": 1.0,
            })
            cer_scores.append(1.0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    out_dir, avg_cer = save_run(predictions, test_rows, cer_scores, OUT_DIR)
    print("\n" + "="*70)
    print("DONE (Gemma-3 zero-shot on staircase test split)")
    print(f"Saved to: {out_dir}")
    if avg_cer is not None:
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
