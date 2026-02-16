#!/usr/bin/env python3
# staircase_gemma_fewshot.py
# Few-shot OCR on the Staircase dataset with Gemma-3 (Unsloth snapshot)
# - Uses the SAME schema-based prompt as the good zero-shot version
# - Uses 2 fixed few-shot examples from train.jsonl:
#     "FMIS_FormblätterMielke_gefüllt (8).jpg"
#     "FMIS_FormblätterMielke_gefüllt (149).jpg"
# - Dual-view (original + autocontrast) for all images (few-shot + test)
# - Robust image-path resolution
# - NO key normalisation: CER on raw GT JSON vs raw prediction
# - Greedy decode with sampling fallback
# - Saves predictions + CER summary in timestamped run folder

import os
import json
import re
import gc
import unicodedata
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import jiwer
import warnings

# --------------------------
# Paths
# --------------------------
TRAIN_JSONL = "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl"
TEST_JSONL  = "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl"
IMAGES_DIR  = "/home/woody/iwi5/iwi5298h/staircase_images"
OUT_DIR     = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/shots_stair"

MODEL_PATH  = "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"


# --------------------------
# Small utilities
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
    """Consistent string representation of a dict (no key reordering)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def extract_json_from_response(response: str) -> Dict:
    """Extract JSON object from model response (robust to extra text / fences)."""
    if not isinstance(response, str):
        return {}
    s = response.strip()

    # Strip markdown code fences if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            body = parts[1]
            # handle ```json
            if body.lstrip().lower().startswith("json"):
                body = body[4:].strip()
            s = body.strip()

    # Try to find the largest JSON-looking block
    matches = re.findall(r"\{.*\}", s, flags=re.DOTALL)
    if matches:
        for m in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(m)
            except Exception:
                continue

    # Final fallback: try whole string
    try:
        return json.loads(s)
    except Exception:
        return {}


# --------------------------
# Robust image path resolution
# --------------------------
def strip_accents(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )


def norm_filename_for_match(p: str) -> str:
    p = strip_accents(p)
    p = p.replace("\\", "/").lower()
    return p


def iter_all_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def find_image_path(images_root: str, image_name_from_json: str) -> Optional[str]:
    """Try multiple strategies (exact, basename, pattern, substring) to find image."""
    if not image_name_from_json:
        return None

    # 1) Direct join
    direct = os.path.join(images_root, image_name_from_json)
    if os.path.exists(direct):
        return direct

    target_norm = norm_filename_for_match(image_name_from_json)

    # 2) Exact basename match
    for path in iter_all_files(images_root):
        if norm_filename_for_match(os.path.basename(path)) == target_norm:
            return path

    # 3) Pattern like "(123).jpg"
    m = re.search(r"\(\s*\d+\s*\)\.(jpg|jpeg|png|tif|tiff)$", image_name_from_json, flags=re.IGNORECASE)
    if m:
        suffix = image_name_from_json[image_name_from_json.rfind("("):]
        suffix_norm = norm_filename_for_match(suffix)
        for path in iter_all_files(images_root):
            base = norm_filename_for_match(os.path.basename(path))
            if base.endswith(suffix_norm):
                return path

    # 4) Fallback: substring anywhere
    for path in iter_all_files(images_root):
        if target_norm in norm_filename_for_match(path):
            return path

    return None


# --------------------------
# Schema-based prompt (same as good zero-shot)
# --------------------------
def staircase_schema_prompt() -> str:
    """
    Single generic schema for the main staircase form layout.
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
  "Schweifung": boolean,
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
# Gemma-3 few-shot runner (built on zero-shot variant)
# --------------------------
class Gemma3StairFewShot:
    def __init__(self, model_path: str):
        print("Loading Gemma-3 4B (Unsloth snapshot) for Staircase few-shot...")

        # BF16 if A100+ else FP16 (V100-safe)
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

        tok = self.processor.tokenizer
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = tok.pad_token_id
        if getattr(self.model.generation_config, "eos_token_id", None) is None:
            self.model.generation_config.eos_token_id = tok.eos_token_id

        warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")
        warnings.filterwarnings("ignore", message="Using a slow image processor.*")
        self._token_candidates = ["<image>", "<start_of_image>"]
        print(f"Loaded Gemma-3. dtype={dtype}")

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

        # Count image markers vs images
        token_count = 0
        for t in self._token_candidates:
            token_count = max(token_count, prompt_text.count(t))

        images = self._gather_images(messages)
        print(f"    [DEBUG] prompt image markers: {token_count}, collected images: {len(images)}")
        if token_count != len(images):
            raise ValueError(
                f"Image token mismatch: prompt has {token_count} image markers "
                f"but collected {len(images)} images."
            )

        inputs = self.processor(
            text=[prompt_text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Strip trailing EOS to avoid empty completion
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
        inputs = self._prepare_inputs(messages)
        input_len = inputs["input_ids"].shape[-1]

        def _decode(gen_ids):
            out = gen_ids[0][input_len:]
            return self.processor.decode(out, skip_special_tokens=True)

        used_sampling = False
        with torch.inference_mode():
            # Stage 1: greedy
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=64,
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
            )
            text = _decode(out).strip()

            # Fallback if empty / not JSON-like
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

        # Memory clean
        del inputs, out
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        return text, used_sampling


# --------------------------
# Few-shot message builder (dual-view)
# --------------------------
def create_few_shot_messages(
    few_shot_examples: List[Dict],
    images_dir: str,
) -> List[Dict]:
    """
    Build a conversation:
      system
      user (dual image + schema prompt) / assistant (pure JSON) for each few-shot example.
    """
    messages: List[Dict] = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You extract structured OCR data from German architectural staircase forms "
                        "and respond with JSON only."
                    ),
                }
            ],
        }
    ]

    schema_text = staircase_schema_prompt()

    for i, ex in enumerate(few_shot_examples):
        img_name = ex.get("image_name", "")
        img_path = find_image_path(images_dir, img_name)
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Few-shot image not found: {img_name}")

        print(f"Few-shot example {i+1}: {img_name}")
        print(f"  Path: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img2 = ImageOps.autocontrast(img, cutoff=1)

        # Ground truth as-is (minus image_name)
        ex_clean = {k: v for k, v in ex.items() if k != "image_name"}
        ex_json_str = json_to_string(ex_clean)

        # USER: dual image + schema prompt
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "image", "image": img2},
                    {"type": "text", "text": schema_text},
                ],
            }
        )

        # ASSISTANT: pure JSON
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ex_json_str},
                ],
            }
        )

    return messages


# --------------------------
# Few-shot selection from train.jsonl
# --------------------------
def select_fixed_few_shot_examples(train_data: List[Dict]) -> List[Dict]:
    """
    Select the two specified few-shot examples from train.jsonl:
      - "FMIS_FormblätterMielke_gefüllt (8).jpg"
      - "FMIS_FormblätterMielke_gefüllt (149).jpg"
    """
    targets = [
        "FMIS_FormblätterMielke_gefüllt (8).jpg",
        "FMIS_FormblätterMielke_gefüllt (149).jpg",
    ]
    target_norms = [norm_filename_for_match(t) for t in targets]

    picked: List[Dict] = []

    for t_norm, t_name in zip(target_norms, targets):
        found = False
        for row in train_data:
            img_name = row.get("image_name", "")
            if norm_filename_for_match(img_name) == t_norm:
                picked.append(row)
                print(f"✓ Found few-shot example for {t_name}: {img_name}")
                found = True
                break
        if not found:
            print(f"✗ Could not find requested few-shot image: {t_name}")

    # Fallback if one/both missing
    if len(picked) < 2:
        print("WARNING: Not all requested few-shot examples found. Falling back to first 2 rows.")
        extra_needed = 2 - len(picked)
        for row in train_data:
            if row in picked:
                continue
            picked.append(row)
            extra_needed -= 1
            if extra_needed <= 0:
                break

    return picked[:2]


# --------------------------
# Save + CER summary
# --------------------------
def save_predictions_with_timestamp(
    predictions,
    test_data,
    all_cer_scores,
    few_shot_info,
    base_dir: str,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_fewshot_{timestamp}")
    ensure_dir(output_dir)

    print(f"\nSaving results to: {output_dir}")

    pred_file = os.path.join(output_dir, "predictions_fewshot.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Predictions saved to: {pred_file}")

    if all_cer_scores:
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)

        total_chars = 0
        total_errors = 0
        for i, test_item in enumerate(test_data[:len(predictions)]):
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_str = json_to_string(gt_json)
            n = len(gt_str)
            cer = all_cer_scores[i]
            total_chars += n
            total_errors += int(round(cer * n))

        weighted_cer = (total_errors / total_chars) if total_chars > 0 else 0.0
        perfect_matches = sum(1 for c in all_cer_scores if c == 0.0)

        print("\n" + "=" * 60)
        print("GEMMA-3 FEW-SHOT CER EVALUATION RESULTS (Staircase)")
        print("=" * 60)
        print(f"\nCER over {len(all_cer_scores)} images:")
        print("-" * 50)
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Minimum CER: {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER: {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"\nWeighted CER: {weighted_cer:.4f} ({weighted_cer*100:.2f}%)")
        print(f"Total characters: {total_chars}")
        print(f"Total errors: {total_errors}")
        print(f"Perfect matches: {perfect_matches}/{len(all_cer_scores)} "
              f"({perfect_matches/len(all_cer_scores)*100:.1f}%)")

        summary_file = os.path.join(output_dir, "evaluation_summary_fewshot.json")
        summary = {
            "timestamp": timestamp,
            "model": "Gemma-3-4B (Unsloth, bnb-4bit)",
            "method": "few-shot",
            "total_images": len(predictions),
            "few_shot_examples": few_shot_info,
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "notes": [
                "Few-shot with 2 staircase forms from train.jsonl",
                "Same schema-based prompt as the good zero-shot version",
                "NO key normalisation – GT keys are gold",
                "Greedy decode with sampling fallback",
                "Dual-view images: original + autocontrast",
            ],
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_file}")

        return output_dir, avg_cer

    return output_dir, None


# --------------------------
# Main
# --------------------------
def main():
    ensure_dir(OUT_DIR)
    print("=" * 60)
    print("GEMMA-3 FEW-SHOT OCR (Staircase Forms, dual-view, schema prompt)")
    print("=" * 60)

    print("Loading train data...")
    train_data = load_jsonl(TRAIN_JSONL)
    print(f"Loaded {len(train_data)} train samples")

    print("Loading test data...")
    test_data = load_jsonl(TEST_JSONL)
    print(f"Loaded {len(test_data)} test samples")

    # Select fixed few-shot examples
    few_shot_examples = select_fixed_few_shot_examples(train_data)
    print("\nUsing the following few-shot examples:")
    few_shot_info = []
    for ex in few_shot_examples:
        img_name = ex.get("image_name", "NO_NAME")
        stair_type = ex.get("stair_type", "UNKNOWN")
        img_path = find_image_path(IMAGES_DIR, img_name)
        print(f"  - {img_name}  (stair_type: {stair_type}, path: {img_path})")
        few_shot_info.append({"image_name": img_name, "stair_type": stair_type})

    runner = Gemma3StairFewShot(MODEL_PATH)

    print("\nBuilding few-shot conversation messages (dual-view)...")
    few_shot_messages = create_few_shot_messages(few_shot_examples, IMAGES_DIR)
    print(f"Built few-shot conversation with {len(few_shot_messages)} messages "
          f"({len(few_shot_examples)} examples, dual-view)")

    predictions = []
    cer_scores = []

    print(f"\nProcessing {len(test_data)} test images (few-shot)...")
    for i, row in enumerate(test_data):
        img_name = row.get("image_name", "")
        print(f"\n[{i+1}/{len(test_data)}] {img_name}")

        img_path = find_image_path(IMAGES_DIR, img_name)
        print(f"    Resolved path: {img_path}")
        if not img_path or not os.path.exists(img_path):
            print("    WARNING: image not found; CER=1.0")
            gt_json = {k: v for k, v in row.items() if k != "image_name"}

            predictions.append(
                {
                    "image_name": img_name,
                    "matched_image_path": img_path or "NOT_FOUND",
                    "predicted_json": {},
                    "ground_truth": gt_json,
                    "raw_response": "Error: Image not found",
                    "cer_score": 1.0,
                }
            )
            cer_scores.append(1.0)
            continue

        try:
            test_img = Image.open(img_path).convert("RGB")
            test_img2 = ImageOps.autocontrast(test_img, cutoff=1)

            # Build full message list: few-shot + final query (dual-view)
            messages = list(few_shot_messages)  # shallow copy is enough
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_img},
                        {"type": "image", "image": test_img2},
                        {"type": "text", "text": staircase_schema_prompt()},
                    ],
                }
            )

            raw, used_sampling = runner.generate(messages, max_new_tokens=1400)
            print(f"    [DEBUG] used_sampling_fallback: {used_sampling}")
            print(f"    [DEBUG] raw output length: {len(raw)}")
            preview = (raw[:200] or "").replace("\n", " ")
            print(f"    [DEBUG] raw preview: {preview!r}")

            pred_json_raw = extract_json_from_response(raw)
            pred_json = pred_json_raw if pred_json_raw else {}

            gt_json = {k: v for k, v in row.items() if k != "image_name"}

            gt_s = json_to_string(gt_json)
            pr_s = json_to_string(pred_json) if pred_json else ""
            cer = jiwer.cer(gt_s, pr_s) if pr_s else 1.0

            print(f"    CER: {cer:.4f} ({cer*100:.2f}%)")

            predictions.append(
                {
                    "image_name": img_name,
                    "matched_image_path": img_path,
                    "predicted_json": pred_json,
                    "ground_truth": gt_json,
                    "raw_response": raw,
                    "cer_score": cer,
                }
            )
            cer_scores.append(cer)

        except Exception as e:
            print(f"    ERROR: {e}")
            gt_json = {k: v for k, v in row.items() if k != "image_name"}

            predictions.append(
                {
                    "image_name": img_name,
                    "matched_image_path": img_path,
                    "predicted_json": {},
                    "ground_truth": gt_json,
                    "raw_response": f"Error: {e}",
                    "cer_score": 1.0,
                }
            )
            cer_scores.append(1.0)

        # Clean per step
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    out_dir, avg_cer = save_predictions_with_timestamp(
        predictions,
        test_data,
        cer_scores,
        few_shot_info,
        base_dir=OUT_DIR,
    )

    print("\n" + "=" * 60)
    print("GEMMA-3 FEW-SHOT STAIRCASE OCR DONE")
    print("=" * 60)
    print(f"Saved to: {out_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
