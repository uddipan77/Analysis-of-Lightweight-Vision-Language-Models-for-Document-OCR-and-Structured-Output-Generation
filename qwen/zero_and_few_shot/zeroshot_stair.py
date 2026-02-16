#!/usr/bin/env python3
# staircase_qwen_zeroshot_simple.py
# Zero-shot OCR on the Staircase dataset with Qwen2.5-VL
# - Single-stage, single generic JSON schema for all layouts
# - Prompt in English, JSON keys in German (matching labels)
# - Dual-view image (original + autocontrast)
# - CER computed against ground-truth JSON
# - Results + summary saved in timestamped run folder
# - No key normalization: CER uses raw model prediction vs raw ground truth
# - TREPPENTYP key fixed to "TREPPENTYP (Laufform)"

import os
import json
import re
import gc
import warnings
from datetime import datetime
from typing import List, Dict, Optional

import torch
from PIL import Image, ImageOps
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import jiwer
import unicodedata

# --------------------------
# Paths (EDIT HERE if needed)
# --------------------------
TEST_JSONL = "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl"
IMAGES_DIR = "/home/woody/iwi5/iwi5298h/staircase_images"
OUT_DIR    = "/home/vault/iwi5/iwi5298h/models_image_text/qwen/zero_shot_stair"
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# --------------------------
# Basic utilities
# --------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def json_to_string(json_obj: Dict) -> str:
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(',', ':'))

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
        ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)
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
# Generic English instruction + JSON schema
# (TREPPENTYP key fixed to "TREPPENTYP (Laufform)")
# --------------------------
def generic_schema_prompt() -> str:
    """
    English prompt, but JSON keys remain German and match (roughly) your label structure.
    One generic schema that works for most staircase forms.
    """
    return (
        "You are an OCR model for historical German staircase survey forms.\n\n"
        "TASK:\n"
        "Given ONE filled-in staircase form image, read all printed text, all handwritten text, "
        "and all checked/unchecked boxes. Then output exactly ONE JSON object that represents "
        "the complete form.\n\n"
        "RULES:\n"
        "- Use exactly the JSON keys shown in the schema below (including capitalization, spaces and umlauts).\n"
        "- Do NOT add extra keys and do NOT omit any keys.\n"
        "- If a field is empty or not filled on the form, still include it with an empty string \"\" "
        "or false for an unchecked checkbox.\n"
        "- For checkboxes: true = checked, false = unchecked.\n"
        "- For numeric values (measurements, years, counts, etc.), return them as strings (e.g. \"4,7\", \"1738\").\n"
        "- Return ONLY the JSON object, starting with '{' and ending with '}'. No explanations, no natural language.\n\n"
        "JSON SCHEMA (structure and keys):\n"
        "{\n"
        "  \"stair_type\": \"\",\n"
        "  \"Name des Hauses\": \"\",\n"
        "  \"Adresse\": \"\",\n"
        "  \"Bauherr\": \"\",\n"
        "  \"Baumeister\": \"\",\n"
        "  \"Bauzeit\": \"\",\n"
        "  \"LAGE am/im Hause\": {\n"
        "    \"im\": false,\n"
        "    \"am\": false\n"
        "  },\n"
        "  \"Gesamt Durchmesser cm\": \"\",\n"
        "  \"MATERIAL\": \"\",\n"
        "  \"TREPPENTYP (Laufform)\": \"\",\n"
        "  \"LÄUFE\": {\n"
        "    \"EG\": {\n"
        "      \"Stufen\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Steig.\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Breite\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Plateaus\": [\"\", \"\", \"\"]\n"
        "    },\n"
        "    \"1.OG\": {\n"
        "      \"Stufen\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Steig.\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Breite\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Plateaus\": [\"\", \"\", \"\"]\n"
        "    },\n"
        "    \"2.OG\": {\n"
        "      \"Stufen\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Steig.\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Breite\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Plateaus\": [\"\", \"\", \"\"]\n"
        "    },\n"
        "    \"3.OG\": {\n"
        "      \"Stufen\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Steig.\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Breite\": [\"\", \"\", \"\", \"\"],\n"
        "      \"Plateaus\": [\"\", \"\", \"\"]\n"
        "    }\n"
        "  },\n"
        "  \"STUFENPROFIL\": \"\",\n"
        "  \"GEHLINIE\": \"\",\n"
        "  \"Schweifung\": \"\",\n"
        "  \"Untertritt\": \"\",\n"
        "  \"WANGE\": false,\n"
        "  \"HOLM\": false,\n"
        "  \"eingespannte Stufen\": false,\n"
        "  \"aufgesattelte Stufen\": false,\n"
        "  \"GELÄNDER\": {\n"
        "    \"Stabwerk\":  { \"Holz\": false, \"Stein\": false, \"Eisen\": false },\n"
        "    \"Traljen\":   { \"Holz\": false, \"Stein\": false, \"Eisen\": false },\n"
        "    \"Docken\":    { \"Holz\": false, \"Stein\": false, \"Eisen\": false },\n"
        "    \"vollwandig\":{ \"Holz\": false, \"Stein\": false, \"Eisen\": false },\n"
        "    \"Ornament\":  { \"Holz\": false, \"Stein\": false, \"Eisen\": false },\n"
        "    \"Baluster\":  { \"Holz\": false, \"Stein\": false, \"Eisen\": false },\n"
        "    \"Bal.-Brett\":{ \"Holz\": false, \"Stein\": false, \"Eisen\": false },\n"
        "    \"Paneel\":    { \"Holz\": false, \"Stein\": false, \"Eisen\": false }\n"
        "  },\n"
        "  \"zur Lauflinie\": {\n"
        "    \"parallel\": false,\n"
        "    \"gekurvt\": false\n"
        "  },\n"
        "  \"Höhe\": \"\",\n"
        "  \"ANFÄNGER\": \"\",\n"
        "  \"DEKORATION\": \"\",\n"
        "  \"HANDLAUFPROFIL\": {\n"
        "    \"Hohe\": \"\",\n"
        "    \"Breite\": \"\"\n"
        "  },\n"
        "  \"Notes\": \"\",\n"
        "  \"Datum\": \"\",\n"
        "  \"ORT\": \"\",\n"
        "  \"Approved by\": \"\"\n"
        "}\n"
    )

# --------------------------
# Dual-view helper (original + autocontrast)
# --------------------------
def make_autocontrast_copy(image_path: str) -> Optional[str]:
    try:
        img = Image.open(image_path).convert("RGB")
        hi = ImageOps.autocontrast(img)
        tmp_path = os.path.join("/tmp", f"autocontrast_{os.path.basename(image_path)}")
        hi.save(tmp_path)
        img.close()
        hi.close()
        return tmp_path
    except Exception:
        return None

# --------------------------
# Qwen runner (single-stage)
# --------------------------
class QwenZeroShotStairSimple:
    def __init__(self, model_name: str = MODEL_NAME):
        print("Loading Qwen2.5-VL model for zero-shot Staircase OCR...")
        use_bf16 = False
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            use_bf16 = major >= 8  # Ampere and newer
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Slightly higher resolution for detailed forms
        min_pixels = 256 * 28 * 28
        max_pixels = 768 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            use_fast=True,
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        ).eval()

        print(f"Model loaded. dtype={dtype}, min_pixels={min_pixels}, max_pixels={max_pixels}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved  = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")

        self.schema_prompt = generic_schema_prompt()

    def _generate(self, messages: List[Dict], max_new_tokens: int = 1536) -> str:
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)

        with torch.no_grad():
            # Greedy first with repetition_penalty
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
            )
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
            out_text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # If the model returns nothing, fall back to mild sampling
            if not out_text:
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    use_cache=True,
                )
                trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
                out_text = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

        del inputs, gen, trimmed, image_inputs, video_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return out_text

    def extract_json(self, image_path: str, max_new_tokens: int = 1536) -> str:
        """Single-stage: image(s) + generic schema prompt."""
        ac_path = make_autocontrast_copy(image_path)

        contents = [{"type": "image", "image": image_path}]
        if ac_path:
            contents.append({"type": "image", "image": ac_path})
        contents.append({"type": "text", "text": self.schema_prompt})

        messages = [{"role": "user", "content": contents}]
        return self._generate(messages, max_new_tokens=max_new_tokens)

# --------------------------
# Save + CER summary
# --------------------------
def save_predictions_with_timestamp(predictions, test_data, all_cer_scores, base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    predictions_file = os.path.join(output_dir, "predictions.jsonl")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Predictions saved to: {predictions_file}")

    if all_cer_scores:
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)

        total_chars = 0
        total_errors = 0
        for i, test_item in enumerate(test_data[:len(predictions)]):
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)
            char_count = len(gt_json_string)
            cer_score = all_cer_scores[i]
            total_chars += char_count
            total_errors += int(round(cer_score * char_count))

        weighted_cer = total_errors / total_chars if total_chars > 0 else 0.0
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)

        print("\n" + "="*60)
        print("CER EVALUATION RESULTS (Zero-shot, Qwen2.5-VL, single generic schema)")
        print("="*60)
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

        summary_file = os.path.join(output_dir, "cer_evaluation_summary.json")
        summary_data = {
            "timestamp": timestamp,
            "total_images": len(predictions),
            "mode": "zero-shot single-schema (no normalization, fixed TREPPENTYP key)",
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "notes": [
                "Single-stage zero-shot with Qwen2.5-VL-7B-Instruct",
                "Generic English prompt, fixed German JSON schema",
                "Dual-view image: original + autocontrast",
                "Resolution: max_pixels=768*28*28",
                "Generation: greedy with repetition_penalty + sampling fallback",
                "No key normalization: CER on raw GT vs raw prediction",
            ],
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_file}")

        return output_dir, avg_cer

    return output_dir, None

# --------------------------
# Main
# --------------------------
def main():
    ensure_dir(OUT_DIR)
    print("="*60)
    print("QWEN2.5-VL ZERO-SHOT OCR (Staircase Forms) — single generic schema (no normalization)")
    print("="*60)

    print("Loading test data...")
    test_data = load_jsonl(TEST_JSONL)
    print(f"Loaded {len(test_data)} samples from {TEST_JSONL}")

    ocr = QwenZeroShotStairSimple()

    predictions = []
    all_cer_scores = []

    print(f"\nProcessing {len(test_data)} images (zero-shot, single-schema)...")
    for i, test_item in enumerate(test_data):
        image_name = test_item.get("image_name", "")
        print(f"\n[{i+1}/{len(test_data)}] Processing: {image_name}")

        image_path = find_image_path(IMAGES_DIR, image_name)
        print(f"  Resolved path: {image_path}")

        if not image_path or not os.path.exists(image_path):
            print("  WARNING: Image not found")
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            prediction_entry = {
                "image_name": image_name,
                "matched_image_path": image_path or "NOT_FOUND",
                "predicted_json": {},
                "ground_truth": gt_json,
                "raw_response": "Error: Image not found",
                "cer_score": 1.0,
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)
            continue

        try:
            response = ocr.extract_json(image_path, max_new_tokens=1536)

            print(f"  raw output length: {len(response)}")
            preview = (response[:160] or "").replace("\n", " ")
            print(f"  raw preview: {preview}")

            predicted_json = extract_json_from_response(response)
            if not isinstance(predicted_json, dict):
                predicted_json = {}

            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)
            pred_json_string = json_to_string(predicted_json) if predicted_json else ""

            cer_score = jiwer.cer(gt_json_string, pred_json_string) if pred_json_string else 1.0
            print(f"  CER: {cer_score:.4f} ({cer_score*100:.2f}%)")

            prediction_entry = {
                "image_name": image_name,
                "matched_image_path": image_path,
                "predicted_json": predicted_json,
                "ground_truth": gt_json,
                "raw_response": response,
                "cer_score": cer_score,
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(cer_score)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                print(f"  GPU Memory: {allocated:.2f} GB")

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            prediction_entry = {
                "image_name": image_name,
                "matched_image_path": image_path,
                "predicted_json": {},
                "ground_truth": gt_json,
                "raw_response": f"Error: {str(e)}",
                "cer_score": 1.0,
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    output_dir, avg_cer = save_predictions_with_timestamp(
        predictions, test_data, all_cer_scores, OUT_DIR
    )

    print(f"\n{'='*60}")
    print("Processing complete! (Zero-shot, single generic schema, Qwen2.5-VL, no normalization)")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER on test set: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR:", e)
        import traceback
        traceback.print_exc()
        exit(1)
