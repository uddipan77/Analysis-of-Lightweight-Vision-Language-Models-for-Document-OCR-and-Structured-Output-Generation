#!/usr/bin/env python3
# staircase_qwen_fewshot.py
# Improved Few-shot OCR on Staircase dataset with Qwen2.5-VL-7B
# - Uses the same JSON-schema prompt as zero-shot schema version
# - Uses 2 specific few-shot examples from train.jsonl:
#     "FMIS_FormblätterMielke_gefüllt (8).jpg"
#     "FMIS_FormblätterMielke_gefüllt (149).jpg"
# - Evaluates on test.jsonl
# - No key normalization: CER is computed on raw model prediction vs raw ground truth
# - Uses dual-view images (original + autocontrast) and repetition_penalty in decoding

import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import re
import gc
import unicodedata
import warnings

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageOps
import jiwer


# --------------------------
# JSON utilities
# --------------------------
def json_to_string(json_obj: Dict) -> str:
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


# --------------------------
# Robust image finding
# --------------------------
def strip_accents(s: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def norm_filename_for_match(p: str) -> str:
    p = strip_accents(p)
    p = p.replace("\\", "/").lower()
    return p


def iter_all_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def find_image_path(images_root: str, image_name_from_json: str) -> Optional[str]:
    if not image_name_from_json:
        return None

    # 1) Direct
    direct = os.path.join(images_root, image_name_from_json)
    if os.path.exists(direct):
        return direct

    target_norm = norm_filename_for_match(image_name_from_json)

    # 2) Exact basename match
    for path in iter_all_files(images_root):
        if norm_filename_for_match(os.path.basename(path)) == target_norm:
            return path

    # 3) Suffix "(n).ext"
    m = re.search(r"\(\s*\d+\s*\)\.(jpg|jpeg|png|tif|tiff)$", image_name_from_json, flags=re.IGNORECASE)
    if m:
        suffix = image_name_from_json[image_name_from_json.rfind("("):]
        suffix_norm = norm_filename_for_match(suffix)
        for path in iter_all_files(images_root):
            base = norm_filename_for_match(os.path.basename(path))
            if base.endswith(suffix_norm):
                return path

    # 4) Substring fallback
    for path in iter_all_files(images_root):
        if target_norm in norm_filename_for_match(path):
            return path

    return None


# --------------------------
# Image resizing (optional, to avoid OOM)
# --------------------------
def resize_image_if_needed(image_path: str, max_size: int = 1024) -> str:
    """
    Resize image if too large to avoid OOM. Returns path to resized image (or original).
    """
    try:
        image = Image.open(image_path)
        width, height = image.size

        if width <= max_size and height <= max_size:
            image.close()
            return image_path

        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)

        print(f"  Resizing {os.path.basename(image_path)} from {width}x{height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Save next to original with suffix
        root, ext = os.path.splitext(image_path)
        temp_path = root + "_resized" + ext
        image.save(temp_path, "JPEG", quality=95)
        image.close()
        return temp_path
    except Exception as e:
        print(f"  WARNING: resize failed for {image_path}: {e}")
        return image_path


# --------------------------
# Autocontrast helper (dual-view)
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
    except Exception as e:
        print(f"  WARNING: autocontrast failed for {image_path}: {e}")
        return None


# --------------------------
# JSON-schema prompt (aligned with zero-shot; fixed TREPPENTYP key)
# --------------------------
def generic_schema_prompt() -> str:
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
# Model wrapper (few-shot)
# --------------------------
class StaircaseOCRProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        print("Loading Qwen2.5-VL model for few-shot Staircase OCR...")

        use_bf16 = False
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            use_bf16 = major >= 8  # Ampere and newer
        dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Slightly higher resolution as before
        min_pixels = 256 * 28 * 28
        max_pixels = 768 * 28 * 28

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()

        print(f"Model loaded. dtype={dtype}, min_pixels={min_pixels}, max_pixels={max_pixels}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
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

        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
            )
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)]
            out_text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Fallback to mild sampling if output is empty
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
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, gen)]
                out_text = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

        # clean up
        del inputs, gen, trimmed, image_inputs, video_inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        return out_text

    def create_few_shot_messages(self, few_shot_examples: List[Dict], images_dir: str) -> List[Dict]:
        """Create few-shot conversation messages from chosen examples."""
        messages = []

        for i, example in enumerate(few_shot_examples):
            image_name = example.get("image_name", "")
            image_path = find_image_path(images_dir, image_name)

            print(f"  Few-shot example {i+1}: {image_name}")
            print(f"    Resolved path: {image_path}")
            print(f"    Exists: {os.path.exists(image_path) if image_path else False}")

            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Few-shot image not found: {image_name}")

            # optional resize
            processed_path = resize_image_if_needed(image_path, max_size=1024)
            ac_path = make_autocontrast_copy(processed_path)

            user_content = [{"type": "image", "image": processed_path}]
            if ac_path:
                user_content.append({"type": "image", "image": ac_path})
            user_content.append({"type": "text", "text": self.schema_prompt})

            # user message
            messages.append({
                "role": "user",
                "content": user_content,
            })

            # assistant message with expected JSON (raw keys, no image_name)
            gt = {k: v for k, v in example.items() if k != "image_name"}
            gt_s = json_to_string(gt)

            messages.append({
                "role": "assistant",
                "content": gt_s,
            })

        return messages

    def process_image_with_few_shot(self, image_path: str, few_shot_messages: List[Dict]) -> str:
        """Run model on a single image given fixed few-shot messages."""
        # clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        processed_image_path = resize_image_if_needed(image_path, max_size=1024)
        ac_path = None

        try:
            ac_path = make_autocontrast_copy(processed_image_path)

            messages = list(few_shot_messages)

            user_content = [{"type": "image", "image": processed_image_path}]
            if ac_path:
                user_content.append({"type": "image", "image": ac_path})
            user_content.append({"type": "text", "text": self.schema_prompt})

            # append test image + schema prompt
            messages.append({
                "role": "user",
                "content": user_content,
            })

            result = self._generate(messages, max_new_tokens=1536)

        finally:
            # clean up temp files
            if processed_image_path != image_path:
                try:
                    os.remove(processed_image_path)
                except Exception:
                    pass
            if ac_path and os.path.exists(ac_path):
                try:
                    os.remove(ac_path)
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        return result


# --------------------------
# Data loading / JSON extraction
# --------------------------
def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_json_from_response(response: str) -> Dict:
    response = response.strip()
    # Strip markdown code fences if present
    if response.startswith("```"):
        parts = response.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.lstrip().lower().startswith("json"):
                body = body[4:].strip()
            response = body.strip()

    matches = re.findall(r"\{.*\}", response, re.DOTALL)
    if matches:
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


# --------------------------
# Few-shot selection
# --------------------------
def select_fixed_few_shot_examples(train_data: List[Dict], images_dir: str) -> List[Dict]:
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

    few_shot_examples = []

    for target in target_images:
        found = False
        for item in train_data:
            img_name = item.get("image_name", "")
            if norm_filename_for_match(img_name) == norm_filename_for_match(target):
                img_path = find_image_path(images_dir, img_name)
                if img_path and os.path.exists(img_path):
                    few_shot_examples.append(item)
                    print(f"  ✓ Found: {img_name}")
                    print(f"    Path: {img_path}")
                    found = True
                    break
        if not found:
            print(f"  ✗ NOT FOUND in train.jsonl or images: {target}")

    if len(few_shot_examples) < 2:
        raise RuntimeError("Could not find both requested few-shot images (8) and (149).")

    return few_shot_examples


# --------------------------
# Saving & CER summary
# --------------------------
def save_predictions_with_timestamp(predictions, test_data, all_cer_scores, base_dir, few_shot_info):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # predictions.jsonl
    pred_file = os.path.join(output_dir, "predictions.jsonl")
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

        for i, test_item in enumerate(test_data[: len(predictions)]):
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)
            char_count = len(gt_json_string)
            cer_score = all_cer_scores[i]

            total_chars += char_count
            total_errors += int(cer_score * char_count)

        weighted_cer = total_errors / total_chars if total_chars > 0 else 0.0
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)

        print("\n" + "=" * 60)
        print("CER EVALUATION RESULTS (Improved Few-shot Qwen2.5-VL)")
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

        summary_file = os.path.join(output_dir, "cer_evaluation_summary.json")
        summary_data = {
            "timestamp": timestamp,
            "total_images": len(predictions),
            "few_shot_count": len(few_shot_info),
            "few_shot_examples": few_shot_info,
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "notes": [
                "Few-shot Qwen2.5-VL-7B-Instruct (improved)",
                "Schema-based prompt (same as zero-shot schema setup, fixed TREPPENTYP key)",
                "Few-shot images: (8) and (149) from train.jsonl",
                "Dual-view images: original + autocontrast (where available)",
                "Resolution: max_pixels=768*28*28 + additional resize to <=1024px",
                "Generation: greedy with repetition_penalty + sampling fallback",
                "No key normalization: CER on raw GT vs raw prediction",
            ],
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_file}")

        return output_dir, avg_cer

    return output_dir, None


# --------------------------
# Main
# --------------------------
def main():
    # Paths
    TRAIN_JSONL = "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl"
    TEST_JSONL = "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl"
    IMAGES_DIR = "/home/woody/iwi5/iwi5298h/staircase_images"
    OUT_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/qwen/few_shot_stair"

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print("Loading test data...")
    test_data = load_jsonl(TEST_JSONL)
    print(f"Loaded {len(test_data)} test samples from {TEST_JSONL}")

    print("\nLoading training data...")
    train_data = load_jsonl(TRAIN_JSONL)
    print(f"Loaded {len(train_data)} training samples from {TRAIN_JSONL}")

    # Select fixed few-shot examples
    few_shot_examples = select_fixed_few_shot_examples(train_data, IMAGES_DIR)

    print("\n" + "=" * 60)
    print(f"Using {len(few_shot_examples)} fixed few-shot examples:")
    few_shot_info = []
    for i, ex in enumerate(few_shot_examples):
        img_name = ex.get("image_name", "NO_NAME")
        img_path = find_image_path(IMAGES_DIR, img_name)
        print(f"  {i+1}. {img_name}")
        print(f"     Path: {img_path}")
        few_shot_info.append({"image_name": img_name})
    print("=" * 60)

    # Init model
    ocr_processor = StaircaseOCRProcessor()

    # Build few-shot messages
    print("\nCreating few-shot messages...")
    few_shot_messages = ocr_processor.create_few_shot_messages(few_shot_examples, IMAGES_DIR)
    print(f"Created few-shot conversation with {len(few_shot_messages)} messages")
    print(f"Total images per inference: {len(few_shot_examples)} few-shot + 1 test (each dual-view where available)")

    # Inference on test set
    predictions = []
    all_cer_scores = []

    print(f"\nProcessing {len(test_data)} test images (few-shot)...")

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
            response = ocr_processor.process_image_with_few_shot(image_path, few_shot_messages)
            predicted_json_raw = extract_json_from_response(response)
            predicted_json = predicted_json_raw if isinstance(predicted_json_raw, dict) else {}

            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}

            gt_json_string = json_to_string(gt_json)
            pred_json_string = json_to_string(predicted_json)

            cer_score = jiwer.cer(gt_json_string, pred_json_string)

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

            print(f"  CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
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
            continue

    output_dir, avg_cer = save_predictions_with_timestamp(
        predictions, test_data, all_cer_scores, OUT_DIR, few_shot_info
    )

    print("\n" + "=" * 60)
    print("Improved few-shot Staircase processing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER on test set: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
