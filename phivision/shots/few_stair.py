#!/usr/bin/env python3
# staircase_phi35_fewshot_simple.py
# Few-shot OCR with Phi-3.5-vision-instruct on staircase dataset
# - Simple prompt: "copy the structure and keys from the examples"
# - 2 diverse few-shot examples (INNENTREPPE + WENDELTREPPE/fallback)
# - No image resizing (Phi sees full-resolution scans)
# - Greedy decode with sampling fallback
# - Robust image matching + JSON extraction + CER logging

import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import AutoModelForCausalLM, AutoProcessor
from huggingface_hub import snapshot_download
from PIL import Image
import torch
import json
from typing import List, Dict, Optional
from datetime import datetime
import re
import jiwer
import gc
import unicodedata
import logging


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
LOCAL_MODEL_DIR = "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision"


# ---------------------------------------------------------------------
# Model download helper
# ---------------------------------------------------------------------
def download_model_if_needed(model_id: str = MODEL_ID, local_dir: str = LOCAL_MODEL_DIR) -> str:
    """
    Download Phi-3.5-vision model from HuggingFace if not already present.
    Returns the path to the model directory (local path or model_id).
    """
    required_files = ["config.json", "pytorch_model.bin.index.json", "tokenizer.json"]
    model_exists = os.path.exists(local_dir) and all(
        os.path.exists(os.path.join(local_dir, f)) for f in required_files
    )

    if model_exists:
        print(f"Model found locally at: {local_dir}")
        return local_dir

    print(f"Model not found locally. Downloading from HuggingFace...")
    print(f"  Source: {model_id}")
    print(f"  Destination: {local_dir}")

    try:
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # keep it lighter
        )
        print(f"✓ Model downloaded successfully to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("Falling back to remote model_id (no local snapshot).")
        return model_id


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------
def json_to_string(json_obj: Dict) -> str:
    """Consistent JSON string (no reordering of keys)."""
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


# ---------------- Robust image finding ----------------
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
    """Robust image path resolver with multiple fallback strategies."""
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

    # 3) Match by "(n).ext" suffix
    m = re.search(
        r"\(\s*\d+\s*\)\.(jpg|jpeg|png|tif|tiff)$",
        image_name_from_json,
        flags=re.IGNORECASE,
    )
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


# ---------------------------------------------------------------------
# Phi-3.5 OCR Processor (few-shot)
# ---------------------------------------------------------------------
class StaircaseOCRProcessor:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the OCR processor with Phi-3.5-vision model.
        If model_path is None, it downloads (or reuses) a local snapshot.
        """
        if model_path is None:
            print("Checking for Phi-3.5-vision-instruct model...")
            model_path = download_model_if_needed()

        print("Loading Phi-3.5-vision-instruct...")
        print(f"Model path: {model_path}")

        # Suppress some noisy config logs
        logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

        # Load model (no flash-attn, eager)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",
            local_files_only=(model_path == LOCAL_MODEL_DIR),
        )

        # Processor with multi-crop support (few-shot + test)
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_crops=4,
            local_files_only=(model_path == LOCAL_MODEL_DIR),
        )

        print("Model loaded successfully.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

        # 🔹 SIMPLE PROMPT: rely on examples to convey the schema
        self.instruction = (
            "Read the staircase form image and transcribe ALL information into a single JSON object.\n"
            "- Use exactly the same keys and nested structure as in the previous examples.\n"
            "- If a value is missing or unreadable, use empty string \"\" or false / [] as appropriate.\n"
            "- Return ONLY a JSON object, with no extra text or markdown."
        )

    # ---------------- Few-shot messages ----------------
    def create_few_shot_messages(
        self,
        few_shot_examples: List[Dict],
        images_dir: str,
    ) -> (List[Dict], List[Image.Image]):
        """
        Create few-shot messages in Phi-3.5-vision chat format.
        Returns: (messages_list, images_list)
        """
        messages: List[Dict] = []
        images: List[Image.Image] = []

        image_index = 1
        for i, example in enumerate(few_shot_examples):
            image_name = example.get("image_name", "")
            stair_type = example.get("stair_type", "UNKNOWN")
            image_path = find_image_path(images_dir, image_name)

            print(f"  Few-shot example {i+1}: {image_name}")
            print(f"    Type: {stair_type}")
            print(f"    Resolved path: {image_path}")
            print(f"    Exists: {os.path.exists(image_path) if image_path else False}")

            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Few-shot image not found: {image_name}")

            img = Image.open(image_path).convert("RGB")
            images.append(img)

            placeholder = f"<|image_{image_index}|>\n"
            # USER: show the image + simple instruction
            messages.append(
                {
                    "role": "user",
                    "content": placeholder + self.instruction,
                }
            )

            # ASSISTANT: ground-truth JSON (minus image_name)
            expected_json = {k: v for k, v in example.items() if k != "image_name"}
            json_str = json.dumps(expected_json, ensure_ascii=False, separators=(",", ":"))
            messages.append(
                {
                    "role": "assistant",
                    "content": json_str,
                }
            )

            image_index += 1

        return messages, images

    # ---------------- Inference with few-shot ----------------
    def process_image_with_few_shot(
        self,
        image_path: str,
        few_shot_messages: List[Dict],
        few_shot_images: List[Image.Image],
        max_new_tokens: int = 1536,
    ) -> str:
        """
        Process a single staircase image with Phi-3.5-vision using few-shot examples.
        No image resizing: Phi sees the full-resolution scan.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        test_image = Image.open(image_path).convert("RGB")
        all_images = few_shot_images + [test_image]
        test_idx = len(few_shot_images) + 1

        # USER message for test image – simple, minimal
        user_msg = {
            "role": "user",
            "content": f"<|image_{test_idx}|>\n{self.instruction}",
        }

        all_messages = list(few_shot_messages) + [user_msg]

        inputs = None
        generate_ids = None
        response = ""

        try:
            prompt = self.processor.tokenizer.apply_chat_template(
                all_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                prompt,
                all_images,
                return_tensors="pt",
            ).to("cuda")

            def _decode(gen_ids):
                new_ids = gen_ids[:, inputs["input_ids"].shape[1]:]
                return self.processor.batch_decode(
                    new_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

            with torch.inference_mode():
                # Stage 1: greedy decode
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                text = _decode(generate_ids)

                # Stage 2: sampling fallback if empty / no JSON
                if (not text) or ("{" not in text):
                    generate_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9,
                        repetition_penalty=1.05,
                        use_cache=True,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                    text = _decode(generate_ids)

            response = text

        except Exception as e:
            print(f"    Generation error: {e}")
            import traceback
            traceback.print_exc()
            response = f"Generation failed: {str(e)}"

        finally:
            if inputs is not None:
                del inputs
            if generate_ids is not None:
                del generate_ids
            del test_image, all_images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return response or "No response generated"


# ---------------------------------------------------------------------
# Data loading & JSON extraction
# ---------------------------------------------------------------------
def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_json_from_response(response: str) -> Dict:
    """Extract JSON object from a free-form Phi response."""
    response = response.strip()

    # Try to find largest JSON-looking block
    matches = re.findall(r"\{.*\}", response, flags=re.DOTALL)
    if matches:
        for m in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(m)
            except json.JSONDecodeError:
                continue

    # Fallback: try entire string
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------
# Few-shot example selection
# ---------------------------------------------------------------------
def select_diverse_few_shot_examples(train_data: List[Dict], images_dir: str) -> List[Dict]:
    """
    Select 2 diverse few-shot examples:
      - INNENTREPPE: fixed image (8)
      - WENDELTREPPE: first available from train.jsonl; if none, fallback
    """
    print("\nSelecting diverse few-shot examples...")

    # Log available WENDELTREPPE options
    print("\nAvailable WENDELTREPPE images in train.jsonl:")
    wendeltreppe_options = []
    for item in train_data:
        if item.get("stair_type") == "WENDELTREPPE":
            img_name = item.get("image_name", "")
            wendeltreppe_options.append(img_name)
            print(f"  - {img_name}")

    target_images = [
        ("FMIS_FormblätterMielke_gefüllt (8).jpg", "INNENTREPPE"),
    ]

    if wendeltreppe_options:
        target_images.append((wendeltreppe_options[0], "WENDELTREPPE"))
        print(f"\nUsing WENDELTREPPE: {wendeltreppe_options[0]}")
    else:
        print("\nWARNING: No WENDELTREPPE found! Using arbitrary fallback.")
        for item in train_data:
            img_name = item.get("image_name", "")
            if img_name != "FMIS_FormblätterMielke_gefüllt (8).jpg":
                target_images.append((img_name, "FALLBACK"))
                break

    few_shot_examples: List[Dict] = []

    for target_image, expected_type in target_images:
        found = False
        for item in train_data:
            img_name = item.get("image_name", "")
            if norm_filename_for_match(img_name) == norm_filename_for_match(target_image):
                img_path = find_image_path(images_dir, img_name)
                if img_path and os.path.exists(img_path):
                    stair_type = item.get("stair_type", "UNKNOWN")
                    few_shot_examples.append(item)
                    print(f"✓ Selected {expected_type}: {img_name}")
                    print(f"  Actual stair_type: {stair_type}")
                    print(f"  Path: {img_path}")
                    found = True
                    break
        if not found:
            print(f"✗ NOT FOUND in train.jsonl: {target_image}")

    return few_shot_examples


# ---------------------------------------------------------------------
# Saving + CER summary
# ---------------------------------------------------------------------
def save_predictions_with_timestamp(
    predictions,
    test_data,
    all_cer_scores,
    base_dir,
    few_shot_info,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_fewshot_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    # Predictions
    pred_path = os.path.join(output_dir, "predictions_fewshot.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Predictions saved to: {pred_path}")

    # CER stats
    if all_cer_scores:
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)

        total_chars = 0
        total_errors = 0
        for i, test_item in enumerate(test_data[: len(predictions)]):
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_s = json_to_string(gt_json)
            char_count = len(gt_s)
            cer_score = all_cer_scores[i]
            total_chars += char_count
            total_errors += int(cer_score * char_count)

        weighted_cer = total_errors / total_chars if total_chars > 0 else 0.0
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)

        print("\n" + "=" * 60)
        print("PHI-3.5-VISION FEW-SHOT CER EVALUATION RESULTS (simple prompt)")
        print("STAIRCASE DOCUMENTATION DATASET")
        print("=" * 60)
        print(f"\nCER across {len(all_cer_scores)} images:")
        print("-" * 50)
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Minimum CER: {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER: {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"\nWeighted CER: {weighted_cer:.4f} ({weighted_cer*100:.2f}%)")
        print(f"Total characters: {total_chars}")
        print(f"Total errors: {total_errors}")
        print(
            f"Perfect matches: {perfect_matches}/{len(all_cer_scores)} "
            f"({perfect_matches/len(all_cer_scores)*100:.1f}%)"
        )

        summary_file = os.path.join(output_dir, "evaluation_summary_fewshot_simple.json")
        summary_data = {
            "timestamp": timestamp,
            "model": "microsoft/Phi-3.5-vision-instruct",
            "dataset": "staircase_documentation",
            "method": "few-shot-simple-prompt",
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
            "optimizations": [
                "Simple instruction: copy structure & keys from examples",
                "No explicit schema in prompt (examples define format)",
                "No image resizing (full-resolution scans)",
                "Greedy decode with sampling fallback",
                "num_crops=4 in AutoProcessor",
            ],
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_file}")

        return output_dir, avg_cer

    return output_dir, None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    TRAIN_JSONL = "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl"
    TEST_JSONL = "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl"
    IMAGES_DIR = "/home/woody/iwi5/iwi5298h/staircase_images"
    OUT_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/phi/stair_shots"

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("PHI-3.5-VISION FEW-SHOT STAIRCASE OCR (simple prompt)")
    print("=" * 60)

    print("\nLoading test data...")
    test_data = load_jsonl(TEST_JSONL)
    print(f"Loaded {len(test_data)} test samples")

    print("\nLoading training data...")
    train_data = load_jsonl(TRAIN_JSONL)
    print(f"Loaded {len(train_data)} training samples")

    # Few-shot selection
    few_shot_examples = select_diverse_few_shot_examples(train_data, IMAGES_DIR)
    if len(few_shot_examples) < 2:
        print("\nERROR: Could not find 2 valid few-shot examples.")
        return

    print("\n" + "=" * 60)
    print(f"Using {len(few_shot_examples)} diverse few-shot examples:")
    few_shot_info = []
    for i, ex in enumerate(few_shot_examples):
        img_name = ex.get("image_name", "NO_NAME")
        stair_type = ex.get("stair_type", "UNKNOWN")
        img_path = find_image_path(IMAGES_DIR, img_name)
        print(f"  {i+1}. {img_name}")
        print(f"     Type: {stair_type}")
        print(f"     Path: {img_path}")
        few_shot_info.append({"image_name": img_name, "stair_type": stair_type})
    print("=" * 60)

    # Init model
    ocr_processor = StaircaseOCRProcessor()

    # Build few-shot messages
    print("\nCreating few-shot messages...")
    few_shot_messages, few_shot_images = ocr_processor.create_few_shot_messages(
        few_shot_examples,
        IMAGES_DIR,
    )
    print(f"Created {len(few_shot_messages)} few-shot messages")
    print(
        f"Total images per inference: {len(few_shot_images)} few-shot + 1 test = "
        f"{len(few_shot_images) + 1}"
    )

    # Inference loop
    predictions = []
    all_cer_scores = []

    print(f"\nProcessing {len(test_data)} test images...")
    print("-" * 60)

    for i, test_item in enumerate(test_data):
        image_name = test_item.get("image_name", "")
        print(f"\n[{i+1}/{len(test_data)}] {image_name}")

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
            response = ocr_processor.process_image_with_few_shot(
                image_path,
                few_shot_messages,
                few_shot_images,
                max_new_tokens=1536,
            )
            predicted_json = extract_json_from_response(response)

            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_s = json_to_string(gt_json)
            pr_s = json_to_string(predicted_json) if predicted_json else ""
            cer_score = jiwer.cer(gt_s, pr_s) if pr_s else 1.0

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
                print(f"  GPU Memory (allocated): {allocated:.2f} GB")

        except Exception as e:
            print(f"  ERROR: {e}")
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            prediction_entry = {
                "image_name": image_name,
                "matched_image_path": image_path,
                "predicted_json": {},
                "ground_truth": gt_json,
                "raw_response": f"Error: {e}",
                "cer_score": 1.0,
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Save + summary
    output_dir, avg_cer = save_predictions_with_timestamp(
        predictions,
        test_data,
        all_cer_scores,
        OUT_DIR,
        few_shot_info,
    )

    print("\n" + "=" * 60)
    print("FEW-SHOT PROCESSING COMPLETE! (simple prompt)")
    print("=" * 60)
    print("Model: microsoft/Phi-3.5-vision-instruct")
    print("Dataset: Staircase Documentation")
    print("Method: Few-shot (simple prompt, full-res images)")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER on test set: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nFATAL ERROR:", e)
        import traceback
        traceback.print_exc()
        exit(1)
