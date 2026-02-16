from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import re
import jiwer


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "/home/vault/iwi5/iwi5298h/models/models--google--gemma-3-4b-it/snapshots/093f9f388b31de276ce2de164bdc2081324b9767"

TEST_JSONL_PATH = "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl"
IMAGES_DIR = "/home/woody/iwi5/iwi5298h/inventory_images"

BASE_OUTPUT_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/few_zero_inven"

PROMPT = """You are an expert OCR system for German historical inventory documents. 
Carefully read ALL text in this image and extract the information into a JSON object.

Return ONLY a JSON object with exactly these keys:
{
  "Überschrift": "the heading or title at the top",
  "Inventarnummer": "the inventory number (often starts with letters like J=)",
  "Maße": {
    "L": "Länge/Length measurement",
    "B": "Breite/Breadth measurement", 
    "D": "Dicke/Depth measurement"
  },
  "Objektbezeichnung": "object type or material description",
  "Fundort": "find location or origin",
  "Fundzeit": "find date or time period",
  "Beschreibungstext": "the main descriptive text - transcribe completely"
}

IMPORTANT RULES:
1. Transcribe text EXACTLY as written in the image (preserve original spelling)
2. Use empty string "" for any field not visible in the image
3. The Maße field MUST be an object with L, B, D keys (use "" if not present)
4. Return ONLY the JSON object - no explanations or commentary
5. Read carefully - this is old German handwriting"""


# -----------------------------
# HELPERS
# -----------------------------
def json_to_string(json_obj: Any) -> str:
    """Convert JSON object to a consistent string representation for CER."""
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def empty_prediction_schema() -> Dict:
    """Return an empty schema matching the new dataset label format."""
    return {
        "Überschrift": "",
        "Inventarnummer": "",
        "Maße": {"L": "", "B": "", "D": ""},
        "Objektbezeichnung": "",
        "Fundort": "",
        "Fundzeit": "",
        "Beschreibungstext": "",
    }


def coerce_to_schema(pred: Any) -> Dict:
    """
    Ensure output is a dict with exactly the expected keys and Maße object.
    Extra keys are dropped; missing keys are filled with "".
    """
    if not isinstance(pred, dict):
        return empty_prediction_schema()

    out = empty_prediction_schema()

    # Copy known top-level keys (German)
    for k in ["Überschrift", "Inventarnummer", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]:
        if k in pred and isinstance(pred[k], (str, int, float)):
            out[k] = str(pred[k])
        elif k in pred and pred[k] is None:
            out[k] = ""
        elif k in pred and isinstance(pred[k], str):
            out[k] = pred[k]

    # Maße object
    mae = pred.get("Maße", {})
    if isinstance(mae, dict):
        for dim in ["L", "B", "D"]:
            val = mae.get(dim, "")
            if isinstance(val, (str, int, float)):
                out["Maße"][dim] = str(val)
            elif val is None:
                out["Maße"][dim] = ""

    return out


def extract_json_from_response(response: str) -> Dict:
    """
    Extract JSON object from model response robustly.
    Returns coerced schema even if partial/extra content is present.
    """
    response = (response or "").strip()

    # Prefer first balanced {...} region
    # (simple regex is OK here; we also try whole response parse)
    json_pattern = r"\{.*\}"
    matches = re.findall(json_pattern, response, re.DOTALL)

    # Try matches first
    for m in matches:
        try:
            parsed = json.loads(m)
            return coerce_to_schema(parsed)
        except json.JSONDecodeError:
            continue

    # Try entire response
    try:
        parsed = json.loads(response)
        return coerce_to_schema(parsed)
    except json.JSONDecodeError:
        return empty_prediction_schema()


# -----------------------------
# OCR PROCESSOR
# -----------------------------
class InventoryOCRProcessorGemma:
    def __init__(self, model_path: str = MODEL_PATH):
        print("Loading Gemma 3 4B IT model...")
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        print("Gemma 3 model loaded successfully!")

    def process_image_zero_shot(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a careful OCR system that outputs only valid JSON."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1,
            )
            generation = generation[0][input_len:]

        return self.processor.decode(generation, skip_special_tokens=True)


# -----------------------------
# OUTPUT SAVING
# -----------------------------
def save_predictions_with_timestamp(
    predictions: List[Dict],
    test_data: List[Dict],
    all_cer_scores: List[float],
    base_dir: str = BASE_OUTPUT_DIR,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_zeroshot_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    predictions_file = os.path.join(output_dir, "predictions_zeroshot.jsonl")
    with open(predictions_file, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Predictions saved to: {predictions_file}")

    if all_cer_scores:
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)

        # Weighted CER
        total_chars = 0
        total_errors = 0

        for i, test_item in enumerate(test_data[: len(predictions)]):
            # IMPORTANT: Exclude image_name from CER
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)

            char_count = len(gt_json_string)
            cer_score = all_cer_scores[i]

            total_chars += char_count
            total_errors += int(cer_score * char_count)

        weighted_cer = total_errors / total_chars if total_chars > 0 else 0.0
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)

        print("\n" + "=" * 60)
        print("GEMMA 3 ZERO-SHOT CER EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nCER Statistics across {len(all_cer_scores)} images:")
        print("-" * 50)
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Minimum CER: {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER: {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"\nWeighted CER: {weighted_cer:.4f} ({weighted_cer*100:.2f}%)")
        print(f"Total characters: {total_chars}")
        print(f"Total errors: {total_errors}")
        print(f"Perfect matches: {perfect_matches}/{len(all_cer_scores)} ({perfect_matches/len(all_cer_scores)*100:.1f}%)")

        summary_file = os.path.join(output_dir, "evaluation_summary_zeroshot.json")
        summary_data = {
            "timestamp": timestamp,
            "model": "google/gemma-3-4b-it (local snapshot)",
            "model_path": MODEL_PATH,
            "method": "zero-shot",
            "total_images": len(predictions),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
        }

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_file}")

        return output_dir, avg_cer

    return output_dir, None


# -----------------------------
# MAIN
# -----------------------------
def main():
    print("=" * 60)
    print("GEMMA 3 ZERO-SHOT INVENTORY OCR PROCESSING (UPDATED)")
    print("=" * 60)

    print("\nLoading test data...")
    test_data = load_jsonl(TEST_JSONL_PATH)
    print(f"Loaded {len(test_data)} test samples")
    print("\nNote: Using ZERO-SHOT learning (no training examples)")

    ocr_processor = InventoryOCRProcessorGemma(MODEL_PATH)

    predictions = []
    all_cer_scores = []

    print(f"\nProcessing {len(test_data)} test images with zero-shot prompting...")
    print("-" * 60)

    for i, test_item in enumerate(test_data):
        img_name = test_item.get("image_name", "")
        print(f"Processing image {i+1}/{len(test_data)}: {img_name}")

        image_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            response = ocr_processor.process_image_zero_shot(image_path)
            predicted_json = extract_json_from_response(response)

            # CER (exclude image_name from GT)
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)
            pred_json_string = json_to_string(predicted_json)

            cer_score = jiwer.cer(gt_json_string, pred_json_string)

            prediction_entry = {
                "image_name": img_name,
                "predicted_json": predicted_json,
                "ground_truth": gt_json,
                "raw_response": response,
                "cer_score": cer_score,
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(cer_score)

            print(f"  Processed successfully. CER: {cer_score:.4f}")

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            prediction_entry = {
                "image_name": img_name,
                "predicted_json": empty_prediction_schema(),
                "ground_truth": {k: v for k, v in test_item.items() if k != "image_name"},
                "raw_response": f"Error: {str(e)}",
                "cer_score": 1.0,
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)
            continue

    output_dir, avg_cer = save_predictions_with_timestamp(predictions, test_data, all_cer_scores)

    print(f"\n{'='*60}")
    print("GEMMA 3 ZERO-SHOT PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print("Model: google/gemma-3-4b-it (local snapshot)")
    print("Method: Zero-Shot Learning (no training examples)")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER on test set: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
