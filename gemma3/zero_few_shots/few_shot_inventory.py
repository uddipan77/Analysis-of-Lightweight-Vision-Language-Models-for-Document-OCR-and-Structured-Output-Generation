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
TRAIN_JSONL_PATH = "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl"
IMAGES_DIR = "/home/woody/iwi5/iwi5298h/inventory_images"

BASE_OUTPUT_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/few_zero_inven"

FEW_SHOT_IMAGE = "inventarbuch-156.jpg"

FEWSHOT_PROMPT = """Extrahiere den Inventar-Eintrag aus dem Bild und gib NUR ein JSON zurück.
Nutze GENAU diese Struktur und Keys:

{
  "Überschrift": "",
  "Inventarnummer": "",
  "Maße": { "L": "", "B": "", "D": "" },
  "Objektbezeichnung": "",
  "Fundort": "",
  "Fundzeit": "",
  "Beschreibungstext": ""
}

Regeln:
- Text exakt wie im Bild (Originalschreibung beibehalten)
- Fehlende Werte als "" (Keys niemals entfernen)
- "Maße" MUSS immer ein Objekt mit L, B, D sein
- Antworte NUR mit dem JSON (keine Erklärungen)
"""


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
    """Ensure prediction matches the expected schema, never dropping keys."""
    if not isinstance(pred, dict):
        return empty_prediction_schema()

    out = empty_prediction_schema()

    for k in ["Überschrift", "Inventarnummer", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]:
        if k in pred:
            v = pred.get(k)
            if v is None:
                out[k] = ""
            elif isinstance(v, (str, int, float)):
                out[k] = str(v)

    mae = pred.get("Maße", {})
    if isinstance(mae, dict):
        for dim in ["L", "B", "D"]:
            v = mae.get(dim, "")
            if v is None:
                out["Maße"][dim] = ""
            elif isinstance(v, (str, int, float)):
                out["Maße"][dim] = str(v)

    return out


def extract_json_from_response(response: str) -> Dict:
    """Extract JSON object from model response, return coerced schema."""
    response = (response or "").strip()

    # Try to find JSON in the response
    json_pattern = r"\{.*\}"
    matches = re.findall(json_pattern, response, re.DOTALL)

    for m in matches:
        try:
            return coerce_to_schema(json.loads(m))
        except json.JSONDecodeError:
            continue

    try:
        return coerce_to_schema(json.loads(response))
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

    def create_few_shot_messages(self, few_shot_examples: List[Dict], images_dir: str) -> List[Dict]:
        """Create few-shot messages with images and expected JSON outputs."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a careful OCR system that outputs only valid JSON."}],
            }
        ]

        for example in few_shot_examples:
            img_path = os.path.join(images_dir, example["image_name"])
            image = Image.open(img_path).convert("RGB")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": FEWSHOT_PROMPT},
                    ],
                }
            )

            expected_json = {k: v for k, v in example.items() if k != "image_name"}
            expected_json = coerce_to_schema(expected_json)

            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": json.dumps(expected_json, ensure_ascii=False, separators=(",", ":"))}],
                }
            )

        return messages

    def process_image_with_few_shot(self, image_path: str, few_shot_messages: List[Dict]) -> str:
        """Process a single image with few-shot examples."""
        image = Image.open(image_path).convert("RGB")

        messages = list(few_shot_messages)
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": FEWSHOT_PROMPT},
                ],
            }
        )

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
    output_dir = os.path.join(base_dir, f"run_fewshot_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving results to: {output_dir}")

    predictions_file = os.path.join(output_dir, "predictions_fewshot.jsonl")
    with open(predictions_file, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Predictions saved to: {predictions_file}")

    if all_cer_scores:
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)

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
        print("GEMMA 3 FEW-SHOT CER EVALUATION RESULTS")
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

        summary_file = os.path.join(output_dir, "evaluation_summary_fewshot.json")
        summary_data = {
            "timestamp": timestamp,
            "model": "google/gemma-3-4b-it (local snapshot)",
            "model_path": MODEL_PATH,
            "method": "few-shot",
            "total_images": len(predictions),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "few_shot_examples": [FEW_SHOT_IMAGE],
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
    print("GEMMA 3 FEW-SHOT INVENTORY OCR PROCESSING (UPDATED)")
    print("=" * 60)

    print("\nLoading test data...")
    test_data = load_jsonl(TEST_JSONL_PATH)
    print(f"Loaded {len(test_data)} test samples")

    print("Loading train data for few-shot example...")
    train_data = load_jsonl(TRAIN_JSONL_PATH)

    few_shot_examples = [ex for ex in train_data if ex.get("image_name") == FEW_SHOT_IMAGE]
    if not few_shot_examples:
        raise FileNotFoundError(
            f"Few-shot example '{FEW_SHOT_IMAGE}' not found in train.jsonl at: {TRAIN_JSONL_PATH}"
        )

    print(f"Using {len(few_shot_examples)} few-shot example(s): {[ex['image_name'] for ex in few_shot_examples]}")

    ocr_processor = InventoryOCRProcessorGemma(MODEL_PATH)

    print("Creating few-shot messages with images...")
    few_shot_messages = ocr_processor.create_few_shot_messages(few_shot_examples, IMAGES_DIR)
    print(f"Created few-shot conversation with {len(few_shot_messages)} messages")

    predictions = []
    all_cer_scores = []

    print(f"\nProcessing {len(test_data)} test images with few-shot prompting...")
    print("-" * 60)

    for i, test_item in enumerate(test_data):
        img_name = test_item.get("image_name", "")
        print(f"Processing image {i+1}/{len(test_data)}: {img_name}")

        image_path = os.path.join(IMAGES_DIR, img_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            response = ocr_processor.process_image_with_few_shot(image_path, few_shot_messages)
            predicted_json = extract_json_from_response(response)

            # CER (exclude image_name)
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)
            pred_json_string = json_to_string(predicted_json)

            cer_score = jiwer.cer(gt_json_string, pred_json_string)

            predictions.append(
                {
                    "image_name": img_name,
                    "predicted_json": predicted_json,
                    "ground_truth": gt_json,
                    "raw_response": response,
                    "cer_score": cer_score,
                }
            )
            all_cer_scores.append(cer_score)

            print(f"  Processed successfully. CER: {cer_score:.4f}")

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            predictions.append(
                {
                    "image_name": img_name,
                    "predicted_json": empty_prediction_schema(),
                    "ground_truth": {k: v for k, v in test_item.items() if k != "image_name"},
                    "raw_response": f"Error: {str(e)}",
                    "cer_score": 1.0,
                }
            )
            all_cer_scores.append(1.0)

    output_dir, avg_cer = save_predictions_with_timestamp(predictions, test_data, all_cer_scores)

    print(f"\n{'='*60}")
    print("GEMMA 3 FEW-SHOT PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print("Model: google/gemma-3-4b-it (local snapshot)")
    print("Method: Few-Shot Learning (1 training example)")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER on test set: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
