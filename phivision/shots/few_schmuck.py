#!/usr/bin/env python3
# phi_3_5_vision_schmuck_one_shot.py
# ✅ One-shot inference on Schmuck jewelry dataset
# ✅ Using SCH_3149.jpg as the example

import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import jiwer
import unicodedata
from typing import List, Dict
from datetime import datetime

# Paths
LOCAL_MODEL_PATH = "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision"
train_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl"
test_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
images_dir = "/home/woody/iwi5/iwi5298h/schmuck_images"

# Create timestamped output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/home/vault/iwi5/iwi5298h/models_image_text/phi/schmuck/few_shot_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "predictions.jsonl")
metrics_file = os.path.join(output_dir, "metrics.json")

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def create_label_string(json_data: Dict) -> str:
    """Ground truth without file_name."""
    label_data = {k: v for k, v in json_data.items() if k != "file_name"}
    return json.dumps(label_data, ensure_ascii=False, sort_keys=False)

def extract_json_from_response(response: str) -> str:
    """Try to extract JSON from response."""
    response = response.strip()
    
    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except:
            return json_str
    
    return response

# Prompt for jewelry catalog
BASE_PROMPT = """Extract all information from this German jewelry catalog image and return ONLY a JSON object.

Required keys:
- Gegenstand: Object type
- Inv.Nr: Inventory number
- Herkunft: Origin/provenance
- Foto Notes: Photo notes
- Standort: Location/storage
- Material: Material description
- Datierung: Dating/time period
- Maße: Measurements/dimensions
- Gewicht: Weight
- erworben von: Acquired from
- am: Acquisition date
- Preis: Price
- Vers.-Wert: Insurance value
- Beschreibung: Description
- Literatur: Literature references
- Ausstellungen: Exhibitions

IMPORTANT: If a value is not present or unclear, use empty string "".

Return ONLY the JSON object, no markdown, no extra text."""

print("=" * 80)
print("🚀 Phi-3.5-Vision One-Shot Inference on Schmuck Dataset")
print("=" * 80)

# Load training data
print(f"\n📂 Loading training data...")
train_data = load_jsonl(train_jsonl_path)
print(f"✅ Loaded {len(train_data)} training samples")

# ✅ Search for SCH_3149.jpg by file_name
target_file = "SCH_3149.jpg"
one_shot_example = None

for item in train_data:
    if item["file_name"] == target_file:
        one_shot_example = item
        print(f"  ✅ Found one-shot example: {target_file}")
        break

if one_shot_example is None:
    print(f"  ❌ Not found in train.jsonl: {target_file}")
    exit(1)

print(f"\n📂 Loading test data from: {test_jsonl_path}")
test_data = load_jsonl(test_jsonl_path)
print(f"✅ Loaded {len(test_data)} test samples")

print(f"\n⏳ Loading model from: {LOCAL_MODEL_PATH}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Device: {device}")

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    num_crops=4
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation='eager'
)

print("✅ Model and processor loaded")

# Load one-shot example image
print(f"\n📸 Loading one-shot example image...")
example_img_path = os.path.join(images_dir, one_shot_example["file_name"])
if not os.path.exists(example_img_path):
    print(f"  ❌ Missing: {example_img_path}")
    exit(1)

example_image = Image.open(example_img_path).convert("RGB")
print(f"  ✅ Loaded: {one_shot_example['file_name']}")

results = []
cer_scores = []
perfect_matches = 0
json_parsed = 0
json_like = 0

print(f"\n{'='*80}")
print("RUNNING ONE-SHOT INFERENCE")
print(f"{'='*80}\n")

for idx, item in enumerate(test_data):
    print(f"────────────────────────────────────────────────────────────")
    print(f"[{idx+1}/{len(test_data)}] {item['file_name']}")
    print(f"────────────────────────────────────────────────────────────")
    
    image_path = os.path.join(images_dir, item["file_name"])
    
    if not os.path.exists(image_path):
        print(f"  ❌ Image not found: {image_path}")
        results.append({
            "file_name": item["file_name"],
            "predicted_text": "",
            "ground_truth_text": create_label_string(item),
            "cer_score": 1.0,
            "error": "Image not found"
        })
        cer_scores.append(1.0)
        continue
    
    try:
        query_image = Image.open(image_path).convert("RGB")
        
        # One-shot conversation
        messages = [
            # Example
            {
                "role": "user",
                "content": f"<|image_1|>\n{BASE_PROMPT}"
            },
            {
                "role": "assistant",
                "content": create_label_string(one_shot_example)
            },
            # Query
            {
                "role": "user",
                "content": f"<|image_2|>\n{BASE_PROMPT}"
            }
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process 2 images (1 example + 1 query)
        all_images = [example_image, query_image]
        inputs = processor(prompt, all_images, return_tensors="pt").to(device)
        
        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        print(f"  🔄 Generating (one-shot)...")
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                **generation_args
            )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        
        raw_output = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()
        
        raw_output = normalize_unicode(raw_output)
        
        prediction = extract_json_from_response(raw_output)
        
        try:
            json.loads(prediction)
            json_parsed += 1
            print(f"  ✅ Valid JSON parsed")
        except:
            print(f"  ⚠️ JSON parse failed - using raw output")
        
        ground_truth = create_label_string(item)
        
        if len(ground_truth) > 0:
            cer = jiwer.cer(ground_truth, prediction)
        else:
            cer = 1.0 if len(prediction) > 0 else 0.0
        
        cer_scores.append(cer)
        
        if cer == 0.0:
            perfect_matches += 1
        
        if "{" in prediction and "}" in prediction:
            json_like += 1
        
        print(f"  📊 CER: {cer:.4f} ({cer*100:.2f}%)")
        if cer == 0.0:
            print(f"  ✨ Perfect match!")
        print(f"  📝 Prediction (first 150 chars): {prediction[:150]}...")
        print()
        
        results.append({
            "file_name": item["file_name"],
            "predicted_text": prediction,
            "ground_truth_text": ground_truth,
            "raw_output": raw_output,
            "cer_score": cer,
        })
        
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "file_name": item["file_name"],
            "predicted_text": "",
            "ground_truth_text": create_label_string(item),
            "cer_score": 1.0,
            "error": str(e)
        })
        cer_scores.append(1.0)

# Compute metrics
avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
perfect_rate = perfect_matches / len(cer_scores) if cer_scores else 0.0
json_parsed_rate = json_parsed / len(results) if results else 0.0
json_like_rate = json_like / len(results) if results else 0.0

metrics = {
    "total_samples": len(test_data),
    "average_cer": avg_cer,
    "perfect_matches": perfect_matches,
    "perfect_match_rate": perfect_rate,
    "valid_json_parsed": json_parsed,
    "json_parsed_rate": json_parsed_rate,
    "json_like_outputs": json_like,
    "json_like_rate": json_like_rate,
    "one_shot_example": one_shot_example["file_name"],
}

print(f"\n{'='*80}")
print("📊 FINAL RESULTS")
print(f"{'='*80}")
print(f"  Total Samples: {len(test_data)}")
print(f"  Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
print(f"  Perfect Matches: {perfect_matches}/{len(cer_scores)} ({perfect_rate:.2%})")
print(f"  Valid JSON Parsed: {json_parsed}/{len(results)} ({json_parsed_rate:.2%})")
print(f"  JSON-like Outputs: {json_like}/{len(results)} ({json_like_rate:.2%})")
print(f"\n  One-shot example: {one_shot_example['file_name']}")
print(f"{'='*80}")

# Save results
save_jsonl(results, output_file)
print(f"\n💾 Predictions saved to: {output_file}")

with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"💾 Metrics saved to: {metrics_file}")

print(f"\n📂 Output directory: {output_dir}")
print("\n✅ One-shot inference complete!")
