#!/usr/bin/env python3
# phi_3_5_vision_schmuck_zero_shot.py
# ✅ Zero-shot inference with JSON parse fallback
# ✅ If JSON parse fails, compare raw string with ground truth JSON string

import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import jiwer
import unicodedata
from typing import List, Dict

# Paths
LOCAL_MODEL_PATH = "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision"
test_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
images_dir = "/home/woody/iwi5/iwi5298h/schmuck_images"
output_file = "/home/vault/iwi5/iwi5298h/models_image_text/phi/schmuck/phi35_vision_schmuck_zero_shot.jsonl"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
    
    # Try to find JSON between curly braces
    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        
        # Try to parse it
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except:
            return json_str
    
    return response

PROMPT = """Extract all information from this German jewelry catalog image and return ONLY a JSON object with exactly these keys:

{
  "Gegenstand": "Object type (e.g., Halskette, Ring, etc.)",
  "Inv.Nr": "Inventory number (e.g., Sch 2891/118)",
  "Herkunft": "Origin/provenance",
  "Foto Notes": "Photo notes (e.g., Foto Nr.XXXX)",
  "Standort": "Location/storage",
  "Material": "Material description",
  "Datierung": "Dating/time period",
  "Maße": "Measurements/dimensions",
  "Gewicht": "Weight",
  "erworben von": "Acquired from",
  "am": "Acquisition date",
  "Preis": "Price",
  "Vers.-Wert": "Insurance value",
  "Beschreibung": "Description",
  "Literatur": "Literature references",
  "Ausstellungen": "Exhibitions"
}

Return ONLY the JSON object with these exact keys. Use empty string "" for missing values. No additional commentary."""

print("=" * 80)
print("🚀 Phi-3.5-Vision Zero-Shot Inference on Schmuck Dataset")
print("=" * 80)

print(f"\n📂 Loading test data from: {test_jsonl_path}")
test_data = load_jsonl(test_jsonl_path)
print(f"✅ Loaded {len(test_data)} test samples")

print(f"\n⏳ Loading model from: {LOCAL_MODEL_PATH}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Device: {device}")

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    num_crops=16
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation='eager'
)

print("✅ Model and processor loaded")

results = []
cer_scores = []
perfect_matches = 0
json_parsed = 0
json_like = 0

print(f"\n{'='*80}")
print("RUNNING ZERO-SHOT INFERENCE")
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
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{PROMPT}"
            }
        ]
        
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = processor(prompt, [image], return_tensors="pt").to(device)
        
        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": processor.tokenizer.eos_token_id,
        }
        
        print(f"  🔄 Generating...")
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
        
        # ✅ Try to extract and parse JSON
        prediction = extract_json_from_response(raw_output)
        
        # Check if it's valid JSON
        try:
            json.loads(prediction)
            json_parsed += 1
            print(f"  ✅ Valid JSON parsed")
        except:
            print(f"  ⚠️ JSON parse failed - using raw output")
        
        # Ground truth (JSON string without file_name)
        ground_truth = create_label_string(item)
        
        # ✅ Compute CER between prediction and ground truth JSON string
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
            "raw_output": raw_output,  # ✅ Store raw output too
            "cer_score": cer,
        })
        
        if (idx + 1) % 5 == 0:
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

print(f"\n{'='*80}")
print("📊 FINAL RESULTS")
print(f"{'='*80}")
print(f"  Total Samples: {len(test_data)}")
print(f"  Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
print(f"  Perfect Matches: {perfect_matches}/{len(cer_scores)} ({perfect_rate:.2%})")
print(f"  Valid JSON Parsed: {json_parsed}/{len(results)} ({json_parsed_rate:.2%})")
print(f"  JSON-like Outputs: {json_like}/{len(results)} ({json_like_rate:.2%})")
print(f"{'='*80}")

save_jsonl(results, output_file)
print(f"\n💾 Results saved to: {output_file}")

print("\n✅ Zero-shot inference complete!")
