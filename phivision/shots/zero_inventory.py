#!/usr/bin/env python3
# phi_3_5_vision_inventory_zero_shot.py
# ✅ Zero-shot inference on German historical inventory documents
# ✅ JSON parse fallback + proper error handling

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
test_jsonl_path = "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl"
images_dir = "/home/woody/iwi5/iwi5298h/inventory_images"
output_base_dir = "/home/vault/iwi5/iwi5298h/models_image_text/phi/inven"

# Create timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(output_base_dir, f"run_zeroshot_{timestamp}")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "predictions_zeroshot.jsonl")

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
    """Ground truth without image_name, projected to schema and canonicalized."""
    label_data = {k: v for k, v in json_data.items() if k != "image_name"}
    return canonical_json_str(project_to_schema(label_data))

# ---- Schema helpers (stable eval + reduce CER noise) ----
SCHEMA_KEYS = ["Überschrift", "Inventarnummer", "Maße", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]
MEAS_KEYS = ["L", "B", "D"]

def project_to_schema(d: Dict) -> Dict:
    """Keep only expected keys, ensure Maße object with L/B/D."""
    out = {}
    for k in SCHEMA_KEYS:
        if k == "Maße":
            meas = d.get("Maße", {})
            if not isinstance(meas, dict):
                meas = {}
            out["Maße"] = {mk: "" if meas.get(mk, "") is None else str(meas.get(mk, "")) for mk in MEAS_KEYS}
        else:
            v = d.get(k, "")
            out[k] = "" if v is None else str(v)
    return out

def canonical_json_str(d: Dict) -> str:
    """Deterministic JSON formatting for CER (no spaces)."""
    return json.dumps(d, ensure_ascii=False, sort_keys=False, separators=(",", ":"))

def extract_json_from_response(response: str) -> str:
    """Extract first complete JSON object from response, handling nested braces."""
    response = response.strip()
    
    if "{" in response and "}" in response:
        start = response.find("{")
        json_str = response[start:]
        
        # Find the first balanced JSON object
        brace_count = 0
        end_idx = -1
        for i, ch in enumerate(json_str):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if end_idx > 0:
            first_json = json_str[:end_idx]
            try:
                parsed = json.loads(first_json)
                # Project to schema for consistent output
                parsed = project_to_schema(parsed)
                return canonical_json_str(parsed)
            except:
                pass
        
        # Fallback: try whole slice between first '{' and last '}'
        end = response.rfind("}") + 1
        json_str = response[start:end]
        try:
            parsed = json.loads(json_str)
            parsed = project_to_schema(parsed)
            return canonical_json_str(parsed)
        except:
            return json_str
    
    return response

# ✅ Prompt for inventory documents with nested measurements (German field names)
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

print("=" * 80)
print("🚀 Phi-3.5-Vision Zero-Shot Inference on Inventory Dataset")
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
    num_crops=16  # Best for single-frame documents
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
    print(f"[{idx+1}/{len(test_data)}] {item['image_name']}")
    print(f"────────────────────────────────────────────────────────────")
    
    image_path = os.path.join(images_dir, item["image_name"])
    
    if not os.path.exists(image_path):
        print(f"  ❌ Image not found: {image_path}")
        results.append({
            "image_name": item["image_name"],
            "predicted_text": "",
            "ground_truth_text": create_label_string(item),
            "cer_score": 1.0,
            "error": "Image not found"
        })
        cer_scores.append(1.0)
        continue
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Phi-3.5-vision input format
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
        
        # Try to extract and parse JSON
        prediction = extract_json_from_response(raw_output)
        
        # Check if valid JSON
        try:
            json.loads(prediction)
            json_parsed += 1
            print(f"  ✅ Valid JSON parsed")
        except:
            print(f"  ⚠️ JSON parse failed - using raw output")
        
        # Ground truth (JSON string without image_name)
        ground_truth = create_label_string(item)
        
        # Compute CER
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
            "image_name": item["image_name"],
            "predicted_text": prediction,
            "ground_truth_text": ground_truth,
            "raw_output": raw_output,
            "cer_score": cer,
        })
        
        # Clear cache every 3 samples (fewer samples in test)
        if (idx + 1) % 3 == 0:
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "image_name": item["image_name"],
            "predicted_text": "",
            "ground_truth_text": create_label_string(item),
            "cer_score": 1.0,
            "error": str(e)
        })
        cer_scores.append(1.0)

# Compute metrics
# Macro CER: average of per-sample CER scores
macro_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0

# Micro CER: total errors / total characters (character-weighted)
total_chars = 0
total_errors = 0
for i, result in enumerate(results):
    gt_text = result.get("ground_truth_text", "")
    char_count = len(gt_text)
    cer_score = cer_scores[i] if i < len(cer_scores) else 1.0
    total_chars += char_count
    total_errors += int(cer_score * char_count)

micro_cer = total_errors / total_chars if total_chars > 0 else 1.0

perfect_rate = perfect_matches / len(cer_scores) if cer_scores else 0.0
json_parsed_rate = json_parsed / len(results) if results else 0.0
json_like_rate = json_like / len(results) if results else 0.0

print(f"\n{'='*80}")
print("📊 FINAL RESULTS - ZERO-SHOT")
print(f"{'='*80}")
print(f"  Total Samples: {len(test_data)}")
print(f"  Macro CER (avg per-sample): {macro_cer:.4f} ({macro_cer*100:.2f}%)")
print(f"  Micro CER (char-weighted):  {micro_cer:.4f} ({micro_cer*100:.2f}%)")
print(f"  Perfect Matches: {perfect_matches}/{len(cer_scores)} ({perfect_rate:.2%})")
print(f"  Valid JSON Parsed: {json_parsed}/{len(results)} ({json_parsed_rate:.2%})")
print(f"  JSON-like Outputs: {json_like}/{len(results)} ({json_like_rate:.2%})")
print(f"{'='*80}")

save_jsonl(results, output_file)
print(f"\n💾 Results saved to: {output_file}")

# Save summary file
summary_data = {
    "timestamp": timestamp,
    "method": "zero-shot",
    "model": "Phi-3.5-Vision",
    "total_samples": len(test_data),
    "macro_cer": float(macro_cer),
    "micro_cer": float(micro_cer),
    "perfect_matches": int(perfect_matches),
    "perfect_rate": float(perfect_rate),
    "valid_json_parsed": int(json_parsed),
    "json_parsed_rate": float(json_parsed_rate),
    "json_like_outputs": int(json_like),
    "total_characters": int(total_chars),
    "total_errors": int(total_errors)
}
summary_file = os.path.join(output_dir, "evaluation_summary_zeroshot.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)
print(f"💾 Summary saved to: {summary_file}")

print(f"\n📁 Output directory: {output_dir}")
print("\n✅ Zero-shot inference complete!")
