#!/usr/bin/env python3
# phi_3_5_vision_inventory_few_shot_fixed.py
# ✅ Updated dataset (updated_json_inven)
# ✅ 1-shot (inventarbuch-156.jpg)
# ✅ FIXED: JSON extraction + parsing + schema projection + canonical pred for CER
# ✅ FIXED: Increase max_new_tokens safely to reduce truncation
# ✅ Saves per-sample CER + macro CER + micro CER

import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import jiwer
import unicodedata
from typing import List, Dict, Any
from datetime import datetime
import re


# -------------------------
# Paths (Code-2 dataset)
# -------------------------
LOCAL_MODEL_PATH = "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision"
train_jsonl_path = "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl"
test_jsonl_path  = "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl"
images_dir       = "/home/woody/iwi5/iwi5298h/inventory_images"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir   = f"/home/vault/iwi5/iwi5298h/models_image_text/phi/inven/few_shot_fixed_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
output_file  = os.path.join(output_dir, "predictions.jsonl")
metrics_file = os.path.join(output_dir, "metrics.json")


# -------------------------
# Helpers
# -------------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")

def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ---- Ground truth canonicalization ----
SCHEMA_KEYS = ["Überschrift", "Inventarnummer", "Maße", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]
MEAS_KEYS = ["L", "B", "D"]

def project_to_schema(d: Dict) -> Dict:
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
    return json.dumps(d, ensure_ascii=False, sort_keys=False, separators=(",", ":"))

def gt_string(item: Dict) -> str:
    d = {k: v for k, v in item.items() if k != "image_name"}
    return canonical_json_str(project_to_schema(d))


# -------------------------
# JSON extraction + light repair
# -------------------------
def extract_json_block(text: str) -> str:
    """
    Extract the most likely JSON block.
    - Prefer first '{' to last '}' span if both exist.
    - Else return original text.
    """
    text = (text or "").strip()
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        return text[start:end].strip()
    return text

def try_parse_json(text: str) -> Dict:
    """
    Try strict JSON parse; if fails, attempt very light fixes:
    - Remove leading/trailing junk around JSON
    - If output truncated but contains a JSON prefix, cut at last complete brace
    """
    text = (text or "").strip()

    # First, try direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting JSON region
    extracted = extract_json_block(text)
    try:
        return json.loads(extracted)
    except Exception:
        pass

    # If truncated: cut at last occurrence of '}' (already handled) OR try cut at last quote+comma pattern.
    # Keep this conservative; if it still fails, return empty dict.
    return {}


def pred_canonical_from_raw(raw_output: str) -> (str, bool, Dict):
    """
    Convert raw model output -> canonical JSON string for CER if possible.
    Returns: (pred_text_for_cer, parsed_ok, pred_obj_projected)
    """
    raw_output = normalize_unicode(raw_output)
    extracted = extract_json_block(raw_output)
    parsed = try_parse_json(extracted)

    if isinstance(parsed, dict) and parsed:
        projected = project_to_schema(parsed)
        return canonical_json_str(projected), True, projected

    # Could not parse → fall back to extracted (still better than full raw sometimes)
    return extracted, False, {}


# -------------------------
# Prompt (keep short but strict)
# -------------------------
PROMPT = """Gib NUR ein JSON-Objekt zurück (keine Erklärungen) mit GENAU diesen Keys:

{
  "Überschrift": "",
  "Inventarnummer": "",
  "Maße": {"L":"","B":"","D":""},
  "Objektbezeichnung": "",
  "Fundort": "",
  "Fundzeit": "",
  "Beschreibungstext": ""
}

Regeln:
- Text exakt wie im Bild
- Fehlende Werte als "" (Keys niemals entfernen)
- "Maße" muss immer ein Objekt mit L,B,D sein
- Beispiele nur für das FORMAT benutzen, keine Werte kopieren
"""


# -------------------------
# Main
# -------------------------
print("=" * 80)
print("🚀 Phi-3.5-Vision 1-shot Inference — FIXED CER (JSON canonical)")
print("=" * 80)

train_data = load_jsonl(train_jsonl_path)
test_data  = load_jsonl(test_jsonl_path)
print(f"✅ Train samples: {len(train_data)}")
print(f"✅ Test samples:  {len(test_data)}")

# 1-shot example
FEW_SHOT_IMAGE = "inventarbuch-156.jpg"
few_shot_example = next((x for x in train_data if x.get("image_name") == FEW_SHOT_IMAGE), None)
if few_shot_example is None:
    raise RuntimeError(f"❌ Required few-shot example not found in train.jsonl: {FEW_SHOT_IMAGE}")

print("✅ Using 1 example:", FEW_SHOT_IMAGE)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"📍 Device: {device}")

processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True,
    num_crops=16
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    device_map="cuda" if device == "cuda" else None,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    _attn_implementation="eager"
)

print("✅ Model and processor loaded")

# Load example image once
ex_path = os.path.join(images_dir, FEW_SHOT_IMAGE)
if not os.path.exists(ex_path):
    raise FileNotFoundError(f"❌ Missing few-shot example image: {ex_path}")
example_image = Image.open(ex_path).convert("RGB")

results = []
cer_scores = []

# For micro CER
total_chars = 0
total_errors = 0

perfect_matches = 0
json_parsed = 0

print(f"\n{'='*80}")
print("RUNNING 1-SHOT INFERENCE — FIXED EVAL")
print(f"{'='*80}")

for idx, item in enumerate(test_data):
    img_name = item["image_name"]
    print(f"\n[{idx+1}/{len(test_data)}] {img_name}")

    image_path = os.path.join(images_dir, img_name)
    gt_text = gt_string(item)

    if not os.path.exists(image_path):
        print(f"  ❌ Image not found: {image_path}")
        cer = 1.0 if gt_text else 0.0
        results.append({
            "image_name": img_name,
            "predicted_text": "",
            "ground_truth_text": gt_text,
            "raw_output": "",
            "cer_score": cer,
            "error": "Image not found",
            "few_shot_examples": [FEW_SHOT_IMAGE],
        })
        cer_scores.append(cer)
        total_chars += len(gt_text)
        total_errors += int(round(cer * len(gt_text)))
        continue

    try:
        query_image = Image.open(image_path).convert("RGB")

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{PROMPT}"},
            {"role": "assistant", "content": gt_string(few_shot_example)},
            {"role": "user", "content": f"<|image_2|>\n{PROMPT}"},
        ]

        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        all_images = [example_image, query_image]
        inputs = processor(prompt, all_images, return_tensors="pt").to(device)

        # ---- Clamp generation to remaining context, but allow LONGER outputs ----
        seq_len = inputs["input_ids"].shape[1]
        model_max = getattr(model.config, "max_position_embeddings", None) or \
                    getattr(model.config, "max_seq_len", None) or \
                    getattr(model.config, "max_sequence_length", None) or 8192

        margin = 32
        remaining = model_max - seq_len - margin
        if remaining <= 0:
            raise RuntimeError(
                f"Input too long: seq_len={seq_len}, model_max={model_max}. "
                f"Reduce crops/resize/shorten example."
            )

        # BIG FIX: allow more tokens than 1024 if context permits (Beschreibungstext is long)
        max_new = min(4096, remaining)  # was 1024

        generation_args = {
            "max_new_tokens": max_new,
            "temperature": 0.0,
            "do_sample": False,
            "eos_token_id": processor.tokenizer.eos_token_id,
            "pad_token_id": processor.tokenizer.eos_token_id,
        }

        print(f"  🧠 seq_len={seq_len}, model_max={model_max}, max_new_tokens={max_new}")

        with torch.no_grad():
            generate_ids = model.generate(**inputs, **generation_args)

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

        raw_output = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        raw_output = normalize_unicode(raw_output)

        # ✅ FIXED: canonicalize prediction if parseable
        pred_text, parsed_ok, pred_obj = pred_canonical_from_raw(raw_output)
        if parsed_ok:
            json_parsed += 1

        cer = jiwer.cer(gt_text, pred_text) if gt_text else (1.0 if pred_text else 0.0)
        cer_scores.append(cer)

        if cer == 0.0:
            perfect_matches += 1

        # micro CER accounting
        char_count = len(gt_text)
        total_chars += char_count
        total_errors += int(round(cer * char_count))

        print(f"  ✅ JSON parsed: {parsed_ok}")
        print(f"  📊 CER: {cer:.4f} ({cer*100:.2f}%)")
        print(f"  📝 Pred (first 150 chars): {pred_text[:150]}...")

        results.append({
            "image_name": img_name,
            "predicted_text": pred_text,      # canonical if parsed, else extracted
            "ground_truth_text": gt_text,     # canonical GT
            "raw_output": raw_output,         # raw model output
            "cer_score": cer,
            "few_shot_examples": [FEW_SHOT_IMAGE],
            "seq_len": int(seq_len),
            "max_new_tokens_used": int(max_new),
            "model_max_context": int(model_max),
            "json_parsed": bool(parsed_ok),
        })

        if (idx + 1) % 3 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        cer = 1.0 if gt_text else 0.0
        cer_scores.append(cer)

        total_chars += len(gt_text)
        total_errors += int(round(cer * len(gt_text)))

        print(f"  ❌ Error: {e}")
        results.append({
            "image_name": img_name,
            "predicted_text": "",
            "ground_truth_text": gt_text,
            "raw_output": "",
            "cer_score": cer,
            "error": str(e),
            "few_shot_examples": [FEW_SHOT_IMAGE],
        })

# -------------------------
# Metrics: macro + micro CER
# -------------------------
macro_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
micro_cer = (total_errors / total_chars) if total_chars > 0 else 1.0

metrics = {
    "timestamp": timestamp,
    "method": "few-shot-fixed-json-canonical",
    "model": "Phi-3.5-Vision",
    "total_samples": len(test_data),
    "macro_cer": float(macro_cer),
    "micro_cer": float(micro_cer),
    "perfect_matches": int(perfect_matches),
    "perfect_match_rate": (perfect_matches / len(cer_scores)) if cer_scores else 0.0,
    "total_characters": int(total_chars),
    "total_errors": int(total_errors),
    "valid_json_parsed": int(json_parsed),
    "json_parsed_rate": float(json_parsed / len(test_data)) if test_data else 0.0,
    "few_shot_examples": [FEW_SHOT_IMAGE],
}

print(f"\n{'='*80}")
print("📊 FINAL RESULTS — FIXED EVAL")
print(f"{'='*80}")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print(f"{'='*80}")

# Save outputs
save_jsonl(results, output_file)
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"\n💾 Predictions saved to: {output_file}")
print(f"💾 Metrics saved to: {metrics_file}")
print(f"📂 Output directory: {output_dir}")
