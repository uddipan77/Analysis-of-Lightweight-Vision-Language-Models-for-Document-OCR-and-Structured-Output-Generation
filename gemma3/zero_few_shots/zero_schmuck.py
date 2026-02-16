#!/usr/bin/env python3
# gemma3_schmuck_zeroshot_chat_fixed.py
# Zero-shot OCR on Schmuck dataset with Gemma-3 (transformers)
# Fixes:
#  - DO NOT cast input_ids to float16 (critical bug)
#  - Use chat_template tokenize=False + processor(text, images)
#  - Strip trailing EOS from input to avoid immediate stop
#  - Greedy decode with sampling fallback if empty/not JSON
#  - Optional dual-view image (original + autocontrast)
#  - Enforce full schema keys

import sys
import os
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import torch
import jiwer
from PIL import Image, ImageOps
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import gc
import warnings


# --------------------------
# Paths
# --------------------------
TEST_JSONL  = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
IMAGES_DIR  = "/home/woody/iwi5/iwi5298h/schmuck_images"

MODEL_PATH  = "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"

OUT_BASE    = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/schmuck/few_zero_schmuck"


SCHMUCK_KEYS = [
    "Gegenstand",
    "Inv.Nr",
    "Herkunft",
    "Foto Notes",
    "Standort",
    "Material",
    "Datierung",
    "Maße",
    "Gewicht",
    "erworben von",
    "am",
    "Preis",
    "Vers.-Wert",
    "Beschreibung",
    "Literatur",
    "Ausstellungen",
]


# --------------------------
# Utilities
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
    return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))

def enforce_schema(pred: Dict) -> Dict:
    """Ensure all keys exist; keep only known keys."""
    out = {k: "" for k in SCHMUCK_KEYS}
    if isinstance(pred, dict):
        for k in SCHMUCK_KEYS:
            if k in pred and pred[k] is not None:
                out[k] = str(pred[k])
    return out

def extract_json_from_response(response: str) -> Dict:
    """Extract the largest JSON object from a free-form response."""
    if not isinstance(response, str):
        return {}

    s = response.strip()

    # Remove markdown fences if present
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.lstrip().startswith("json"):
                body = body[4:].strip()
            s = body.strip()

    # Find JSON blocks
    matches = re.findall(r"\{.*\}", s, flags=re.DOTALL)
    if matches:
        for m in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(m)
            except Exception:
                continue

    # Try full parse
    try:
        return json.loads(s)
    except Exception:
        return {}

def schmuck_prompt() -> str:
    # Keep prompt short but strict; long prompts can reduce OCR fidelity on small VLMs.
    keys = ", ".join(SCHMUCK_KEYS)
    return f"""
You are an OCR system for German museum jewelry catalog cards.
Read ALL printed + handwritten text carefully and extract the fields into ONE JSON object.

Rules:
- Output ONLY valid JSON (no commentary, no markdown).
- Use exactly these keys: {keys}
- Use empty string "" for missing fields.
- Keep German spelling, punctuation, and spacing as seen.
- For inventory numbers etc. do not invent values.

Return JSON only.
""".strip()


# --------------------------
# Runner
# --------------------------
class Gemma3SchmuckZeroShot:
    def __init__(self, model_path: str):
        print("=" * 70)
        print("Loading Gemma-3 for Schmuck zero-shot OCR (transformers)")
        print("=" * 70)

        # BF16 on Ampere+; FP16 on V100/Volta
        use_bf16 = False
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            print(f"Detected GPU: {gpu} (cc {major}.{minor})")
            use_bf16 = major >= 8

        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"Using dtype: {self.dtype}")

        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=self.dtype,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        # Ensure EOS/PAD configured
        tok = self.processor.tokenizer
        if getattr(self.model.generation_config, "pad_token_id", None) is None:
            self.model.generation_config.pad_token_id = tok.pad_token_id
        if getattr(self.model.generation_config, "eos_token_id", None) is None:
            self.model.generation_config.eos_token_id = tok.eos_token_id

        warnings.filterwarnings("ignore", message="`do_sample` is set to `False`.*")
        warnings.filterwarnings("ignore", message="Using a slow image processor.*")

        # Different builds use different image placeholder tokens
        self._token_candidates = ["<image>", "<start_of_image>"]

        print("✅ Model + processor loaded.\n")

    @staticmethod
    def _build_messages(image: Image.Image, dual_view: bool = True) -> List[Dict]:
        """System + user with optional dual-view image."""
        image = image.convert("RGB")
        contents = []

        if dual_view:
            img2 = ImageOps.autocontrast(image, cutoff=1)
            contents.append({"type": "image", "image": image})
            contents.append({"type": "image", "image": img2})
        else:
            contents.append({"type": "image", "image": image})

        contents.append({"type": "text", "text": schmuck_prompt()})

        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You output JSON only."}],
            },
            {
                "role": "user",
                "content": contents,
            },
        ]

    @staticmethod
    def _gather_images(messages: List[Dict]) -> List[Image.Image]:
        imgs = []
        for m in messages:
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "image":
                    imgs.append(c.get("image"))
        return imgs

    def _prepare_inputs(self, messages: List[Dict]):
        # IMPORTANT: tokenize=False here (we pass images separately to processor)
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Debug marker count (optional but useful)
        token_count = 0
        for t in self._token_candidates:
            token_count = max(token_count, prompt_text.count(t))
        images = self._gather_images(messages)
        # If mismatch, generation can behave weirdly; we hard error to catch it.
        if token_count != len(images):
            raise ValueError(
                f"Image token mismatch: prompt has {token_count} markers but {len(images)} images."
            )

        # Build tensors
        inputs = self.processor(
            text=[prompt_text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # Move to device WITHOUT forcing dtype (critical fix)
        device = self.model.device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        # Cast ONLY pixel values to model dtype (leave input_ids as long!)
        for pix_key in ("pixel_values", "pixel_values_videos"):
            if pix_key in inputs and torch.is_tensor(inputs[pix_key]):
                inputs[pix_key] = inputs[pix_key].to(dtype=self.dtype)

        # Strip trailing EOS from input_ids to avoid immediate stop
        eos_id = self.model.generation_config.eos_token_id
        if eos_id is not None and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            attn = inputs.get("attention_mask", None)

            eos_scalar = eos_id if isinstance(eos_id, int) else eos_id[0]
            if input_ids.shape[-1] > 1 and input_ids[0, -1].item() == eos_scalar:
                inputs["input_ids"] = input_ids[:, :-1]
                if attn is not None:
                    inputs["attention_mask"] = attn[:, :-1]

        input_len = inputs["input_ids"].shape[-1]
        return inputs, input_len

    def generate(self, image: Image.Image, max_new_tokens: int = 900) -> Tuple[str, bool]:
        """Greedy decode with sampling fallback if empty or not JSON-like."""
        messages = self._build_messages(image, dual_view=True)
        inputs, input_len = self._prepare_inputs(messages)

        def _decode(gen_ids):
            out = gen_ids[0][input_len:]
            return self.processor.decode(out, skip_special_tokens=True).strip()

        used_sampling = False
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=64,          # force some output
                do_sample=False,
                repetition_penalty=1.05,
                use_cache=True,
            )
            text = _decode(out)

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
                text = _decode(out)

        del inputs, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return text, used_sampling


# --------------------------
# Save results
# --------------------------
def save_predictions_with_timestamp(predictions, test_data, all_cer_scores, base_dir: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"run_zeroshot_{timestamp}")
    ensure_dir(out_dir)

    pred_path = os.path.join(out_dir, "predictions_zeroshot.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"Saved predictions: {pred_path}")

    if all_cer_scores:
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)

        # Weighted CER
        total_chars = 0
        total_errors = 0
        for i, test_item in enumerate(test_data[:len(predictions)]):
            gt_json = {k: v for k, v in test_item.items() if k != "file_name"}
            gt_s = json_to_string(gt_json)
            n = len(gt_s)
            cer = all_cer_scores[i]
            total_chars += n
            total_errors += int(round(cer * n))
        weighted = (total_errors / total_chars) if total_chars else 0.0

        perfect = sum(1 for c in all_cer_scores if c == 0.0)

        summary = {
            "timestamp": timestamp,
            "model": "Gemma 3 4B (transformers)",
            "dataset": "Schmuck",
            "method": "zero-shot",
            "total_images": len(predictions),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted),
            "perfect_matches": int(perfect),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
        }

        summary_path = os.path.join(out_dir, "evaluation_summary_zeroshot.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Summary: {summary_path}")
        print(f"Avg CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print("=" * 70 + "\n")
        return out_dir, avg_cer

    return out_dir, None


# --------------------------
# Main
# --------------------------
def main():
    ensure_dir(OUT_BASE)

    print("=" * 70)
    print("GEMMA-3 ZERO-SHOT SCHMUCK OCR (FIXED)")
    print("=" * 70)
    print("Loading test JSONL ...")
    test_data = load_jsonl(TEST_JSONL)
    print(f"✅ Loaded {len(test_data)} samples\n")

    runner = Gemma3SchmuckZeroShot(MODEL_PATH)

    predictions = []
    cer_scores = []

    print("=" * 70)
    print(f"Processing {len(test_data)} images ...")
    print("=" * 70)

    for i, row in enumerate(test_data, 1):
        fn = row.get("file_name", "")
        img_path = os.path.join(IMAGES_DIR, fn)

        print(f"\n{'─'*70}")
        print(f"[{i}/{len(test_data)}] {fn}")
        print(f"{'─'*70}")

        if not os.path.exists(img_path):
            print(f"⚠️ Missing image: {img_path}")
            pred_json = enforce_schema({})
            gt_json = {k: v for k, v in row.items() if k != "file_name"}
            predictions.append({
                "file_name": fn,
                "predicted_json": pred_json,
                "ground_truth": gt_json,
                "raw_response": "Error: Image not found",
                "cer_score": 1.0,
            })
            cer_scores.append(1.0)
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            raw, used_sampling = runner.generate(image, max_new_tokens=900)

            print(f"[DEBUG] used_sampling_fallback: {used_sampling}")
            print(f"[DEBUG] raw_output_len: {len(raw)}")
            print(f"[DEBUG] raw_preview: {raw[:220].replace(chr(10), ' ')}")

            pred = extract_json_from_response(raw)
            pred_json = enforce_schema(pred)

            gt_json = {k: v for k, v in row.items() if k != "file_name"}

            gt_s = json_to_string(gt_json)
            pr_s = json_to_string(pred_json)

            cer = jiwer.cer(gt_s, pr_s)
            print(f"✅ CER: {cer:.4f} ({cer*100:.2f}%)")

            predictions.append({
                "file_name": fn,
                "predicted_json": pred_json,
                "ground_truth": gt_json,
                "raw_response": raw,
                "cer_score": cer,
            })
            cer_scores.append(cer)

            running_avg = sum(cer_scores) / len(cer_scores)
            print(f"📈 Running Avg CER: {running_avg:.4f} ({running_avg*100:.2f}%)")

        except Exception as e:
            print(f"❌ Error: {e}")
            pred_json = enforce_schema({})
            gt_json = {k: v for k, v in row.items() if k != "file_name"}
            predictions.append({
                "file_name": fn,
                "predicted_json": pred_json,
                "ground_truth": gt_json,
                "raw_response": f"Error: {e}",
                "cer_score": 1.0,
            })
            cer_scores.append(1.0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    out_dir, avg_cer = save_predictions_with_timestamp(predictions, test_data, cer_scores, OUT_BASE)

    print("=" * 70)
    print("DONE ✅")
    print(f"Saved to: {out_dir}")
    if avg_cer is not None:
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
