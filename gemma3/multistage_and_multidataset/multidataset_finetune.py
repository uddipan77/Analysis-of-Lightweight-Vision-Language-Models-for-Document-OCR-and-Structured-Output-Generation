#!/usr/bin/env python3
# gemma_multidataset.py
#
# ✅ Multi-dataset fine-tuning of Gemma-3-4B-IT (Unsloth 4-bit) on:
#    - Inventory
#    - Schmuck
#    - Staircase
#
# ✅ One shared model + one shared LoRA adapter trained on combined TRAIN of all datasets.
# ✅ Best model selected by LOWEST GLOBAL validation CER (autoregressive generation each epoch).
# ✅ Saves ONLY ONE best adapter: <run_dir>/best_model
# ✅ Resume via: python gemma_multidataset.py --run_dir <existing_run_dir>
# ✅ TensorBoard logging: <run_dir>/tb_logs
#
# IMPORTANT FIXES:
# - PyTorch 2.6+ changed torch.load default to weights_only=True -> resume may fail on RNG state (numpy)
#   => allowlist numpy globals (+ concrete dtype classes) for safe unpickling.
# - Also add a safe Trainer override so resume never crashes even if RNG restore still fails.
#
# First run:
#   python gemma_multidataset.py
#
# Resume after timeout:
#   python gemma_multidataset.py --run_dir /path/to/existing/run_dir
#
# TensorBoard:
#   tensorboard --logdir <run_dir>/tb_logs

import sys
import os

# ✅ CRITICAL: Enable logits return for compute_metrics (Unsloth requirement)
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import json
import re
import gc
import glob
import shutil
import random
import argparse
from datetime import datetime
from typing import List, Dict, Callable, Optional, Any

import jiwer
import numpy as np
import numpy  # keep (we reference numpy._core...)
import numpy._core.multiarray  # ✅ keep (some installs need this path)
from PIL import Image

# =============================================================================
# ✅ PyTorch 2.6+ / 2.8 resume fix (safe-unpickling allowlist for RNG state)
# =============================================================================
# SFTTrainer/Trainer resume loads rng_state.pth via torch.load(...).
# With PyTorch>=2.6 default weights_only=True, numpy objects are rejected unless allowlisted.
try:
    # Concrete dtype classes (NumPy 2.x uses e.g. numpy.dtypes.UInt32DType)
    _DTYPE_CLASSES = {
        np.dtype(t).__class__ for t in [
            np.bool_,
            np.uint8, np.uint16, np.uint32, np.uint64,
            np.int8, np.int16, np.int32, np.int64,
            np.float16, np.float32, np.float64,
        ]
    }
    torch.serialization.add_safe_globals([
        np.ndarray,
        np.dtype,
        *_DTYPE_CLASSES,  # ✅ covers numpy.dtypes.UInt32DType etc.
        numpy._core.multiarray._reconstruct,
        numpy._core.multiarray.scalar,
    ])
except Exception as e:
    print(f"[WARN] add_safe_globals failed (continuing): {e}")

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

# =============================================================================
# ✅ Bulletproof fallback: don’t crash if RNG state still can’t be loaded
# =============================================================================
class SafeRNGResumeSFTTrainer(SFTTrainer):
    def _load_rng_state(self, resume_from_checkpoint):
        try:
            return super()._load_rng_state(resume_from_checkpoint)
        except Exception as e:
            print(
                f"[WARN] Could not load RNG state from checkpoint ({resume_from_checkpoint}). "
                f"Continuing without RNG restore. Error: {e}"
            )
            return

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe",
    "datasets": {
        "inventory": {
            "train_jsonl": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
            "val_jsonl": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
            "test_jsonl": "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
            "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",
            "image_key": "image_name",
        },
        "schmuck": {
            "train_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
            "val_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
            "test_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
            "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",
            "image_key": "file_name",
        },
        "staircase": {
            "train_jsonl": "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl",
            "val_jsonl": "/home/woody/iwi5/iwi5298h/json_staircase/val.jsonl",
            "test_jsonl": "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl",
            "images_dir": "/home/woody/iwi5/iwi5298h/staircase_images",
            "image_key": "image_name",
        },
    },
    "output_base_dir": "/home/vault/iwi5/iwi5298h/models_image_text/gemma/general",

    # Training hyperparams
    "num_epochs": 12,
    "batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 3e-5,
    "max_seq_length": 2048,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "label_smoothing_factor": 0.05,

    # LoRA
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,

    # CER validation
    "max_val_samples_per_dataset": 30,
    "gen_max_new_tokens": 1024,

    # Test-time chunk flush interval
    "test_chunk_flush_interval": 5,

    # ✅ Resume-friendly rolling checkpoint
    "save_strategy": "epoch",
    "save_steps": 500,          # ignored when save_strategy="epoch"; kept for compatibility
    "save_total_limit": 1,
}

# =============================================================================
# INSTRUCTIONS PER DATASET
# =============================================================================

INSTRUCTION_INVENTORY = """Du bist ein OCR- und Information-Extraction-Modell für deutsche historische Inventardokumente.

Aufgabe:
Lies ALLE Informationen aus dem Bild dieses Inventarblatts und gib GENAU EIN JSON-Objekt zurück.

Das JSON MUSS folgende Felder enthalten:
- Überschrift: Dokumenttitel
- Inventarnummer: Inventar- oder Katalognummer
- Maße: Objekt mit L, B, D (Länge, Breite, Tiefe)
- Objektbezeichnung: Beschreibung/Name des Objekts
- Fundort: Fundort des Objekts
- Fundzeit: Zeit der Auffindung
- Beschreibungstext: Ausführlicher Beschreibungstext

Regeln:
- Gib NUR ein gültiges JSON-Objekt zurück (kein extra Text davor oder danach).
- Verwende GENAU diese Feldnamen und Groß-/Kleinschreibung.
- Wenn ein Feld leer ist oder nicht sichtbar, gib einen leeren String "" zurück.
- Das Feld "Maße" MUSS immer ein Objekt mit den Schlüsseln "L", "B", "D" sein, auch wenn leer.
- Erfinde keine zusätzlichen Felder."""

INSTRUCTION_SCHMUCK = """Extract all information from this German jewelry catalog document image as a structured JSON object.

The JSON should contain these fields:
- Gegenstand: Object/item name
- Inv.Nr: Inventory number
- Herkunft: Origin/provenance
- Foto Notes: Photo notes
- Standort: Location
- Material: Material description
- Datierung: Dating/time period
- Maße: Measurements
- Gewicht: Weight
- erworben von: Acquired from
- am: Acquired on (date)
- Preis: Price
- Vers.-Wert: Insurance value
- Beschreibung: Description
- Literatur: Literature references
- Ausstellungen: Exhibitions

Return ONLY the JSON object, properly formatted."""

INSTRUCTION_STAIRCASE = """You are an OCR model for historical staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this form type (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
- Never drop a key that appears in the form's JSON structure. If a field is empty on the form, still include it with an empty string "" (or false for an unchecked box).
- Use booleans for checkbox options: true if the box is checked, false if it is empty.
- Use strings for numbers and free-text fields (measurements, dates, names, notes).
- Do NOT invent new fields."""

DATASET_INSTRUCTIONS = {
    "inventory": INSTRUCTION_INVENTORY,
    "schmuck": INSTRUCTION_SCHMUCK,
    "staircase": INSTRUCTION_STAIRCASE,
}

# =============================================================================
# UTILS
# =============================================================================

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

def extract_json_from_response(response: str) -> Dict:
    """
    Robust JSON extraction:
    1. Strip markdown fences.
    2. Greedy brace matching.
    3. Walk-back prefix finding.
    4. Regex nested-brace heuristic.
    5. Fallback to whole response.
    """
    if isinstance(response, list):
        response = response[0] if response else ""
    if response is None:
        return {}

    text = str(response).strip()
    if not text:
        return {}

    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.lstrip().startswith("json"):
                body = body[4:].strip()
            text = body.strip()

    def try_parse(candidate: str):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        core = text[start : end + 1]
    else:
        core = text

    parsed = try_parse(core)
    if parsed is not None:
        return parsed

    import re as _re
    brace_positions = [m.start() for m in _re.finditer(r"\}", core)]
    for pos in reversed(brace_positions):
        candidate = core[: pos + 1]
        parsed = try_parse(candidate)
        if parsed is not None:
            return parsed

    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, core, re.DOTALL)
    if matches:
        for match in sorted(matches, key=len, reverse=True):
            parsed = try_parse(match)
            if parsed is not None:
                return parsed

    parsed = try_parse(text)
    if parsed is not None:
        return parsed

    return {}

def json_to_readable_string(obj: Dict, exclude_keys: Optional[List[str]] = None) -> str:
    if exclude_keys:
        obj = {k: v for k, v in obj.items() if k not in exclude_keys}
    return json.dumps(obj, ensure_ascii=False, indent=2)

DATASET_EXCLUDE_KEYS = {
    "inventory": ["image_name", "image_path"],
    "schmuck": ["file_name", "image_path"],
    "staircase": ["image_name", "image_path"],
}

DATASET_IMAGE_KEYS = {
    "inventory": "image_name",
    "schmuck": "file_name",
    "staircase": "image_name",
}

def create_label_string(item: Dict, dataset_name: str) -> str:
    exclude = DATASET_EXCLUDE_KEYS[dataset_name]
    return json_to_readable_string(item, exclude_keys=exclude)

def recover_existing_best_model(run_dir: str, cer_callback=None) -> bool:
    best_dir = os.path.join(run_dir, "best_model")
    meta_path = os.path.join(best_dir, "best_meta.json")
    if not os.path.isdir(best_dir):
        return False

    if cer_callback is not None:
        cer_callback.best_checkpoint_path = best_dir
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cer_callback.best_cer = float(meta.get("best_global_cer", cer_callback.best_cer))
            except Exception:
                pass

    print(f"✅ Using existing best_model from previous run: {best_dir}")
    return True

# =============================================================================
# DATA PREPARATION (Unsloth conversation format)
# =============================================================================

def prepare_conversation_data(
    jsonl_path: str,
    images_dir: str,
    dataset_name: str,
    instruction: str,
) -> List[Dict]:
    data = load_jsonl(jsonl_path)
    image_key = DATASET_IMAGE_KEYS[dataset_name]

    converted = []
    valid = 0
    for item in data:
        img_name = item.get(image_key, None)
        if img_name is None:
            continue

        image_path = os.path.join(images_dir, img_name)
        if not os.path.exists(image_path):
            continue

        gt_json_str = create_label_string(item, dataset_name)

        conversation = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": instruction},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": gt_json_str},
                    ],
                },
            ]
        }
        converted.append(conversation)
        valid += 1

    print(f"   [{dataset_name}] {valid} valid / {len(data)} total")
    return converted

# =============================================================================
# MULTI-DATASET CER CALLBACK
# =============================================================================

class MultiDatasetCERCallback(TrainerCallback):
    def __init__(
        self,
        model,
        tokenizer,
        val_sets: Dict[str, List[Dict]],
        images_dirs: Dict[str, str],
        image_keys: Dict[str, str],
        instructions: Dict[str, str],
        exclude_keys_map: Dict[str, List[str]],
        output_dir: str,
        max_new_tokens: int = 1024,
        flush_interval: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.val_sets = val_sets
        self.images_dirs = images_dirs
        self.image_keys = image_keys
        self.instructions = instructions
        self.exclude_keys_map = exclude_keys_map
        self.output_dir = output_dir
        self.max_new_tokens = max_new_tokens
        self.flush_interval = max(1, flush_interval)

        self.best_cer = float("inf")
        self.best_checkpoint_path: Optional[str] = None
        self.cer_history: List[Dict] = []

    @staticmethod
    def _mean_cer(preds: List[str], gts: List[str]) -> float:
        if not preds or not gts:
            return 1.0
        total = 0.0
        n = 0
        for p, g in zip(preds, gts):
            if g is None:
                continue
            g = str(g)
            p = "" if p is None else str(p)
            if len(g) == 0 and len(p) == 0:
                n += 1
            elif len(g) == 0:
                total += 1.0
                n += 1
            else:
                total += jiwer.cer(g, p)
                n += 1
        return total / max(n, 1)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if state.epoch is None:
            return control

        epoch_num = int(state.epoch)

        print("\n" + "=" * 80)
        print(f"🔍 MULTI-DATASET CER VALIDATION (Epoch {epoch_num})")
        print("=" * 80)

        FastVisionModel.for_inference(self.model)
        self.model.eval()
        device = next(self.model.parameters()).device

        global_preds: List[str] = []
        global_gts: List[str] = []
        per_dataset_cer: Dict[str, float] = {}

        for ds_name, val_items in self.val_sets.items():
            print(f"\n--- Dataset: {ds_name} ({len(val_items)} samples) ---")

            preds, gts = [], []
            images_dir = self.images_dirs[ds_name]
            image_key = self.image_keys[ds_name]
            instruction = self.instructions[ds_name]
            exclude_keys = self.exclude_keys_map[ds_name]

            for i, item in enumerate(val_items):
                img_id = item.get(image_key, None)
                if img_id is None:
                    continue

                image_path = os.path.join(images_dir, img_id)
                if not os.path.exists(image_path):
                    continue

                gt_str = json_to_readable_string(item, exclude_keys=exclude_keys)
                image = None

                try:
                    image = Image.open(image_path).convert("RGB")

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ]

                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to(device)

                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=False,
                            repetition_penalty=1.0,
                            use_cache=True,
                        )

                    input_len = inputs["input_ids"].shape[-1]
                    gen_ids = outputs[0][input_len:]
                    raw_output = self.tokenizer.decode(
                        gen_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    pred_json = extract_json_from_response(raw_output)
                    pred_str = json_to_readable_string(pred_json) if pred_json else ""

                    preds.append(pred_str)
                    gts.append(gt_str)

                except Exception as e:
                    print(f"\n   ⚠️  Validation error [{ds_name}] on {img_id}: {e}")
                    preds.append("")
                    gts.append(gt_str)

                finally:
                    if image is not None:
                        image.close()
                        del image
                    for _v in ["inputs", "outputs", "gen_ids"]:
                        if _v in locals():
                            del locals()[_v]
                    if (i + 1) % self.flush_interval == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

            ds_cer = self._mean_cer(preds, gts)
            per_dataset_cer[ds_name] = ds_cer

            global_preds.extend(preds)
            global_gts.extend(gts)

            if metrics is not None:
                metrics[f"eval_cer_{ds_name}"] = float(ds_cer)

            print(f"\n   ✅ CER [{ds_name}] = {ds_cer:.4f} ({ds_cer*100:.2f}%)")

        global_cer = self._mean_cer(global_preds, global_gts)
        if metrics is not None:
            metrics["eval_cer"] = float(global_cer)

        self.cer_history.append({
            "epoch": epoch_num,
            "global_cer": float(global_cer),
            "per_dataset": {k: float(v) for k, v in per_dataset_cer.items()},
        })

        print("\n" + "-" * 80)
        print(f"🌐 Global CER (all samples combined) = {global_cer:.4f} ({global_cer*100:.2f}%)")
        for ds_name, cer_val in per_dataset_cer.items():
            print(f"   • {ds_name}: {cer_val:.4f} ({cer_val*100:.2f}%)")
        print("-" * 80)

        if global_cer < self.best_cer:
            improvement = self.best_cer - global_cer
            self.best_cer = float(global_cer)

            best_model_path = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)

            print(f"   🎯 NEW BEST GLOBAL CER = {self.best_cer:.4f} (improved by {improvement:.4f})")
            print(f"   💾 Saving best adapter to: {best_model_path}")

            self.model.save_pretrained(best_model_path)
            try:
                self.tokenizer.save_pretrained(best_model_path)
                print("   ✅ Adapter + tokenizer saved")
            except Exception as e:
                print(f"   ⚠️  Tokenizer save warning: {e}")

            meta = {
                "best_global_cer": float(self.best_cer),
                "best_epoch": epoch_num,
                "timestamp": datetime.now().isoformat(),
            }
            with open(os.path.join(best_model_path, "best_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            self.best_checkpoint_path = best_model_path
        else:
            print(f"   📊 No improvement. Best remains: {self.best_cer:.4f}")

        print("=" * 80 + "\n")

        FastVisionModel.for_training(self.model)
        self.model.train()
        return control

# =============================================================================
# TEST-TIME EVALUATION
# =============================================================================

def evaluate_on_dataset(
    dataset_name: str,
    model,
    tokenizer,
    test_jsonl: str,
    images_dir: str,
    image_key: str,
    instruction: str,
    exclude_keys: List[str],
    output_dir: str,
    max_new_tokens: int = 1024,
    flush_interval: int = 5,
) -> Dict:
    print("\n" + "=" * 80)
    print(f"TEST INFERENCE - {dataset_name.upper()}")
    print("=" * 80)

    test_data = load_jsonl(test_jsonl)
    print(f"📊 Test samples: {len(test_data)}")

    FastVisionModel.for_inference(model)
    model.eval()
    device = next(model.parameters()).device

    results = []
    all_cer_scores = []

    chunk_size = flush_interval
    num_chunks = (len(test_data) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(test_data))
        chunk_data = test_data[start_idx:end_idx]

        for i, item in enumerate(chunk_data):
            abs_idx = start_idx + i + 1
            img_id = item.get(image_key, None)
            if img_id is None:
                continue

            print(f"   [{abs_idx}/{len(test_data)}] {img_id}", end="")

            image_path = os.path.join(images_dir, img_id)
            gt_str = json_to_readable_string(item, exclude_keys=exclude_keys)

            if not os.path.exists(image_path):
                results.append({
                    image_key: img_id,
                    "predicted_text": "",
                    "ground_truth_text": gt_str,
                    "cer_score": 1.0,
                    "error": "Image not found",
                })
                all_cer_scores.append(1.0)
                print(" ⚠️ not found")
                continue

            image = None
            try:
                image = Image.open(image_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": instruction},
                        ],
                    }
                ]

                inputs = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.0,
                        use_cache=True,
                    )

                input_len = inputs["input_ids"].shape[-1]
                gen_ids = outputs[0][input_len:]
                raw_output = tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                pred_json = extract_json_from_response(raw_output)
                pred_str = json_to_readable_string(pred_json) if pred_json else ""

                cer = jiwer.cer(gt_str, pred_str) if (gt_str and pred_str) else 1.0
                all_cer_scores.append(cer)

                status = "✨" if cer == 0.0 else "✅" if cer < 0.1 else "⚠️" if cer < 0.3 else "❌"
                print(f"  {status} CER: {cer:.4f}")

                results.append({
                    image_key: img_id,
                    "predicted_text": pred_str,
                    "ground_truth_text": gt_str,
                    "raw_output": raw_output,
                    "cer_score": float(cer),
                })

            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    image_key: img_id,
                    "predicted_text": "",
                    "ground_truth_text": gt_str,
                    "cer_score": 1.0,
                    "error": str(e),
                })
                all_cer_scores.append(1.0)

            finally:
                if image is not None:
                    image.close()
                    del image
                for _v in ["inputs", "outputs", "gen_ids"]:
                    if _v in locals():
                        del locals()[_v]
                torch.cuda.empty_cache()
                gc.collect()

        if (chunk_idx + 1) % 3 == 0 or chunk_idx == num_chunks - 1:
            intermediate_file = os.path.join(
                output_dir, f"test_predictions_{dataset_name}_chunk_{chunk_idx+1}.jsonl"
            )
            save_jsonl(results, intermediate_file)

    results_file = os.path.join(output_dir, f"test_predictions_{dataset_name}.jsonl")
    save_jsonl(results, results_file)

    avg_cer = float(np.mean(all_cer_scores)) if all_cer_scores else 1.0
    median_cer = float(np.median(all_cer_scores)) if all_cer_scores else 1.0
    std_cer = float(np.std(all_cer_scores)) if all_cer_scores else 0.0
    n = len(all_cer_scores)
    perfect_matches = sum(1 for c in all_cer_scores if c == 0.0)
    good_matches = sum(1 for c in all_cer_scores if c < 0.1)

    print(f"\n{'='*80}")
    print(f"📊 FINAL TEST RESULTS - {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"   Total Samples: {n}")
    print(f"   Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"   Median CER: {median_cer:.4f}")
    print(f"   Std CER: {std_cer:.4f}")
    print(f"   Perfect Matches (CER=0): {perfect_matches}/{n}")
    print(f"   Good Matches (CER<0.1): {good_matches}/{n}")
    print(f"   Results: {results_file}")
    print(f"{'='*80}")

    summary_file = os.path.join(output_dir, f"summary_{dataset_name}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"GEMMA-3 {dataset_name.upper()} TEST RESULTS (MULTI-DATASET MODEL)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)\n")
        f.write(f"Median CER: {median_cer:.4f}\n")
        f.write(f"Std CER: {std_cer:.4f}\n")
        f.write(f"Perfect Matches: {perfect_matches}/{n}\n")
        f.write(f"Good Matches (CER<0.1): {good_matches}/{n}\n")

    return {
        "avg_cer": avg_cer,
        "median_cer": median_cer,
        "std_cer": std_cer,
        "perfect_matches": perfect_matches,
        "good_matches": good_matches,
        "total": n,
    }

# =============================================================================
# ARGPARSE
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir", type=str, default=None,
        help="Existing run directory to resume from. If not set, creates a new run.",
    )
    p.add_argument(
        "--cleanup_checkpoints", action="store_true",
        help="Delete train_state/checkpoint-* folders after training completes (keeps best_model + logs).",
    )
    return p.parse_args()

# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("GEMMA-3 FINE-TUNING - MULTI-DATASET (RESUME + BEST GLOBAL CER)")
    print("(INVENTORY + SCHMUCK + STAIRCASE)")
    print("Single model, Unsloth 4-bit QLoRA, BEST by epoch-wise autoregressive CER")
    print("=" * 80)

    base_dir = CONFIG["output_base_dir"]

    if args.run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, f"run_{timestamp}_multidataset_bestCER")
        os.makedirs(run_dir, exist_ok=True)
        is_new_run = True
        print(f"\n🆕 New run directory: {run_dir}")
    else:
        run_dir = args.run_dir
        if not os.path.isdir(run_dir):
            raise ValueError(f"--run_dir does not exist: {run_dir}")
        is_new_run = False
        print(f"\n🔁 Resuming run directory: {run_dir}")

    CONFIG["output_dir"] = run_dir

    cfg_path = os.path.join(run_dir, "training_config.json")
    if is_new_run:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print(f"   • epochs: {CONFIG['num_epochs']}")
    print(f"   • batch_size: {CONFIG['batch_size']}")
    print(f"   • grad_accum: {CONFIG['gradient_accumulation_steps']}")
    print(f"   • lr: {CONFIG['learning_rate']}")
    print(f"   • LoRA: r={CONFIG['lora_r']} alpha={CONFIG['lora_alpha']} dropout={CONFIG['lora_dropout']}")
    print(f"   • save_strategy: {CONFIG['save_strategy']} (keep {CONFIG['save_total_limit']})")
    print(f"   • best adapter saved to: best_model/")
    print(f"   • tensorboard: {os.path.join(run_dir, 'tb_logs')}")

    cache_dir = "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/gemma3/unsloth_compiled_cache"
    if os.path.exists(cache_dir):
        print(f"\n🧹 Clearing Unsloth compiled cache: {cache_dir}")
        shutil.rmtree(cache_dir)

    print(f"\n⏳ Loading Gemma-3 vision model with Unsloth 4-bit...")
    model, tokenizer = FastVisionModel.from_pretrained(
        CONFIG["model_path"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("   ✅ Model loaded")

    model = FastVisionModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
    )
    print("   ✅ LoRA applied (Unsloth)")

    ds_cfg = CONFIG["datasets"]

    print("\n📊 Preparing combined training data...")
    train_datasets = []
    val_datasets = []

    for ds_name in ["inventory", "schmuck", "staircase"]:
        cfg = ds_cfg[ds_name]
        inst = DATASET_INSTRUCTIONS[ds_name]

        print(f"\n   📁 {ds_name.upper()} train:")
        train_data = prepare_conversation_data(
            cfg["train_jsonl"], cfg["images_dir"], ds_name, inst,
        )
        train_datasets.extend(train_data)

        print(f"   📁 {ds_name.upper()} val:")
        val_data = prepare_conversation_data(
            cfg["val_jsonl"], cfg["images_dir"], ds_name, inst,
        )
        val_datasets.extend(val_data)

    print(f"\n   📊 TOTAL combined train samples: {len(train_datasets)}")
    print(f"   📊 TOTAL combined val samples: {len(val_datasets)}")

    dummy_n = 3
    dummy_eval = val_datasets[:min(dummy_n * 3, len(val_datasets))]
    print(f"   📊 Dummy eval samples (for Trainer eval_loss): {len(dummy_eval)}")

    max_val = CONFIG["max_val_samples_per_dataset"]
    val_sets = {
        ds_name: load_jsonl(ds_cfg[ds_name]["val_jsonl"])[:max_val]
        for ds_name in ["inventory", "schmuck", "staircase"]
    }
    images_dirs = {ds_name: ds_cfg[ds_name]["images_dir"] for ds_name in ds_cfg}
    image_keys = {ds_name: ds_cfg[ds_name]["image_key"] for ds_name in ds_cfg}

    cer_callback = MultiDatasetCERCallback(
        model=model,
        tokenizer=tokenizer,
        val_sets=val_sets,
        images_dirs=images_dirs,
        image_keys=image_keys,
        instructions=DATASET_INSTRUCTIONS,
        exclude_keys_map=DATASET_EXCLUDE_KEYS,
        output_dir=run_dir,
        max_new_tokens=CONFIG["gen_max_new_tokens"],
        flush_interval=CONFIG["test_chunk_flush_interval"],
    )

    if not is_new_run:
        recover_existing_best_model(run_dir, cer_callback)

    FastVisionModel.for_training(model)

    train_output_dir = os.path.join(run_dir, "train_state")
    os.makedirs(train_output_dir, exist_ok=True)

    trainer = SafeRNGResumeSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_datasets,
        eval_dataset=dummy_eval,
        compute_metrics=None,
        args=SFTConfig(
            per_device_train_batch_size=CONFIG["batch_size"],
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            eval_accumulation_steps=4,
            batch_eval_metrics=True,
            warmup_steps=100,
            num_train_epochs=CONFIG["num_epochs"],
            learning_rate=CONFIG["learning_rate"],
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy=CONFIG["save_strategy"],
            save_steps=CONFIG["save_steps"],
            save_total_limit=CONFIG["save_total_limit"],
            max_grad_norm=CONFIG["max_grad_norm"],
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            weight_decay=CONFIG["weight_decay"],
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            optim="adamw_torch_fused",
            load_best_model_at_end=False,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=CONFIG["max_seq_length"],
            report_to="tensorboard",
            logging_dir=os.path.join(run_dir, "tb_logs"),
            logging_first_step=True,
            seed=3407,
            output_dir=train_output_dir,
            save_safetensors=True,
            prediction_loss_only=True,
            disable_tqdm=False,
            label_smoothing_factor=CONFIG["label_smoothing_factor"],
        ),
        callbacks=[cer_callback],
    )

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        print(f"\n{'='*80}")
        print("GPU INFORMATION")
        print(f"{'='*80}")
        print(f"   GPU: {gpu_stats.name}")
        print(f"   Max memory: {max_memory} GB")
        print(f"   Reserved: {start_gpu_memory} GB")
        print(f"{'='*80}\n")

    last_ckpt = None
    if os.path.isdir(train_output_dir):
        last_ckpt = get_last_checkpoint(train_output_dir)

    print("\n" + "=" * 80)
    if last_ckpt is not None:
        print(f"🔁 Resuming training from: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("🚀 Starting multi-dataset Gemma-3 QLoRA training from scratch...")
        trainer.train()
    print("=" * 80)

    if args.cleanup_checkpoints:
        for ckpt_dir in glob.glob(os.path.join(train_output_dir, "checkpoint-*")):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        print("🧹 Deleted train_state/checkpoint-* folders (kept best_model + logs).")

    if not cer_callback.best_checkpoint_path:
        recover_existing_best_model(run_dir, cer_callback)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED - MULTI-DATASET MODEL")
    print("=" * 80)
    print(f"   🎯 Best GLOBAL Validation CER: {cer_callback.best_cer:.4f} ({cer_callback.best_cer*100:.2f}%)")
    print(f"   💾 Best adapter saved at: {cer_callback.best_checkpoint_path}")
    print("=" * 80)

    if not cer_callback.best_checkpoint_path:
        raise RuntimeError("No best model was saved and no existing best_model/ found.")

    print("\n" + "=" * 80)
    print("PHASE 2: PER-DATASET EVALUATION ON TEST SETS")
    print("=" * 80)

    print(f"\n⏳ Loading best adapter weights from: {cer_callback.best_checkpoint_path}")

    import safetensors.torch
    adapter_file = os.path.join(cer_callback.best_checkpoint_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        adapter_file = os.path.join(cer_callback.best_checkpoint_path, "adapter_model.bin")
        adapter_state = torch.load(adapter_file, map_location="cpu")
    else:
        adapter_state = safetensors.torch.load_file(adapter_file)

    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, adapter_state)
    del adapter_state
    torch.cuda.empty_cache()
    gc.collect()
    print("   ✅ Best adapter weights loaded into existing Unsloth model")

    flush_interval = CONFIG["test_chunk_flush_interval"]
    test_results = {}

    for ds_name in ["inventory", "schmuck", "staircase"]:
        ds = ds_cfg[ds_name]
        result = evaluate_on_dataset(
            dataset_name=ds_name,
            model=model,
            tokenizer=tokenizer,
            test_jsonl=ds["test_jsonl"],
            images_dir=ds["images_dir"],
            image_key=ds["image_key"],
            instruction=DATASET_INSTRUCTIONS[ds_name],
            exclude_keys=DATASET_EXCLUDE_KEYS[ds_name],
            output_dir=run_dir,
            max_new_tokens=CONFIG["gen_max_new_tokens"],
            flush_interval=flush_interval,
        )
        test_results[ds_name] = result

    summary_file = os.path.join(run_dir, "training_summary_global.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GEMMA-3 MULTI-DATASET FINE-TUNING RESULTS (BEST BY CER)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best GLOBAL Validation CER: {cer_callback.best_cer:.4f} ({cer_callback.best_cer*100:.2f}%)\n\n")
        f.write("Test Results per Dataset:\n")
        f.write("-" * 50 + "\n")
        for ds_name, res in test_results.items():
            f.write(f"  {ds_name.upper()}:\n")
            f.write(f"    Average CER: {res['avg_cer']:.4f} ({res['avg_cer']*100:.2f}%)\n")
            f.write(f"    Median CER: {res['median_cer']:.4f}\n")
            f.write(f"    Perfect Matches: {res['perfect_matches']}/{res['total']}\n")
            f.write(f"    Good Matches (CER<0.1): {res['good_matches']}/{res['total']}\n\n")
        f.write("\nCER History (per epoch):\n")
        f.write("-" * 50 + "\n")
        for entry in cer_callback.cer_history:
            epoch = entry["epoch"]
            gcer = entry["global_cer"]
            f.write(f"  Epoch {epoch}: Global CER={gcer:.4f} ({gcer*100:.2f}%) ")
            for ds_name, ds_cer in entry["per_dataset"].items():
                f.write(f"| {ds_name}: {ds_cer:.4f} ({ds_cer*100:.2f}%) ")
            f.write("\n")

    csv_file = os.path.join(run_dir, "per_epoch_cer.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        header = ["epoch", "global", "inventory", "schmuck", "staircase"]
        f.write(",".join(header) + "\n")
        for entry in cer_callback.cer_history:
            epoch = entry["epoch"]
            gcer = entry["global_cer"]
            per_ds = entry["per_dataset"]
            inv = per_ds.get("inventory", float("nan"))
            sch = per_ds.get("schmuck", float("nan"))
            sta = per_ds.get("staircase", float("nan"))
            f.write(f"{epoch},{gcer:.6f},{inv:.6f},{sch:.6f},{sta:.6f}\n")

    cer_hist_file = os.path.join(run_dir, "cer_history.json")
    with open(cer_hist_file, "w", encoding="utf-8") as f:
        json.dump(cer_callback.cer_history, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Global summary saved to: {summary_file}")
    print(f"✅ Per-epoch CER CSV saved to: {csv_file}")
    print(f"✅ CER history JSON saved to: {cer_hist_file}")
    print(f"\nAll outputs saved under:\n  {run_dir}")
    print("\n🎉 Gemma-3 multi-dataset fine-tuning + per-dataset test evaluation complete!\n")

if __name__ == "__main__":
    main()