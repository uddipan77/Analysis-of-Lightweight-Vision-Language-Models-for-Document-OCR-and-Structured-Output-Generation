#!/usr/bin/env python3
# phi_multidataset_bestCER_resume_tb.py
#
# ✅ Multi-dataset fine-tuning for Phi-3.5-Vision (QLoRA 4-bit) on:
#    - Inventory
#    - Schmuck
#    - Staircase
#
# ✅ One shared base model + one shared LoRA adapter trained on combined TRAIN of all datasets.
# ✅ Best model selected by LOWEST GLOBAL validation CER using autoregressive generation (each epoch).
# ✅ Saves ONLY ONE best adapter folder: <run_dir>/best_model
# ✅ TensorBoard logging enabled (<run_dir>/tb_logs)
# ✅ Uses tiny dummy eval set only to trigger epoch-end evaluation callback cheaply.
#
# ✅ Resume logic:
#    - Keeps ONE rolling checkpoint (save_total_limit=1) for resume safety.
#    - Resume via: python <script> --run_dir <existing_run_dir>
#    - Automatically resumes from last checkpoint if present.
#    - Optional: --cleanup_checkpoints to delete checkpoint folder after training completes.
#
# IMPORTANT: Best model is a PEFT adapter; at test time we load base model + attach adapter (PeftModel.from_pretrained).
#
# First run:
#   python phi_multidataset2.py
#
# Resume after timeout:
#   python phi_multidataset2.py --run_dir /path/to/existing/run_dir
#
# TensorBoard:
#   tensorboard --logdir <run_dir>/tb_logs

import os
import json
import glob
import shutil
import random
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Callable, Optional

import torch
import unicodedata
import jiwer
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, ConcatDataset, Subset

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision",
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
    "output_base_dir": "/home/vault/iwi5/iwi5298h/models_image_text/phi/general",

    # Training hyperparams
    "num_epochs": 15,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "weight_decay": 0.05,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "max_grad_norm": 1.0,
    "data_augmentation": True,

    # Quantization
    "use_4bit": True,
    "use_nested_quant": True,

    # CER validation
    "max_val_samples_per_dataset": 30,
    "val_cache_flush_interval": 4,   # flush CUDA cache every N val samples (stability)

    # Test-time CUDA cache flush interval
    "test_chunk_flush_interval": 3,

    # Dummy eval size to trigger Trainer eval cheaply (epoch-end)
    "dummy_eval_size_per_dataset": 3,

    # Resume-friendly rolling checkpoint (keeps only ONE for resume safety)
    "save_strategy": "steps",
    "save_steps": 500,           # save every ~500 steps; tune to ~15-30 min intervals
    "save_total_limit": 1,       # keep only latest checkpoint on disk
}

# =============================================================================
# INSTRUCTIONS PER DATASET
# =============================================================================

INSTRUCTION_INVENTORY = """You are an OCR model for German historical inventory documents.

Task:
Given ONE image of an inventory document, read all printed and handwritten text and output a single JSON object that represents the complete document.

The JSON must have EXACTLY the following structure with these German field names:

{
  "Überschrift": "Document heading/title",
  "Inventarnummer": "Inventory number",
  "Maße": {
    "L": "Length value",
    "B": "Breadth/width value",
    "D": "Depth value"
  },
  "Objektbezeichnung": "Object description/name",
  "Fundort": "Find location/origin",
  "Fundzeit": "Find time/dating",
  "Beschreibungstext": "Main descriptive text (full transcription)"
}

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly these field names with exact German spelling and capitalization.
- Include ALL fields in EVERY response, even if the field has no visible text in the image.
- If a field is empty or not visible, use an empty string "" for that field.
- The "Maße" field must ALWAYS be an object with the keys "L", "B", and "D", even if they are empty.
- Use strings for all values (including numbers and measurements).
- Do NOT invent new fields or add any extra keys.
- Do NOT add comments, explanations, or prose around the JSON."""

INSTRUCTION_SCHMUCK = """Extract all information from this German historical jewelry/schmuck catalog image and return ONLY a JSON object with exactly these keys:

{
  "Gegenstand": "Object/item type",
  "Inv.Nr": "Inventory number (e.g., Sch 3051)",
  "Herkunft": "Origin/provenance",
  "Foto Notes": "Photo notes/number",
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

INSTRUCTION_STAIRCASE = """You are an OCR model for historical staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this dataset (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
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

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def load_jsonl(file_path: str) -> List[Dict]:
    data: List[Dict] = []
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


def extract_json_from_response(response: str) -> str:
    """Extract FIRST complete JSON object from response; fallback to raw substring/text."""
    response = response.strip()
    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]

        # Try isolate first complete JSON object
        try:
            brace_count = 0
            first_end = -1
            for i, ch in enumerate(json_str):
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        first_end = i + 1
                        break
            if first_end > 0:
                first_json = json_str[:first_end]
                parsed = json.loads(first_json)
                return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        except Exception:
            pass

        # Fallback parse full substring
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        except Exception:
            return json_str

    return response


# ---- label string creators per dataset --------------------------------------

def create_label_string_inventory(json_data: Dict) -> str:
    label_data = {k: v for k, v in json_data.items() if k != "image_name"}
    return json.dumps(label_data, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def create_label_string_schmuck(json_data: Dict) -> str:
    label_data = {k: v for k, v in json_data.items() if k != "file_name"}
    return json.dumps(label_data, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


def create_label_string_staircase(json_data: Dict) -> str:
    label_data = {k: v for k, v in json_data.items() if k != "image_name"}
    return json.dumps(label_data, ensure_ascii=False, separators=(",", ":"), sort_keys=False)


LABEL_FNS: Dict[str, Callable[[Dict], str]] = {
    "inventory": create_label_string_inventory,
    "schmuck": create_label_string_schmuck,
    "staircase": create_label_string_staircase,
}

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

class DocumentImageAugmenter:
    """Light augmentation for document images."""
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def augment(self, image: Image.Image) -> Image.Image:
        if not self.enabled or random.random() > 0.7:
            return image

        if random.random() > 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() > 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
        if random.random() > 0.5:
            image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.9, 1.1))
        if random.random() > 0.7:
            angle = random.uniform(-2, 2)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        return image

# =============================================================================
# DATASETS
# =============================================================================

class InventoryDataset(Dataset):
    def __init__(self, jsonl_path, images_dir, processor, instruction, augment=False):
        self.data = load_jsonl(jsonl_path)
        self.images_dir = images_dir
        self.processor = processor
        self.instruction = instruction
        self.augmenter = DocumentImageAugmenter(enabled=augment)

        self.valid_samples = []
        for item in self.data:
            image_path = os.path.join(self.images_dir, item["image_name"])
            if os.path.exists(image_path):
                self.valid_samples.append(item)

        print(f"   [inventory] Loaded {len(self.valid_samples)} valid samples "
              f"(out of {len(self.data)} total)")
        if augment:
            print("   [inventory] Data augmentation ENABLED")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        image_path = os.path.join(self.images_dir, item["image_name"])
        image = Image.open(image_path).convert("RGB")
        image = self.augmenter.augment(image)

        gt_json_str = create_label_string_inventory(item)

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{self.instruction}"},
            {"role": "assistant", "content": gt_json_str},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(prompt, [image], return_tensors="pt")

        labels = inputs["input_ids"].clone()
        assistant_token = self.processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        input_ids_list = inputs["input_ids"][0].tolist()

        try:
            for i in range(len(input_ids_list) - len(assistant_token)):
                if input_ids_list[i:i + len(assistant_token)] == assistant_token:
                    labels[0, : i + len(assistant_token)] = -100
                    break
        except Exception:
            pass

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
            "image_sizes": inputs["image_sizes"].squeeze(0) if "image_sizes" in inputs else None,
            "labels": labels.squeeze(0),
        }


class SchmuckDataset(Dataset):
    def __init__(self, jsonl_path, images_dir, processor, instruction, augment=False):
        self.data = load_jsonl(jsonl_path)
        self.images_dir = images_dir
        self.processor = processor
        self.instruction = instruction
        self.augmenter = DocumentImageAugmenter(enabled=augment)

        self.valid_samples = []
        for item in self.data:
            image_path = os.path.join(self.images_dir, item["file_name"])
            if os.path.exists(image_path):
                self.valid_samples.append(item)

        print(f"   [schmuck] Loaded {len(self.valid_samples)} valid samples "
              f"(out of {len(self.data)} total)")
        if augment:
            print("   [schmuck] Data augmentation ENABLED")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        image_path = os.path.join(self.images_dir, item["file_name"])
        image = Image.open(image_path).convert("RGB")
        image = self.augmenter.augment(image)

        gt_json_str = create_label_string_schmuck(item)

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{self.instruction}"},
            {"role": "assistant", "content": gt_json_str},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(prompt, [image], return_tensors="pt")

        labels = inputs["input_ids"].clone()
        assistant_token = self.processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        input_ids_list = inputs["input_ids"][0].tolist()

        try:
            for i in range(len(input_ids_list) - len(assistant_token)):
                if input_ids_list[i:i + len(assistant_token)] == assistant_token:
                    labels[0, : i + len(assistant_token)] = -100
                    break
        except Exception:
            pass

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
            "image_sizes": inputs["image_sizes"].squeeze(0) if "image_sizes" in inputs else None,
            "labels": labels.squeeze(0),
        }


class StaircaseDataset(Dataset):
    def __init__(self, jsonl_path, images_dir, processor, instruction, augment=False):
        self.data = load_jsonl(jsonl_path)
        self.images_dir = images_dir
        self.processor = processor
        self.instruction = instruction
        self.augmenter = DocumentImageAugmenter(enabled=augment)

        self.valid_samples = []
        for item in self.data:
            image_path = os.path.join(self.images_dir, item["image_name"])
            if os.path.exists(image_path):
                self.valid_samples.append(item)

        print(f"   [staircase] Loaded {len(self.valid_samples)} valid samples "
              f"(out of {len(self.data)} total)")
        if augment:
            print("   [staircase] Data augmentation ENABLED")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        image_path = os.path.join(self.images_dir, item["image_name"])
        image = Image.open(image_path).convert("RGB")
        image = self.augmenter.augment(image)

        gt_json_str = create_label_string_staircase(item)

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{self.instruction}"},
            {"role": "assistant", "content": gt_json_str},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(prompt, [image], return_tensors="pt")

        labels = inputs["input_ids"].clone()
        assistant_token = self.processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        input_ids_list = inputs["input_ids"][0].tolist()

        try:
            for i in range(len(input_ids_list) - len(assistant_token)):
                if input_ids_list[i:i + len(assistant_token)] == assistant_token:
                    labels[0, : i + len(assistant_token)] = -100
                    break
        except Exception:
            pass

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
            "image_sizes": inputs["image_sizes"].squeeze(0) if "image_sizes" in inputs else None,
            "labels": labels.squeeze(0),
        }

# =============================================================================
# DATA COLLATOR
# =============================================================================

@dataclass
class DataCollatorForPhi3Vision:
    processor: AutoProcessor

    def __call__(self, features):
        batch = {
            "input_ids": torch.stack([f["input_ids"] for f in features]),
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }
        if features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        if features[0]["image_sizes"] is not None:
            batch["image_sizes"] = torch.stack([f["image_sizes"] for f in features])
        return batch

# =============================================================================
# MULTI-DATASET CER CALLBACK
# =============================================================================

class MultiDatasetCERCallback(TrainerCallback):
    """
    Runs autoregressive greedy generation CER evaluation on subsets of val data
    for each dataset at the end of each epoch (triggered by Trainer eval).
    Saves ONLY the best adapter to <run_dir>/best_model.
    """
    def __init__(
        self,
        processor,
        val_sets: Dict[str, List[Dict]],
        images_dirs: Dict[str, str],
        image_keys: Dict[str, str],
        instructions: Dict[str, str],
        label_fns: Dict[str, Callable[[Dict], str]],
        output_dir: str,
        flush_interval: int = 4,
    ):
        self.processor = processor
        self.val_sets = val_sets
        self.images_dirs = images_dirs
        self.image_keys = image_keys
        self.instructions = instructions
        self.label_fns = label_fns
        self.output_dir = output_dir
        self.flush_interval = max(1, int(flush_interval))

        self.best_cer = float("inf")
        self.best_checkpoint_path: Optional[str] = None
        self.cer_history = []

    @staticmethod
    def calculate_cer(predictions, targets):
        if not predictions or not targets:
            return 1.0
        total_cer = 0.0
        valid_pairs = 0
        for pred, target in zip(predictions, targets):
            if target is None:
                continue
            if len(target) > 0:
                total_cer += jiwer.cer(target, pred)
                valid_pairs += 1
        return total_cer / valid_pairs if valid_pairs > 0 else 1.0

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        if state.epoch is None:
            return control

        epoch_num = int(state.epoch)

        print("\n" + "=" * 80)
        print(f"🔍 MULTI-DATASET CER VALIDATION (Epoch {epoch_num})")
        print("=" * 80)

        model.eval()
        device = next(model.parameters()).device

        global_predictions = []
        global_targets = []
        per_dataset_cer: Dict[str, float] = {}

        for ds_name, val_items in self.val_sets.items():
            print(f"\n--- Dataset: {ds_name} ---")

            predictions = []
            targets = []

            images_dir = self.images_dirs[ds_name]
            instruction = self.instructions[ds_name]
            image_key = self.image_keys[ds_name]
            label_fn = self.label_fns[ds_name]

            for i, item in enumerate(val_items):
                img_id = item[image_key]
                print(f"   [{ds_name}] Validating {i+1}/{len(val_items)}: {img_id}", end="\r")

                image_path = os.path.join(images_dir, img_id)
                if not os.path.exists(image_path):
                    continue

                try:
                    image = Image.open(image_path).convert("RGB")

                    messages = [{"role": "user", "content": f"<|image_1|>\n{instruction}"}]
                    prompt = self.processor.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )

                    inputs = self.processor(prompt, [image], return_tensors="pt").to(device)

                    with torch.no_grad():
                        generate_ids = model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            temperature=0.0,
                            do_sample=False,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                        )

                    # Strip the prompt portion
                    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

                    raw_output = self.processor.batch_decode(
                        generate_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()

                    raw_output = normalize_unicode(raw_output)
                    prediction = extract_json_from_response(raw_output)
                    ground_truth = label_fn(item)

                    predictions.append(prediction)
                    targets.append(ground_truth)

                    # clean
                    del inputs, generate_ids, image
                    if (i + 1) % self.flush_interval == 0:
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"\n   ⚠️  Validation error on {img_id}: {e}")
                    predictions.append("")
                    targets.append(label_fn(item))
                    continue

            ds_cer = self.calculate_cer(predictions, targets)
            per_dataset_cer[ds_name] = ds_cer

            global_predictions.extend(predictions)
            global_targets.extend(targets)

            if metrics is not None:
                metrics[f"eval_cer_{ds_name}"] = ds_cer

            print(f"\n   ✅ Validation CER [{ds_name}] (Epoch {epoch_num}): {ds_cer:.4f} ({ds_cer*100:.2f}%)")

        global_cer = self.calculate_cer(global_predictions, global_targets)
        if metrics is not None:
            metrics["eval_cer"] = global_cer

        self.cer_history.append({
            "epoch": epoch_num,
            "global_cer": float(global_cer),
            "per_dataset": {k: float(v) for k, v in per_dataset_cer.items()},
        })

        print("\n" + "-" * 80)
        print(f"🌐 Global Validation CER (all datasets combined): {global_cer:.4f} ({global_cer*100:.2f}%)")
        for ds_name, cer_val in per_dataset_cer.items():
            print(f"   • {ds_name}: {cer_val:.4f} ({cer_val*100:.2f}%)")
        print("-" * 80)

        if global_cer < self.best_cer:
            improvement = self.best_cer - global_cer
            self.best_cer = float(global_cer)

            print(f"   🎯 NEW BEST GLOBAL CER: {self.best_cer:.4f} (improved by {improvement:.4f})")

            best_model_path = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)

            print(f"   💾 Saving best adapter to: {best_model_path}")
            # This saves the PEFT adapter (because model is a PeftModel)
            model.save_pretrained(best_model_path)

            # Save tokenizer as well (optional but handy)
            try:
                self.processor.tokenizer.save_pretrained(best_model_path)
                print("   ✅ Adapter + tokenizer saved")
            except Exception as e:
                print(f"   ⚠️  Tokenizer save warning: {e}")

            # Save metadata about best epoch/score
            meta = {
                "best_global_cer": float(self.best_cer),
                "best_epoch": int(epoch_num),
                "timestamp": datetime.now().isoformat(),
            }
            with open(os.path.join(best_model_path, "best_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            self.best_checkpoint_path = best_model_path
        else:
            print(f"   📊 No improvement. Best global CER remains: {self.best_cer:.4f}")

        print("=" * 80 + "\n")
        model.train()
        return control

# =============================================================================
# TEST EVALUATION
# =============================================================================

def evaluate_on_dataset(
    dataset_name: str,
    model,
    processor,
    test_jsonl: str,
    images_dir: str,
    image_key: str,
    instruction: str,
    label_fn,
    output_dir: str,
    flush_interval: int = 3,
):
    print("\n" + "=" * 80)
    print(f"EVALUATION ON TEST SET - {dataset_name.upper()}")
    print("=" * 80)

    test_data = load_jsonl(test_jsonl)
    print(f"\n📊 Running inference on {len(test_data)} test samples for [{dataset_name}]...\n")

    results = []
    cer_scores = []

    model.eval()
    device = next(model.parameters()).device

    for idx, item in enumerate(test_data):
        img_id = item[image_key]
        print(f"[{idx+1}/{len(test_data)}] Processing {img_id}")

        image_path = os.path.join(images_dir, img_id)
        if not os.path.exists(image_path):
            print("   ❌ Image not found")
            results.append({
                image_key: img_id,
                "predicted_text": "",
                "ground_truth_text": label_fn(item),
                "cer_score": 1.0,
                "error": "Image not found",
            })
            cer_scores.append(1.0)
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            messages = [{"role": "user", "content": f"<|image_1|>\n{instruction}"}]
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(prompt, [image], return_tensors="pt").to(device)

            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

            raw_output = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            raw_output = normalize_unicode(raw_output)
            prediction = extract_json_from_response(raw_output)
            ground_truth = label_fn(item)

            cer = (
                jiwer.cer(ground_truth, prediction)
                if len(ground_truth) > 0
                else (1.0 if len(prediction) > 0 else 0.0)
            )
            cer_scores.append(cer)

            status = "✨" if cer == 0.0 else "✅" if cer < 0.1 else "⚠️" if cer < 0.3 else "❌"
            print(f"   {status} CER: {cer:.4f} ({cer*100:.2f}%)")

            results.append({
                image_key: img_id,
                "predicted_text": prediction,
                "ground_truth_text": ground_truth,
                "raw_output": raw_output,
                "cer_score": float(cer),
            })

            del inputs, generate_ids, image
            if (idx + 1) % flush_interval == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                image_key: img_id,
                "predicted_text": "",
                "ground_truth_text": label_fn(item),
                "cer_score": 1.0,
                "error": str(e),
            })
            cer_scores.append(1.0)

    results_file = os.path.join(output_dir, f"test_predictions_{dataset_name}.jsonl")
    save_jsonl(results, results_file)

    avg_cer = float(sum(cer_scores) / len(cer_scores)) if cer_scores else 1.0
    perfect_matches = int(sum(1 for c in cer_scores if c == 0.0))
    good_matches = int(sum(1 for c in cer_scores if c < 0.1))
    n = int(len(cer_scores)) if cer_scores else 1

    print(f"\n{'='*80}")
    print(f"📊 FINAL TEST RESULTS - {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    print(f"   Total Samples: {len(test_data)}")
    print(f"   Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"   Perfect Matches (CER=0): {perfect_matches}/{n} ({perfect_matches/n*100:.1f}%)")
    print(f"   Good Matches (CER<0.1): {good_matches}/{n} ({good_matches/n*100:.1f}%)")
    print(f"   Results saved to: {results_file}")
    print(f"{'='*80}")

    summary_file = os.path.join(output_dir, f"summary_{dataset_name}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"PHI-3.5-VISION {dataset_name.upper()} TEST RESULTS (MULTI-DATASET MODEL)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Test CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)\n")
        f.write(f"Median CER: {float(np.median(cer_scores)):.4f}\n")
        f.write(f"Std CER: {float(np.std(cer_scores)):.4f}\n")
        f.write(f"Perfect Matches: {perfect_matches}/{n} ({perfect_matches/n*100:.1f}%)\n")
        f.write(f"Good Matches (CER<0.1): {good_matches}/{n} ({good_matches/n*100:.1f}%)\n")

    print(f"   ✅ Summary saved to: {summary_file}\n")

    return {
        "avg_cer": avg_cer,
        "perfect_matches": perfect_matches,
        "good_matches": good_matches,
        "total": n,
    }

# =============================================================================
# MAIN
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


def main():
    args = parse_args()

    print("\n" + "=" * 80)
    print("PHI-3.5-VISION FINE-TUNING - MULTI-DATASET (RESUME + BEST GLOBAL CER)")
    print("(INVENTORY + SCHMUCK + STAIRCASE)")
    print("Single base model, 4-bit QLoRA, BEST by epoch-wise autoregressive CER")
    print("=" * 80)

    base_dir = CONFIG["output_base_dir"]

    # Decide run_dir: new or resume
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

    # Save config (only on new run; don't overwrite on resume)
    cfg_path = os.path.join(run_dir, "training_config.json")
    if is_new_run:
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print(f"   • epochs: {CONFIG['num_epochs']}")
    print(f"   • batch_size: {CONFIG['batch_size']}")
    print(f"   • grad_accum: {CONFIG['gradient_accumulation_steps']}")
    print(f"   • lr: {CONFIG['learning_rate']}")
    print(f"   • LoRA: r={CONFIG['lora_r']} alpha={CONFIG['lora_alpha']} dropout={CONFIG['lora_dropout']}")
    print(f"   • save_strategy: {CONFIG['save_strategy']} (every {CONFIG['save_steps']} steps, keep {CONFIG['save_total_limit']})")
    print(f"   • best adapter saved to: best_model/")
    print(f"   • tensorboard: {os.path.join(run_dir, 'tb_logs')}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["use_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=CONFIG["use_nested_quant"],
    )

    print(f"\n⏳ Loading processor from: {CONFIG['model_path']}")
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"], trust_remote_code=True, num_crops=16
    )
    print("   ✅ Processor loaded")

    print(f"\n⏳ Loading Phi-3.5-Vision base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )
    print("   ✅ Base model loaded")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   📊 GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Prepare for LoRA training
    print("\n📝 Preparing model for QLoRA training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print("   ✅ LoRA applied")
    print(f"   📊 Trainable params: {trainable_params:,} / {all_params:,} ({100*trainable_params/all_params:.2f}%)")

    # Load datasets
    ds_cfg = CONFIG["datasets"]

    print("\n📊 Loading datasets...")
    print("\n   📁 INVENTORY train/val:")
    inv_train = InventoryDataset(
        ds_cfg["inventory"]["train_jsonl"], ds_cfg["inventory"]["images_dir"],
        processor, DATASET_INSTRUCTIONS["inventory"],
        augment=CONFIG["data_augmentation"],
    )
    inv_val = InventoryDataset(
        ds_cfg["inventory"]["val_jsonl"], ds_cfg["inventory"]["images_dir"],
        processor, DATASET_INSTRUCTIONS["inventory"], augment=False,
    )

    print("\n   📁 SCHMUCK train/val:")
    schmuck_train = SchmuckDataset(
        ds_cfg["schmuck"]["train_jsonl"], ds_cfg["schmuck"]["images_dir"],
        processor, DATASET_INSTRUCTIONS["schmuck"],
        augment=CONFIG["data_augmentation"],
    )
    schmuck_val = SchmuckDataset(
        ds_cfg["schmuck"]["val_jsonl"], ds_cfg["schmuck"]["images_dir"],
        processor, DATASET_INSTRUCTIONS["schmuck"], augment=False,
    )

    print("\n   📁 STAIRCASE train/val:")
    stair_train = StaircaseDataset(
        ds_cfg["staircase"]["train_jsonl"], ds_cfg["staircase"]["images_dir"],
        processor, DATASET_INSTRUCTIONS["staircase"],
        augment=CONFIG["data_augmentation"],
    )
    stair_val = StaircaseDataset(
        ds_cfg["staircase"]["val_jsonl"], ds_cfg["staircase"]["images_dir"],
        processor, DATASET_INSTRUCTIONS["staircase"], augment=False,
    )

    train_dataset = ConcatDataset([inv_train, schmuck_train, stair_train])

    # Dummy eval dataset: tiny subset to trigger epoch end eval; real CER in callback
    dummy_n = int(CONFIG["dummy_eval_size_per_dataset"])
    dummy_eval_size = min(dummy_n, len(inv_val), len(schmuck_val), len(stair_val))
    dummy_eval_dataset = ConcatDataset([
        Subset(inv_val, list(range(dummy_eval_size))),
        Subset(schmuck_val, list(range(dummy_eval_size))),
        Subset(stair_val, list(range(dummy_eval_size))),
    ])

    print(f"\n   📊 TOTAL combined train samples: {len(train_dataset)}")
    print(f"   📊 Dummy eval samples (for Trainer eval_loss): {len(dummy_eval_dataset)}")
    print(f"   📊 Real CER eval: up to {CONFIG['max_val_samples_per_dataset']} per dataset via callback")

    data_collator = DataCollatorForPhi3Vision(processor=processor)

    # Callback validation subsets
    max_val = int(CONFIG["max_val_samples_per_dataset"])
    val_sets = {
        "inventory": load_jsonl(ds_cfg["inventory"]["val_jsonl"])[:max_val],
        "schmuck": load_jsonl(ds_cfg["schmuck"]["val_jsonl"])[:max_val],
        "staircase": load_jsonl(ds_cfg["staircase"]["val_jsonl"])[:max_val],
    }
    images_dirs = {
        "inventory": ds_cfg["inventory"]["images_dir"],
        "schmuck": ds_cfg["schmuck"]["images_dir"],
        "staircase": ds_cfg["staircase"]["images_dir"],
    }
    image_keys = {
        "inventory": ds_cfg["inventory"]["image_key"],
        "schmuck": ds_cfg["schmuck"]["image_key"],
        "staircase": ds_cfg["staircase"]["image_key"],
    }

    cer_callback = MultiDatasetCERCallback(
        processor=processor,
        val_sets=val_sets,
        images_dirs=images_dirs,
        image_keys=image_keys,
        instructions=DATASET_INSTRUCTIONS,
        label_fns=LABEL_FNS,
        output_dir=run_dir,
        flush_interval=int(CONFIG["val_cache_flush_interval"]),
    )

    # Training arguments
    train_output_dir = os.path.join(run_dir, "train_state")
    os.makedirs(train_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=train_output_dir,
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        max_grad_norm=CONFIG["max_grad_norm"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy=CONFIG["save_strategy"],   # ✅ rolling checkpoint for resume
        save_steps=CONFIG["save_steps"],
        save_total_limit=CONFIG["save_total_limit"],  # keep only latest checkpoint
        eval_strategy="epoch",   # ✅ triggers callback each epoch
        load_best_model_at_end=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir=os.path.join(run_dir, "tb_logs"),
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dummy_eval_dataset,
        data_collator=data_collator,
        callbacks=[cer_callback],
    )

    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"\n{'='*80}")
        print("GPU INFORMATION")
        print(f"{'='*80}")
        print(f"   GPU: {gpu_stats.name}")
        print(f"   Max memory: {round(gpu_stats.total_memory / 1024**3, 3)} GB")
        print(f"{'='*80}\n")

    # Resume logic: detect last checkpoint in train_output_dir
    last_ckpt = None
    if os.path.isdir(train_output_dir):
        last_ckpt = get_last_checkpoint(train_output_dir)

    # Train
    print("\n" + "=" * 80)
    if last_ckpt is not None:
        print(f"🔁 Resuming training from: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        print("🚀 Starting multi-dataset QLoRA training from scratch...")
        trainer.train()
    print("=" * 80)

    # Optional: clean up rolling checkpoint to save disk
    if args.cleanup_checkpoints:
        for ckpt_dir in glob.glob(os.path.join(train_output_dir, "checkpoint-*")):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        print("🧹 Deleted train_state/checkpoint-* folders (kept best_model + logs).")

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED - MULTI-DATASET MODEL")
    print("=" * 80)
    print(f"   🎯 Best GLOBAL Validation CER: {cer_callback.best_cer:.4f} ({cer_callback.best_cer*100:.2f}%)")
    print(f"   💾 Best adapter saved at: {cer_callback.best_checkpoint_path}")
    print("=" * 80)

    if not cer_callback.best_checkpoint_path:
        raise RuntimeError("No best model was saved. Check validation callback and val sets paths.")

    # -----------------------------
    # PHASE 2: Load best adapter correctly and evaluate on test sets
    # -----------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: PER-DATASET EVALUATION ON TEST SETS")
    print("=" * 80)

    print(f"\n⏳ Loading base model + best adapter from: {cer_callback.best_checkpoint_path}")

    base_for_test = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )
    best_model = PeftModel.from_pretrained(base_for_test, cer_callback.best_checkpoint_path)
    best_model.eval()

    best_processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"], trust_remote_code=True, num_crops=16
    )
    print("   ✅ Best adapter attached to base model")

    flush_interval = int(CONFIG["test_chunk_flush_interval"])
    test_results = {}

    for ds_name in ["inventory", "schmuck", "staircase"]:
        ds = ds_cfg[ds_name]
        result = evaluate_on_dataset(
            dataset_name=ds_name,
            model=best_model,
            processor=best_processor,
            test_jsonl=ds["test_jsonl"],
            images_dir=ds["images_dir"],
            image_key=ds["image_key"],
            instruction=DATASET_INSTRUCTIONS[ds_name],
            label_fn=LABEL_FNS[ds_name],
            output_dir=run_dir,
            flush_interval=flush_interval,
        )
        test_results[ds_name] = result

    # Save global summary and CER history
    summary_file = os.path.join(run_dir, "training_summary_global.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PHI-3.5-VISION MULTI-DATASET FINE-TUNING RESULTS (BEST BY CER)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Best GLOBAL Validation CER: {cer_callback.best_cer:.4f} ({cer_callback.best_cer*100:.2f}%)\n\n")
        f.write("Test Results per Dataset:\n")
        f.write("-" * 50 + "\n")
        for ds_name, res in test_results.items():
            f.write(f"  {ds_name.upper()}:\n")
            f.write(f"    Average CER: {res['avg_cer']:.4f} ({res['avg_cer']*100:.2f}%)\n")
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
    print("\n🎉 Multi-dataset fine-tuning + per-dataset test evaluation complete!\n")


if __name__ == "__main__":
    main()
