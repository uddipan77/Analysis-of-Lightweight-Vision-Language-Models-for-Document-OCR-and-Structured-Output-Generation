#!/usr/bin/env python3
# qwen_multidataset_standard.py
#
# ✅ Single-stage (standard) multi-dataset fine-tuning for Qwen2.5-VL-7B (Unsloth) on:
#    - Staircase
#    - Schmuck
#    - Inventory
#
# ✅ One shared base model + one shared LoRA adapter trained on combined TRAIN of all datasets
# ✅ Best model selected by LOWEST GLOBAL validation CER (autoregressive generation) each epoch
# ✅ CER computed via greedy decoding (temperature=0.0, do_sample=False) on a fixed val subset per dataset
# ✅ After training: loads best-by-CER weights and runs autoregressive greedy predictions on TEST of all datasets
# ✅ Resume support: re-run with --run-dir and it resumes from latest checkpoint-* in run_dir/train/
#
# Notes:
# - Trainer does NOT use compute_metrics (avoids logits RAM blow-up).
# - We manage "best model" via callback, saving a HF folder best_model_cer/ + optional state_dict.
# - Checkpoints are saved by Trainer each epoch (train/checkpoint-*), enabling resume after preemption/timeout.

import os

# ----------------------------------------------------------------------
# IMPORTANT: set Unsloth / PyTorch env BEFORE importing unsloth/torch
# ----------------------------------------------------------------------
os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"          # keep grads on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import glob
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np
import jiwer

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainerCallback

from qwen_vl_utils import process_vision_info


# =============================================================================
# Paths & Global Config
# =============================================================================

BASE_OUTPUT_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/qwen/multidataset_standard"

CONFIG = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/qwen7b",

    "datasets": {
        "staircase": {
            "train_jsonl": "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl",
            "val_jsonl":   "/home/woody/iwi5/iwi5298h/json_staircase/val.jsonl",
            "test_jsonl":  "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl",
            "images_dir":  "/home/woody/iwi5/iwi5298h/staircase_images",
            "id_field":    "image_name",
        },
        "schmuck": {
            "train_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
            "val_jsonl":   "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
            "test_jsonl":  "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
            "images_dir":  "/home/woody/iwi5/iwi5298h/schmuck_images",
            "id_field":    "file_name",
        },
        "inventory": {
            "train_jsonl": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
            "val_jsonl":   "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
            "test_jsonl":  "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
            "images_dir":  "/home/woody/iwi5/iwi5298h/inventory_images",
            "id_field":    "image_name",
        },
    },

    # Training hyperparams
    "num_epochs": 15,
    "batch_size": 2,
    "learning_rate": 5e-5,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,         # safer for Schmuck JSON
    "weight_decay": 0.05,
    "warmup_ratio": 0.1,

    # CER evaluation
    "max_eval_samples_per_dataset": 30,

    # Generation settings for val/test CER (greedy deterministic)
    "gen_max_new_tokens": 1024,
    "gen_temperature": 0.0,
    "gen_do_sample": False,
    "gen_repetition_penalty": 1.0,
}


# =============================================================================
# Dataset-specific instructions (prompts)
# =============================================================================

STAIRCASE_INSTRUCTION = """You are an OCR model for historical German staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this dataset (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
- Never drop a key that appears in the form’s JSON structure. If a field is empty on the form, still include it with an empty string "" (or false for an unchecked box).
- Use booleans for checkbox options: true if the box is checked, false if it is empty.
- Use strings for numbers and free-text fields (measurements, dates, names, notes).
- Do NOT invent new fields.
"""

SCHMUCK_INSTRUCTION = """Extract ALL the jewelry information from this German historical document image as a complete JSON object.

The JSON must include ALL of these fields with their exact German names:
- Gegenstand (type of jewelry - REQUIRED, never skip this)
- Inv.Nr (inventory number)
- Herkunft (origin/provenance)
- Foto Notes (photo information)
- Standort (location/storage)
- Material (materials used)
- Datierung (dating information)
- Maße (measurements/dimensions)
- Gewicht (weight)
- erworben von (acquired from)
- am (date acquired)
- Preis (price)
- Vers.-Wert (insurance value)
- Beschreibung (description)
- Literatur (literature references)
- Ausstellungen (exhibitions)

Include ALL fields, even if empty (use empty string ""). Preserve exact German spelling and punctuation.
Return ONLY the JSON object without any additional text or formatting.
"""

INVENTORY_INSTRUCTION = """You are an OCR model for German museum inventory books and catalog cards.

Task:
Given ONE image of a historical inventory entry, read all printed text and handwritten notes and output a single JSON object that represents the complete entry.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the field names and nesting as in the training JSON for this dataset.
- Never drop a key that appears in the JSON structure. If a field is empty on the document, still include it with an empty string "".
- Use strings for all values (including numbers and measurements).
- Do NOT invent new fields.
"""

DATASET_INSTRUCTIONS = {
    "staircase": STAIRCASE_INSTRUCTION,
    "schmuck": SCHMUCK_INSTRUCTION,
    "inventory": INVENTORY_INSTRUCTION,
}


# =============================================================================
# JSON & IO helpers
# =============================================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_first_json(response: str) -> Dict[str, Any]:
    """Find first {...} block which parses; fallback to parsing full string; else {}."""
    if isinstance(response, list):
        response = response[0] if response else ""
    response = str(response).strip()

    matches = re.findall(r"\{.*\}", response, re.DOTALL)
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(response)
    except Exception:
        return {}


def calculate_cer(predictions: List[str], targets: List[str]) -> float:
    """Average CER over (pred, target) pairs using jiwer.cer."""
    if not predictions or not targets or len(predictions) != len(targets):
        return 1.0
    total = 0.0
    count = 0
    for p, t in zip(predictions, targets):
        t = str(t)
        p = str(p)
        if len(t) == 0:
            continue
        total += jiwer.cer(t, p)
        count += 1
    return total / count if count > 0 else 1.0


def calculate_cer_stats(cer_scores: List[float]) -> Dict[str, Any]:
    if not cer_scores:
        return {}
    avg_cer = sum(cer_scores) / len(cer_scores)
    return {
        "total_images": len(cer_scores),
        "average_cer": float(avg_cer),
        "median_cer": float(np.median(cer_scores)),
        "minimum_cer": float(min(cer_scores)),
        "maximum_cer": float(max(cer_scores)),
        "std_cer": float(np.std(cer_scores)),
        "perfect_matches": int(sum(1 for c in cer_scores if c == 0.0)),
    }


def find_latest_checkpoint(stage_dir: str) -> Optional[str]:
    """Return path to latest 'checkpoint-*' directory inside stage_dir, else None."""
    if not os.path.isdir(stage_dir):
        return None
    ckpts = [
        d for d in os.listdir(stage_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(stage_dir, d))
    ]
    if not ckpts:
        return None

    def ckpt_num(name: str) -> int:
        try:
            return int(name.split("-")[-1])
        except Exception:
            return -1

    ckpts.sort(key=ckpt_num)
    return os.path.join(stage_dir, ckpts[-1])


# =============================================================================
# Image path helpers
# =============================================================================

def find_image_path_staircase(image_name: str, images_dir: str) -> str:
    """Flexible matching for staircase like '... (123).jpg'."""
    exact = os.path.join(images_dir, image_name)
    if os.path.exists(exact):
        return exact

    pattern_match = re.search(r"\((\d+)\)\.jpg$", image_name)
    if pattern_match:
        pattern = f"({pattern_match.group(1)}).jpg"
        matches = glob.glob(os.path.join(images_dir, f"*{pattern}"))
        if matches:
            return matches[0]

    base_name = os.path.splitext(image_name)[0]
    matches = glob.glob(os.path.join(images_dir, f"*{base_name}*"))
    if matches:
        return matches[0]

    return exact


def find_image_path_generic(file_name: str, images_dir: str) -> str:
    """Generic flexible match: exact, else any file containing basename."""
    exact = os.path.join(images_dir, file_name)
    if os.path.exists(exact):
        return exact

    base_name, _ = os.path.splitext(file_name)
    matches = glob.glob(os.path.join(images_dir, f"*{base_name}*"))
    if matches:
        return matches[0]

    return exact


def clean_label_for_dataset(dataset_name: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    """Remove file/image meta keys per dataset."""
    if dataset_name == "staircase":
        remove_keys = {"image_name", "image_path"}
    elif dataset_name == "schmuck":
        remove_keys = {"file_name", "image_path"}
    else:  # inventory
        remove_keys = {"image_name", "image_path"}
    return {k: v for k, v in obj.items() if k not in remove_keys}


def label_to_str(dataset_name: str, obj: Dict[str, Any]) -> str:
    clean = clean_label_for_dataset(dataset_name, obj)
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


# =============================================================================
# Multi-dataset finetuner
# =============================================================================

class MultiDatasetQwenStandardFinetune:
    def __init__(self, model_path: str):
        print(f"Loading Qwen2.5-VL-7B from {model_path} with Unsloth...")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=CONFIG["max_seq_length"],
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        # Shared LoRA (single adapter) for all datasets
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )

        self.best_state_path: Optional[str] = None
        self.best_global_cer: float = float("inf")
        self.cer_history: List[Dict[str, Any]] = []

        print("Model + LoRA initialized successfully!")

    # ---------------------- Data preparation ---------------------- #

    def convert_sample_to_conversation(
        self,
        dataset_name: str,
        sample: Dict[str, Any],
        images_dir: str,
    ) -> Optional[Dict[str, Any]]:
        ds_cfg = CONFIG["datasets"][dataset_name]
        id_field = ds_cfg["id_field"]
        identifier = sample[id_field]

        if dataset_name == "staircase":
            image_path = find_image_path_staircase(identifier, images_dir)
        else:
            image_path = find_image_path_generic(identifier, images_dir)

        if not os.path.exists(image_path):
            print(f"[WARN] {dataset_name}: image not found for {identifier}, skipping.")
            return None

        sample_with_path = sample.copy()
        sample_with_path["image_path"] = image_path

        instruction = DATASET_INSTRUCTIONS[dataset_name]
        gt_str = label_to_str(dataset_name, sample_with_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image_path},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gt_str}],
            },
        ]
        return {"messages": messages}

    def prepare_datasets(
        self,
        datasets_cfg: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, List[Dict]]]:
        """
        Returns:
          - combined_train_dataset: list of {"messages": ...}
          - combined_eval_dataset:  list of {"messages": ...}
          - val_raw_by_dataset: dataset_name -> list[raw samples] (used in CER callback)
        """
        train_dataset: List[Dict] = []
        eval_dataset: List[Dict] = []
        val_raw: Dict[str, List[Dict]] = {}

        print("\nPreparing training & validation datasets for all 3 datasets...")
        for ds_name, ds_cfg in datasets_cfg.items():
            print(f"\n[{ds_name.upper()}]")

            # Train
            train_records = load_jsonl(ds_cfg["train_jsonl"])
            ds_train_convs = []
            for rec in train_records:
                conv = self.convert_sample_to_conversation(ds_name, rec, ds_cfg["images_dir"])
                if conv is not None:
                    ds_train_convs.append(conv)
            print(f"  Train: {len(ds_train_convs)} samples (valid)")
            train_dataset.extend(ds_train_convs)

            # Val
            val_records = load_jsonl(ds_cfg["val_jsonl"])
            val_raw[ds_name] = val_records
            ds_val_convs = []
            for rec in val_records:
                conv = self.convert_sample_to_conversation(ds_name, rec, ds_cfg["images_dir"])
                if conv is not None:
                    ds_val_convs.append(conv)
            print(f"  Val:   {len(ds_val_convs)} samples (valid)")
            eval_dataset.extend(ds_val_convs)

        print(f"\nTOTAL combined train samples: {len(train_dataset)}")
        print(f"TOTAL combined val samples:   {len(eval_dataset)}\n")

        return train_dataset, eval_dataset, val_raw

    # ---------------------- Training ---------------------- #

    def train_standard(
        self,
        run_dir: str,
        datasets_cfg: Dict[str, Dict[str, Any]],
    ) -> SFTTrainer:
        """
        Single-stage training on combined datasets.
        Resume from checkpoint if found in run_dir/train/.
        Best model selected by LOWEST GLOBAL validation CER (autoregressive) via callback.
        """
        train_dataset, eval_dataset, val_raw_by_dataset = self.prepare_datasets(datasets_cfg)

        FastVisionModel.for_training(self.model)
        data_collator = UnslothVisionDataCollator(self.model, self.tokenizer)

        train_dir = os.path.join(run_dir, "train")
        os.makedirs(train_dir, exist_ok=True)

        best_dir = os.path.join(run_dir, "best_model_cer")
        os.makedirs(best_dir, exist_ok=True)

        cer_callback = MultiDatasetCERCallback(
            finetuner=self,
            datasets_cfg=datasets_cfg,
            val_raw_by_dataset=val_raw_by_dataset,
            best_dir=best_dir,
            max_eval_samples_per_dataset=CONFIG["max_eval_samples_per_dataset"],
        )

        args = SFTConfig(
            per_device_train_batch_size=CONFIG["batch_size"],
            per_device_eval_batch_size=CONFIG["batch_size"],
            gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
            warmup_steps=50,
            num_train_epochs=CONFIG["num_epochs"],
            learning_rate=CONFIG["learning_rate"],
            logging_steps=10,

            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,

            load_best_model_at_end=False,      # callback manages best-by-CER
            metric_for_best_model="eval_cer_global",
            greater_is_better=False,

            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            weight_decay=CONFIG["weight_decay"],
            lr_scheduler_type="cosine",
            warmup_ratio=CONFIG["warmup_ratio"],
            optim="adamw_8bit",
            gradient_checkpointing=True,

            remove_unused_columns=False,
            dataset_text_field=None,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=CONFIG["max_seq_length"],

            report_to="tensorboard",
            logging_dir=os.path.join(train_dir, "logs"),
            seed=3407,
            output_dir=train_dir,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            callbacks=[cer_callback],
        )

        # GPU stats
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_mem = round(gpu_stats.total_memory / 1024**3, 3)
        print(f"GPU = {gpu_stats.name}, Max memory = {max_mem} GB.")
        print(f"{start_gpu_mem} GB reserved before training.")

        latest_ckpt = find_latest_checkpoint(train_dir)
        if latest_ckpt is not None:
            print(f"\n[TRAIN] Resuming from checkpoint: {latest_ckpt}\n")
            trainer_stats = trainer.train(resume_from_checkpoint=latest_ckpt)
        else:
            print("\n[TRAIN] No checkpoint found, starting from scratch.\n")
            trainer_stats = trainer.train()

        used_mem = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        used_mem_for_lora = round(used_mem - start_gpu_mem, 3)
        used_pct = round(used_mem / max_mem * 100, 3)
        lora_pct = round(used_mem_for_lora / max_mem * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes.")
        print(f"Peak reserved memory = {used_mem} GB "
              f"({used_pct}% of max, {lora_pct}% for training).")

        # Load best-by-CER weights (saved by callback)
        if cer_callback.best_state_path is not None:
            print(f"\nLoading best-by-CER weights from {cer_callback.best_state_path} ...")
            loaded_state = torch.load(cer_callback.best_state_path, map_location="cpu")

            current_state = self.model.state_dict()
            filtered_state = {k: v for k, v in loaded_state.items() if k in current_state}

            missing_keys, unexpected_keys = self.model.load_state_dict(filtered_state, strict=False)
            if missing_keys:
                print(f"[WARN] Missing keys when loading best CER weights ({len(missing_keys)}).")
            if unexpected_keys:
                print(f"[WARN] Unexpected keys when loading best CER weights ({len(unexpected_keys)}).")

            trainer.model = self.model
            self.best_state_path = cer_callback.best_state_path
            self.best_global_cer = cer_callback.best_cer
            self.cer_history = cer_callback.cer_history

            print(f"Restored best global validation CER = {self.best_global_cer:.4f} "
                  f"({self.best_global_cer*100:.2f}%)")
        else:
            print("\n[WARN] No best CER weights saved; keeping final training weights.")
            self.best_state_path = None
            self.best_global_cer = float("inf")
            self.cer_history = cer_callback.cer_history

        torch.cuda.empty_cache()
        return trainer

    # ---------------------- Test evaluation per dataset ---------------------- #

    def evaluate_dataset(
        self,
        dataset_name: str,
        test_jsonl: str,
        images_dir: str,
        output_dir: str,
    ) -> Dict[str, Any]:
        """
        Evaluate current in-memory model on a dataset's test split (autoregressive greedy).
        Saves predictions JSONL + CER report.
        """
        print(f"\n{'='*80}")
        print(f"EVALUATION ON TEST SPLIT - {dataset_name.upper()}")
        print(f"{'='*80}")

        FastVisionModel.for_inference(self.model)

        test_data = load_jsonl(test_jsonl)
        print(f"Loaded {len(test_data)} test samples for {dataset_name}.")

        predictions_log: List[Dict[str, Any]] = []
        cer_scores: List[float] = []

        instruction = DATASET_INSTRUCTIONS[dataset_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        id_field = CONFIG["datasets"][dataset_name]["id_field"]

        for idx, item in enumerate(test_data):
            identifier = item[id_field]
            print(f"[{dataset_name}] Test {idx+1}/{len(test_data)}: {identifier}", end="\r")

            if dataset_name == "staircase":
                image_path = find_image_path_staircase(identifier, images_dir)
            else:
                image_path = find_image_path_generic(identifier, images_dir)

            if not os.path.exists(image_path):
                gt_clean = clean_label_for_dataset(dataset_name, item)
                gt_str = label_to_str(dataset_name, gt_clean)
                predictions_log.append(
                    {
                        "id": identifier,
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": gt_clean,
                        "target_text": gt_str,
                        "raw_response": "Error: Image not found",
                        "cer_score": 1.0,
                        "image_path": image_path,
                    }
                )
                cer_scores.append(1.0)
                continue

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": image_path},
                        ],
                    }
                ]

                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.tokenizer(
                    text=[input_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=CONFIG["gen_max_new_tokens"],
                        use_cache=True,
                        temperature=CONFIG["gen_temperature"],
                        do_sample=CONFIG["gen_do_sample"],
                        repetition_penalty=CONFIG["gen_repetition_penalty"],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]

                generated_text = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                pred_json_raw = extract_first_json(generated_text)
                pred_clean = clean_label_for_dataset(dataset_name, pred_json_raw or {})
                gt_clean = clean_label_for_dataset(dataset_name, item)

                pred_str = label_to_str(dataset_name, pred_clean)
                gt_str = label_to_str(dataset_name, gt_clean)

                cer = jiwer.cer(gt_str, pred_str)
                cer_scores.append(float(cer))

                predictions_log.append(
                    {
                        "id": identifier,
                        "predicted_json": pred_clean,
                        "predicted_text": pred_str,
                        "target_json": gt_clean,
                        "target_text": gt_str,
                        "raw_response": generated_text,
                        "cer_score": float(cer),
                        "image_path": image_path,
                    }
                )

            except Exception as e:
                gt_clean = clean_label_for_dataset(dataset_name, item)
                gt_str = label_to_str(dataset_name, gt_clean)
                predictions_log.append(
                    {
                        "id": identifier,
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": gt_clean,
                        "target_text": gt_str,
                        "raw_response": f"Error: {e}",
                        "cer_score": 1.0,
                        "image_path": image_path,
                    }
                )
                cer_scores.append(1.0)

            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        pred_file = os.path.join(output_dir, f"test_predictions_{dataset_name}.jsonl")
        save_jsonl(predictions_log, pred_file)

        stats = calculate_cer_stats(cer_scores)
        cer_file = os.path.join(output_dir, f"cer_evaluation_{dataset_name}.txt")
        self._save_cer_report(dataset_name, stats, cer_file)

        print(f"\n[{dataset_name}] Test predictions saved to: {pred_file}")
        print(f"[{dataset_name}] CER summary saved to: {cer_file}")

        return {
            "predictions_file": pred_file,
            "cer_file": cer_file,
            "cer_stats": stats,
        }

    def _save_cer_report(self, dataset_name: str, stats: Dict[str, Any], path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CER EVALUATION RESULTS - {dataset_name.upper()} TEST SPLIT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"CER Statistics across {stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average CER: {stats['average_cer']:.4f} ({stats['average_cer']*100:.2f}%)\n")
            f.write(f"Median CER: {stats['median_cer']:.4f} ({stats['median_cer']*100:.2f}%)\n")
            f.write(f"Minimum CER: {stats['minimum_cer']:.4f} ({stats['minimum_cer']*100:.2f}%)\n")
            f.write(f"Maximum CER: {stats['maximum_cer']:.4f} ({stats['maximum_cer']*100:.2f}%)\n")
            f.write(f"Standard Deviation: {stats['std_cer']:.4f}\n\n")
            f.write(
                f"Perfect matches: {stats['perfect_matches']}/"
                f"{stats['total_images']} "
                f"({stats['perfect_matches']/stats['total_images']*100:.2f}%)\n"
            )
        print(f"CER evaluation results for {dataset_name} saved to: {path}")

    def save_final_model(self, trainer: SFTTrainer, run_dir: str) -> None:
        """Save final trainer model (in addition to best-by-CER)."""
        output_dir = os.path.join(run_dir, "final_trainer_model")
        print(f"\nSaving final Trainer model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("Final Trainer model saved.")


# =============================================================================
# CER Callback: best model by GLOBAL validation CER (autoregressive greedy)
# =============================================================================

class MultiDatasetCERCallback(TrainerCallback):
    def __init__(
        self,
        finetuner: MultiDatasetQwenStandardFinetune,
        datasets_cfg: Dict[str, Dict[str, Any]],
        val_raw_by_dataset: Dict[str, List[Dict[str, Any]]],
        best_dir: str,
        max_eval_samples_per_dataset: int = 30,
    ):
        self.finetuner = finetuner
        self.datasets_cfg = datasets_cfg
        self.val_raw_by_dataset = val_raw_by_dataset
        self.best_dir = best_dir
        self.max_eval_samples_per_dataset = max_eval_samples_per_dataset

        self.best_cer: float = float("inf")
        self.best_state_path: Optional[str] = None
        self.cer_history: List[Dict[str, Any]] = []

        os.makedirs(self.best_dir, exist_ok=True)

    def _generate_pred_str(
        self,
        model,
        ds_name: str,
        item: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[str, str]:
        """
        Returns (pred_str, gt_str) for a single val item using autoregressive greedy generation.
        """
        ds_cfg = self.datasets_cfg[ds_name]
        instruction = DATASET_INSTRUCTIONS[ds_name]
        id_field = ds_cfg["id_field"]
        images_dir = ds_cfg["images_dir"]

        identifier = item[id_field]
        if ds_name == "staircase":
            image_path = find_image_path_staircase(identifier, images_dir)
        else:
            image_path = find_image_path_generic(identifier, images_dir)

        gt_clean = clean_label_for_dataset(ds_name, item)
        gt_str = label_to_str(ds_name, gt_clean)

        if not os.path.exists(image_path):
            return ("", gt_str)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": image_path},
                ],
            }
        ]

        input_text = self.finetuner.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.finetuner.tokenizer(
            text=[input_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=CONFIG["gen_max_new_tokens"],
                use_cache=True,
                temperature=CONFIG["gen_temperature"],
                do_sample=CONFIG["gen_do_sample"],
                repetition_penalty=CONFIG["gen_repetition_penalty"],
                pad_token_id=self.finetuner.tokenizer.pad_token_id,
                eos_token_id=self.finetuner.tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]

        generated_text = self.finetuner.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        pred_json_raw = extract_first_json(generated_text)
        pred_clean = clean_label_for_dataset(ds_name, pred_json_raw or {})
        pred_str = label_to_str(ds_name, pred_clean)

        del inputs, outputs, generated_ids_trimmed
        torch.cuda.empty_cache()

        return (pred_str, gt_str)

    def on_evaluate(self, args, state, control, model, tokenizer=None, metrics=None, **kwargs):
        if state.epoch is None:
            return control

        print("\n" + "=" * 80)
        print(f"🔍 Multi-dataset CER Callback: validation CER at Epoch {int(state.epoch)}")
        print("=" * 80)

        model.eval()
        device = next(model.parameters()).device

        global_predictions: List[str] = []
        global_targets: List[str] = []
        cer_by_dataset: Dict[str, float] = {}

        for ds_name in self.datasets_cfg.keys():
            val_raw = self.val_raw_by_dataset[ds_name]
            subset = val_raw[: self.max_eval_samples_per_dataset]
            print(f"\n[{ds_name.upper()}] Evaluating on {len(subset)} validation samples...")

            ds_preds: List[str] = []
            ds_tgts: List[str] = []

            for idx, item in enumerate(subset):
                identifier = item[self.datasets_cfg[ds_name]["id_field"]]
                print(f"  [{ds_name}] val {idx+1}/{len(subset)}: {identifier}", end="\r")
                try:
                    pred_str, gt_str = self._generate_pred_str(model, ds_name, item, device)
                except Exception as e:
                    gt_clean = clean_label_for_dataset(ds_name, item)
                    gt_str = label_to_str(ds_name, gt_clean)
                    pred_str = ""
                    print(f"\n  [WARN] CER val error on {ds_name} sample {identifier}: {e}")

                ds_preds.append(pred_str)
                ds_tgts.append(gt_str)

            ds_cer = calculate_cer(ds_preds, ds_tgts)
            cer_by_dataset[ds_name] = float(ds_cer)
            print(f"\n  ✅ {ds_name} CER: {ds_cer:.4f} ({ds_cer*100:.2f}%)")

            global_predictions.extend(ds_preds)
            global_targets.extend(ds_tgts)

        global_cer = float(calculate_cer(global_predictions, global_targets))
        print(f"\n🌐 GLOBAL validation CER (all datasets): {global_cer:.4f} ({global_cer*100:.2f}%)")

        # Log for Trainer
        if metrics is not None:
            metrics["eval_cer_global"] = global_cer
            for ds_name, ds_cer in cer_by_dataset.items():
                metrics[f"eval_cer_{ds_name}"] = ds_cer

        # CER history
        entry = {"epoch": float(state.epoch), "global_cer": global_cer}
        for ds_name, ds_cer in cer_by_dataset.items():
            entry[f"cer_{ds_name}"] = float(ds_cer)
        self.cer_history.append(entry)

        # Save best model by GLOBAL CER
        if global_cer < self.best_cer:
            improvement = self.best_cer - global_cer
            self.best_cer = global_cer
            print(f"\n🎯 NEW BEST GLOBAL CER: {self.best_cer:.4f} (improved by {improvement:.4f})")

            # 1) save a raw state_dict for fast reload inside same run
            best_state_path = os.path.join(self.best_dir, "pytorch_model_best_cer.bin")
            torch.save(model.state_dict(), best_state_path)
            self.best_state_path = best_state_path
            print(f"   💾 Saved best-by-CER state_dict to: {best_state_path}")

            # 2) also save a HF folder for future use
            try:
                hf_dir = os.path.join(self.best_dir, "hf")
                os.makedirs(hf_dir, exist_ok=True)
                self.finetuner.model.save_pretrained(hf_dir)
                self.finetuner.tokenizer.save_pretrained(hf_dir)
                print(f"   💾 Saved best-by-CER HF folder to: {hf_dir}")
            except Exception as e:
                print(f"   [WARN] Could not save HF folder for best model: {e}")
        else:
            print(f"\n📊 No improvement. Best global CER remains: {self.best_cer:.4f}")

        print("=" * 80 + "\n")
        model.train()
        return control


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Standard (single-stage) multi-dataset finetuning for Qwen2.5-VL-7B"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help=(
            "Existing run directory to resume from. "
            "If not provided, a new run directory will be created."
        ),
    )
    args = parser.parse_args()

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    if args.run_dir is not None:
        run_dir = args.run_dir
        print(f"[INFO] Resuming / continuing run in: {run_dir}")
        if not os.path.isdir(run_dir):
            raise ValueError(f"--run-dir {run_dir} does not exist.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"Created new run directory: {run_dir}")

        # Save config snapshot
        config_file = os.path.join(run_dir, "training_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    finetuner = MultiDatasetQwenStandardFinetune(model_path=CONFIG["model_path"])

    print("=" * 80)
    print("STARTING STANDARD MULTI-DATASET TRAINING FOR QWEN2.5-VL-7B")
    print("=" * 80)

    trainer = finetuner.train_standard(
        run_dir=run_dir,
        datasets_cfg=CONFIG["datasets"],
    )

    # Save final trainer model (separate from best)
    finetuner.save_final_model(trainer, run_dir)

    # =======================
    # Evaluation on test sets
    # =======================
    print("\n" + "=" * 80)
    print("STARTING EVALUATION ON ALL TEST SETS (USING BEST / FINAL WEIGHTS)")
    print("=" * 80)

    eval_summary: Dict[str, Dict[str, Any]] = {}
    for ds_name, ds_cfg in CONFIG["datasets"].items():
        results = finetuner.evaluate_dataset(
            dataset_name=ds_name,
            test_jsonl=ds_cfg["test_jsonl"],
            images_dir=ds_cfg["images_dir"],
            output_dir=run_dir,
        )
        eval_summary[ds_name] = results["cer_stats"]

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY - MULTI-DATASET QWEN2.5-VL (TEST)")
    print("=" * 80)
    for ds_name, stats in eval_summary.items():
        print(f"\n[{ds_name.upper()}]")
        print(f"Average CER: {stats['average_cer']:.4f} ({stats['average_cer']*100:.2f}%)")
        print(f"Median CER:  {stats['median_cer']:.4f} ({stats['median_cer']*100:.2f}%)")
        print(
            f"Perfect matches: {stats['perfect_matches']}/"
            f"{stats['total_images']} "
            f"({stats['perfect_matches']/stats['total_images']*100:.2f}%)"
        )

    # Save CER history from training callback
    cer_hist_file = os.path.join(run_dir, "cer_history.json")
    with open(cer_hist_file, "w", encoding="utf-8") as f:
        json.dump(finetuner.cer_history, f, indent=2, ensure_ascii=False)
    print(f"\nCER history saved to: {cer_hist_file}")

    print("\nAll outputs saved under:")
    print(f"  {run_dir}")
    print("\nStandard multi-dataset training and evaluation complete!\n")


if __name__ == "__main__":
    main()
