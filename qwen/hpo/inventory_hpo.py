#!/usr/bin/env python3
# qwen_inventory_finetune_unsloth_optuna.py
# Qwen2.5-VL inventory fine-tune + Optuna HPO
# Shared DB: /home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db
# Study name: qwen_inventory
# Objective: minimize validation CER (generation-based CER on a FIXED val subset)

import torch
import json
import os
from typing import List, Dict, Any
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
import re
import jiwer
from transformers import EarlyStoppingCallback
import numpy as np
import random
import gc
import optuna
import glob


# ============================================================
# INVENTORY JSON-schema-style prompt (from qwen_finetune.py)
# ============================================================

INVENTORY_SCHEMA_PROMPT = """You are an OCR model for German museum inventory books and catalog cards.

Task:
Given ONE image of a historical inventory entry, read all printed text and handwritten notes and output a single JSON object that represents the complete entry.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the following field names with exact German spelling and capitalization:
  - "Überschrift" (heading/title of the entry)
  - "Inventarnummer" (inventory number)
  - "Maße" (measurements object with keys "L", "B", "D" for length, breadth, depth)
  - "Objektbezeichnung" (object description/name)
  - "Fundort" (find location)
  - "Fundzeit" (find time/date)
  - "Beschreibungstext" (description text)

IMPORTANT:
- Include ALL fields in EVERY response, even if the field has no visible text in the image.
- If a field is empty or not present in the document, use an empty string "" for that field.
- For "Maße", ALWAYS include all three keys "L", "B", "D" even if they are empty (e.g. {"L": "", "B": "", "D": ""}).
- Use strings for all values including numbers and measurements.
- Do NOT invent new fields or skip any fields.
- Preserve punctuation and German diacritics as accurately as possible.
"""


# ============================================================
# GLOBAL CONFIG
# ============================================================
CONFIG = {
    # model
    "model_path": "/home/vault/iwi5/iwi5298h/models/qwen7b",

    # data (updated JSON path with new structure)
    "train_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
    "val_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
    "test_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",

    # output
    "output_dir": "/home/vault/iwi5/iwi5298h/models_image_text/qwen/hpo/new_inven",

    # optuna
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "qwen_new_inventory_dataset",
    "n_trials": 20,

    # ✅ fixed (NOT optimized)
    "batch_size": 1,

    # ✅ fixed validation subset for HPO objective
    # Always use the SAME 50 samples every trial (deterministic selection)
    "val_eval_max_samples": 50,

    # ✅ fixed epochs per trial (NOT optimized)
    "num_epochs": 5,

    # ---------------------------
    # Search spaces (Optuna)
    # ---------------------------
    "learning_rate_range": (1e-5, 5e-4),
    "weight_decay_range": (0.0, 0.1),

    "grad_accum_choices": [4, 8, 16],
    "lora_r_choices": [8, 16, 32, 64],
    "lora_alpha_choices": [16, 32, 64, 128],
    "lora_dropout_range": (0.0, 0.2),
    "warmup_ratio_range": (0.0, 0.2),

    # IMPORTANT: avoid 512 to prevent image-token mismatch
    "max_seq_length_choices": [1024, 2048],
}


class InventoryOCRFinetune:
    def __init__(
        self,
        model_name: str,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        max_seq_length: int = 2048,
    ):
        """Initialize with configurable LoRA + max_seq_length (for Optuna)."""
        print("Loading Qwen2.5-VL model with Unsloth...")
        print(f"  • max_seq_length: {max_seq_length}")
        print(f"  • LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )

        print("Model loaded successfully with LoRA adapters!")

    # ---------- basic utils ----------
    def load_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def save_jsonl(self, data: List[Dict], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def dict_without_image_name(self, obj):
        return {k: v for k, v in obj.items() if k not in ["image_name", "image_path"]}

    def json_to_string_no_sort(self, obj):
        return json.dumps(self.dict_without_image_name(obj), ensure_ascii=False, separators=(",", ":"))

    def safe_json_loads(self, s):
        try:
            return json.loads(s)
        except Exception:
            return None

    def canonical_json_string(self, obj):
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def extract_json_from_response(self, response: str) -> Dict:
        response = response.strip()
        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        return {}

    def convert_to_conversation(self, sample):
        gt_json_string = self.json_to_string_no_sort(sample)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": INVENTORY_SCHEMA_PROMPT},
                    {"type": "image", "image": sample["image_path"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": gt_json_string},
                ],
            },
        ]
        return {"messages": conversation}

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)
        for item in data:
            item["image_path"] = os.path.join(images_dir, item["image_name"])
        converted_dataset = [self.convert_to_conversation(sample) for sample in data]
        return converted_dataset

    # ---------- CER helpers ----------
    def calculate_cer(self, predictions, targets):
        """Compute average CER using jiwer.cer over string pairs (same as qwen_finetune.py)."""
        if not predictions or not targets or len(predictions) != len(targets):
            return 1.0

        total_cer = 0.0
        valid_pairs = 0

        for pred, target in zip(predictions, targets):
            pred_str = str(pred)
            target_str = str(target)

            if len(target_str) == 0:
                continue

            try:
                cer_val = jiwer.cer(target_str, pred_str)
            except Exception:
                continue

            total_cer += cer_val
            valid_pairs += 1

        return (total_cer / valid_pairs) if valid_pairs > 0 else 1.0

    def calculate_cer_json(self, predictions, targets):
        preds_json = [self.safe_json_loads(p) for p in predictions]
        targets_json = [self.safe_json_loads(t) for t in targets]
        pred_strings = [self.canonical_json_string(j) if j is not None else "" for j in preds_json]
        target_strings = [self.canonical_json_string(j) if j is not None else "" for j in targets_json]
        return self.calculate_cer(pred_strings, target_strings)

    # ✅ FIXED SUBSET (deterministic) validation CER for HPO
    def evaluate_on_validation_fixed_subset(
        self,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        epoch: int,
        max_samples: int = 50,
    ) -> float:
        """
        Deterministic CER eval:
        - sort validation records by image_name
        - take first max_samples
        => SAME subset every trial
        """
        print(f"Evaluating on validation FIXED subset at epoch {epoch}...")

        FastVisionModel.for_inference(self.model)

        val_data = self.load_jsonl(val_jsonl_path)

        # deterministic subset selection
        val_data_sorted = sorted(val_data, key=lambda x: x.get("image_name", ""))
        val_subset = val_data_sorted[: min(max_samples, len(val_data_sorted))]

        print(f"  Using fixed validation subset size = {len(val_subset)}")

        predictions = []
        targets = []

        for test_item in val_subset:
            try:
                image_path = os.path.join(images_dir, test_item["image_name"])
                if not os.path.exists(image_path):
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": INVENTORY_SCHEMA_PROMPT},
                            {"type": "image", "image": image_path},
                        ],
                    }
                ]

                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.tokenizer(
                    text=[input_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_cache=True,
                        temperature=0.0,
                        do_sample=False,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]

                generated_text = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                predicted_json = self.extract_json_from_response(generated_text)
                pred_json_string = self.json_to_string_no_sort(predicted_json) if predicted_json else ""
                gt_json_string = self.json_to_string_no_sort(test_item)

                predictions.append(pred_json_string)
                targets.append(gt_json_string)

            except Exception:
                predictions.append("")
                targets.append(self.json_to_string_no_sort(test_item))
                continue

        val_cer = self.calculate_cer_json(predictions, targets)

        log_file = os.path.join(output_dir, "validation_log.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch}: Validation CER (fixed subset) = {val_cer:.4f}\n")

        print(f"Validation CER (fixed subset) at epoch {epoch}: {val_cer:.4f}")

        FastVisionModel.for_training(self.model)
        return val_cer

    # ---------- TRAIN ----------
    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 15,
        batch_size: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.05,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.1,
        max_seq_length: int = 2048,
        save_strategy: str = "epoch",
        report_to: str = "tensorboard",
        load_best_model_at_end: bool = True,
    ):
        print("Preparing training and validation datasets...")
        train_dataset = self.prepare_training_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_training_data(val_jsonl_path, images_dir)
        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

        FastVisionModel.for_training(self.model)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,

                # NOTE: keeping your original warmup_steps=50
                # warmup_ratio is still set (some versions prioritize steps)
                warmup_steps=50,
                warmup_ratio=warmup_ratio,

                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=10,

                eval_strategy="epoch",
                save_strategy=save_strategy,  # "no" in HPO trials
                save_total_limit=1,

                dataloader_num_workers=0,
                dataloader_pin_memory=False,

                weight_decay=weight_decay,
                lr_scheduler_type="cosine",

                load_best_model_at_end=load_best_model_at_end,
                metric_for_best_model="eval_loss",
                greater_is_better=False,

                optim="adamw_8bit",
                gradient_checkpointing=True,

                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},

                max_seq_length=max_seq_length,

                report_to=report_to,
                logging_dir=f"{output_dir}/logs",
                seed=3407,
                output_dir=output_dir,
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001,
                )
            ],
        )

        print("Starting training with early stopping...")
        trainer.train()
        return trainer

    # ---------- TEST ----------
    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict:
        print("Starting evaluation on test.jsonl...")

        FastVisionModel.for_inference(self.model)

        test_data = self.load_jsonl(test_jsonl_path)
        print(f"Loaded {len(test_data)} test samples")

        predictions = []
        all_cer_scores = []

        for i, test_item in enumerate(test_data):
            print(f"Processing test image {i+1}/{len(test_data)}: {test_item['image_name']}")
            image_path = os.path.join(images_dir, test_item["image_name"])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": INVENTORY_SCHEMA_PROMPT},
                            {"type": "image", "image": image_path},
                        ],
                    }
                ]

                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.tokenizer(
                    text=[input_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_cache=True,
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]

                generated_text = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                predicted_json = self.extract_json_from_response(generated_text)
                gt_json_string = self.json_to_string_no_sort(test_item)
                pred_json_string = self.json_to_string_no_sort(predicted_json) if predicted_json else ""

                cer_score = jiwer.cer(gt_json_string, pred_json_string)

                prediction_entry = {
                    "image_name": test_item["image_name"],
                    "predicted_json": predicted_json,
                    "predicted_text": pred_json_string,
                    "target_json": self.dict_without_image_name(test_item),
                    "target_text": gt_json_string,
                    "raw_response": generated_text,
                    "cer_score": cer_score,
                }
                predictions.append(prediction_entry)
                all_cer_scores.append(cer_score)
                print(f"  Processed successfully. CER: {cer_score:.3f}")

            except Exception as e:
                print(f"Error processing {test_item['image_name']}: {str(e)}")
                prediction_entry = {
                    "image_name": test_item["image_name"],
                    "predicted_json": {},
                    "predicted_text": "",
                    "target_json": self.dict_without_image_name(test_item),
                    "target_text": self.json_to_string_no_sort(test_item),
                    "raw_response": f"Error: {str(e)}",
                    "cer_score": 1.0,
                }
                predictions.append(prediction_entry)
                all_cer_scores.append(1.0)
                continue

        predictions_file = os.path.join(output_dir, "test_predictions.jsonl")
        self.save_jsonl(predictions, predictions_file)

        cer_stats = self.calculate_cer_statistics(all_cer_scores)

        cer_file = os.path.join(output_dir, "cer_evaluation_results.txt")
        self.save_cer_results(cer_stats, cer_file, len(predictions))

        return {
            "predictions": predictions,
            "cer_stats": cer_stats,
            "predictions_file": predictions_file,
            "cer_file": cer_file,
        }

    def calculate_cer_statistics(self, all_cer_scores: List[float]) -> Dict:
        if not all_cer_scores:
            return {}

        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        median_cer = np.median(all_cer_scores)
        std_cer = np.std(all_cer_scores)

        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)

        return {
            "total_images": len(all_cer_scores),
            "average_cer": avg_cer,
            "median_cer": median_cer,
            "minimum_cer": min_cer,
            "maximum_cer": max_cer,
            "std_cer": std_cer,
            "perfect_matches": perfect_matches,
        }

    def save_cer_results(self, cer_stats: Dict, cer_file: str, num_predictions: int):
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("CER EVALUATION RESULTS ON TEST.JSONL\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n")
            f.write(f"Median CER: {cer_stats['median_cer']:.4f}\n")
            f.write(f"Minimum CER: {cer_stats['minimum_cer']:.4f}\n")
            f.write(f"Maximum CER: {cer_stats['maximum_cer']:.4f}\n")
            f.write(f"Standard Deviation: {cer_stats['std_cer']:.4f}\n\n")
            f.write(
                f"Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']} "
                f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n"
            )
            f.write(f"Total images processed: {num_predictions}\n")

        print(f"CER evaluation results saved to: {cer_file}")

    def save_model(self, trainer, output_dir: str):
        print(f"Saving model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================
def optuna_objective(trial: optuna.trial.Trial):
    print("\n" + "=" * 80)
    print(f"OPTUNA TRIAL {trial.number} ({CONFIG['optuna_study_name']})")
    print("=" * 80)

    # Fixed epochs per trial
    num_epochs = CONFIG["num_epochs"]
    
    learning_rate = trial.suggest_float(
        "learning_rate",
        CONFIG["learning_rate_range"][0],
        CONFIG["learning_rate_range"][1],
        log=True,
    )
    weight_decay = trial.suggest_float("weight_decay", CONFIG["weight_decay_range"][0], CONFIG["weight_decay_range"][1])

    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", CONFIG["grad_accum_choices"])
    lora_r = trial.suggest_categorical("lora_r", CONFIG["lora_r_choices"])
    lora_alpha = trial.suggest_categorical("lora_alpha", CONFIG["lora_alpha_choices"])
    lora_dropout = trial.suggest_float("lora_dropout", CONFIG["lora_dropout_range"][0], CONFIG["lora_dropout_range"][1])
    warmup_ratio = trial.suggest_float("warmup_ratio", CONFIG["warmup_ratio_range"][0], CONFIG["warmup_ratio_range"][1])
    max_seq_length = trial.suggest_categorical("max_seq_length", CONFIG["max_seq_length_choices"])

    batch_size = CONFIG["batch_size"]  # fixed

    print(f"  • num_epochs:     {num_epochs} (fixed)")
    print(f"  • learning_rate:  {learning_rate:.2e}")
    print(f"  • weight_decay:   {weight_decay:.4f}")
    print(f"  • grad_accum:     {gradient_accumulation_steps}")
    print(f"  • lora_r:         {lora_r}")
    print(f"  • lora_alpha:     {lora_alpha}")
    print(f"  • lora_dropout:   {lora_dropout:.4f}")
    print(f"  • warmup_ratio:   {warmup_ratio:.4f}")
    print(f"  • max_seq_length: {max_seq_length}")
    print(f"  • batch_size:     {batch_size} (fixed)")

    # per-trial temp dir (no checkpoints)
    trial_tmp_dir = os.path.join(CONFIG["output_dir"], "optuna_tmp")
    os.makedirs(trial_tmp_dir, exist_ok=True)

    finetuner = InventoryOCRFinetune(
        model_name=CONFIG["model_path"],
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_length=max_seq_length,
    )

    trainer = finetuner.train_model(
        train_jsonl_path=CONFIG["train_jsonl_path"],
        val_jsonl_path=CONFIG["val_jsonl_path"],
        images_dir=CONFIG["images_dir"],
        output_dir=trial_tmp_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        max_seq_length=max_seq_length,
        save_strategy="no",
        report_to="none",
        load_best_model_at_end=False,
    )

    # objective CER on fixed subset (same 50 samples every trial)
    val_cer = finetuner.evaluate_on_validation_fixed_subset(
        val_jsonl_path=CONFIG["val_jsonl_path"],
        images_dir=CONFIG["images_dir"],
        output_dir=trial_tmp_dir,
        epoch=num_epochs,
        max_samples=CONFIG["val_eval_max_samples"],
    )

    print(f"  ✅ Trial {trial.number} validation CER (fixed subset) = {val_cer:.4f}")

    del trainer
    del finetuner
    torch.cuda.empty_cache()
    gc.collect()

    return val_cer


# ============================================================
# FINAL TRAIN + TEST
# ============================================================
def train_final_with_best_params(best_params: Dict):
    print("\n" + "=" * 80)
    print("TRAINING FINAL QWEN MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    batch_size = CONFIG["batch_size"]  # fixed

    finetuner = InventoryOCRFinetune(
        model_name=CONFIG["model_path"],
        lora_r=best_params["lora_r"],
        lora_alpha=best_params["lora_alpha"],
        lora_dropout=best_params["lora_dropout"],
        max_seq_length=best_params["max_seq_length"],
    )

    # combine train + val
    train_data = finetuner.load_jsonl(CONFIG["train_jsonl_path"])
    val_data = finetuner.load_jsonl(CONFIG["val_jsonl_path"])
    combined = train_data + val_data

    combined_jsonl = os.path.join(CONFIG["output_dir"], "combined_train_val.jsonl")
    finetuner.save_jsonl(combined, combined_jsonl)
    print(f"📄 Combined dataset saved to {combined_jsonl} ({len(combined)} samples)")

    final_model_dir = os.path.join(CONFIG["output_dir"], "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    trainer = finetuner.train_model(
        train_jsonl_path=combined_jsonl,
        val_jsonl_path=CONFIG["val_jsonl_path"],
        images_dir=CONFIG["images_dir"],
        output_dir=final_model_dir,
        num_epochs=CONFIG["num_epochs"],  # use fixed epochs from config
        batch_size=batch_size,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        warmup_ratio=best_params["warmup_ratio"],
        max_seq_length=best_params["max_seq_length"],
        save_strategy="epoch",
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    finetuner.save_model(trainer, final_model_dir)
    return finetuner, final_model_dir


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("QWEN2.5-VL INVENTORY FINE-TUNING + OPTUNA HPO")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["optuna_db_path"]), exist_ok=True)

    # save config
    cfg_path = os.path.join(CONFIG["output_dir"], "training_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    print(f"Config saved to {cfg_path}")

    # --------------------------
    # PHASE 1: OPTUNA
    # --------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: OPTUNA HPO (minimize validation CER)")
    print("=" * 60)

    storage = f"sqlite:///{CONFIG['optuna_db_path']}"
    study = optuna.create_study(
        study_name=CONFIG["optuna_study_name"],
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    remaining = CONFIG["n_trials"] - len(completed)
    print(f"Completed trials: {len(completed)}/{CONFIG['n_trials']}")
    print(f"Remaining trials: {remaining}")

    if remaining > 0:
        study.optimize(optuna_objective, n_trials=remaining, show_progress_bar=True)
    else:
        print("All trials already done, skipping HPO.")

    print("\n" + "=" * 60)
    print("OPTUNA DONE")
    print("=" * 60)
    print(f"Best trial #: {study.best_trial.number}")
    print(f"Best validation CER: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # save best params
    best_params_path = os.path.join(CONFIG["output_dir"], "best_hyperparameters.json")
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial_number": study.best_trial.number,
                "best_validation_cer": study.best_value,
                "best_params": study.best_params,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Best params saved to {best_params_path}")

    # --------------------------
    # PHASE 2: FINAL TRAIN
    # --------------------------
    finetuner, final_model_dir = train_final_with_best_params(study.best_params)

    # --------------------------
    # PHASE 3: TEST
    # --------------------------
    print("\n" + "=" * 60)
    print("PHASE 3: TEST EVALUATION")
    print("=" * 60)

    test_results = finetuner.evaluate_on_test_set(
        test_jsonl_path=CONFIG["test_jsonl_path"],
        images_dir=CONFIG["images_dir"],
        output_dir=CONFIG["output_dir"],
    )

    cer_stats = test_results["cer_stats"]

    summary_file = os.path.join(CONFIG["output_dir"], "final_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("QWEN2.5-VL INVENTORY - OPTUNA HPO RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best validation CER (objective): {study.best_value:.4f}\n")
        f.write("Best hyperparameters:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        f.write("Test CER:\n")
        f.write(f"  Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n")
        f.write(f"  Median CER: {cer_stats['median_cer']:.4f}\n")
        f.write(f"  Min CER: {cer_stats['minimum_cer']:.4f}\n")
        f.write(f"  Max CER: {cer_stats['maximum_cer']:.4f}\n")
        f.write(f"  Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']}\n")
        f.write("\n")
        f.write(f"Final model dir: {final_model_dir}\n")

    print(f"\n✅ Final summary saved to: {summary_file}")
    print("\n🎉 Qwen HPO pipeline finished.\n")


if __name__ == "__main__":
    main()
