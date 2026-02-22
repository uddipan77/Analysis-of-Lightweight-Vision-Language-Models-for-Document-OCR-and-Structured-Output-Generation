#!/usr/bin/env python3
# qwen_inventory_finetune_unsloth_optuna_code2_aligned.py
#
# ✅ Optuna HPO aligned to Code2 evaluation logic:
#   - Validation CER computed on RAW JSON strings (NO sort_keys canonicalization)
#   - Deterministic greedy decoding (temperature=0.0, do_sample=False)
#   - Same JSON serialization style as Code2: json_to_string_no_sort()
#
# ✅ Still fast:
#   - Train only 5 epochs per trial
#   - Evaluate once at end on validation (your val has 44 anyway)
#
# Notes:
# - Keep training "similar" to Code2 (as much as possible while still allowing HPO).
# - Removed stochastic test decoding; and test eval uses greedy too (optional).
# - Uses robust image path finding like Code2 (optional but safer).

import os
os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import gc
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import jiwer
import optuna
import glob

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

from transformers import EarlyStoppingCallback, TrainerCallback
from qwen_vl_utils import process_vision_info


# ============================================================
# INVENTORY PROMPT (same as Code2)
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

    # data
    "train_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
    "val_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
    "test_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",

    # output
    "output_dir": "/home/vault/iwi5/iwi5298h/models_image_text/qwen/hpo/new_inven_code2_aligned",

    # optuna
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "qwen_new_inventory_dataset_code2_aligned",
    "n_trials": 20,

    # fixed to keep trials fast
    "num_epochs": 5,
    "batch_size": 1,

    # validation objective: use ALL val if smaller than this
    "val_eval_max_samples": 50,

    # ---------------------------
    # Search spaces (Optuna)
    # ---------------------------
    "learning_rate_range": (1e-5, 5e-4),
    "weight_decay_range": (0.0, 0.1),
    "grad_accum_choices": [4, 8, 16],
    "lora_r_choices": [16, 32, 64],
    "lora_alpha_choices": [16, 32, 64, 128],
    "lora_dropout_range": (0.0, 0.2),
    "warmup_ratio_range": (0.0, 0.2),
    "max_seq_length_choices": [1024, 2048],  # avoid 512
}


# ============================================================
# Finetuner (aligned to Code2)
# ============================================================
class InventoryOCRFinetune:
    def __init__(
        self,
        model_path: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        max_seq_length: int,
    ):
        self.instruction = INVENTORY_SCHEMA_PROMPT

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        # NOTE: Code2 used use_gradient_checkpointing=False & use_rslora=False.
        # Here we keep it consistent with Code2 to reduce mismatch.
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=3407,
            use_rslora=False,
        )

    # ---------- IO ----------
    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def save_jsonl(self, data: List[Dict[str, Any]], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ---------- JSON helpers (Code2 style: NO sort keys) ----------
    def dict_without_image_meta(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in obj.items() if k not in ("image_name", "image_path")}

    def json_to_string_no_sort(self, obj: Dict[str, Any]) -> str:
        return json.dumps(
            self.dict_without_image_meta(obj),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        response = response.strip()
        matches = re.findall(r"\{.*\}", response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except Exception:
                    continue
        try:
            return json.loads(response)
        except Exception:
            return {}

    # ---------- image path (robust, Code2-ish) ----------
    def find_image_path(self, image_name: str, images_dir: str) -> str:
        exact_path = os.path.join(images_dir, image_name)
        if os.path.exists(exact_path):
            return exact_path

        base_name = os.path.splitext(image_name)[0]
        matches = glob.glob(os.path.join(images_dir, f"*{base_name}*"))
        if matches:
            return matches[0]
        return exact_path

    # ---------- dataset prep ----------
    def convert_to_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        gt_json_string = self.json_to_string_no_sort(sample)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.instruction},
                    {"type": "image", "image": sample["image_path"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gt_json_string}],
            },
        ]
        return {"messages": conversation}

    def prepare_data(self, jsonl_path: str, images_dir: str) -> List[Dict[str, Any]]:
        data = self.load_jsonl(jsonl_path)
        converted = []
        for item in data:
            img_path = self.find_image_path(item["image_name"], images_dir)
            if not os.path.exists(img_path):
                continue
            item = dict(item)
            item["image_path"] = img_path
            converted.append(self.convert_to_conversation(item))
        return converted

    # ---------- CER (raw-string CER like Code2) ----------
    def calculate_cer(self, preds: List[str], tgts: List[str]) -> float:
        if not preds or not tgts or len(preds) != len(tgts):
            return 1.0
        total = 0.0
        n = 0
        for p, t in zip(preds, tgts):
            if not t:
                continue
            try:
                total += jiwer.cer(t, p)
                n += 1
            except Exception:
                continue
        return (total / n) if n > 0 else 1.0

    # ✅ Objective evaluation aligned to Code2:
    # - greedy decode
    # - extract JSON
    # - CER over json_to_string_no_sort(pred_json) vs json_to_string_no_sort(gt_obj)
    def evaluate_on_validation_code2_aligned(
        self,
        val_jsonl_path: str,
        images_dir: str,
        max_samples: int = 50,
    ) -> float:
        FastVisionModel.for_inference(self.model)
        self.model.eval()

        val_data = self.load_jsonl(val_jsonl_path)

        # deterministic selection (same as before)
        val_data_sorted = sorted(val_data, key=lambda x: x.get("image_name", ""))
        val_subset = val_data_sorted[: min(max_samples, len(val_data_sorted))]

        preds, tgts = [], []

        device = "cuda"
        for item in val_subset:
            image_path = self.find_image_path(item["image_name"], images_dir)
            if not os.path.exists(image_path):
                continue

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.instruction},
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
                        max_new_tokens=1024,
                        use_cache=True,
                        temperature=0.0,
                        do_sample=False,          # ✅ deterministic (Code2)
                        repetition_penalty=1.0,
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

                pred_json_raw = self.extract_json_from_response(generated_text)
                pred_json = self.dict_without_image_meta(pred_json_raw or {})

                gt_obj = self.dict_without_image_meta(item)

                pred_str = self.json_to_string_no_sort(pred_json) if pred_json else ""
                gt_str = self.json_to_string_no_sort(gt_obj)

                preds.append(pred_str)
                tgts.append(gt_str)

                del inputs, outputs, generated_ids_trimmed
                torch.cuda.empty_cache()

            except Exception:
                gt_obj = self.dict_without_image_meta(item)
                tgts.append(self.json_to_string_no_sort(gt_obj))
                preds.append("")
                continue

        val_cer = self.calculate_cer(preds, tgts)

        FastVisionModel.for_training(self.model)
        return float(val_cer)

    # ---------- train (keep similar to Code2; still supports HPO knobs) ----------
    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        gradient_accumulation_steps: int,
        warmup_ratio: float,
        max_seq_length: int,
        save_strategy: str,
        report_to: str,
    ):
        train_dataset = self.prepare_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_data(val_jsonl_path, images_dir)

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

                # keep your style; warmup_ratio is meaningful; warmup_steps fixed
                warmup_steps=50,
                warmup_ratio=warmup_ratio,

                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=10,

                eval_strategy="epoch",
                save_strategy=save_strategy,
                save_total_limit=1,

                dataloader_num_workers=0,
                dataloader_pin_memory=False,

                weight_decay=weight_decay,
                lr_scheduler_type="cosine",

                # Code2: you manage best-by-CER externally; in HPO we do end-of-trial eval
                load_best_model_at_end=False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,

                optim="adamw_8bit",
                gradient_checkpointing=False,   # ✅ match Code2
                remove_unused_columns=False,

                dataset_text_field=None,
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

        trainer.train()
        return trainer


# ============================================================
# OPTUNA OBJECTIVE (Code2-aligned val CER)
# ============================================================
def optuna_objective(trial: optuna.trial.Trial):
    print("\n" + "=" * 80)
    print(f"OPTUNA TRIAL {trial.number} ({CONFIG['optuna_study_name']})")
    print("=" * 80)

    num_epochs = CONFIG["num_epochs"]
    batch_size = CONFIG["batch_size"]

    learning_rate = trial.suggest_float(
        "learning_rate",
        CONFIG["learning_rate_range"][0],
        CONFIG["learning_rate_range"][1],
        log=True,
    )
    weight_decay = trial.suggest_float(
        "weight_decay",
        CONFIG["weight_decay_range"][0],
        CONFIG["weight_decay_range"][1],
    )
    gradient_accumulation_steps = trial.suggest_categorical(
        "gradient_accumulation_steps",
        CONFIG["grad_accum_choices"],
    )
    lora_r = trial.suggest_categorical("lora_r", CONFIG["lora_r_choices"])
    lora_alpha = trial.suggest_categorical("lora_alpha", CONFIG["lora_alpha_choices"])
    lora_dropout = trial.suggest_float(
        "lora_dropout",
        CONFIG["lora_dropout_range"][0],
        CONFIG["lora_dropout_range"][1],
    )
    warmup_ratio = trial.suggest_float(
        "warmup_ratio",
        CONFIG["warmup_ratio_range"][0],
        CONFIG["warmup_ratio_range"][1],
    )
    max_seq_length = trial.suggest_categorical(
        "max_seq_length",
        CONFIG["max_seq_length_choices"],
    )

    print(f"  • num_epochs:     {num_epochs} (fixed)")
    print(f"  • batch_size:     {batch_size} (fixed)")
    print(f"  • learning_rate:  {learning_rate:.2e}")
    print(f"  • weight_decay:   {weight_decay:.4f}")
    print(f"  • grad_accum:     {gradient_accumulation_steps}")
    print(f"  • lora_r:         {lora_r}")
    print(f"  • lora_alpha:     {lora_alpha}")
    print(f"  • lora_dropout:   {lora_dropout:.4f}")
    print(f"  • warmup_ratio:   {warmup_ratio:.4f}")
    print(f"  • max_seq_length: {max_seq_length}")

    # per-trial dir
    trial_tmp_dir = os.path.join(CONFIG["output_dir"], "optuna_tmp")
    os.makedirs(trial_tmp_dir, exist_ok=True)

    finetuner = InventoryOCRFinetune(
        model_path=CONFIG["model_path"],
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
    )

    # ✅ objective: Code2-aligned validation CER (raw strings, no sort, greedy)
    val_cer = finetuner.evaluate_on_validation_code2_aligned(
        val_jsonl_path=CONFIG["val_jsonl_path"],
        images_dir=CONFIG["images_dir"],
        max_samples=CONFIG["val_eval_max_samples"],
    )

    print(f"  ✅ Trial {trial.number} validation CER (Code2-aligned) = {val_cer:.4f}")

    # cleanup
    del trainer
    del finetuner
    torch.cuda.empty_cache()
    gc.collect()

    return val_cer


# ============================================================
# FINAL TRAIN (still 5 epochs, because you said you'll paste HPs into Code2 anyway)
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("QWEN2.5-VL INVENTORY OPTUNA HPO (CODE2-ALIGNED VAL CER)")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["optuna_db_path"]), exist_ok=True)

    cfg_path = os.path.join(CONFIG["output_dir"], "training_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)
    print(f"Config saved to {cfg_path}")

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
    print("OPTUNA DONE (CODE2-ALIGNED)")
    print("=" * 60)
    print(f"Best trial #: {study.best_trial.number}")
    print(f"Best validation CER: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

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

    print("\nDone. Paste best_params into Code2 and run full training/eval there.")


if __name__ == "__main__":
    main()
