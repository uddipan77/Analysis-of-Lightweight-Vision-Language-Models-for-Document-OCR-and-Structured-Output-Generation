#!/usr/bin/env python3
# qwen_schmuck_hpo.py
#
# Hyperparameter optimization for Qwen2.5-VL-7B on the SCHMUCK dataset (OCR → JSON).
#
# - Uses Unsloth FastVisionModel (4-bit) + Qwen VL chat template
# - Trains on train.jsonl
# - AFTER training for all epochs of a trial, runs generation on a subset
#   of val.jsonl and computes canonical JSON CER
# - Optuna objective = lowest validation CER per trial
# - No per-epoch CER callbacks, no checkpoints inside trials
# - Best hyperparameters & config are saved under a timestamped HPO folder

import os

# ----------------------------------------------------------------------
# Environment for Unsloth / PyTorch
# ----------------------------------------------------------------------
os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"          # keep grads on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import glob
from datetime import datetime
from typing import List, Dict, Any

import torch
import numpy as np
import jiwer
from PIL import Image

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

from qwen_vl_utils import process_vision_info

import optuna
from optuna.trial import TrialState

# ======================================================================
# Global HPO config
# ======================================================================

CONFIG = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/qwen7b",
    "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
    "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",

    # Base (reference) hyperparameters – center of the search space
    "batch_size": 2,
    "learning_rate": 0.0001680720977739039,
    "weight_decay": 0.08927180304353628,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.052058449429580246,
    "gradient_accumulation_steps": 8,
    "num_epochs": 8,

    # Optuna
    "optuna_n_trials": 30,
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "qwen_schmuck",

    # Where HPO run folders (with timestamp) will be created
    "hpo_output_root": "/home/vault/iwi5/iwi5298h/models_image_text/qwen/hpo/schmuck",
}

# ======================================================================
# Schmuck-specific instruction (same semantics as your finetune script)
# ======================================================================

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

Include ALL fields, even if empty (use empty string ""). Preserve exact German spelling and punctuation. Return ONLY the JSON object without any additional text or formatting.
"""

# ======================================================================
# Helper class for Qwen SCHMUCK HPO
# ======================================================================


class SchmuckOCRHPO:
    def __init__(self, model_path: str, lora_r: int, lora_alpha: int, lora_dropout: float):
        print(f"[HPO] Loading Qwen2.5-VL model from {model_path} with Unsloth...")

        self.instruction = SCHMUCK_INSTRUCTION

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        # LoRA params are trial-specific
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
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )

        print(
            f"[HPO] Model loaded with LoRA (r={lora_r}, alpha={lora_alpha}, "
            f"dropout={lora_dropout:.4f})"
        )

    # ---------------- I/O helpers ----------------

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def clean_dict_for_ground_truth(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        # Remove Schmück metadata keys
        return {
            k: v
            for k, v in obj.items()
            if k not in ["file_name", "image_path"]
        }

    def json_to_string_no_sort(self, obj: Dict[str, Any]) -> str:
        clean_obj = self.clean_dict_for_ground_truth(obj)
        return json.dumps(clean_obj, ensure_ascii=False, separators=(",", ":"))

    def safe_json_loads(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    def canonical_json_string(self, obj: Any) -> str:
        if obj is None:
            return ""
        return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))

    # ---------------- JSON extraction ----------------

    def extract_json_from_response(self, response: str) -> Dict:
        # Copied logic from your finetune code, with minor robustness tweaks
        if isinstance(response, list):
            response = response[0] if response else ""
        response = str(response).strip()

        if "{" in response and "}" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]

            # Try to isolate first complete JSON
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
                    if isinstance(parsed, dict):
                        return parsed
            except Exception:
                pass

            # Fallback: try entire span
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}

        # Final fallback
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}

        return {}

    # ---------------- Image utilities ----------------

    def find_image_path(self, file_name: str, images_dir: str) -> str:
        exact_path = os.path.join(images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path

        base_name, _ = os.path.splitext(file_name)
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matches = glob.glob(search_pattern)
        if matches:
            return matches[0]

        return exact_path  # might not exist – handled later

    # ---------------- Conversation conversion ----------------

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
                "content": [
                    {"type": "text", "text": gt_json_string},
                ],
            },
        ]
        return {"messages": conversation}

    # ---------------- Dataset prep ----------------

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)

        valid_data = []
        for item in data:
            if "file_name" not in item:
                print("[HPO] Warning: 'file_name' missing in item, skipping.")
                continue
            image_path = self.find_image_path(item["file_name"], images_dir)
            if os.path.exists(image_path):
                item_copy = item.copy()
                item_copy["image_path"] = image_path
                valid_data.append(item_copy)
            else:
                print(f"[HPO] Warning: Image not found: {item['file_name']}, skipping...")

        print(f"[HPO] Training samples (with valid images): {len(valid_data)}")
        converted_dataset = [self.convert_to_conversation(sample) for sample in valid_data]
        return converted_dataset

    # ---------------- CER helpers ----------------

    def calculate_cer(self, predictions: List[str], targets: List[str]) -> float:
        if (
            not predictions
            or not targets
            or len(predictions) != len(targets)
        ):
            return 1.0

        total_cer = 0.0
        num = 0
        for pred, target in zip(predictions, targets):
            pred_str, target_str = str(pred), str(target)
            if len(target_str) == 0:
                continue
            try:
                cer_val = jiwer.cer(target_str, pred_str)
            except Exception:
                continue
            total_cer += cer_val
            num += 1

        return (total_cer / num) if num > 0 else 1.0

    def calculate_cer_json(self, pred_json_strings: List[str], target_json_strings: List[str]) -> float:
        return self.calculate_cer(pred_json_strings, target_json_strings)

    # ---------------- Validation evaluation after training ----------------

    def evaluate_on_validation(
        self,
        val_jsonl_path: str,
        images_dir: str,
        max_eval_samples: int = 30,
    ) -> float:
        print("\n[HPO] === Running validation CER evaluation (one-shot, post-training) ===")

        FastVisionModel.for_inference(self.model)
        device = next(self.model.parameters()).device

        val_data = self.load_jsonl(val_jsonl_path)
        if not val_data:
            print("[HPO] WARNING: Validation file is empty; returning CER=1.0")
            return 1.0

        # Deterministic subset: first N samples
        subset = val_data[:max_eval_samples]
        print(f"[HPO] Using {len(subset)} validation samples for CER")

        predictions_str = []
        targets_str = []

        for i, item in enumerate(subset):
            file_name = item.get("file_name", f"val_{i}")
            print(
                f"[HPO]   Val sample {i+1}/{len(subset)}: {file_name}",
                end="\r",
            )

            image_path = self.find_image_path(file_name, images_dir)
            if not os.path.exists(image_path):
                print(f"\n[HPO]   ⚠️ Image not found for {file_name}, skipping.")
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
                        do_sample=False,
                        repetition_penalty=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]

                decoded = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                generated_text = decoded[0] if decoded else ""

                predicted_json = self.extract_json_from_response(generated_text)
                pred_json_string = (
                    self.json_to_string_no_sort(predicted_json)
                    if predicted_json
                    else ""
                )

                gt_json_string = self.json_to_string_no_sort(item)

                predictions_str.append(pred_json_string)
                targets_str.append(gt_json_string)

                del inputs, outputs, generated_ids_trimmed
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n[HPO]   ⚠️ Validation error on {file_name}: {e}")
                predictions_str.append("")
                targets_str.append(self.json_to_string_no_sort(item))
                continue

        val_cer = self.calculate_cer_json(predictions_str, targets_str)

        print(
            f"\n[HPO]   ✅ Validation CER (post-training): "
            f"{val_cer:.4f} ({val_cer*100:.2f}%)\n"
        )

        FastVisionModel.for_training(self.model)
        return val_cer


# ======================================================================
# Single trial training (no eval during training, eval only after)
# ======================================================================

def train_one_config(
    config: Dict[str, Any],
    train_jsonl_path: str,
    val_jsonl_path: str,
    images_dir: str,
    trial_output_dir: str,
    trial: optuna.trial.Trial,
):
    """
    Train once with a given hyperparameter config and return:
       val_cer

    - SFTTrainer trains for num_epochs on train.jsonl
    - After training finishes, we run one-shot generation-based CER on val.jsonl
    - No checkpoints are saved inside the trial folder.
    """

    print("\n" + "=" * 80)
    print("[HPO] Qwen SCHMUCK OCR HPO TRIAL")
    print("=" * 80)
    print(f"[HPO] Trial #{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)

    # Save trial config
    cfg_path = os.path.join(trial_output_dir, "training_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[HPO] Trial output directory: {trial_output_dir}")
    print(f"[HPO]   • num_epochs: {config['num_epochs']}")
    print(f"[HPO]   • batch_size: {config['batch_size']}")
    print(f"[HPO]   • grad_accum: {config['gradient_accumulation_steps']}")
    print(f"[HPO]   • learning_rate: {config['learning_rate']}")
    print(f"[HPO]   • weight_decay: {config['weight_decay']}")
    print(f"[HPO]   • lora_r: {config['lora_r']}")
    print(f"[HPO]   • lora_alpha: {config['lora_alpha']}")
    print(f"[HPO]   • lora_dropout: {config['lora_dropout']:.4f}")

    finetuner = SchmuckOCRHPO(
        model_path=config["model_path"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )

    train_dataset = finetuner.prepare_training_data(train_jsonl_path, images_dir)
    print(f"[HPO] Training samples: {len(train_dataset)}")

    FastVisionModel.for_training(finetuner.model)

    trainer = SFTTrainer(
        model=finetuner.model,
        tokenizer=finetuner.tokenizer,
        data_collator=UnslothVisionDataCollator(finetuner.model, finetuner.tokenizer),
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],

            logging_steps=10,
            eval_strategy="no",             # <--- NO eval during training
            save_strategy="no",             # <--- NO checkpoints
            load_best_model_at_end=False,
            greater_is_better=False,

            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            optim="adamw_8bit",
            gradient_checkpointing=True,

            remove_unused_columns=False,
            dataset_text_field=None,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,

            report_to="tensorboard",
            logging_dir=os.path.join(trial_output_dir, "logs"),
            seed=3407,
            output_dir=trial_output_dir,
        ),
    )

    print("[HPO] Starting training for this trial (no eval during training)...\n")
    trainer.train()

    # After full training, run one-shot validation CER
    val_cer = finetuner.evaluate_on_validation(
        val_jsonl_path=val_jsonl_path,
        images_dir=images_dir,
        max_eval_samples=30,
    )

    # Attach CER to trial for logging / analysis
    trial.set_user_attr("val_cer", float(val_cer))

    # Cleanup
    del trainer
    del finetuner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(val_cer)


# ======================================================================
# Optuna driver
# ======================================================================

def run_optuna_hpo():
    # Create timestamped HPO run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hpo_run_dir = os.path.join(CONFIG["hpo_output_root"], f"hpo_run_{timestamp}")
    os.makedirs(hpo_run_dir, exist_ok=True)

    db_path = os.path.abspath(CONFIG["optuna_db_path"])
    storage_url = f"sqlite:///{db_path}"

    print("\n" + "=" * 80)
    print("[HPO] Starting / Resuming Optuna HPO for Qwen SCHMUCK OCR")
    print("=" * 80)
    print(f"[HPO]   Storage: {storage_url}")
    print(f"[HPO]   Study name: {CONFIG['optuna_study_name']}")
    print(f"[HPO]   HPO run folder: {hpo_run_dir}")
    print(f"[HPO]   Target COMPLETE trials: {CONFIG['optuna_n_trials']}")

    storage = optuna.storages.RDBStorage(url=storage_url)
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)

    study = optuna.create_study(
        study_name=CONFIG["optuna_study_name"],
        direction="minimize",    # minimize validation CER
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    all_trials = study.get_trials(deepcopy=False)
    completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]
    n_completed = len(completed_trials)
    target = CONFIG["optuna_n_trials"]
    remaining = max(target - n_completed, 0)

    print(f"[HPO]   Completed trials so far: {n_completed}")
    print(f"[HPO]   Remaining trials to run: {remaining}")

    train_path = CONFIG["train_jsonl_path"]
    val_path = CONFIG["val_jsonl_path"]
    images_dir = CONFIG["images_dir"]

    if remaining > 0:

        def objective(trial: optuna.trial.Trial) -> float:
            config = CONFIG.copy()

            # ---------------- Hyperparameter search space ----------------
            config["learning_rate"] = trial.suggest_float(
                "learning_rate",
                5e-5,
                5e-4,
                log=True,
            )
            config["weight_decay"] = trial.suggest_float(
                "weight_decay",
                0.01,
                0.1,
                log=True,
            )
            config["lora_r"] = trial.suggest_categorical(
                "lora_r",
                [8, 16, 32],
            )
            config["lora_alpha"] = trial.suggest_categorical(
                "lora_alpha",
                [16, 32, 64],
            )
            config["lora_dropout"] = trial.suggest_float(
                "lora_dropout",
                0.0,
                0.15,
            )
            config["gradient_accumulation_steps"] = trial.suggest_categorical(
                "gradient_accumulation_steps",
                [4, 8, 16],
            )
            config["num_epochs"] = trial.suggest_int(
                "num_epochs",
                5,
                10,
            )

            trial_output_dir = os.path.join(
                hpo_run_dir,
                f"trial_{trial.number:03d}",
            )

            val_cer = train_one_config(
                config=config,
                train_jsonl_path=train_path,
                val_jsonl_path=val_path,
                images_dir=images_dir,
                trial_output_dir=trial_output_dir,
                trial=trial,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return val_cer

        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
        )

    # ======================= After HPO =======================

    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("[HPO] Qwen SCHMUCK HPO finished / resumed")
    print("=" * 80)
    print(f"[HPO]   Best trial number: {best_trial.number}")
    print(
        f"[HPO]   Best Validation CER: {best_trial.value:.4f} "
        f"({best_trial.value*100:.2f}%)"
    )
    print("[HPO]   Best params:")
    for k, v in best_trial.params.items():
        print(f"[HPO]      {k}: {v}")

    # Save best hyperparams + config into this run folder
    best_hparams = {
        "best_trial_number": best_trial.number,
        "best_validation_cer": float(best_trial.value),
        "best_params": best_trial.params,
    }
    best_hparams_path = os.path.join(hpo_run_dir, "best_hyperparameters.json")
    with open(best_hparams_path, "w", encoding="utf-8") as f:
        json.dump(best_hparams, f, indent=2, ensure_ascii=False)

    best_config = CONFIG.copy()
    for k, v in best_trial.params.items():
        best_config[k] = v
    best_config_path = os.path.join(hpo_run_dir, "best_config.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(hpo_run_dir, "hpo_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Optuna HPO Summary - Qwen2.5-VL-7B SCHMUCK OCR\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Study name: {CONFIG['optuna_study_name']}\n")
        f.write(f"Storage: sqlite:///{os.path.abspath(CONFIG['optuna_db_path'])}\n")
        f.write(f"Total requested COMPLETE trials: {CONFIG['optuna_n_trials']}\n")
        f.write(f"Best trial number: {best_trial.number}\n")
        f.write(
            f"Best Validation CER: {best_trial.value:.4f} "
            f"({best_trial.value*100:.2f}%)\n\n"
        )
        f.write("Best Hyperparameters:\n")
        for k, v in best_trial.params.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n[HPO] ✅ Best hyperparameters saved to: {best_hparams_path}")
    print(f"[HPO] ✅ Best config saved to:          {best_config_path}")
    print(f"[HPO] ✅ HPO summary saved to:         {summary_path}")
    print(
        "\n[HPO] 🎉 HPO done. You can now copy these values into your main "
        "Qwen SCHMUCK finetune script.\n"
    )


# ======================================================================
# main()
# ======================================================================

def main():
    run_optuna_hpo()


if __name__ == "__main__":
    main()
