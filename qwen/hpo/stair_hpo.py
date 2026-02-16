#!/usr/bin/env python3
# staircase_qwen_hpo_enhanced.py
#
# ENHANCED Hyperparameter optimization for Qwen2.5-VL-7B Staircase OCR with Unsloth.
#
# Key enhancements:
#   - Added more hyperparameters: LoRA rank, LoRA alpha, LoRA dropout, warmup ratio, max_seq_length
#   - Study name: "qwen_stair_enhanced_v2"
#   - Expanded search spaces for better optimization
#   - Fixed num_epochs = 5 for all trials (for speed)
#   - Single CER evaluation per trial on validation subset
#   - NO data augmentation during HPO
#   - NO checkpoints, NO TensorBoard logging during trials
#
# FIXES (to avoid the crash you saw):
#   1) Removed max_seq_length=512 from the search space (it can truncate image tokens and cause
#      "Image features and image tokens do not match")
#   2) Prune trials that hit the image-token mismatch instead of crashing the entire study
#   3) Prune CUDA OOM trials (sometimes thrown as torch.cuda.OutOfMemoryError, sometimes as RuntimeError)

import os

# ----------------------------------------------------------------------
# IMPORTANT: Unsloth / PyTorch env BEFORE importing unsloth/torch
# ----------------------------------------------------------------------
os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"          # keep grads on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import glob
import random
import tempfile
from typing import List, Dict, Any

import torch
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
    "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl",
    "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/val.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/staircase_images",

    # Fixed training setup for HPO
    "num_epochs": 5,          # fixed across trials for speed
    "batch_size": 1,

    # CER eval settings (post-training, once per trial)
    "cer_max_val_samples": 10,    # controls how many val images you use
    "cer_max_new_tokens": 512,

    # Optuna
    "optuna_n_trials": 30,  # total COMPLETE trials target
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "qwen_stair_enhanced_v2",

    # Where to store HPO outputs (best hyperparams etc.)
    "hpo_output_dir": "/home/vault/iwi5/iwi5298h/models_image_text/qwen/hpo/stair_enhanced",

    # Base directory for per-trial logs (no checkpoints)
    "trial_output_root": "/home/vault/iwi5/iwi5298h/models_image_text/qwen/hpo/stair_enhanced/trials",
}


# ======================================================================
# Compact JSON-schema prompt (same as your finetune script)
# ======================================================================

STAIRCASE_SCHEMA_PROMPT = """You are an OCR model for historical German staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this dataset (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
- Never drop a key that appears in the form's JSON structure. If a field is empty on the form, still include it with an empty string "" (or false for an unchecked box).
- Use booleans for checkbox options: true if the box is checked, false if it is empty.
- Use strings for numbers and free-text fields (measurements, dates, names, notes).
- Do NOT invent new fields.
"""


KEY_NORMALIZATION = {
    "Gesamt Ø cm": "Gesamt Durchmesser cm",
    "Gesamt Durchmesser cm": "Gesamt Durchmesser cm",
    "Gehlinie": "GEHLINIE",
    "Hohe": "Hohe",
}


# ======================================================================
# Helper class (enhanced for HPO with dynamic LoRA params)
# ======================================================================

class StaircaseOCRFinetuneHPO:
    def __init__(
        self,
        model_path: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        max_seq_length: int = 1024,
    ):
        print(f"[HPO] Loading Qwen2.5-VL model from {model_path} with Unsloth...")
        print(f"[HPO]   LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        print(f"[HPO]   max_seq_length={max_seq_length}")

        self.instruction = STAIRCASE_SCHEMA_PROMPT

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        # LoRA params configurable from HPO trial
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

        # No augmentation in HPO, but keep temp dir for compatibility/cleanup
        self.temp_dir = tempfile.mkdtemp(prefix="qwen_hpo_tmp_")
        print(f"[HPO] Created temporary dir: {self.temp_dir}")
        print("[HPO] Model loaded with LoRA for HPO.")

    # ------------- IO & JSON helpers ------------- #

    def load_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def dict_without_image_meta(self, obj: Dict) -> Dict:
        return {k: v for k, v in obj.items() if k not in ("image_name", "image_path")}

    def json_to_string_no_sort(self, obj: Dict) -> str:
        return json.dumps(
            self.dict_without_image_meta(obj),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def normalize_keys(self, d: Any) -> Any:
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                nk = KEY_NORMALIZATION.get(k, k)
                out[nk] = self.normalize_keys(v)
            return out
        elif isinstance(d, list):
            return [self.normalize_keys(x) for x in d]
        else:
            return d

    # ------------- JSON extraction ------------- #

    def extract_json_from_response(self, response: str) -> Dict:
        response = response.strip()
        matches = re.findall(r"\{.*\}", response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    # ------------- Image helper ------------- #

    def find_image_path(self, image_name: str, images_dir: str) -> str:
        exact_path = os.path.join(images_dir, image_name)
        if os.path.exists(exact_path):
            return exact_path

        pattern_match = re.search(r"\((\d+)\)\.jpg$", image_name)
        if pattern_match:
            pattern = f"({pattern_match.group(1)}).jpg"
            search_pattern = os.path.join(images_dir, f"*{pattern}")
            matching_files = glob.glob(search_pattern)
            if matching_files:
                return matching_files[0]

        base_name = os.path.splitext(image_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]

        return exact_path

    # ------------- Conversation conversion ------------- #

    def convert_to_conversation(self, sample: Dict) -> Dict:
        label_obj = self.dict_without_image_meta(sample)
        gt_json_string = self.json_to_string_no_sort(label_obj)

        image_path = sample.get("image_path", None)
        contents = [{"type": "text", "text": self.instruction}]

        if isinstance(image_path, str) and image_path and os.path.exists(image_path):
            contents.append({"type": "image", "image": image_path})
        else:
            print("[HPO] [WARN] Missing/invalid image_path, using text-only.")

        conversation = [
            {"role": "user", "content": contents},
            {"role": "assistant", "content": [{"type": "text", "text": gt_json_string}]},
        ]
        return {"messages": conversation}

    # ------------- Dataset prep (NO augmentation) ------------- #

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)
        converted_dataset = []

        print("[HPO] Preparing training data (no augmentation for HPO)")
        for i, item in enumerate(data):
            image_path = self.find_image_path(item["image_name"], images_dir)
            if os.path.exists(image_path):
                sample = item.copy()
                sample["image_path"] = image_path
                converted_dataset.append(self.convert_to_conversation(sample))
            else:
                print(f"[HPO] Warning: Image not found for {item['image_name']}, skipping...")

        print(
            f"[HPO] Total training samples: {len(converted_dataset)} "
            f"(original: {len(data)}, augmented: 0)"
        )
        return converted_dataset

    # ------------- CER helpers ------------- #

    def calculate_cer(self, predictions, targets):
        if not predictions or not targets or len(predictions) != len(targets):
            return 1.0

        total_cer = 0.0
        valid_pairs = 0

        for pred, target in zip(predictions, targets):
            pred_str, target_str = str(pred), str(target)
            if len(target_str) == 0:
                continue
            try:
                cer_val = jiwer.cer(target_str, pred_str)
            except Exception as e:
                print(f"[HPO] [WARN] CER computation failed for one sample: {e}")
                continue
            total_cer += cer_val
            valid_pairs += 1

        return (total_cer / valid_pairs) if valid_pairs > 0 else 1.0

    def cleanup_temp_files(self):
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"[HPO] Cleaned temp dir: {self.temp_dir}")
        except Exception as e:
            print(f"[HPO] Warning: could not clean temp dir {self.temp_dir}: {e}")


# ======================================================================
# Single CER evaluation (post-training, once per trial)
# ======================================================================

def evaluate_cer_on_validation(
    finetuner: StaircaseOCRFinetuneHPO,
    model,
    tokenizer,
    val_data: List[Dict],
    images_dir: str,
    max_samples: int,
    max_new_tokens: int,
) -> float:
    """
    Run generation on a small random subset of validation data and
    compute average CER. This is the HPO objective.

    Called ONCE per trial, after training.
    """
    print("\n" + "=" * 80)
    print("[HPO] RUNNING CER EVALUATION ON VALIDATION SET (HPO OBJECTIVE)")
    print("=" * 80)

    device = next(model.parameters()).device
    model.eval()

    # Subsample val data
    if len(val_data) > max_samples:
        val_subset = random.sample(val_data, max_samples)
    else:
        val_subset = list(val_data)

    print(f"[HPO]   Using {len(val_subset)} validation samples for CER")

    predictions: List[str] = []
    targets: List[str] = []

    for i, item in enumerate(val_subset):
        image_name = item["image_name"]
        image_path = finetuner.find_image_path(image_name, images_dir)

        print(
            f"[HPO]   Evaluating {i+1}/{len(val_subset)}: {image_name}",
            end="\r",
        )

        if not os.path.exists(image_path):
            continue

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": finetuner.instruction},
                        {"type": "image", "image": image_path},
                    ],
                }
            ]

            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            image_inputs, video_inputs = process_vision_info(messages)

            inputs = tokenizer(
                text=[input_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]

            generated_text = tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            predicted_json_raw = finetuner.extract_json_from_response(generated_text)
            predicted_json = finetuner.normalize_keys(
                finetuner.dict_without_image_meta(predicted_json_raw or {})
            )
            gt_obj = finetuner.normalize_keys(
                finetuner.dict_without_image_meta(item)
            )

            gt_str = finetuner.json_to_string_no_sort(gt_obj)
            pred_str = finetuner.json_to_string_no_sort(predicted_json)

            predictions.append(pred_str)
            targets.append(gt_str)

            del inputs, outputs, generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[HPO]   ⚠️  CER eval error on {image_name}: {e}")
            gt_obj = finetuner.normalize_keys(
                finetuner.dict_without_image_meta(item)
            )
            gt_str = finetuner.json_to_string_no_sort(gt_obj)
            predictions.append("")
            targets.append(gt_str)
            continue

    val_cer = finetuner.calculate_cer(predictions, targets)

    print(
        f"\n[HPO]   ✅ Validation CER (HPO objective): "
        f"{val_cer:.4f} ({val_cer*100:.2f}%)"
    )
    print("=" * 80 + "\n")

    return val_cer


# ======================================================================
# Single training run for one hyperparameter config
# ======================================================================

def train_one_config(
    config: Dict[str, Any],
    train_raw_jsonl_path: str,
    val_raw_jsonl_path: str,
    images_dir: str,
    trial_output_dir: str,
):
    """
    Train once with given config and return:
      best_cer, cer_history_dict

    NO model checkpoints are saved. Only logs + metrics.
    """
    print("\n" + "=" * 80)
    print("[HPO] Qwen Staircase OCR HPO TRIAL (ENHANCED MODE)")
    print("=" * 80)

    os.makedirs(trial_output_dir, exist_ok=True)

    config_file = os.path.join(trial_output_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[HPO] Trial output directory: {trial_output_dir}")
    print(f"[HPO]   • num_epochs: {config['num_epochs']}")
    print(f"[HPO]   • batch_size: {config['batch_size']}")
    print(f"[HPO]   • grad_accum: {config['gradient_accumulation_steps']}")
    print(f"[HPO]   • learning_rate: {config['learning_rate']}")
    print(f"[HPO]   • weight_decay: {config['weight_decay']}")
    print(f"[HPO]   • lora_r: {config['lora_r']}")
    print(f"[HPO]   • lora_alpha: {config['lora_alpha']}")
    print(f"[HPO]   • lora_dropout: {config['lora_dropout']}")
    print(f"[HPO]   • warmup_ratio: {config['warmup_ratio']}")
    print(f"[HPO]   • max_seq_length: {config['max_seq_length']}")

    finetuner = StaircaseOCRFinetuneHPO(
        model_path=config["model_path"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        max_seq_length=config["max_seq_length"],
    )

    train_dataset = finetuner.prepare_training_data(train_raw_jsonl_path, images_dir)
    val_raw_data = finetuner.load_jsonl(val_raw_jsonl_path)

    print(
        f"[HPO] Training samples: {len(train_dataset)}, "
        f"Validation raw records: {len(val_raw_data)}"
    )

    FastVisionModel.for_training(finetuner.model)

    trainer = SFTTrainer(
        model=finetuner.model,
        tokenizer=finetuner.tokenizer,
        data_collator=UnslothVisionDataCollator(finetuner.model, finetuner.tokenizer),
        train_dataset=train_dataset,
        eval_dataset=None,  # <- NO eval during training
        args=SFTConfig(
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],

            logging_steps=20,
            eval_strategy="no",    # <- IMPORTANT: no per-epoch eval
            save_strategy="no",    # <- IMPORTANT: no checkpoints
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=None,

            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            lr_scheduler_type="cosine",
            warmup_ratio=config["warmup_ratio"],
            optim="adamw_8bit",
            gradient_checkpointing=False,

            remove_unused_columns=False,
            dataset_text_field=None,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=config["max_seq_length"],

            report_to="none",  # <- no TensorBoard during HPO
            logging_dir=os.path.join(trial_output_dir, "logs"),
            seed=3407,
            output_dir=trial_output_dir,
        ),
    )

    print("[HPO] Starting training for this trial (NO eval during training)...\n")
    trainer.train()
    print("[HPO] Training finished for this trial.")

    # Single CER eval on small validation subset (HPO objective)
    best_cer = evaluate_cer_on_validation(
        finetuner=finetuner,
        model=finetuner.model,
        tokenizer=finetuner.tokenizer,
        val_data=val_raw_data,
        images_dir=images_dir,
        max_samples=CONFIG["cer_max_val_samples"],
        max_new_tokens=CONFIG["cer_max_new_tokens"],
    )

    cer_history = [{"epoch": float(config["num_epochs"]), "cer": float(best_cer)}]

    print(
        f"[HPO]   Best Validation CER in this trial: {best_cer:.4f} "
        f"({best_cer*100:.2f}%)"
    )

    finetuner.cleanup_temp_files()
    del trainer
    del finetuner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_cer, cer_history


# ======================================================================
# Optuna HPO driver (ENHANCED with more hyperparameters)
# ======================================================================

def run_optuna_hpo():
    db_path = os.path.abspath(CONFIG["optuna_db_path"])
    storage_url = f"sqlite:///{db_path}"

    print("\n" + "=" * 80)
    print("[HPO] Starting / Resuming Optuna HPO for Qwen Staircase OCR (ENHANCED MODE)")
    print("=" * 80)
    print(f"[HPO]   Storage: {storage_url}")
    print(f"[HPO]   Study name: {CONFIG['optuna_study_name']}")
    print(f"[HPO]   Target total trials (COMPLETE): {CONFIG['optuna_n_trials']}")

    storage = optuna.storages.RDBStorage(url=storage_url)
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)

    study = optuna.create_study(
        study_name=CONFIG["optuna_study_name"],
        direction="minimize",
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

        def objective(trial: optuna.trial.Trial):
            config = CONFIG.copy()

            # Fixed epochs & batch size for speed
            config["num_epochs"] = CONFIG["num_epochs"]
            config["batch_size"] = CONFIG["batch_size"]

            # Core training hyperparameters
            config["learning_rate"] = trial.suggest_float(
                "learning_rate", 5e-6, 2e-4, log=True
            )
            config["weight_decay"] = trial.suggest_float(
                "weight_decay", 0.0, 0.15
            )
            config["gradient_accumulation_steps"] = trial.suggest_categorical(
                "gradient_accumulation_steps", [4, 8, 16]
            )

            # LoRA architecture hyperparameters
            config["lora_r"] = trial.suggest_categorical(
                "lora_r", [8, 16, 32, 64]
            )
            config["lora_alpha"] = trial.suggest_categorical(
                "lora_alpha", [16, 32, 64, 128]
            )
            config["lora_dropout"] = trial.suggest_float(
                "lora_dropout", 0.0, 0.2
            )

            # Training schedule hyperparameters
            config["warmup_ratio"] = trial.suggest_float(
                "warmup_ratio", 0.0, 0.2
            )

            # ------------------------------------------------------------------
            # FIX 1: Remove 512 from max_seq_length search space
            # ------------------------------------------------------------------
            config["max_seq_length"] = trial.suggest_categorical(
                "max_seq_length", [1024, 2048]
            )

            trial_output_dir = os.path.join(
                CONFIG["trial_output_root"],
                f"trial_{trial.number}",
            )

            try:
                best_cer, cer_history = train_one_config(
                    config=config,
                    train_raw_jsonl_path=train_path,
                    val_raw_jsonl_path=val_path,
                    images_dir=images_dir,
                    trial_output_dir=trial_output_dir,
                )

            except ValueError as e:
                # ------------------------------------------------------------------
                # FIX 2: Prune image-token mismatch failures
                # ------------------------------------------------------------------
                msg = str(e)
                if "Image features and image tokens do not match" in msg:
                    raise optuna.TrialPruned(msg)
                raise

            except torch.cuda.OutOfMemoryError as e:
                # ------------------------------------------------------------------
                # FIX 3a: Prune CUDA OOM trials (explicit OOM exception)
                # ------------------------------------------------------------------
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"CUDA OOM: {e}")

            except RuntimeError as e:
                # ------------------------------------------------------------------
                # FIX 3b: Prune CUDA OOM trials (OOM sometimes appears as RuntimeError)
                # ------------------------------------------------------------------
                if "CUDA out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned(f"CUDA OOM(RuntimeError): {e}")
                raise

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            trial.set_user_attr("cer_history", cer_history)
            return best_cer

        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
        )

    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("[HPO] Qwen Staircase HPO finished / resumed (ENHANCED MODE)")
    print("=" * 80)
    print(f"[HPO]   Best trial number: {best_trial.number}")
    print(f"[HPO]   Best Validation CER: {best_trial.value:.4f} "
          f"({best_trial.value*100:.2f}%)")
    print("[HPO]   Best params:")
    for k, v in best_trial.params.items():
        print(f"[HPO]      {k}: {v}")

    hpo_out_dir = CONFIG["hpo_output_dir"]
    os.makedirs(hpo_out_dir, exist_ok=True)

    best_hparams = {
        "best_trial_number": best_trial.number,
        "best_validation_cer": best_trial.value,
        "best_params": best_trial.params,
    }
    best_hparams_path = os.path.join(hpo_out_dir, "best_hyperparameters.json")
    with open(best_hparams_path, "w", encoding="utf-8") as f:
        json.dump(best_hparams, f, indent=2, ensure_ascii=False)

    best_config = CONFIG.copy()
    for k, v in best_trial.params.items():
        best_config[k] = v
    best_config_path = os.path.join(hpo_out_dir, "best_config.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(hpo_out_dir, "hpo_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Optuna HPO Summary - Qwen2.5-VL-7B Staircase OCR (ENHANCED MODE)\n")
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
        f.write("\n" + "=" * 80 + "\n")
        f.write("ENHANCED HYPERPARAMETERS SEARCHED:\n")
        f.write("=" * 80 + "\n")
        f.write("Core Training:\n")
        f.write("  - learning_rate: [5e-6, 2e-4] (log scale)\n")
        f.write("  - weight_decay: [0.0, 0.15]\n")
        f.write("  - gradient_accumulation_steps: [4, 8, 16]\n")
        f.write("\nLoRA Architecture:\n")
        f.write("  - lora_r: [8, 16, 32, 64]\n")
        f.write("  - lora_alpha: [16, 32, 64, 128]\n")
        f.write("  - lora_dropout: [0.0, 0.2]\n")
        f.write("\nTraining Schedule:\n")
        f.write("  - warmup_ratio: [0.0, 0.2]\n")
        f.write("\nModel Architecture:\n")
        f.write("  - max_seq_length: [1024, 2048]  (512 removed to prevent image-token mismatch)\n")

    print(f"\n[HPO] ✅ Best hyperparameters saved to: {best_hparams_path}")
    print(f"[HPO] ✅ Best config saved to:          {best_config_path}")
    print(f"[HPO] ✅ HPO summary saved to:         {summary_path}")
    print("\n[HPO] 🎉 HPO done. You can now copy these values into your main Qwen finetune .py file.\n")


# ======================================================================
# main()
# ======================================================================

def main():
    run_optuna_hpo()


if __name__ == "__main__":
    main()
