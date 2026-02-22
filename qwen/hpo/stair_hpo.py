#!/usr/bin/env python3
# staircase_qwen_hpo_better_fullval_aug.py
#
# ✅ Better Optuna HPO for Qwen2.5-VL-7B on Staircase, aligned to your real pipeline:
#   - Train 5 epochs per trial (fast)
#   - Augmentation ENABLED during HPO (same spirit as your finetune script)
#   - Evaluate on the ENTIRE validation set (no random subset)
#   - Autoregressive generation CER (greedy, deterministic)
#   - max_new_tokens increased to 1024 (match your main scripts)
#   - Deterministic seeds fixed per trial
#   - No checkpoints / no TB during trials
#
# Note:
# - This will be slower than your old HPO because it uses full val + augmentation.
# - If it becomes too slow, reduce augmentation factor (0 or 1) or reduce #trials.

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
import gc
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import jiwer
from PIL import Image
import torchvision.transforms as transforms

import optuna
from optuna.trial import TrialState

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

from qwen_vl_utils import process_vision_info


# ======================================================================
# CONFIG
# ======================================================================

CONFIG = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/qwen7b",
    "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl",
    "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/val.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/staircase_images",

    # HPO behavior
    "num_epochs": 5,
    "batch_size": 1,

    # Full-val autoreg evaluation
    "eval_max_new_tokens": 1024,   # ✅ match your main scripts
    "eval_use_cache": True,

    # Augmentation inside HPO
    "use_augmentation": True,
    "augment_factor": 1,           # 0 disables; 1 matches your training

    # Optuna
    "optuna_n_trials": 30,
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "qwen_stair_fullval_aug_v1",   # ✅ NEW STUDY NAME

    # Output
    "hpo_output_dir": "/home/vault/iwi5/iwi5298h/models_image_text/qwen/hpo/stair_fullval_aug_v1",
    "trial_output_root": "/home/vault/iwi5/iwi5298h/models_image_text/qwen/hpo/stair_fullval_aug_v1/trials",

    # Determinism
    "seed": 3407,
}


# ======================================================================
# PROMPT (same as your finetune scripts)
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
# SEEDING
# ======================================================================

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# FINETUNER (HPO) - includes augmentation + full-val autoreg CER
# ======================================================================

class StaircaseOCRFinetuneHPO:
    def __init__(
        self,
        model_path: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        max_seq_length: int,
        seed: int,
        augment_factor: int = 0,
        use_augmentation: bool = False,
    ):
        self.instruction = STAIRCASE_SCHEMA_PROMPT
        self.seed = seed
        self.augment_factor = int(augment_factor)
        self.use_augmentation = bool(use_augmentation)

        # Temp dir for augmented images per trial
        self.temp_dir = tempfile.mkdtemp(prefix="qwen_hpo_aug_")

        # Load model
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

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
            random_state=seed,
            use_rslora=False,
        )

    # --------------------- IO --------------------- #

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def dict_without_image_meta(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in obj.items() if k not in ("image_name", "image_path")}

    def json_to_string_no_sort(self, obj: Dict[str, Any]) -> str:
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
        return d

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

    # --------------------- Image path --------------------- #

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

    # --------------------- Augmentation --------------------- #

    def create_augmentation_transforms(self):
        # Same “spirit” as your main script
        return [
            transforms.RandomRotation(degrees=(-2, 2), fill=255),
            transforms.ColorJitter(brightness=(0.9, 1.1)),
            transforms.ColorJitter(contrast=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
            transforms.RandomResizedCrop(
                size=(512, 512),
                scale=(0.95, 1.0),
                ratio=(0.9, 1.1),
            ),
        ]

    def augment_image(self, image_path: str, aug_id: int) -> Optional[str]:
        try:
            image = Image.open(image_path).convert("RGB")

            # Deterministic-ish: since we seed before trial, random calls will be repeatable
            transforms_list = self.create_augmentation_transforms()
            num_transforms = random.randint(2, min(3, len(transforms_list)))
            selected_transforms = random.sample(transforms_list, num_transforms)

            augmented_image = image
            for t in selected_transforms:
                augmented_image = t(augmented_image)

            original_name = os.path.basename(image_path)
            name_without_ext, ext = os.path.splitext(original_name)
            augmented_name = f"{name_without_ext}_aug{aug_id}{ext}"
            augmented_path = os.path.join(self.temp_dir, augmented_name)
            augmented_image.save(augmented_path, quality=95)
            return augmented_path
        except Exception:
            return None

    # --------------------- Conversation conversion --------------------- #

    def convert_to_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        label_obj = self.dict_without_image_meta(sample)
        gt_json_string = self.json_to_string_no_sort(label_obj)

        contents = [{"type": "text", "text": self.instruction}]
        contents.append({"type": "image", "image": sample["image_path"]})

        return {
            "messages": [
                {"role": "user", "content": contents},
                {"role": "assistant", "content": [{"type": "text", "text": gt_json_string}]},
            ]
        }

    # --------------------- Dataset prep --------------------- #

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict[str, Any]]:
        data = self.load_jsonl(jsonl_path)
        out = []

        for item in data:
            image_path = self.find_image_path(item["image_name"], images_dir)
            if not os.path.exists(image_path):
                continue

            base = dict(item)
            base["image_path"] = image_path
            out.append(self.convert_to_conversation(base))

            if self.use_augmentation and self.augment_factor > 0:
                for aug_id in range(1, self.augment_factor + 1):
                    aug_path = self.augment_image(image_path, aug_id)
                    if aug_path and os.path.exists(aug_path):
                        aug_item = dict(item)
                        aug_item["image_path"] = aug_path
                        # Keep image_name unique; labels remain same (same JSON)
                        aug_item["image_name"] = f"{item['image_name']}_aug{aug_id}"
                        out.append(self.convert_to_conversation(aug_item))

        return out

    # --------------------- CER --------------------- #

    def calculate_cer(self, preds: List[str], tgts: List[str]) -> float:
        if not preds or not tgts or len(preds) != len(tgts):
            return 1.0

        total, n = 0.0, 0
        for p, t in zip(preds, tgts):
            if not t:
                continue
            try:
                total += jiwer.cer(t, p)
                n += 1
            except Exception:
                continue
        return (total / n) if n > 0 else 1.0

    # ✅ FULL validation evaluation (entire val.jsonl), autoregressive generate, greedy
    def evaluate_full_validation_cer(
        self,
        val_jsonl_path: str,
        images_dir: str,
        max_new_tokens: int = 1024,
    ) -> float:
        FastVisionModel.for_inference(self.model)
        self.model.eval()

        val_data = self.load_jsonl(val_jsonl_path)

        preds, tgts = [], []
        device = "cuda"

        for item in val_data:
            image_path = self.find_image_path(item["image_name"], images_dir)
            if not os.path.exists(image_path):
                # count as failure for that sample
                gt_obj = self.normalize_keys(self.dict_without_image_meta(item))
                tgts.append(self.json_to_string_no_sort(gt_obj))
                preds.append("")
                continue

            try:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.instruction},
                        {"type": "image", "image": image_path},
                    ],
                }]

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
                        max_new_tokens=max_new_tokens,
                        use_cache=CONFIG["eval_use_cache"],
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

                generated_text = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                pred_json_raw = self.extract_json_from_response(generated_text)
                pred_json = self.normalize_keys(self.dict_without_image_meta(pred_json_raw or {}))
                gt_obj = self.normalize_keys(self.dict_without_image_meta(item))

                preds.append(self.json_to_string_no_sort(pred_json))
                tgts.append(self.json_to_string_no_sort(gt_obj))

                del inputs, outputs, generated_ids_trimmed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception:
                gt_obj = self.normalize_keys(self.dict_without_image_meta(item))
                tgts.append(self.json_to_string_no_sort(gt_obj))
                preds.append("")
                continue

        val_cer = self.calculate_cer(preds, tgts)

        FastVisionModel.for_training(self.model)
        return float(val_cer)

    def cleanup(self):
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass


# ======================================================================
# ONE TRIAL: train 5 epochs, then full-val CER
# ======================================================================

def run_one_trial(trial: optuna.trial.Trial) -> float:
    # Deterministic per-trial seed (stable but different across trials)
    base_seed = int(CONFIG["seed"])
    trial_seed = base_seed + int(trial.number)
    set_all_seeds(trial_seed)

    # Suggest hyperparams
    learning_rate = trial.suggest_float("learning_rate", 5e-6, 2e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.15)
    grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [4, 8, 16])

    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128])
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)

    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    max_seq_length = trial.suggest_categorical("max_seq_length", [1024, 2048])

    # Per-trial output dir (logs only)
    trial_output_dir = os.path.join(CONFIG["trial_output_root"], f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)

    # Save trial config
    trial_cfg = {
        "seed": trial_seed,
        "num_epochs": CONFIG["num_epochs"],
        "batch_size": CONFIG["batch_size"],
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_accumulation_steps": grad_accum,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "warmup_ratio": warmup_ratio,
        "max_seq_length": max_seq_length,
        "use_augmentation": CONFIG["use_augmentation"],
        "augment_factor": CONFIG["augment_factor"],
        "eval_max_new_tokens": CONFIG["eval_max_new_tokens"],
    }
    with open(os.path.join(trial_output_dir, "trial_config.json"), "w", encoding="utf-8") as f:
        json.dump(trial_cfg, f, indent=2, ensure_ascii=False)

    finetuner = StaircaseOCRFinetuneHPO(
        model_path=CONFIG["model_path"],
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_length=max_seq_length,
        seed=trial_seed,
        augment_factor=CONFIG["augment_factor"],
        use_augmentation=CONFIG["use_augmentation"],
    )

    try:
        # Data
        train_dataset = finetuner.prepare_training_data(CONFIG["train_jsonl_path"], CONFIG["images_dir"])
        # (We don't need an eval_dataset during training for HPO)
        FastVisionModel.for_training(finetuner.model)

        trainer = SFTTrainer(
            model=finetuner.model,
            tokenizer=finetuner.tokenizer,
            data_collator=UnslothVisionDataCollator(finetuner.model, finetuner.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=None,
            args=SFTConfig(
                per_device_train_batch_size=CONFIG["batch_size"],
                per_device_eval_batch_size=CONFIG["batch_size"],
                gradient_accumulation_steps=grad_accum,

                num_train_epochs=CONFIG["num_epochs"],
                learning_rate=learning_rate,
                weight_decay=weight_decay,

                lr_scheduler_type="cosine",
                warmup_ratio=warmup_ratio,
                warmup_steps=50,  # keep consistent with your main script style

                logging_steps=20,
                eval_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,

                dataloader_num_workers=0,
                dataloader_pin_memory=False,

                optim="adamw_8bit",
                gradient_checkpointing=False,

                remove_unused_columns=False,
                dataset_text_field=None,
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=max_seq_length,

                report_to="none",
                logging_dir=os.path.join(trial_output_dir, "logs"),
                seed=trial_seed,
                output_dir=trial_output_dir,
            ),
        )

        trainer.train()

        # ✅ Full validation CER, autoregressive generate
        val_cer = finetuner.evaluate_full_validation_cer(
            val_jsonl_path=CONFIG["val_jsonl_path"],
            images_dir=CONFIG["images_dir"],
            max_new_tokens=CONFIG["eval_max_new_tokens"],
        )

        return float(val_cer)

    except torch.cuda.OutOfMemoryError as e:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise optuna.TrialPruned(f"CUDA OOM: {e}")

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.TrialPruned(f"CUDA OOM(RuntimeError): {e}")
        raise

    finally:
        try:
            del trainer
        except Exception:
            pass
        finetuner.cleanup()
        del finetuner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ======================================================================
# OPTUNA DRIVER
# ======================================================================

def run_optuna():
    os.makedirs(CONFIG["hpo_output_dir"], exist_ok=True)
    os.makedirs(CONFIG["trial_output_root"], exist_ok=True)

    # Save master config
    with open(os.path.join(CONFIG["hpo_output_dir"], "hpo_master_config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    storage_url = f"sqlite:///{os.path.abspath(CONFIG['optuna_db_path'])}"
    storage = optuna.storages.RDBStorage(url=storage_url)
    sampler = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        group=True,
    )

    study = optuna.create_study(
        study_name=CONFIG["optuna_study_name"],
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    all_trials = study.get_trials(deepcopy=False)
    completed = [t for t in all_trials if t.state == TrialState.COMPLETE]
    remaining = max(CONFIG["optuna_n_trials"] - len(completed), 0)

    print("\n" + "=" * 80)
    print("[HPO] Qwen Staircase HPO (FULL VAL + AUG) starting / resuming")
    print("=" * 80)
    print(f"[HPO] Storage: {storage_url}")
    print(f"[HPO] Study:   {CONFIG['optuna_study_name']}")
    print(f"[HPO] Done:    {len(completed)} / {CONFIG['optuna_n_trials']}")
    print(f"[HPO] Remain:  {remaining}")
    print("=" * 80 + "\n")

    if remaining > 0:
        study.optimize(run_one_trial, n_trials=remaining, gc_after_trial=True)

    best = study.best_trial

    print("\n" + "=" * 80)
    print("[HPO] DONE - Best trial")
    print("=" * 80)
    print(f"Best trial #: {best.number}")
    print(f"Best val CER: {best.value:.6f} ({best.value*100:.2f}%)")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Save best params/config
    best_hparams = {
        "best_trial_number": best.number,
        "best_validation_cer": best.value,
        "best_params": best.params,
    }
    with open(os.path.join(CONFIG["hpo_output_dir"], "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_hparams, f, indent=2, ensure_ascii=False)

    best_config = dict(CONFIG)
    for k, v in best.params.items():
        best_config[k] = v
    with open(os.path.join(CONFIG["hpo_output_dir"], "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    print(f"\n[HPO] Saved best_hyperparameters.json and best_config.json to:")
    print(f"      {CONFIG['hpo_output_dir']}\n")


def main():
    run_optuna()


if __name__ == "__main__":
    main()
