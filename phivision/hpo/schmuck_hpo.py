#!/usr/bin/env python3
# phi_vision_schmuck_hpo_fast.py
#
# Fast Hyperparameter Optimization for Phi-3.5-Vision on SCHMUCK dataset.
# Key design:
#   - num_epochs fixed to 8 (centered on your good config)
#   - NO per-epoch CER callback (no generation during training)
#   - Single CER evaluation per trial on a small random val subset
#   - No augmentation for HPO (speed + stability)
#   - 4-bit QLoRA + gradient checkpointing for memory
#   - No checkpoints, no TensorBoard logging during HPO
#
# Objective: minimize validation CER (generation-based) per trial.

import os
import json
import random
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Any

# ========= IMPORTANT: help with fragmentation BEFORE torch import =========
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ==========================================================================

import torch
from PIL import Image, ImageEnhance
import jiwer
import optuna
from optuna.trial import TrialState

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================================
# Configuration - Schmuck dataset & HPO
# ============================================================================

CONFIG: Dict[str, Any] = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision",
    "train_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
    "val_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",

    # ---------- HPO training setup (fixed across trials) ----------
    "num_epochs": 8,
    "batch_size": 1,

    # Center of search space
    "gradient_accumulation_steps": 8,
    "learning_rate": 1.680720977739039e-4,
    "weight_decay": 0.08927180304353628,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.052058449429580246,
    "max_grad_norm": 1.0,

    # No augmentation for HPO
    "data_augmentation": False,

    # Quantization
    "use_4bit": True,
    "use_nested_quant": True,

    # ========= NEW: memory-related config =========
    "max_seq_length": 2048,        # keep context bounded
    "processor_num_crops": 8,      # fewer crops than 16 to reduce vision tokens
    # =================================================

    # Optuna
    "optuna_n_trials": 30,  # total desired COMPLETE trials across all runs
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "phi_schmuck",

    # Base directory for HPO outputs; timestamped run folder is created inside
    "hpo_output_base_dir": "/home/vault/iwi5/iwi5298h/models_image_text/phi/hpo_schmuck",

    # CER evaluation settings (for HPO objective)
    "cer_max_val_samples": 20,    # evaluate CER on at most this many val images
    "cer_max_new_tokens": 512,    # shorter generations to speed up eval
}


# ============================================================================
# Instruction Prompt for Schmuck Dataset
# ============================================================================

INSTRUCTION = """Extract all information from this German historical jewelry/schmuck catalog image and return ONLY a JSON object with exactly these keys:

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


# ============================================================================
# Utility functions
# ============================================================================

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def create_label_string(json_data: Dict) -> str:
    """Ground truth without file_name (metadata)."""
    label_data = {k: v for k, v in json_data.items() if k != "file_name"}
    return json.dumps(label_data, ensure_ascii=False, sort_keys=False)


def extract_json_from_response(response: str) -> str:
    """Extract FIRST JSON from response to prevent repetition issues."""
    response = response.strip()

    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]

        # Try to isolate first complete JSON object
        try:
            brace_count = 0
            first_end = -1
            for i, char in enumerate(json_str):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        first_end = i + 1
                        break

            if first_end > 0:
                first_json = json_str[:first_end]
                parsed = json.loads(first_json)
                return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            pass

        # Fallback: try parsing full substring
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            return json_str

    return response


# ============================================================================
# Data Augmentation (kept, but disabled for HPO)
# ============================================================================

class DocumentImageAugmenter:
    """Augmentation for historical documents/catalog images"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def augment(self, image: Image.Image) -> Image.Image:
        if not self.enabled or random.random() > 0.7:
            return image

        # Random brightness (±15%)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))

        # Random contrast (±15%)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))

        # Random sharpness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))

        # Small rotation (±2 degrees)
        if random.random() > 0.7:
            angle = random.uniform(-2, 2)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        return image


# ============================================================================
# Schmuck Dataset & Collator
# ============================================================================

class SchmuckDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        processor,
        instruction: str,
        augment: bool = False,
    ):
        self.data = load_jsonl(jsonl_path)
        self.images_dir = images_dir
        self.processor = processor
        self.instruction = instruction
        self.augmenter = DocumentImageAugmenter(enabled=augment)

        # Filter valid samples
        self.valid_samples = []
        for item in self.data:
            file_name = item["file_name"]
            image_path = os.path.join(self.images_dir, file_name)
            if os.path.exists(image_path):
                self.valid_samples.append(item)

        print(
            f"   📊 Loaded {len(self.valid_samples)} valid samples "
            f"(out of {len(self.data)} total)"
        )
        if augment:
            print("   🔄 Data augmentation ENABLED")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        image_path = os.path.join(self.images_dir, item["file_name"])

        # Load and (optionally) augment image
        image = Image.open(image_path).convert("RGB")
        image = self.augmenter.augment(image)

        # Ground truth JSON string (excluding file_name)
        gt_json_str = create_label_string(item)

        # Chat messages
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{self.instruction}",
            },
            {
                "role": "assistant",
                "content": gt_json_str,
            },
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # ========= NEW: enforce max_length + truncation =========
        inputs = self.processor(
            prompt,
            [image],
            return_tensors="pt",
            max_length=self.processor.tokenizer.model_max_length,
            truncation=True,
        )
        # ========================================================

        # Teacher forcing: labels = input_ids, with instruction masked as -100
        labels = inputs["input_ids"].clone()

        # Mask everything up to and including "<|assistant|>" token
        assistant_token = self.processor.tokenizer.encode(
            "<|assistant|>", add_special_tokens=False
        )
        input_ids_list = inputs["input_ids"][0].tolist()

        try:
            for i in range(len(input_ids_list) - len(assistant_token)):
                if input_ids_list[i: i + len(assistant_token)] == assistant_token:
                    labels[0, : i + len(assistant_token)] = -100
                    break
        except Exception:
            pass

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0)
            if "pixel_values" in inputs
            else None,
            "image_sizes": inputs["image_sizes"].squeeze(0)
            if "image_sizes" in inputs
            else None,
            "labels": labels.squeeze(0),
            "file_name": item["file_name"],
            "ground_truth": gt_json_str,
        }


@dataclass
class DataCollatorForPhi3Vision:
    processor: AutoProcessor

    def __call__(self, features):
        batch = {}
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack(
            [f["attention_mask"] for f in features]
        )
        batch["labels"] = torch.stack([f["labels"] for f in features])

        if features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.stack(
                [f["pixel_values"] for f in features]
            )

        if features[0]["image_sizes"] is not None:
            batch["image_sizes"] = torch.stack(
                [f["image_sizes"] for f in features]
            )

        return batch


# ============================================================================
# Single CER evaluation (post-training, once per trial)
# ============================================================================

def evaluate_cer_on_validation(
    model,
    processor,
    val_data: List[Dict],
    images_dir: str,
    instruction: str,
    max_samples: int,
    max_new_tokens: int,
) -> float:
    """
    Run generation on a small random subset of validation data and
    compute average CER. This is the HPO objective.
    """
    print("\n" + "=" * 80)
    print("🔍 RUNNING CER EVALUATION ON SCHMUCK VALIDATION SET (HPO OBJECTIVE)")
    print("=" * 80)

    device = next(model.parameters()).device
    model.eval()

    # Subsample val data
    if len(val_data) > max_samples:
        val_subset = random.sample(val_data, max_samples)
    else:
        val_subset = list(val_data)

    print(f"   • Using {len(val_subset)} validation samples for CER")

    predictions: List[str] = []
    targets: List[str] = []

    for i, item in enumerate(val_subset):
        file_name = item["file_name"]
        image_path = os.path.join(images_dir, file_name)

        print(f"   Evaluating {i+1}/{len(val_subset)}: {file_name}", end="\r")

        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": f"<|image_1|>\n{instruction}",
                }
            ]

            prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # ========= NEW: enforce max_length + truncation =========
            inputs = processor(
                prompt,
                [image],
                return_tensors="pt",
                max_length=processor.tokenizer.model_max_length,
                truncation=True,
            ).to(device)
            # ========================================================

            with torch.no_grad():
                generate_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
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
            ground_truth = create_label_string(item)

            predictions.append(prediction)
            targets.append(ground_truth)

            del inputs, generate_ids
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n   ⚠️  CER eval error on {file_name}: {e}")
            predictions.append("")
            targets.append(create_label_string(item))
            continue

    # Compute CER
    if not predictions or not targets:
        return 1.0

    total_cer = 0.0
    valid_pairs = 0
    for pred, target in zip(predictions, targets):
        if len(target) > 0:
            try:
                cer_score = jiwer.cer(target, pred)
            except Exception:
                cer_score = 1.0
            total_cer += cer_score
            valid_pairs += 1

    avg_cer = total_cer / valid_pairs if valid_pairs > 0 else 1.0

    print(
        f"\n   ✅ Validation CER (HPO objective): "
        f"{avg_cer:.4f} ({avg_cer * 100:.2f}%)"
    )
    print("=" * 80 + "\n")

    return avg_cer


# ============================================================================
# Single training run for one hyperparameter config
# ============================================================================

def train_one_config(
    config: Dict[str, Any],
    processor,
    bnb_config: BitsAndBytesConfig,
    train_dataset,
    val_data: List[Dict],
    data_collator,
    output_dir: str,
) -> Tuple[float, Dict]:
    """
    Train once with given config and return:
      val_cer (float), info dict
    """
    print("\n" + "=" * 80)
    print("PHI-3.5-VISION FINE-TUNING (HPO TRIAL) - SCHMUCK DATASET")
    print("BitsAndBytes QLoRA 4-bit (Memory-Optimized)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Save per-run config
    config_file = os.path.join(output_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n📂 Trial output directory: {output_dir}")
    print(f"   • Training epochs: {config['num_epochs']}")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"   • Learning rate: {config['learning_rate']}")
    print(f"   • Weight decay: {config['weight_decay']}")
    print(
        f"   • LoRA r={config['lora_r']}, alpha={config['lora_alpha']}, "
        f"dropout={config['lora_dropout']}"
    )

    print(f"\n🛡️  Memory Optimization:")
    print(f"   • 4-bit NF4 Quantization: {config['use_4bit']}")
    print(f"   • Nested Quantization: {config['use_nested_quant']}")
    print(f"\n🔧 Anti-Repetition for eval:")
    print(f"   • First-JSON extraction: enabled")
    print(f"   • Using temperature=0.0 for deterministic generation")

    print(f"\n⏳ Loading Phi-3.5-Vision with 4-bit quantization...")

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )

    print("   ✅ Model loaded with 4-bit quantization")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(
            f"   📊 GPU Memory - Allocated: {allocated:.2f} GB, "
            f"Reserved: {reserved:.2f} GB"
        )

    # ========= CHANGED: enable gradient checkpointing =========
    print("\n📝 Preparing model for LoRA training (WITH gradient checkpointing)...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    # ==========================================================

    # Apply LoRA
    lora_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print("   ✅ LoRA applied successfully")
    print(
        f"   📊 Trainable params: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.2f}%)"
    )

    # TrainingArguments: NO eval during training, NO checkpoints, NO TB
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="no",   # no checkpoints
        eval_strategy="no",   # no per-epoch eval
        load_best_model_at_end=False,
        fp16=False,
        bf16=True,
        # ========= CHANGED: turn ON gradient checkpointing =========
        gradient_checkpointing=True,
        # ==========================================================
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",     # no tensorboard during HPO
        optim="paged_adamw_8bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    # Train
    print("🚀 Starting QLoRA training on SCHMUCK dataset (HPO trial)...\n")
    print("=" * 80)
    trainer.train()
    print("=" * 80)
    print("✅ TRAINING FINISHED FOR THIS TRIAL")
    print("=" * 80)

    # Single CER eval on small validation subset (HPO objective)
    cer_value = evaluate_cer_on_validation(
        model=model,
        processor=processor,
        val_data=val_data,
        images_dir=config["images_dir"],
        instruction=INSTRUCTION,
        max_samples=CONFIG["cer_max_val_samples"],
        max_new_tokens=CONFIG["cer_max_new_tokens"],
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clean up
    del trainer
    del model

    info = {
        "val_cer": cer_value,
    }

    return cer_value, info


# ============================================================================
# Optuna HPO driver
# ============================================================================

def run_optuna_hpo_only(
    processor,
    bnb_config: BitsAndBytesConfig,
    train_dataset,
    val_data,
    data_collator,
):
    # Build SQLite URL
    db_path = os.path.abspath(CONFIG["optuna_db_path"])
    storage_url = f"sqlite:///{db_path}"

    # Timestamped run directory under base HPO output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CONFIG["hpo_output_base_dir"], f"run_{timestamp}")
    trials_root = os.path.join(run_dir, "trials")
    os.makedirs(trials_root, exist_ok=True)

    print("\n" + "=" * 80)
    print("🔎 Starting / Resuming Optuna HPO (Phi-3.5-Vision, SCHMUCK, FAST MODE)")
    print("=" * 80)
    print(f"   • Storage: {storage_url}")
    print(f"   • Study name: {CONFIG['optuna_study_name']}")
    print(f"   • Target total trials: {CONFIG['optuna_n_trials']}")
    print(f"   • Run directory: {run_dir}")

    storage = optuna.storages.RDBStorage(url=storage_url)
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)

    study = optuna.create_study(
        study_name=CONFIG["optuna_study_name"],
        direction="minimize",  # minimize CER
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    # Count completed trials
    all_trials = study.get_trials(deepcopy=False)
    completed_trials = [t for t in all_trials if t.state == TrialState.COMPLETE]
    n_completed = len(completed_trials)
    target = CONFIG["optuna_n_trials"]
    remaining = max(target - n_completed, 0)

    print(f"   • Completed trials so far: {n_completed}")
    print(f"   • Remaining trials to run: {remaining}")

    if remaining > 0:

        def objective(trial: optuna.trial.Trial):
            # Copy base config and override with sampled hyperparameters
            config = CONFIG.copy()

            # Fixed epochs & batch size for speed
            config["num_epochs"] = CONFIG["num_epochs"]
            config["batch_size"] = CONFIG["batch_size"]

            # Learning rate around your best (log scale)
            config["learning_rate"] = trial.suggest_float(
                "learning_rate", 5e-5, 4e-4, log=True
            )

            # Weight decay
            config["weight_decay"] = trial.suggest_float(
                "weight_decay", 0.0, 0.1
            )

            # LoRA rank, alpha, dropout
            config["lora_r"] = trial.suggest_categorical(
                "lora_r", [8, 16, 24, 32]
            )
            config["lora_alpha"] = trial.suggest_categorical(
                "lora_alpha", [16, 32, 48, 64]
            )
            config["lora_dropout"] = trial.suggest_float(
                "lora_dropout", 0.05, 0.15
            )

            # Gradient accumulation
            config["gradient_accumulation_steps"] = trial.suggest_categorical(
                "gradient_accumulation_steps", [8, 16]
            )

            # Output dir per trial
            trial_output_dir = os.path.join(trials_root, f"trial_{trial.number}")

            val_cer, info = train_one_config(
                config=config,
                processor=processor,
                bnb_config=bnb_config,
                train_dataset=train_dataset,
                val_data=val_data,
                data_collator=data_collator,
                output_dir=trial_output_dir,
            )

            trial.set_user_attr("val_cer", val_cer)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return val_cer

        # Run remaining trials
        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
        )

    # Save best hyperparams to files (inside this run_dir)
    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("🏆 Optuna HPO finished / resumed (Phi-3.5-Vision, SCHMUCK, FAST MODE)")
    print("=" * 80)
    print(f"   Best trial number: {best_trial.number}")
    print(f"   Best Validation CER: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)")
    print(f"   Best params:")
    for k, v in best_trial.params.items():
        print(f"      {k}: {v}")

    hpo_out_dir = run_dir
    os.makedirs(hpo_out_dir, exist_ok=True)

    # 1) Best hyperparameters only
    best_hparams = {
        "best_trial_number": best_trial.number,
        "best_validation_cer": best_trial.value,
        "best_params": best_trial.params,
    }
    best_hparams_path = os.path.join(hpo_out_dir, "best_hyperparameters.json")
    with open(best_hparams_path, "w", encoding="utf-8") as f:
        json.dump(best_hparams, f, indent=2, ensure_ascii=False)

    # 2) Full CONFIG with best params applied
    best_config = CONFIG.copy()
    for k, v in best_trial.params.items():
        best_config[k] = v
    best_config_path = os.path.join(hpo_out_dir, "best_config.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    # 3) Human-readable summary
    summary_path = os.path.join(hpo_out_dir, "hpo_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Optuna HPO Summary - PHI-3.5-Vision SCHMUCK OCR (FAST MODE)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Study name: {CONFIG['optuna_study_name']}\n")
        f.write(f"Storage: sqlite:///{os.path.abspath(CONFIG['optuna_db_path'])}\n")
        f.write(f"Total requested trials: {CONFIG['optuna_n_trials']}\n")
        f.write(f"Best trial number: {best_trial.number}\n")
        f.write(
            f"Best Validation CER: {best_trial.value:.4f} "
            f"({best_trial.value*100:.2f}%)\n\n"
        )
        f.write("Best Hyperparameters:\n")
        for k, v in best_trial.params.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n✅ Best hyperparameters saved to: {best_hparams_path}")
    print(f"✅ Best config saved to:          {best_config_path}")
    print(f"✅ HPO summary saved to:         {summary_path}")
    print("\n🎉 HPO run complete. You can now copy these hyperparameters into your main SCHMUCK fine-tuning script.\n")


# ============================================================================
# Main
# ============================================================================

def main():
    # Shared 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["use_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=CONFIG["use_nested_quant"],
    )

    # Processor shared across trials
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        num_crops=CONFIG["processor_num_crops"],   # fewer crops for HPO
    )
    # Match staircase: left truncation, bounded context
    processor.tokenizer.truncation_side = "left"
    processor.tokenizer.model_max_length = CONFIG["max_seq_length"]
    print("   ✅ Processor loaded")

    # Load datasets once
    print("\n📊 Loading SCHMUCK datasets...")

    print("   📁 Training dataset:")
    train_dataset = SchmuckDataset(
        CONFIG["train_jsonl"],
        CONFIG["images_dir"],
        processor,
        INSTRUCTION,
        augment=CONFIG["data_augmentation"],  # False for HPO
    )

    print("   📁 Validation dataset (for length / sanity check):")
    _ = SchmuckDataset(
        CONFIG["val_jsonl"],
        CONFIG["images_dir"],
        processor,
        INSTRUCTION,
        augment=False,
    )

    # Raw validation JSON for CER eval
    val_data = load_jsonl(CONFIG["val_jsonl"])
    print(f"   📊 Raw validation samples for CER computation: {len(val_data)}")

    data_collator = DataCollatorForPhi3Vision(processor=processor)

    # Run HPO only
    run_optuna_hpo_only(
        processor=processor,
        bnb_config=bnb_config,
        train_dataset=train_dataset,
        val_data=val_data,
        data_collator=data_collator,
    )


if __name__ == "__main__":
    main()
