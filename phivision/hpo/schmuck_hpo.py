#!/usr/bin/env python3
# phi_vision_schmuck_hpo_v2_fixedval_aug_crops16.py
#
# Optuna HPO for Phi-3.5-Vision on SCHMUCK with:
#   ✅ NO checkpoints, NO TensorBoard (fast HPO)
#   ✅ Train 5 epochs per trial
#   ✅ Single CER eval per trial on a FIXED 50-image validation subset (same across all trials)
#   ✅ Match main finetune regime better:
#        - num_crops = 16
#        - max_seq_length enforced + truncation (left truncation)
#        - max_new_tokens = 1024 for eval
#        - augmentation ENABLED (like main training)
#
# Objective: minimize CER on the fixed validation subset.

import os
import json
import random
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# ========= IMPORTANT: help with fragmentation BEFORE torch import =========
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ========================================================================

import numpy as np
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


# =============================================================================
# CONFIG
# =============================================================================

CONFIG: Dict[str, Any] = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision",
    "train_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
    "val_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",

    # --- HPO regime (fixed) ---
    "num_epochs": 5,
    "batch_size": 1,
    "max_seq_length": 2048,
    "processor_num_crops": 16,  # MATCH main finetune

    # --- augmentation: MATCH main finetune ---
    "data_augmentation": True,

    # --- quantization ---
    "use_4bit": True,
    "use_nested_quant": True,

    # --- evaluation regime (fixed across trials) ---
    "fixed_val_subset_size": 50,
    "fixed_val_subset_seed": 42,   # ensures same 50 samples every run
    "eval_max_new_tokens": 1024,   # MATCH main finetune eval
    "eval_temperature": 0.0,
    "eval_do_sample": False,

    # --- Optuna ---
    "optuna_n_trials": 30,
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "phi_schmuck_v2_fixed50_aug_crops16",
    "hpo_output_base_dir": "/home/vault/iwi5/iwi5298h/models_image_text/phi/hpo_schmuck_v2",

    # --- reproducibility / variance control ---
    # Use same global seed for ALL trials to reduce noise from augmentation/shuffling.
    # (Hyperparams still change learning dynamics; but the randomness is consistent.)
    "global_seed": 3407,

    # optional: set this if you want to cap training steps for faster trials
    "max_steps": -1,  # -1 means no cap
}


# =============================================================================
# Instruction Prompt (same as your main)
# =============================================================================

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


# =============================================================================
# Utilities
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    label_data = {k: v for k, v in json_data.items() if k != "file_name"}
    return json.dumps(label_data, ensure_ascii=False, sort_keys=False)

def extract_json_from_response(response: str) -> str:
    response = response.strip()

    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]

        # isolate first complete JSON object
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
                return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            pass

        # fallback parse whole substring
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            return json_str

    return response


# =============================================================================
# Augmentation (enabled here to match main training)
# =============================================================================

class DocumentImageAugmenter:
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
# Dataset & Collator (ENFORCE max_length + truncation + left truncation)
# =============================================================================

class SchmuckDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        processor,
        instruction: str,
        augment: bool = False,
        max_seq_length: int = 2048,
    ):
        self.data = load_jsonl(jsonl_path)
        self.images_dir = images_dir
        self.processor = processor
        self.instruction = instruction
        self.augmenter = DocumentImageAugmenter(enabled=augment)
        self.max_seq_length = max_seq_length

        self.valid_samples = []
        for item in self.data:
            image_path = os.path.join(self.images_dir, item["file_name"])
            if os.path.exists(image_path):
                self.valid_samples.append(item)

        print(f"   📊 Loaded {len(self.valid_samples)} valid samples (out of {len(self.data)} total)")
        if augment:
            print("   🔄 Data augmentation ENABLED")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        image_path = os.path.join(self.images_dir, item["file_name"])

        image = Image.open(image_path).convert("RGB")
        image = self.augmenter.augment(image)

        gt_json_str = create_label_string(item)

        messages = [
            {"role": "user", "content": f"<|image_1|>\n{self.instruction}"},
            {"role": "assistant", "content": gt_json_str},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        inputs = self.processor(
            prompt,
            [image],
            return_tensors="pt",
            max_length=self.max_seq_length,
            truncation=True,
        )

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
            "file_name": item["file_name"],
            "ground_truth": gt_json_str,
        }


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
# Fixed validation subset builder
# =============================================================================

def build_fixed_val_subset(val_data: List[Dict], subset_size: int, seed: int) -> List[Dict]:
    if len(val_data) <= subset_size:
        return list(val_data)
    rng = random.Random(seed)
    # stable selection across runs
    indices = rng.sample(range(len(val_data)), subset_size)
    indices.sort()
    return [val_data[i] for i in indices]


# =============================================================================
# Single CER evaluation on the FIXED validation subset (once per trial)
# =============================================================================

def evaluate_cer_fixed_subset(
    model,
    processor,
    fixed_val_subset: List[Dict],
    images_dir: str,
    instruction: str,
    max_seq_length: int,
    max_new_tokens: int,
) -> float:
    device = next(model.parameters()).device
    model.eval()

    predictions: List[str] = []
    targets: List[str] = []

    for i, item in enumerate(fixed_val_subset):
        file_name = item["file_name"]
        image_path = os.path.join(images_dir, file_name)

        print(f"   Eval {i+1}/{len(fixed_val_subset)}: {file_name}", end="\r")

        if not os.path.exists(image_path):
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            messages = [{"role": "user", "content": f"<|image_1|>\n{instruction}"}]
            prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = processor(
                prompt,
                [image],
                return_tensors="pt",
                max_length=max_seq_length,
                truncation=True,
            ).to(device)

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n   ⚠️  Eval error on {file_name}: {e}")
            predictions.append("")
            targets.append(create_label_string(item))

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

    return total_cer / valid_pairs if valid_pairs > 0 else 1.0


# =============================================================================
# Train once for a single trial (NO checkpoints, NO eval during training)
# =============================================================================

def train_one_trial(
    trial_config: Dict[str, Any],
    processor,
    bnb_config: BitsAndBytesConfig,
    train_dataset,
    data_collator,
    output_dir: str,
) -> AutoModelForCausalLM:
    os.makedirs(output_dir, exist_ok=True)

    # Save per-trial config for traceability (still no checkpoints)
    with open(os.path.join(output_dir, "trial_config.json"), "w", encoding="utf-8") as f:
        json.dump(trial_config, f, indent=2, ensure_ascii=False)

    model = AutoModelForCausalLM.from_pretrained(
        trial_config["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=trial_config["lora_r"],
        lora_alpha=trial_config["lora_alpha"],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=trial_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=trial_config["num_epochs"],
        per_device_train_batch_size=trial_config["batch_size"],
        gradient_accumulation_steps=trial_config["gradient_accumulation_steps"],
        learning_rate=trial_config["learning_rate"],
        weight_decay=trial_config["weight_decay"],
        max_grad_norm=trial_config["max_grad_norm"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=20,

        save_strategy="no",
        eval_strategy="no",
        load_best_model_at_end=False,

        fp16=False,
        bf16=True,
        gradient_checkpointing=True,

        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        optim="paged_adamw_8bit",

        max_steps=trial_config.get("max_steps", -1),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()

    # IMPORTANT: detach Trainer to reduce refs before returning
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


# =============================================================================
# Optuna driver
# =============================================================================

def run_optuna_hpo(
    processor,
    bnb_config: BitsAndBytesConfig,
    train_dataset,
    fixed_val_subset: List[Dict],
    data_collator,
):
    db_path = os.path.abspath(CONFIG["optuna_db_path"])
    storage_url = f"sqlite:///{db_path}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CONFIG["hpo_output_base_dir"], f"run_{timestamp}")
    trials_root = os.path.join(run_dir, "trials")
    os.makedirs(trials_root, exist_ok=True)

    # Save fixed val subset filenames for transparency
    with open(os.path.join(run_dir, "fixed_val_subset_files.json"), "w", encoding="utf-8") as f:
        json.dump([x["file_name"] for x in fixed_val_subset], f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("🔎 Optuna HPO (Phi-3.5-Vision, SCHMUCK) - v2 fixed50 + aug + crops16")
    print("=" * 90)
    print(f"   • Storage: {storage_url}")
    print(f"   • Study name: {CONFIG['optuna_study_name']}")
    print(f"   • Target trials: {CONFIG['optuna_n_trials']}")
    print(f"   • Run directory: {run_dir}")
    print(f"   • Fixed validation subset size: {len(fixed_val_subset)}")
    print(f"   • num_crops: {CONFIG['processor_num_crops']}, max_seq_length: {CONFIG['max_seq_length']}")
    print(f"   • augmentation: {CONFIG['data_augmentation']}")
    print("=" * 90 + "\n")

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

    print(f"Completed trials so far: {n_completed}")
    print(f"Remaining trials to run: {remaining}\n")

    if remaining <= 0:
        print("Nothing to run. Printing best trial.")
        best_trial = study.best_trial
        print(f"Best trial #{best_trial.number} value={best_trial.value:.6f}")
        return

    def objective(trial: optuna.trial.Trial) -> float:
        # Reduce variance: same seed for all trials
        set_all_seeds(CONFIG["global_seed"])

        trial_config = CONFIG.copy()

        # Sample hyperparameters
        trial_config["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 4e-4, log=True)
        trial_config["weight_decay"] = trial.suggest_float("weight_decay", 0.0, 0.1)

        trial_config["lora_r"] = trial.suggest_categorical("lora_r", [8, 16, 24, 32])
        trial_config["lora_alpha"] = trial.suggest_categorical("lora_alpha", [16, 32, 48, 64])
        trial_config["lora_dropout"] = trial.suggest_float("lora_dropout", 0.05, 0.15)

        trial_config["gradient_accumulation_steps"] = trial.suggest_categorical(
            "gradient_accumulation_steps", [8, 16]
        )

        trial_config["max_grad_norm"] = 1.0
        trial_config["num_epochs"] = CONFIG["num_epochs"]
        trial_config["batch_size"] = CONFIG["batch_size"]
        trial_config["max_steps"] = CONFIG["max_steps"]

        # Per-trial output dir (still no checkpoints)
        trial_output_dir = os.path.join(trials_root, f"trial_{trial.number}")

        print("\n" + "-" * 90)
        print(f"TRIAL {trial.number} | params: "
              f"lr={trial_config['learning_rate']:.6g}, wd={trial_config['weight_decay']:.6g}, "
              f"r={trial_config['lora_r']}, alpha={trial_config['lora_alpha']}, "
              f"drop={trial_config['lora_dropout']:.4f}, "
              f"gas={trial_config['gradient_accumulation_steps']}")
        print("-" * 90)

        model = None
        try:
            model = train_one_trial(
                trial_config=trial_config,
                processor=processor,
                bnb_config=bnb_config,
                train_dataset=train_dataset,
                data_collator=data_collator,
                output_dir=trial_output_dir,
            )

            val_cer = evaluate_cer_fixed_subset(
                model=model,
                processor=processor,
                fixed_val_subset=fixed_val_subset,
                images_dir=CONFIG["images_dir"],
                instruction=INSTRUCTION,
                max_seq_length=CONFIG["max_seq_length"],
                max_new_tokens=CONFIG["eval_max_new_tokens"],
            )

            trial.set_user_attr("val_cer_fixed50", float(val_cer))

            print(f"\n✅ TRIAL {trial.number} CER (fixed subset): {val_cer:.6f} ({val_cer*100:.3f}%)")
            return float(val_cer)

        finally:
            # Clean up model between trials
            try:
                if model is not None:
                    del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    # Save best info
    best_trial = study.best_trial
    print("\n" + "=" * 90)
    print("🏆 Optuna HPO DONE (v2)")
    print("=" * 90)
    print(f"Best trial number: {best_trial.number}")
    print(f"Best CER (fixed subset): {best_trial.value:.6f} ({best_trial.value*100:.3f}%)")
    print("Best params:")
    for k, v in best_trial.params.items():
        print(f"  - {k}: {v}")

    os.makedirs(run_dir, exist_ok=True)

    best_hparams = {
        "study_name": CONFIG["optuna_study_name"],
        "best_trial_number": best_trial.number,
        "best_validation_cer_fixed_subset": best_trial.value,
        "best_params": best_trial.params,
        "fixed_val_subset_size": len(fixed_val_subset),
        "fixed_val_subset_seed": CONFIG["fixed_val_subset_seed"],
        "processor_num_crops": CONFIG["processor_num_crops"],
        "max_seq_length": CONFIG["max_seq_length"],
        "augmentation": CONFIG["data_augmentation"],
        "eval_max_new_tokens": CONFIG["eval_max_new_tokens"],
    }

    with open(os.path.join(run_dir, "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_hparams, f, indent=2, ensure_ascii=False)

    best_config = CONFIG.copy()
    for k, v in best_trial.params.items():
        best_config[k] = v

    with open(os.path.join(run_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    with open(os.path.join(run_dir, "hpo_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Optuna HPO Summary - Phi-3.5-Vision SCHMUCK (v2 fixed50 + aug + crops16)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Study name: {CONFIG['optuna_study_name']}\n")
        f.write(f"Storage: sqlite:///{db_path}\n")
        f.write(f"Total requested trials: {CONFIG['optuna_n_trials']}\n\n")
        f.write(f"Best trial number: {best_trial.number}\n")
        f.write(f"Best CER (fixed subset): {best_trial.value:.6f} ({best_trial.value*100:.3f}%)\n\n")
        f.write("Best Hyperparameters:\n")
        for k, v in best_trial.params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nFixed subset files saved to: fixed_val_subset_files.json\n")

    print("\n✅ Saved:")
    print(f"  - {os.path.join(run_dir, 'best_hyperparameters.json')}")
    print(f"  - {os.path.join(run_dir, 'best_config.json')}")
    print(f"  - {os.path.join(run_dir, 'hpo_summary.txt')}")
    print(f"  - {os.path.join(run_dir, 'fixed_val_subset_files.json')}")
    print("\n🎉 Done.\n")


# =============================================================================
# main
# =============================================================================

def main():
    set_all_seeds(CONFIG["global_seed"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["use_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=CONFIG["use_nested_quant"],
    )

    # Processor (MATCH main finetune: num_crops=16 + enforce truncation)
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        num_crops=CONFIG["processor_num_crops"],
    )
    processor.tokenizer.truncation_side = "left"
    processor.tokenizer.model_max_length = CONFIG["max_seq_length"]
    print("✅ Processor loaded")

    print("\n📊 Loading datasets...")
    train_dataset = SchmuckDataset(
        CONFIG["train_jsonl"],
        CONFIG["images_dir"],
        processor,
        INSTRUCTION,
        augment=CONFIG["data_augmentation"],
        max_seq_length=CONFIG["max_seq_length"],
    )

    val_data = load_jsonl(CONFIG["val_jsonl"])
    print(f"📊 Raw validation records: {len(val_data)}")

    fixed_val_subset = build_fixed_val_subset(
        val_data=val_data,
        subset_size=CONFIG["fixed_val_subset_size"],
        seed=CONFIG["fixed_val_subset_seed"],
    )
    print(f"✅ Fixed validation subset built: {len(fixed_val_subset)} samples")

    data_collator = DataCollatorForPhi3Vision(processor=processor)

    run_optuna_hpo(
        processor=processor,
        bnb_config=bnb_config,
        train_dataset=train_dataset,
        fixed_val_subset=fixed_val_subset,
        data_collator=data_collator,
    )


if __name__ == "__main__":
    main()
