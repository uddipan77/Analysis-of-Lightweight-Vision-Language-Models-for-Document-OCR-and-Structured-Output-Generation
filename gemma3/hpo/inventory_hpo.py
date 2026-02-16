#!/usr/bin/env python3
# gemma3_inventory_hpo.py
#
# ENHANCED Hyperparameter optimization for Gemma-3-4B-IT (Unsloth) on INVENTORY dataset.
#
# ✅ ALIGNED with your finetune script hyperparameters:
#    - optim="adamw_8bit" (same as finetune)
#    - lr_scheduler_type="cosine"
#    - warmup_ratio tuned
#    - weight_decay tuned
#    - max_grad_norm tuned
#    - LoRA params tuned (r, alpha, dropout, use_rslora)
#    - learning_rate tuned
#    - gradient_accumulation_steps tuned
#    - num_epochs tuned
#
# ❌ NOT tuning adam_beta1/adam_beta2/adam_epsilon because your finetune script does not set them
#    (so HPO results would not transfer unless you also add them to finetune).
#
# - One stage of training per trial: train on train.jsonl for N epochs.
# - NO eval during training (no per-epoch CER).
# - After training finishes, run generation-based CER on a FIXED subset of val.jsonl.
#   ✅ Fixed subset is deterministic across trials (same 50 samples every time).
# - Optuna objective = final validation CER per trial (lower is better).
# - No model checkpoints are saved (no save_pretrained / state_dict).
# - Best hyperparameters are written to JSON/TXT in a timestamped run folder.
#
# Study name: "gemma3_inven_enhanced"
# DB: sqlite: ///home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db
# Outputs: /home/vault/iwi5/iwi5298h/models_image_text/gemma/HPO/inven/run_HPO_<timestamp>/

import sys
import os

# Optional, but matches your other Gemma scripts
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import random
from datetime import datetime
from typing import List, Dict, Any, Tuple

import torch
import jiwer
from PIL import Image
import gc

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

import optuna
from optuna.trial import TrialState


# ======================================================================
# Global HPO config
# ======================================================================

CONFIG: Dict[str, Any] = {
    # Paths
    "model_path": (
        "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/"
        "models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/"
        "snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"
    ),
    "train_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
    "val_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",

    # Base hyperparameters (defaults; overridden per trial)
    "num_epochs": 7,  # tuned
    "batch_size": 1,  # fixed (matches finetune default)
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 8,

    # LoRA defaults (tuned)
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_rslora": True,

    # Schedule / optimizer defaults (aligned with finetune script)
    "warmup_ratio": 0.1,           # tuned
    "lr_scheduler_type": "cosine", # fixed (same as finetune)
    "max_grad_norm": 1.0,          # tuned
    "optim": "adamw_8bit",         # ✅ SAME AS YOUR FINETUNE CODE

    # HPO settings
    "optuna_n_trials": 25,
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    "optuna_study_name": "gemma3_new_inventory_dataset",

    # Will be overridden in main() to include timestamp
    "hpo_output_base_dir": "/home/vault/iwi5/iwi5298h/models_image_text/gemma/HPO/new_inven",
    "hpo_output_dir": None,
    "trial_output_root": None,

    # CER evaluation settings (FIXED subset across trials)
    "cer_max_val_samples": 50,   # fixed size across trials
    "cer_subset_seed": 3407,     # fixed seed across trials (deterministic subset)
    "cer_max_new_tokens": 512,
}


# ======================================================================
# Inventory helper class for HPO (no multi-stage training)
# ======================================================================

class InventoryGemma3HPO:
    def __init__(self, model_path: str, lora_cfg: Dict[str, Any]):
        print("[HPO] Loading Gemma-3 vision model with Unsloth for INVENTORY...")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,  # Let Unsloth decide (bf16-like)
            load_in_4bit=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        print("[HPO] Base model loaded (4-bit)")

        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=int(lora_cfg["lora_r"]),
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=int(lora_cfg["lora_alpha"]),
            lora_dropout=float(lora_cfg["lora_dropout"]),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=bool(lora_cfg["use_rslora"]),
        )

        print(
            "[HPO] Gemma-3 model wrapped with LoRA: "
            f"r={lora_cfg['lora_r']}, alpha={lora_cfg['lora_alpha']}, "
            f"dropout={lora_cfg['lora_dropout']}, rslora={lora_cfg['use_rslora']}"
        )

    # ----------------------- I/O HELPERS -----------------------

    def load_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def dict_without_image_name(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in obj.items() if k not in ["image_name", "image_path"]}

    # ---------- Canonical JSON (order-invariant) ----------

    def _canonicalize_json(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._canonicalize_json(obj[k]) for k in sorted(obj.keys())}
        elif isinstance(obj, list):
            return [self._canonicalize_json(x) for x in obj]
        else:
            return obj

    def json_to_string_readable(self, obj: Dict[str, Any]) -> str:
        clean_obj = self.dict_without_image_name(obj)
        canonical = self._canonicalize_json(clean_obj)
        return json.dumps(canonical, ensure_ascii=False, indent=2)

    # ----------------------- JSON EXTRACTION -----------------------

    def extract_json_from_response(self, response: str) -> Dict:
        """
        Robust JSON extraction for slightly truncated / noisy outputs.
        """
        if isinstance(response, list):
            response = response[0] if response else ""
        if response is None:
            return {}

        text = str(response).strip()
        if not text:
            return {}

        # Strip Markdown fences if present
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

        # Focus on main JSON-ish core
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            core = text[start : end + 1]
        else:
            core = text

        # 1) Try full core
        parsed = try_parse(core)
        if parsed is not None:
            return parsed

        # 2) Walk backwards over '}' positions
        import re
        brace_positions = [m.start() for m in re.finditer(r"\}", core)]
        for pos in reversed(brace_positions):
            candidate = core[: pos + 1]
            parsed = try_parse(candidate)
            if parsed is not None:
                return parsed

        # 3) Nested balanced candidates, longest first
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, core, re.DOTALL)
        if matches:
            for match in sorted(matches, key=len, reverse=True):
                parsed = try_parse(match)
                if parsed is not None:
                    return parsed

        # 4) Last resort: whole text
        parsed = try_parse(text)
        if parsed is not None:
            return parsed

        return {}

    # ----------------------- CONVERSIONS -----------------------

    def inventory_instruction(self) -> str:
        return (
            "Du bist ein OCR- und Information-Extraction-Modell für deutsche "
            "historische Inventardokumente.\n\n"
            "Aufgabe:\n"
            "Lies ALLE Informationen aus dem Bild dieses Inventarblatts und gib "
            "GENAU EIN JSON-Objekt zurück.\n\n"
            "Das JSON MUSS folgende Felder enthalten:\n"
            "- Überschrift: Dokumenttitel\n"
            "- Inventarnummer: Inventar- oder Katalognummer\n"
            "- Maße: Objekt mit L, B, D (Länge, Breite, Tiefe)\n"
            "- Objektbezeichnung: Beschreibung/Name des Objekts\n"
            "- Fundort: Fundort des Objekts\n"
            "- Fundzeit: Zeit der Auffindung\n"
            "- Beschreibungstext: Ausführlicher Beschreibungstext\n\n"
            "Regeln:\n"
            "- Gib NUR ein gültiges JSON-Objekt zurück (kein extra Text davor oder danach).\n"
            "- Verwende GENAU diese Feldnamen und Groß-/Kleinschreibung.\n"
            "- Wenn ein Feld leer ist oder nicht sichtbar, gib einen leeren String \"\" zurück.\n"
            "- Das Feld \"Maße\" MUSS immer ein Objekt mit den Schlüsseln \"L\", \"B\", \"D\" sein, auch wenn leer.\n"
            "- Erfinde keine zusätzlichen Felder.\n"
        )

    def convert_to_conversation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        instruction = self.inventory_instruction()
        gt_json_string = self.json_to_string_readable(sample)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": instruction},
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

    # ----------------------- DATA PREP (NO AUG) -----------------------

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        """
        Prepare training data WITHOUT augmentation (for HPO, to keep it fast).
        """
        data = self.load_jsonl(jsonl_path)

        processed = []
        missing = 0

        print("[HPO] Preparing INVENTORY TRAIN data (no augmentation)")

        for i, item in enumerate(data):
            if "image_name" not in item:
                print(f"[HPO] Warning: 'image_name' missing in item index {i}, skipping.")
                continue

            image_path = os.path.join(images_dir, item["image_name"])
            if not os.path.exists(image_path):
                missing += 1
                continue

            item_copy = item.copy()
            item_copy["image_path"] = image_path
            processed.append(item_copy)

        print(f"[HPO] Training: {len(processed)} samples (missing images: {missing})")

        converted_dataset = [self.convert_to_conversation(s) for s in processed]
        return converted_dataset

    def load_validation_raw(self, jsonl_path: str) -> List[Dict]:
        return self.load_jsonl(jsonl_path)

    # ----------------------- CER HELPERS -----------------------

    def calculate_cer(self, predictions: List[str], targets: List[str]) -> float:
        if (
            predictions is None
            or targets is None
            or len(predictions) == 0
            or len(targets) == 0
            or len(predictions) != len(targets)
        ):
            return 1.0

        total_cer = 0.0
        count = 0
        for pred, target in zip(predictions, targets):
            pred_str, target_str = str(pred), str(target)
            if len(target_str) > 0:
                try:
                    cer = jiwer.cer(target_str, pred_str)
                except Exception:
                    cer = 1.0
                total_cer += cer
                count += 1

        return total_cer / count if count > 0 else 1.0

    # ----------------------- FIXED VALIDATION SUBSET -----------------------

    def pick_fixed_validation_subset(
        self,
        val_data: List[Dict[str, Any]],
        max_samples: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """
        Deterministically pick the same subset across trials.
        - Uses stable ordering: sorts by image_name (fallback to json dump)
        - Uses a dedicated RNG with fixed seed
        """
        if not val_data:
            return []

        def stable_key(x: Dict[str, Any]) -> str:
            name = x.get("image_name")
            if isinstance(name, str) and name:
                return name
            return json.dumps(x, ensure_ascii=False, sort_keys=True)

        sorted_val = sorted(val_data, key=stable_key)

        if len(sorted_val) <= max_samples:
            return list(sorted_val)

        rng = random.Random(int(seed))
        indices = list(range(len(sorted_val)))
        rng.shuffle(indices)
        picked = [sorted_val[i] for i in indices[:max_samples]]
        return picked

    # ----------------------- VALIDATION CER (GENERATION) -----------------------

    def evaluate_cer_on_validation(
        self,
        val_subset: List[Dict[str, Any]],
        images_dir: str,
        max_new_tokens: int,
    ) -> float:
        """
        Run generation on a FIXED subset of validation data and compute average CER.
        This is the HPO objective.
        """
        print("\n" + "=" * 80)
        print("[HPO] RUNNING GENERATION-BASED CER EVALUATION ON FIXED VAL SUBSET")
        print("=" * 80)

        device = next(self.model.parameters()).device
        FastVisionModel.for_inference(self.model)
        self.model.eval()

        print(f"[HPO] Using {len(val_subset)} fixed validation samples for CER")

        predictions: List[str] = []
        targets: List[str] = []

        instruction = self.inventory_instruction()

        for i, item in enumerate(val_subset):
            image_name = item.get("image_name", f"val_{i}")
            image_path = os.path.join(images_dir, image_name)

            print(f"[HPO]   Evaluating {i+1}/{len(val_subset)}: {image_name}", end="\r")

            if not os.path.exists(image_path):
                targets.append(self.json_to_string_readable(item))
                predictions.append("")
                continue

            image = None
            try:
                image = Image.open(image_path)

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
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        repetition_penalty=1.0,
                    )

                input_len = inputs["input_ids"].shape[-1]
                generated_ids_trimmed = outputs[0][input_len:]
                generated_text = self.tokenizer.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                predicted_json = self.extract_json_from_response(generated_text)
                pred_str = (
                    self.json_to_string_readable(predicted_json) if predicted_json else ""
                )
                gt_str = self.json_to_string_readable(item)

                predictions.append(pred_str)
                targets.append(gt_str)

            except Exception as e:
                print(f"\n[HPO]   ⚠️  CER eval error on {image_name}: {e}")
                gt_str = self.json_to_string_readable(item)
                predictions.append("")
                targets.append(gt_str)

            finally:
                if image is not None:
                    image.close()
                    del image
                if "inputs" in locals():
                    del inputs
                if "outputs" in locals():
                    del outputs
                if "generated_ids_trimmed" in locals():
                    del generated_ids_trimmed

                torch.cuda.empty_cache()
                gc.collect()

        avg_cer = self.calculate_cer(predictions, targets)

        print(
            f"\n[HPO]   ✅ Validation CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)"
        )
        print("=" * 80 + "\n")

        FastVisionModel.for_training(self.model)
        return avg_cer


# ======================================================================
# Single training run for one hyperparameter config
# ======================================================================

def train_one_config(
    config: Dict[str, Any],
    trial_output_dir: str,
    fixed_val_subset: List[Dict[str, Any]],
) -> Tuple[float, Dict[str, Any]]:
    """
    Train once with given config and return:
      val_cer (float), info dict

    IMPORTANT:
      - NO per-epoch eval.
      - NO model checkpoints.
      - Single generation-based CER eval AFTER training on a FIXED val subset.
    """
    os.makedirs(trial_output_dir, exist_ok=True)

    # Save per-run config (for debugging)
    config_file = os.path.join(trial_output_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("[HPO] Gemma-3 INVENTORY HPO TRIAL (FINETUNE-ALIGNED)")
    print("=" * 80)
    print(f"[HPO] Trial output directory: {trial_output_dir}")
    print(f"[HPO]   • num_epochs: {config['num_epochs']}")
    print(f"[HPO]   • batch_size: {config['batch_size']}")
    print(f"[HPO]   • grad_accum: {config['gradient_accumulation_steps']}")
    print(f"[HPO]   • learning_rate: {config['learning_rate']}")
    print(f"[HPO]   • weight_decay: {config['weight_decay']}")
    print(f"[HPO]   • lora_r: {config['lora_r']}")
    print(f"[HPO]   • lora_alpha: {config['lora_alpha']}")
    print(f"[HPO]   • lora_dropout: {config['lora_dropout']}")
    print(f"[HPO]   • use_rslora: {config['use_rslora']}")
    print(f"[HPO]   • warmup_ratio: {config['warmup_ratio']}")
    print(f"[HPO]   • lr_scheduler_type: {config['lr_scheduler_type']}")
    print(f"[HPO]   • max_grad_norm: {config['max_grad_norm']}")
    print(f"[HPO]   • optim: {config['optim']}")

    lora_cfg = {
        "lora_r": config["lora_r"],
        "lora_alpha": config["lora_alpha"],
        "lora_dropout": config["lora_dropout"],
        "use_rslora": config["use_rslora"],
    }
    finetuner = InventoryGemma3HPO(model_path=config["model_path"], lora_cfg=lora_cfg)

    train_dataset = finetuner.prepare_training_data(
        config["train_jsonl_path"],
        config["images_dir"],
    )

    print(
        f"[HPO] Training samples: {len(train_dataset)}, "
        f"Fixed validation subset: {len(fixed_val_subset)}"
    )

    FastVisionModel.for_training(finetuner.model)

    trainer = SFTTrainer(
        model=finetuner.model,
        tokenizer=finetuner.tokenizer,
        data_collator=UnslothVisionDataCollator(finetuner.model, finetuner.tokenizer),
        train_dataset=train_dataset,
        eval_dataset=None,  # NO eval during training
        args=SFTConfig(
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],

            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            num_train_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],

            # finetune-aligned knobs
            lr_scheduler_type=config["lr_scheduler_type"],  # "cosine"
            warmup_ratio=config["warmup_ratio"],
            max_grad_norm=config["max_grad_norm"],
            optim=config["optim"],  # "adamw_8bit"

            # HPO speed + simplicity
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,

            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            # same behavior as your finetune script (bf16 if supported)
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),

            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            remove_unused_columns=False,
            dataset_text_field=None,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,

            report_to="none",
            logging_dir=os.path.join(trial_output_dir, "logs"),
            seed=3407,
            output_dir=trial_output_dir,
            save_safetensors=False,
            prediction_loss_only=False,
            disable_tqdm=False,
        ),
    )

    print("[HPO] Starting training for this trial...\n")
    trainer.train()
    print("\n[HPO] Training finished for this trial.")

    cer_value = finetuner.evaluate_cer_on_validation(
        val_subset=fixed_val_subset,
        images_dir=config["images_dir"],
        max_new_tokens=config["cer_max_new_tokens"],
    )

    # Clean up
    del trainer
    del finetuner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    info = {"val_cer": float(cer_value)}
    return float(cer_value), info


# ======================================================================
# Optuna HPO driver
# ======================================================================

def run_optuna_hpo_only():
    db_path = os.path.abspath(CONFIG["optuna_db_path"])
    storage_url = f"sqlite:///{db_path}"

    hpo_out_dir = CONFIG["hpo_output_dir"]
    trial_root = CONFIG["trial_output_root"]

    print("\n" + "=" * 80)
    print("[HPO] Starting / Resuming Optuna HPO for Gemma-3 INVENTORY (FINETUNE-ALIGNED)")
    print("=" * 80)
    print(f"[HPO]   Storage: {storage_url}")
    print(f"[HPO]   Study name: {CONFIG['optuna_study_name']}")
    print(f"[HPO]   Target total trials (COMPLETE): {CONFIG['optuna_n_trials']}")
    print(f"[HPO]   Output dir for this HPO run: {hpo_out_dir}")

    os.makedirs(hpo_out_dir, exist_ok=True)
    os.makedirs(trial_root, exist_ok=True)

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

    # --------- Build FIXED validation subset ONCE (shared across all trials) ---------
    # We load raw val here and deterministically choose the subset; each trial reuses it.
    tmp = InventoryGemma3HPO(
        model_path=CONFIG["model_path"],
        lora_cfg={
            "lora_r": CONFIG["lora_r"],
            "lora_alpha": CONFIG["lora_alpha"],
            "lora_dropout": CONFIG["lora_dropout"],
            "use_rslora": CONFIG["use_rslora"],
        },
    )
    val_raw = tmp.load_validation_raw(CONFIG["val_jsonl_path"])
    fixed_val_subset = tmp.pick_fixed_validation_subset(
        val_data=val_raw,
        max_samples=int(CONFIG["cer_max_val_samples"]),
        seed=int(CONFIG["cer_subset_seed"]),
    )
    del tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    fixed_list_path = os.path.join(hpo_out_dir, "fixed_val_subset_image_names.json")
    with open(fixed_list_path, "w", encoding="utf-8") as f:
        json.dump(
            [x.get("image_name", "") for x in fixed_val_subset],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[HPO] ✅ Fixed validation subset saved to: {fixed_list_path}")

    if remaining > 0:

        def objective(trial: optuna.trial.Trial):
            config = CONFIG.copy()

            # Always fixed for this codebase
            config["batch_size"] = CONFIG["batch_size"]

            # --- Hyperparameter search spaces (FINETUNE-ALIGNED) ---

            # Core
            config["learning_rate"] = trial.suggest_float(
                "learning_rate", 5e-6, 2e-4, log=True
            )
            config["weight_decay"] = trial.suggest_float(
                "weight_decay", 0.0, 0.15
            )
            config["gradient_accumulation_steps"] = trial.suggest_categorical(
                "gradient_accumulation_steps", [4, 8, 16]
            )

            # Training length - FIXED (not optimized)
            config["num_epochs"] = 5

            # LoRA
            config["lora_r"] = trial.suggest_categorical(
                "lora_r", [8, 16, 32, 64]
            )
            config["lora_alpha"] = trial.suggest_categorical(
                "lora_alpha", [16, 32, 64, 128]
            )
            config["lora_dropout"] = trial.suggest_float(
                "lora_dropout", 0.0, 0.2
            )
            config["use_rslora"] = trial.suggest_categorical(
                "use_rslora", [False, True]
            )

            # Schedule / stability
            config["warmup_ratio"] = trial.suggest_float(
                "warmup_ratio", 0.0, 0.2
            )
            config["max_grad_norm"] = trial.suggest_float(
                "max_grad_norm", 0.5, 2.0
            )

            # Fixed to match finetune code (kept in config for clarity/logging)
            config["lr_scheduler_type"] = "cosine"
            config["optim"] = "adamw_8bit"

            trial_output_dir = os.path.join(trial_root, f"trial_{trial.number}")

            val_cer, _info = train_one_config(
                config=config,
                trial_output_dir=trial_output_dir,
                fixed_val_subset=fixed_val_subset,
            )

            trial.set_user_attr("val_cer", float(val_cer))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return float(val_cer)

        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
        )

    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("[HPO] Gemma-3 INVENTORY HPO finished / resumed (FINETUNE-ALIGNED)")
    print("=" * 80)
    print(f"[HPO]   Best trial number: {best_trial.number}")
    print(
        f"[HPO]   Best Validation CER: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)"
    )
    print("[HPO]   Best params:")
    for k, v in best_trial.params.items():
        print(f"[HPO]      {k}: {v}")

    best_hparams = {
        "best_trial_number": best_trial.number,
        "best_validation_cer": best_trial.value,
        "best_params": best_trial.params,
        "fixed_val_subset_size": int(CONFIG["cer_max_val_samples"]),
        "fixed_val_subset_seed": int(CONFIG["cer_subset_seed"]),
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
        f.write("Optuna HPO Summary - Gemma-3-4B-IT INVENTORY OCR (FINETUNE-ALIGNED)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Study name: {CONFIG['optuna_study_name']}\n")
        f.write(f"Storage: sqlite:///{os.path.abspath(CONFIG['optuna_db_path'])}\n")
        f.write(f"Total requested COMPLETE trials: {CONFIG['optuna_n_trials']}\n")
        f.write(f"Fixed val subset size: {CONFIG['cer_max_val_samples']}\n")
        f.write(f"Fixed val subset seed: {CONFIG['cer_subset_seed']}\n")
        f.write(f"Best trial number: {best_trial.number}\n")
        f.write(
            f"Best Validation CER: {best_trial.value:.4f} ({best_trial.value*100:.2f}%)\n\n"
        )
        f.write("Best Hyperparameters:\n")
        for k, v in best_trial.params.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n[HPO] ✅ Best hyperparameters saved to: {best_hparams_path}")
    print(f"[HPO] ✅ Best config saved to:          {best_config_path}")
    print(f"[HPO] ✅ HPO summary saved to:         {summary_path}")
    print(f"[HPO] ✅ Fixed val subset list saved to:{fixed_list_path}")
    print(
        "\n[HPO] 🎉 HPO done. Copy these values into your main "
        "Gemma-3 INVENTORY finetune script.\n"
    )


# ======================================================================
# main()
# ======================================================================

def main():
    base_dir = CONFIG["hpo_output_base_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_HPO_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    CONFIG["hpo_output_dir"] = run_dir
    CONFIG["trial_output_root"] = os.path.join(run_dir, "trials")

    config_file = os.path.join(run_dir, "hpo_run_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print(f"[HPO] Created HPO run directory: {run_dir}")
    run_optuna_hpo_only()


if __name__ == "__main__":
    main()
