#!/usr/bin/env python3
# gemma3_schmuck_hpo_local.py
#
# Hyperparameter optimization for Gemma-3-4B-IT (Unsloth) on SCHMUCK dataset.
#
# Per trial:
#   1) Load model from LOCAL SNAPSHOT path (4-bit) with local_files_only=True
#   2) Apply LoRA (trial-specific)
#   3) Train for num_epochs (NO eval during training, NO checkpoints)
#   4) Run generation-based CER on a small val subset (objective)
#
# Fix for Optuna error:
#   - "Cannot set different log configuration to the same parameter name."
#     This happens when resuming a study where a param name was previously
#     registered with a different distribution (log=True vs log=False, etc.).
#   - Solution: use a NEW study name (versioned), or delete the old study.
#
# Improvements:
#   - Adds more hyperparameters for better search:
#       * warmup_ratio
#       * lr_scheduler_type
#       * max_grad_norm
#       * optim (fused vs non-fused)
#       * adam_beta1 / adam_beta2 / adam_epsilon
#       * use_rslora (on/off)
#       * lora target rank/alpha/dropout (already there)
#   - Keeps local snapshot loading + Gemma tokenizer.apply_chat_template

import os
import sys
import json
import glob
import random
import re
import gc
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import torch
import jiwer
from PIL import Image

import optuna
from optuna.trial import TrialState

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

# Optional but consistent with your other Gemma scripts
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"

# Safer in some HPC shells
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# ======================================================================
# Global HPO config
# ======================================================================

SEARCH_SPACE_VERSION = "ssv2"  # bump this if you change distributions/param names in the future

CONFIG: Dict[str, Any] = {
    # IMPORTANT: use the SAME local snapshot style path that worked for INVENTORY
    "model_path": (
        "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/"
        "models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/"
        "snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"
    ),

    "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
    "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",

    # Reference hyperparameters (center)
    "batch_size": 1,
    "learning_rate": 2e-4,
    "weight_decay": 0.05,
    "gradient_accumulation_steps": 8,
    "num_epochs": 8,

    # LoRA base (center)
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "use_rslora": True,

    # Trainer / optimizer refs (center)
    "warmup_ratio": 0.10,
    "lr_scheduler_type": "cosine",
    "max_grad_norm": 1.0,
    "optim": "adamw_torch_fused",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,

    # CER evaluation settings
    "cer_max_val_samples": 30,
    "cer_max_new_tokens": 1024,

    # Optuna
    "optuna_n_trials": 30,
    "optuna_db_path": "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/vlmmodels.db",
    # Versioned study name to avoid distribution conflicts
    "optuna_study_name": f"gemma3_schmuck_data_{SEARCH_SPACE_VERSION}",

    # Output folders
    "hpo_output_base_dir": "/home/vault/iwi5/iwi5298h/models_image_text/gemma3/hpo/schmuck",
    "hpo_output_dir": None,
    "trial_output_root": None,
}

# ======================================================================
# Schmuck instruction
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

Include ALL fields, even if empty (use empty string ""). Preserve exact German spelling and punctuation.
Return ONLY the JSON object without any additional text or formatting.
"""

# ======================================================================
# Helpers
# ======================================================================

def seed_everything(seed: int = 3407) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def clean_gt(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in obj.items() if k not in ["file_name", "image_path"]}

def json_to_string(obj: Dict[str, Any]) -> str:
    # stable separators, keep original key order from file
    return json.dumps(clean_gt(obj), ensure_ascii=False, separators=(",", ":"))

def find_image_path(file_name: str, images_dir: str) -> str:
    exact = os.path.join(images_dir, file_name)
    if os.path.exists(exact):
        return exact
    base, _ = os.path.splitext(file_name)
    matches = glob.glob(os.path.join(images_dir, f"*{base}*"))
    return matches[0] if matches else exact

def extract_first_json_object(text: str) -> Dict[str, Any]:
    """
    Robust extraction: try to parse the FIRST complete JSON object.
    """
    if text is None:
        return {}
    if isinstance(text, list):
        text = text[0] if text else ""
    s = str(text).strip()
    if not s:
        return {}

    # Remove markdown fences if any
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.lstrip().startswith("json"):
                body = body.lstrip()[4:].strip()
            s = body.strip()

    if "{" not in s or "}" not in s:
        return {}

    start = s.find("{")
    core = s[start:]

    def try_parse(candidate: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # 1) Try full core up to last brace
    end = core.rfind("}")
    if end != -1:
        parsed = try_parse(core[: end + 1])
        if parsed is not None:
            return parsed

    # 2) Walk backward over brace positions and try prefixes
    brace_positions = [m.start() for m in re.finditer(r"\}", core)]
    for pos in reversed(brace_positions):
        cand = core[: pos + 1]
        parsed = try_parse(cand)
        if parsed is not None:
            return parsed

    # 3) Balanced-ish candidates (longest first)
    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(json_pattern, core, flags=re.DOTALL)
    for m in sorted(matches, key=len, reverse=True):
        parsed = try_parse(m)
        if parsed is not None:
            return parsed

    return {}

def calculate_cer(preds: List[str], tgts: List[str]) -> float:
    if not preds or not tgts or len(preds) != len(tgts):
        return 1.0
    total, n = 0.0, 0
    for p, t in zip(preds, tgts):
        if not t:
            continue
        try:
            total += jiwer.cer(t, p)
        except Exception:
            total += 1.0
        n += 1
    return total / n if n > 0 else 1.0

# ======================================================================
# Gemma-3 SCHMUCK HPO helper
# ======================================================================

class SchmuckGemma3HPO:
    def __init__(
        self,
        model_path: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        use_rslora: bool,
        seed: int = 3407,
    ):
        print("[HPO] Loading Gemma-3 model from LOCAL snapshot with Unsloth...")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
            local_files_only=True,
        )
        print("[HPO] Base model loaded (4-bit, local_files_only=True)")

        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=int(lora_r),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=int(seed),
            use_rslora=bool(use_rslora),
        )

        print(
            f"[HPO] LoRA applied (r={lora_r}, alpha={lora_alpha}, "
            f"dropout={lora_dropout:.4f}, rslora={use_rslora})"
        )

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict[str, Any]]:
        data = load_jsonl(jsonl_path)
        processed = []
        missing = 0

        for item in data:
            fn = item.get("file_name")
            if not fn:
                continue
            img_path = find_image_path(fn, images_dir)
            if not os.path.exists(img_path):
                missing += 1
                continue
            item2 = dict(item)
            item2["image_path"] = img_path
            processed.append(item2)

        print(f"[HPO] Training: {len(processed)} samples (missing images: {missing})")

        converted = []
        for s in processed:
            gt = json_to_string(s)
            converted.append({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": s["image_path"]},
                            {"type": "text", "text": SCHMUCK_INSTRUCTION},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": gt}]},
                ]
            })
        return converted

    def evaluate_cer_on_validation(
        self,
        val_data: List[Dict[str, Any]],
        images_dir: str,
        max_samples: int,
        max_new_tokens: int,
    ) -> float:
        print("\n" + "=" * 80)
        print("[HPO] RUNNING GENERATION-BASED CER EVALUATION ON SCHMUCK VALIDATION")
        print("=" * 80)

        device = next(self.model.parameters()).device
        FastVisionModel.for_inference(self.model)
        self.model.eval()

        subset = random.sample(val_data, max_samples) if len(val_data) > max_samples else list(val_data)
        preds, tgts = [], []

        for i, item in enumerate(subset):
            fn = item.get("file_name", f"val_{i}")
            img_path = find_image_path(fn, images_dir)

            print(f"[HPO]   Evaluating {i+1}/{len(subset)}: {fn}", end="\r")

            gt = json_to_string(item)

            if not os.path.exists(img_path):
                preds.append("")
                tgts.append(gt)
                continue

            image = None
            try:
                image = Image.open(img_path).convert("RGB")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": SCHMUCK_INSTRUCTION},
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
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=False,
                        use_cache=True,
                        temperature=0.0,
                        repetition_penalty=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                input_len = inputs["input_ids"].shape[-1]
                gen_ids = out[0][input_len:]
                text = self.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                pred_json = extract_first_json_object(text)
                pred = json_to_string(pred_json) if pred_json else ""

                preds.append(pred)
                tgts.append(gt)

            except Exception as e:
                print(f"\n[HPO]   ⚠️  CER eval error on {fn}: {e}")
                preds.append("")
                tgts.append(gt)

            finally:
                try:
                    if image is not None:
                        image.close()
                except Exception:
                    pass

                for v in ["inputs", "out", "gen_ids"]:
                    if v in locals():
                        del locals()[v]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        cer = calculate_cer(preds, tgts)
        print(f"\n[HPO]   ✅ Validation CER: {cer:.4f} ({cer*100:.2f}%)")
        print("=" * 80 + "\n")

        FastVisionModel.for_training(self.model)
        return float(cer)

# ======================================================================
# One trial train + post-train validate
# ======================================================================

def train_one_config(config: Dict[str, Any], trial_output_dir: str) -> Tuple[float, Dict[str, Any]]:
    os.makedirs(trial_output_dir, exist_ok=True)

    with open(os.path.join(trial_output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    finetuner = SchmuckGemma3HPO(
        model_path=config["model_path"],
        lora_r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        use_rslora=config["use_rslora"],
        seed=3407,
    )

    train_dataset = finetuner.prepare_training_data(config["train_jsonl_path"], config["images_dir"])
    val_raw = load_jsonl(config["val_jsonl_path"])

    FastVisionModel.for_training(finetuner.model)

    trainer = SFTTrainer(
        model=finetuner.model,
        tokenizer=finetuner.tokenizer,
        data_collator=UnslothVisionDataCollator(finetuner.model, finetuner.tokenizer),
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            per_device_train_batch_size=int(config["batch_size"]),
            per_device_eval_batch_size=int(config["batch_size"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            num_train_epochs=int(config["num_epochs"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),

            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,

            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            lr_scheduler_type=str(config["lr_scheduler_type"]),
            warmup_ratio=float(config["warmup_ratio"]),

            optim=str(config["optim"]),
            bf16=True,
            fp16=False,

            max_grad_norm=float(config["max_grad_norm"]),
            adam_beta1=float(config["adam_beta1"]),
            adam_beta2=float(config["adam_beta2"]),
            adam_epsilon=float(config["adam_epsilon"]),

            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            remove_unused_columns=False,
            dataset_text_field=None,
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,

            report_to="none",
            seed=3407,
            output_dir=trial_output_dir,
            save_safetensors=False,
            disable_tqdm=False,
        ),
    )

    print("\n[HPO] Starting training for this trial...\n")
    trainer.train()
    print("\n[HPO] Training finished for this trial.")

    cer = finetuner.evaluate_cer_on_validation(
        val_data=val_raw,
        images_dir=config["images_dir"],
        max_samples=int(config["cer_max_val_samples"]),
        max_new_tokens=int(config["cer_max_new_tokens"]),
    )

    del trainer
    del finetuner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return float(cer), {"val_cer": float(cer)}

# ======================================================================
# Optuna driver
# ======================================================================

def run_optuna_hpo_only():
    db_path = os.path.abspath(CONFIG["optuna_db_path"])
    storage_url = f"sqlite:///{db_path}"

    hpo_out_dir = CONFIG["hpo_output_dir"]
    trial_root = CONFIG["trial_output_root"]
    os.makedirs(hpo_out_dir, exist_ok=True)
    os.makedirs(trial_root, exist_ok=True)

    print("\n" + "=" * 80)
    print("[HPO] Starting / Resuming Optuna HPO for Gemma-3 SCHMUCK (LOCAL MODEL)")
    print("=" * 80)
    print(f"[HPO]   Storage: {storage_url}")
    print(f"[HPO]   Study name: {CONFIG['optuna_study_name']}")
    print(f"[HPO]   Target COMPLETE trials: {CONFIG['optuna_n_trials']}")
    print(f"[HPO]   Output dir: {hpo_out_dir}")

    storage = optuna.storages.RDBStorage(url=storage_url)
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, group=True)

    study = optuna.create_study(
        study_name=CONFIG["optuna_study_name"],
        direction="minimize",
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
    )

    completed = [t for t in study.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE]
    remaining = max(int(CONFIG["optuna_n_trials"]) - len(completed), 0)

    print(f"[HPO]   Completed trials so far: {len(completed)}")
    print(f"[HPO]   Remaining trials to run: {remaining}")

    if remaining > 0:

        def objective(trial: optuna.trial.Trial) -> float:
            cfg = CONFIG.copy()

            # Core knobs
            cfg["learning_rate"] = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
            cfg["weight_decay"]  = trial.suggest_float("weight_decay", 0.0, 0.12)

            # LoRA knobs
            cfg["lora_r"]        = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
            cfg["lora_alpha"]    = trial.suggest_categorical("lora_alpha", [16, 32, 64, 128, 256])
            cfg["lora_dropout"]  = trial.suggest_float("lora_dropout", 0.0, 0.20)
            cfg["use_rslora"]    = trial.suggest_categorical("use_rslora", [True, False])

            # Batch-efficiency knobs
            cfg["gradient_accumulation_steps"] = trial.suggest_categorical(
                "gradient_accumulation_steps", [4, 8, 16, 32]
            )
            cfg["num_epochs"] = trial.suggest_int("num_epochs", 5, 10)

            # Scheduler / stability knobs
            cfg["warmup_ratio"] = trial.suggest_float("warmup_ratio", 0.0, 0.20)
            cfg["lr_scheduler_type"] = trial.suggest_categorical(
                "lr_scheduler_type", ["cosine", "linear", "constant_with_warmup"]
            )
            cfg["max_grad_norm"] = trial.suggest_float("max_grad_norm", 0.3, 2.0)

            # Optimizer knobs (fused is usually best on A100, but keep option)
            cfg["optim"] = trial.suggest_categorical("optim", ["adamw_torch_fused", "adamw_torch"])

            # Adam knobs (can matter a lot for LoRA stability)
            cfg["adam_beta1"] = trial.suggest_float("adam_beta1", 0.85, 0.95)
            cfg["adam_beta2"] = trial.suggest_float("adam_beta2", 0.95, 0.999)
            cfg["adam_epsilon"] = trial.suggest_float("adam_epsilon", 1e-8, 1e-6, log=True)

            trial_dir = os.path.join(trial_root, f"trial_{trial.number:03d}")
            val_cer, info = train_one_config(cfg, trial_dir)
            trial.set_user_attr("val_cer", float(val_cer))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return float(val_cer)

        study.optimize(objective, n_trials=remaining, gc_after_trial=True)

    best = study.best_trial
    print("\n" + "=" * 80)
    print("[HPO] Gemma-3 SCHMUCK HPO finished / resumed")
    print("=" * 80)
    print(f"[HPO]   Best trial: {best.number}")
    print(f"[HPO]   Best CER:   {best.value:.4f} ({best.value*100:.2f}%)")
    print("[HPO]   Best params:")
    for k, v in best.params.items():
        print(f"[HPO]      {k}: {v}")

    best_hparams = {
        "best_trial_number": best.number,
        "best_validation_cer": float(best.value),
        "best_params": best.params,
        "study_name": CONFIG["optuna_study_name"],
        "search_space_version": SEARCH_SPACE_VERSION,
    }
    with open(os.path.join(hpo_out_dir, "best_hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(best_hparams, f, indent=2, ensure_ascii=False)

    best_config = CONFIG.copy()
    for k, v in best.params.items():
        best_config[k] = v
    with open(os.path.join(hpo_out_dir, "best_config.json"), "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)

    with open(os.path.join(hpo_out_dir, "hpo_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Optuna HPO Summary - Gemma-3 SCHMUCK (LOCAL MODEL)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Study name: {CONFIG['optuna_study_name']}\n")
        f.write(f"Search space version: {SEARCH_SPACE_VERSION}\n")
        f.write(f"Storage: sqlite:///{os.path.abspath(CONFIG['optuna_db_path'])}\n")
        f.write(f"Target COMPLETE trials: {CONFIG['optuna_n_trials']}\n")
        f.write(f"Best trial number: {best.number}\n")
        f.write(f"Best Validation CER: {best.value:.4f} ({best.value*100:.2f}%)\n\n")
        f.write("Best Hyperparameters:\n")
        for k, v in best.params.items():
            f.write(f"  {k}: {v}\n")

    print(f"\n[HPO] ✅ Saved outputs to: {hpo_out_dir}\n")

# ======================================================================
# main()
# ======================================================================

def main():
    seed_everything(3407)

    base_dir = CONFIG["hpo_output_base_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"hpo_run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    CONFIG["hpo_output_dir"] = run_dir
    CONFIG["trial_output_root"] = os.path.join(run_dir, "trials")

    with open(os.path.join(run_dir, "hpo_run_config.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print(f"[HPO] Created HPO run directory: {run_dir}")
    run_optuna_hpo_only()

if __name__ == "__main__":
    main()
