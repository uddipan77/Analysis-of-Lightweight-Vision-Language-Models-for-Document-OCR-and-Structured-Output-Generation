#!/usr/bin/env python3
# gemma3_inventory_finetune_A100_multistage_genCER_best_nosort.py
#
# ✅ A100-OPTIMIZED multi-stage training (warm-up + main)
# ✅ BEST model selected by AUTOREGRESSIVE generation-based CER on val subset
# ✅ Trainer-managed: load_best_model_at_end + metric_for_best_model="gen_cer"
# ✅ SAVE ONLY ONE model folder: best_model_gen_cer/
# ✅ DO NOT sort JSON keys (keeps original JSONL key order)
#
# Notes:
# - Stage 1: teacher forcing only, no eval, higher LR, no checkpoint saves
# - Stage 2: eval each epoch, gen-CER callback provides eval_gen_cer, Trainer picks best
# - After Stage 2: deletes stage2/checkpoint-* so only best_model_gen_cer remains
#
# Paths to change: in main() config + local_model_path

import sys
import os

# Unsloth: enable logits return if needed by TRL internals (we do NOT use compute_metrics here)
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import json
import shutil
from typing import List, Dict, Any, Optional
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
import re
import jiwer
from datetime import datetime
from transformers import EarlyStoppingCallback, TrainerCallback
import numpy as np
import gc
from PIL import Image
import random


# =====================================================================
# Helpers
# =====================================================================

def safe_rmtree(path: str):
    try:
        if path and os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"[WARN] Could not remove {path}: {e}")

def remove_all_stage2_checkpoints(stage2_output_dir: str):
    """
    Deletes stage2/checkpoint-* folders to ensure only one final model folder remains.
    """
    if not stage2_output_dir or not os.path.isdir(stage2_output_dir):
        return
    for name in os.listdir(stage2_output_dir):
        if name.startswith("checkpoint-"):
            safe_rmtree(os.path.join(stage2_output_dir, name))


# =====================================================================
# Generation CER callback (Trainer-compatible)
# =====================================================================

class GenerationCEREvalCallback(TrainerCallback):
    """
    Computes AUTOREGRESSIVE generation CER on a fixed validation subset at each evaluation.
    Injects:
      metrics["eval_gen_cer"]
      metrics["eval_gen_cer_percentage"]

    This enables:
      load_best_model_at_end=True + metric_for_best_model="gen_cer"
    """

    def __init__(
        self,
        finetuner,
        val_raw_items: List[Dict[str, Any]],
        images_dir: str,
        subset_size: int = 50,
        subset_seed: int = 3407,
        max_new_tokens: int = 1024,
    ):
        self.finetuner = finetuner
        self.images_dir = images_dir
        self.max_new_tokens = max_new_tokens

        # Stable ordering + deterministic subset selection
        # (No JSON key sorting. We use image_name if present; else hash of raw JSON string.)
        def stable_key(x: Dict[str, Any]) -> str:
            name = x.get("image_name", "")
            if isinstance(name, str) and name:
                return name
            # fallback: stable string without sorting keys
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(hash(str(x)))

        sorted_val = sorted(val_raw_items, key=stable_key)

        rng = random.Random(int(subset_seed))
        if len(sorted_val) > subset_size:
            idxs = list(range(len(sorted_val)))
            rng.shuffle(idxs)
            self.val_subset = [sorted_val[i] for i in idxs[:subset_size]]
        else:
            self.val_subset = sorted_val

        print(f"[GenCER Callback] Fixed val subset size = {len(self.val_subset)}")

        self.cer_history: List[Dict[str, float]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}

        model = kwargs.get("model", None)
        if model is None:
            return control

        epoch = state.epoch if state.epoch is not None else 0.0

        gen_cer = self.finetuner.evaluate_generation_cer_on_items(
            model=model,
            val_items=self.val_subset,
            images_dir=self.images_dir,
            max_new_tokens=self.max_new_tokens,
        )

        # ✅ Trainer expects eval_* for metric_for_best_model
        metrics["eval_gen_cer"] = float(gen_cer)
        metrics["eval_gen_cer_percentage"] = float(gen_cer) * 100.0

        # Optional: non-prefixed copies for convenience
        metrics["gen_cer"] = float(gen_cer)
        metrics["gen_cer_percentage"] = float(gen_cer) * 100.0

        self.cer_history.append({"epoch": float(epoch), "cer": float(gen_cer)})

        print(f"\n[GenCER Callback] Epoch {epoch:.2f} gen_cer = {gen_cer:.4f} ({gen_cer*100:.2f}%)")
        return control


# =====================================================================
# Gemma-3 inventory finetuner (multi-stage, no key sorting)
# =====================================================================

class InventoryGemma3Finetune:
    def __init__(self, model_path: str):
        print("Loading Gemma-3 vision model with Unsloth for INVENTORY...")

        cache_dir = "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/gemma3/unsloth_compiled_cache"
        if os.path.exists(cache_dir):
            print(f"Clearing Unsloth compiled cache: {cache_dir}")
            shutil.rmtree(cache_dir)

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        print("Model loaded - Unsloth auto-configured dtype and attention")

        # A100-optimized LoRA config
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )

        # bookkeeping
        self.best_model_dir: Optional[str] = None
        self.best_val_cer: Optional[float] = None
        self.cer_history: List[Dict[str, float]] = []

        print("Gemma-3 vision model loaded with A100-optimized LoRA config")

    # ----------------------- I/O HELPERS -----------------------

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

    def json_to_string_readable(self, obj):
        """
        ✅ NO key sorting. Preserves original JSON key order from JSONL.
        """
        clean_obj = self.dict_without_image_name(obj)
        return json.dumps(clean_obj, ensure_ascii=False, indent=2)

    # ----------------------- JSON EXTRACTION -----------------------

    def extract_json_from_response(self, response: str) -> Dict:
        if isinstance(response, list):
            response = response[0] if response else ""
        if response is None:
            return {}

        text = str(response).strip()
        if not text:
            return {}

        # fenced
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

        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            core = text[start : end + 1]
        else:
            core = text

        parsed = try_parse(core)
        if parsed is not None:
            return parsed

        # trim trailing braces progressively
        brace_positions = [m.start() for m in re.finditer(r"\}", core)]
        for pos in reversed(brace_positions):
            candidate = core[: pos + 1]
            parsed = try_parse(candidate)
            if parsed is not None:
                return parsed

        # greedy JSON pattern
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, core, re.DOTALL)
        if matches:
            for match in sorted(matches, key=len, reverse=True):
                parsed = try_parse(match)
                if parsed is not None:
                    return parsed

        parsed = try_parse(text)
        if parsed is not None:
            return parsed

        return {}

    # ----------------------- INSTRUCTION / CONVERSION -----------------------

    def _instruction_text(self) -> str:
        return """Du bist ein OCR- und Information-Extraction-Modell für deutsche historische Inventardokumente.

Aufgabe:
Lies ALLE Informationen aus dem Bild dieses Inventarblatts und gib GENAU EIN JSON-Objekt zurück.

Das JSON MUSS folgende Felder enthalten:
- Überschrift: Dokumenttitel
- Inventarnummer: Inventar- oder Katalognummer
- Maße: Objekt mit L, B, D (Länge, Breite, Tiefe)
- Objektbezeichnung: Beschreibung/Name des Objekts
- Fundort: Fundort des Objekts
- Fundzeit: Zeit der Auffindung
- Beschreibungstext: Ausführlicher Beschreibungstext

Regeln:
- Gib NUR ein gültiges JSON-Objekt zurück (kein extra Text davor oder danach).
- Verwende GENAU diese Feldnamen und Groß-/Kleinschreibung.
- Wenn ein Feld leer ist oder nicht sichtbar, gib einen leeren String "" zurück.
- Das Feld "Maße" MUSS immer ein Objekt mit den Schlüsseln "L", "B", "D" sein, auch wenn leer.
- Erfinde keine zusätzlichen Felder."""

    def convert_to_conversation(self, sample):
        instruction = self._instruction_text()
        gt_json_string = self.json_to_string_readable(sample)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": instruction},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": gt_json_string}]},
        ]
        return {"messages": conversation}

    # ----------------------- DATA PREP -----------------------

    def prepare_data_no_augmentation(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)

        processed = []
        for i, item in enumerate(data):
            if "image_name" not in item:
                print(f"Warning: 'image_name' missing in item index {i}, skipping.")
                continue

            image_path = os.path.join(images_dir, item["image_name"])
            if os.path.exists(image_path):
                item_copy = item.copy()
                item_copy["image_path"] = image_path
                processed.append(item_copy)
            else:
                print(f"Warning: Image not found: {image_path}, skipping...")

        print(f"Data: {len(processed)} valid samples out of {len(data)} total (no augmentation)")
        return [self.convert_to_conversation(sample) for sample in processed]

    # ----------------------- CER -----------------------

    def calculate_cer(self, predictions, targets):
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
                total_cer += jiwer.cer(target_str, pred_str)
                count += 1

        return total_cer / count if count > 0 else 1.0

    # ----------------------- AUTOREGRESSIVE GEN-CER EVAL -----------------------

    def evaluate_generation_cer_on_items(
        self,
        model,
        val_items: List[Dict[str, Any]],
        images_dir: str,
        max_new_tokens: int = 1024,
    ) -> float:
        FastVisionModel.for_inference(model)
        model.eval()
        device = next(model.parameters()).device

        instruction = self._instruction_text()

        preds, tgts = [], []

        for idx, item in enumerate(val_items):
            img_name = item.get("image_name", f"val_{idx}")
            image_path = os.path.join(images_dir, img_name)

            gt_str = self.json_to_string_readable(item)

            if not os.path.exists(image_path):
                preds.append("")
                tgts.append(gt_str)
                continue

            image = None
            try:
                image = Image.open(image_path).convert("RGB")

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
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.0,
                        use_cache=True,
                    )

                input_len = inputs["input_ids"].shape[-1]
                gen_ids = outputs[0][input_len:]
                gen_text = self.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                pred_json = self.extract_json_from_response(gen_text)
                pred_str = self.json_to_string_readable(pred_json) if pred_json else ""

                preds.append(pred_str)
                tgts.append(gt_str)

            except Exception:
                preds.append("")
                tgts.append(gt_str)
            finally:
                if image is not None:
                    image.close()
                    del image
                if "inputs" in locals():
                    del inputs
                if "outputs" in locals():
                    del outputs
                if "gen_ids" in locals():
                    del gen_ids
                torch.cuda.empty_cache()
                gc.collect()

        avg_cer = self.calculate_cer(preds, tgts)

        FastVisionModel.for_training(model)
        model.train()

        return avg_cer

    # ----------------------- MULTI-STAGE TRAINING -----------------------

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 12,
        batch_size: int = 2,
        learning_rate: float = 3e-5,
        stage1_epochs: int = 4,
        gen_val_subset_size: int = 50,
        gen_val_max_new_tokens: int = 1024,
        gen_val_seed: int = 3407,
    ):
        """
        Multi-stage training:
        - Stage 1: warm-up (teacher forcing only, no eval, higher LR)
        - Stage 2: main training (eval each epoch, Trainer-managed BEST by eval_gen_cer)
        - End: Save ONLY best model to best_model_gen_cer, remove stage2 checkpoints
        """
        print("Preparing training and validation datasets for INVENTORY...")

        train_dataset = self.prepare_data_no_augmentation(train_jsonl_path, images_dir)
        val_dataset = self.prepare_data_no_augmentation(val_jsonl_path, images_dir)
        val_raw = self.load_jsonl(val_jsonl_path)

        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

        FastVisionModel.for_training(self.model)

        # -------------------------
        # Stage 1 – warm-up
        # -------------------------
        print("\n" + "=" * 70)
        print("STAGE 1: Warm-up training (no eval, teacher forcing)")
        print("=" * 70 + "\n")

        stage1_output_dir = os.path.join(output_dir, "stage1")
        os.makedirs(stage1_output_dir, exist_ok=True)

        stage1_args = SFTConfig(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,
            warmup_steps=50,
            num_train_epochs=stage1_epochs,
            learning_rate=learning_rate * 3.0,
            logging_steps=5,
            eval_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            optim="adamw_torch_fused",
            load_best_model_at_end=False,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/logs_stage1",
            logging_first_step=True,
            seed=3407,
            output_dir=stage1_output_dir,
            save_safetensors=True,
            prediction_loss_only=False,
            disable_tqdm=False,
            label_smoothing_factor=0.05,
        )

        trainer_stage1 = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=None,
            compute_metrics=None,  # gen-CER comes in stage2 callback
            args=stage1_args,
        )

        trainer_stage1.train()
        del trainer_stage1
        torch.cuda.empty_cache()
        gc.collect()

        # -------------------------
        # Stage 2 – main training
        # -------------------------
        print("\n" + "=" * 70)
        print("STAGE 2: Main training (eval each epoch, BEST by autoregressive gen-CER)")
        print("=" * 70 + "\n")

        remaining_epochs = max(1, num_epochs - stage1_epochs)

        stage2_output_dir = os.path.join(output_dir, "stage2")
        os.makedirs(stage2_output_dir, exist_ok=True)

        gen_cer_callback = GenerationCEREvalCallback(
            finetuner=self,
            val_raw_items=val_raw,
            images_dir=images_dir,
            subset_size=gen_val_subset_size,
            subset_seed=gen_val_seed,
            max_new_tokens=gen_val_max_new_tokens,
        )

        stage2_args = SFTConfig(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=50,
            num_train_epochs=remaining_epochs,
            learning_rate=learning_rate,
            logging_steps=5,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            optim="adamw_torch_fused",

            # ✅ Trainer-managed best checkpoint by eval_gen_cer
            load_best_model_at_end=True,
            metric_for_best_model="gen_cer",
            greater_is_better=False,

            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/logs_stage2",
            logging_first_step=True,
            seed=3407,
            output_dir=stage2_output_dir,
            save_safetensors=True,
            prediction_loss_only=False,
            disable_tqdm=False,
            label_smoothing_factor=0.05,
        )

        trainer_stage2 = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=None,  # gen-CER comes from callback (autoregressive)
            args=stage2_args,
            callbacks=[
                gen_cer_callback,
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01,
                ),
            ],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting Stage 2 training (Trainer selects BEST by eval_gen_cer)...")
        trainer_stats = trainer_stage2.train()

        print("\n=== BEST CHECKPOINT INFO (Stage 2) ===")
        print(f"Best checkpoint: {trainer_stage2.state.best_model_checkpoint}")
        print(f"Best metric (gen_cer): {trainer_stage2.state.best_metric}")

        # ✅ EXPLICIT: Assign best model back to self.model (don't rely on implicit in-place modification)
        self.model = trainer_stage2.model

        # Save ONLY best model folder at the top run_dir
        best_dir = os.path.join(output_dir, "best_model_gen_cer")
        os.makedirs(best_dir, exist_ok=True)
        print(f"\nSaving ONLY best model to: {best_dir}")
        self.model.save_pretrained(best_dir)
        self.tokenizer.save_pretrained(best_dir)

        # Remove all stage2 checkpoint folders so only one remains
        print("\nCleaning up: removing stage2 checkpoint-* folders (keeping only best_model_gen_cer)...")
        remove_all_stage2_checkpoints(stage2_output_dir)

        # bookkeeping
        self.best_model_dir = best_dir
        self.best_val_cer = float(trainer_stage2.state.best_metric) if trainer_stage2.state.best_metric is not None else None
        self.cer_history = list(getattr(gen_cer_callback, "cer_history", []))

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"Peak reserved memory = {used_memory} GB ({used_percentage}%).")

        return trainer_stage2

    # ----------------------- TEST EVAL -----------------------

    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict[str, Any]:
        print("Starting evaluation on INVENTORY test.jsonl with aggressive chunking...")

        FastVisionModel.for_inference(self.model)

        test_data = self.load_jsonl(test_jsonl_path)
        print(f"Loaded {len(test_data)} test samples")

        predictions = []
        all_cer_scores = []

        instruction = self._instruction_text()

        chunk_size = 5
        num_chunks = (len(test_data) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(test_data))
            chunk_data = test_data[start_idx:end_idx]

            print("\n" + "=" * 60)
            print(f"Chunk {chunk_idx+1}/{num_chunks} ({len(chunk_data)} images)")
            print("=" * 60)

            for i, test_item in enumerate(chunk_data):
                abs_idx = start_idx + i + 1
                img_name = test_item.get("image_name", f"idx_{abs_idx}")
                print(f"\n[{abs_idx}/{len(test_data)}] Processing: {img_name}")

                image_path = os.path.join(images_dir, img_name)

                if not os.path.exists(image_path):
                    print("  ⚠️  Image not found")
                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": self.json_to_string_readable(test_item),
                        "raw_response": "Error: Image not found",
                        "cer_score": 1.0,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)
                    continue

                image = None
                try:
                    image = Image.open(image_path).convert("RGB")

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
                    ).to("cuda")

                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            do_sample=False,
                            repetition_penalty=1.0,
                            use_cache=True,
                        )

                    input_len = inputs["input_ids"].shape[-1]
                    generated_ids_trimmed = outputs[0][input_len:]
                    generated_text = self.tokenizer.decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    predicted_json = self.extract_json_from_response(generated_text)
                    gt_json_string = self.json_to_string_readable(test_item)
                    pred_json_string = self.json_to_string_readable(predicted_json) if predicted_json else ""

                    cer_score = jiwer.cer(gt_json_string, pred_json_string) if pred_json_string else 1.0

                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": predicted_json,
                        "predicted_text": pred_json_string,
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": gt_json_string,
                        "raw_response": generated_text,
                        "cer_score": cer_score,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(cer_score)

                    print(f"  ✅ CER: {cer_score:.4f} ({cer_score*100:.2f}%)")

                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": self.json_to_string_readable(test_item),
                        "raw_response": f"Error: {str(e)}",
                        "cer_score": 1.0,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)

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

            torch.cuda.empty_cache()
            gc.collect()

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
        return {
            "total_images": len(all_cer_scores),
            "average_cer": avg_cer,
            "median_cer": float(np.median(all_cer_scores)),
            "minimum_cer": float(min(all_cer_scores)),
            "maximum_cer": float(max(all_cer_scores)),
            "std_cer": float(np.std(all_cer_scores)),
            "perfect_matches": int(sum(1 for cer in all_cer_scores if cer == 0.0)),
        }

    def save_cer_results(self, cer_stats: Dict, cer_file: str, num_predictions: int):
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("CER EVALUATION RESULTS - GEMMA-3 INVENTORY (A100)\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n")
            f.write(f"Median CER: {cer_stats['median_cer']:.4f} ({cer_stats['median_cer']*100:.2f}%)\n")
            f.write(f"Minimum CER: {cer_stats['minimum_cer']:.4f} ({cer_stats['minimum_cer']*100:.2f}%)\n")
            f.write(f"Maximum CER: {cer_stats['maximum_cer']:.4f} ({cer_stats['maximum_cer']*100:.2f}%)\n")
            f.write(f"Standard Deviation: {cer_stats['std_cer']:.4f}\n\n")
            f.write(
                f"Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']} "
                f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n"
            )
            f.write(f"Total images processed: {num_predictions}\n")

            if self.best_val_cer is not None:
                f.write(
                    f"\nBest validation gen-CER (Stage 2, autoregressive): "
                    f"{self.best_val_cer:.4f} ({self.best_val_cer*100:.2f}%)\n"
                )
            if getattr(self, "cer_history", None):
                f.write("\nGEN-CER HISTORY (validation subset):\n")
                for entry in self.cer_history:
                    f.write(
                        f"  Epoch {entry['epoch']:.2f}: {entry['cer']:.4f} "
                        f"({entry['cer']*100:.2f}%)\n"
                    )

        print(f"\nCER evaluation results saved to: {cer_file}")

    def save_model(self, trainer, output_dir: str):
        # In this flow we already saved ONLY best model to best_model_gen_cer
        if self.best_model_dir is not None:
            print(f"\nBest model already saved (ONLY ONE) at: {self.best_model_dir}")
        else:
            fallback_dir = os.path.join(output_dir, "final_model_fallback")
            print(f"\n[WARN] best_model_dir missing. Saving fallback final model to {fallback_dir}...")
            trainer.model.save_pretrained(fallback_dir)
            trainer.tokenizer.save_pretrained(fallback_dir)


# =====================================================================
# main()
# =====================================================================

def main():
    base_checkpoint_dir = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/inventory_dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_A100_multistage_genCER_INV_noSort_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    config = {
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",
        "output_dir": run_dir,
        "num_epochs": 12,
        "stage1_epochs": 4,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "gen_val_subset_size": 50,
        "gen_val_max_new_tokens": 1024,
        "gen_val_seed": 3407,
    }

    with open(os.path.join(run_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    local_model_path = (
        "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/"
        "models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/"
        "snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"
    )

    print("\n" + "=" * 60)
    print("GEMMA-3 INVENTORY - MULTI-STAGE + BEST BY AUTOREGRESSIVE GEN-CER (NO SORT, SAVE ONLY ONE)")
    print("=" * 60)

    finetuner = InventoryGemma3Finetune(model_path=local_model_path)

    print("\n" + "=" * 60)
    print("STARTING MULTI-STAGE TRAINING")
    print("=" * 60)

    trainer = finetuner.train_model(
        train_jsonl_path=config["train_jsonl_path"],
        val_jsonl_path=config["val_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        stage1_epochs=config["stage1_epochs"],
        gen_val_subset_size=config["gen_val_subset_size"],
        gen_val_max_new_tokens=config["gen_val_max_new_tokens"],
        gen_val_seed=config["gen_val_seed"],
    )

    finetuner.save_model(trainer, config["output_dir"])

    print("\n" + "=" * 60)
    print("STARTING EVALUATION ON TEST.JSONL (IN-MEMORY BEST MODEL)")
    print("=" * 60)

    test_results = finetuner.evaluate_on_test_set(
        test_jsonl_path=config["test_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
    )

    print(f"\n🎉 Multi-stage training completed!")
    print(f"All outputs saved to: {run_dir}")
    print(
        f"\nFinal TEST CER: {test_results['cer_stats']['average_cer']:.4f} "
        f"({test_results['cer_stats']['average_cer']*100:.2f}%)"
    )

    if finetuner.best_val_cer is not None:
        print(
            f"Best VALIDATION gen-CER (Stage 2): "
            f"{finetuner.best_val_cer:.4f} ({finetuner.best_val_cer*100:.2f}%)"
        )
        print(f"ONLY saved best model at: {finetuner.best_model_dir}")


if __name__ == "__main__":
    main()
