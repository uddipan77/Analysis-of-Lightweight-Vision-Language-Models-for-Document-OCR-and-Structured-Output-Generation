#!/usr/bin/env python3
# inventory_qwen_finetune_schema_prompt_cer.py
# Qwen2.5-VL-7B Inventory OCR fine-tuning with Unsloth
# - Compact schema-style prompt (structure is in labels)
# - Best model selected by lowest CER on validation set
# - CER is computed via autoregressive generation on a small val subset each epoch
# - No compute_metrics -> avoids huge logits-in-RAM

import os

# ----------------------------------------------------------------------
# IMPORTANT: set Unsloth / PyTorch env BEFORE importing unsloth/torch
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

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

from transformers import TrainerCallback
from qwen_vl_utils import process_vision_info


# ======================================================================
# Inventory JSON-schema-style prompt
# ======================================================================

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


# ======================================================================
# Main Finetuner Class
# ======================================================================

class InventoryOCRFinetune:
    def __init__(
        self,
        model_path: str = "/home/vault/iwi5/iwi5298h/models/qwen7b",
    ):
        """Initialize Unsloth Qwen2.5-VL with LoRA and CER-based selection for inventory OCR."""
        print(f"Loading Qwen2.5-VL model from {model_path} with Unsloth...")

        self.instruction = INVENTORY_SCHEMA_PROMPT

        # Base Unsloth model (4-bit) for training
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,    # shorter to save memory
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        # LoRA: same style as staircase script
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0.1402367506273414,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=3407,
            use_rslora=False,
        )

        print("Model loaded successfully with LoRA (r=16, no offload)!")

    # --------------------- Utility & IO --------------------- #

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

    def dict_without_image_meta(self, obj: Dict) -> Dict:
        """Drop image-specific keys (image_name, image_path)."""
        return {k: v for k, v in obj.items() if k not in ("image_name", "image_path")}

    def json_to_string_no_sort(self, obj: Dict) -> str:
        return json.dumps(
            self.dict_without_image_meta(obj),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def safe_json_loads(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    # --------------------- JSON extraction --------------------- #

    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON object from model response (same pattern as staircase)."""
        response = response.strip()

        # Try to find JSON block
        matches = re.findall(r"\{.*\}", response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Fallback: whole string
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    # --------------------- Image handling --------------------- #

    def find_image_path(self, image_name: str, images_dir: str) -> str:
        """Flexible matching for inventory images."""
        exact_path = os.path.join(images_dir, image_name)
        if os.path.exists(exact_path):
            return exact_path

        # Try basename fuzzy match
        base_name = os.path.splitext(image_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matches = glob.glob(search_pattern)
        if matches:
            return matches[0]

        return exact_path

    # --------------------- Conversation conversion --------------------- #

    def convert_to_conversation(self, sample: Dict) -> Dict:
        """
        Build chat-style training example: inventory instruction + image -> pure JSON text.
        """
        label_obj = self.dict_without_image_meta(sample)
        gt_json_string = self.json_to_string_no_sort(label_obj)

        image_path = sample.get("image_path", None)
        contents = [{"type": "text", "text": self.instruction}]

        if isinstance(image_path, str) and image_path and os.path.exists(image_path):
            contents.append({"type": "image", "image": image_path})
        else:
            print(
                f"[WARN] convert_to_conversation: missing/invalid image_path, using text-only."
            )

        conversation = [
            {
                "role": "user",
                "content": contents,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gt_json_string}],
            },
        ]
        return {"messages": conversation}

    # --------------------- Dataset prep --------------------- #

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        """
        Prepare training data in conversation format (no augmentation, for now).
        """
        data = self.load_jsonl(jsonl_path)
        converted_dataset = []

        print(f"Preparing training data from {jsonl_path} ...")

        for item in data:
            image_path = self.find_image_path(item["image_name"], images_dir)
            if os.path.exists(image_path):
                item = item.copy()
                item["image_path"] = image_path
                converted_dataset.append(self.convert_to_conversation(item))
            else:
                print(
                    f"Warning: Image not found for {item['image_name']}, skipping..."
                )

        print(f"Training samples prepared: {len(converted_dataset)}")
        return converted_dataset

    def prepare_validation_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        """Validation data without augmentation, same as training style."""
        data = self.load_jsonl(jsonl_path)
        converted_dataset = []

        for item in data:
            image_path = self.find_image_path(item["image_name"], images_dir)
            if os.path.exists(image_path):
                item = item.copy()
                item["image_path"] = image_path
                converted_dataset.append(self.convert_to_conversation(item))
            else:
                print(f"Warning: Image not found for {item['image_name']}, skipping...")

        print(f"Validation samples: {len(converted_dataset)}")
        return converted_dataset

    # --------------------- CER helpers --------------------- #

    def calculate_cer(self, predictions, targets):
        """
        Compute average CER using jiwer.cer over string pairs.
        """
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
                print(f"[WARN] CER computation failed for one sample: {e}")
                continue

            total_cer += cer_val
            valid_pairs += 1

        return (total_cer / valid_pairs) if valid_pairs > 0 else 1.0

    # --------------------- Training (best model = lowest CER) --------------------- #

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 20,
        batch_size: int = 2,
        learning_rate: float = 0.00021050710372623888,
    ):
        print("Preparing training and validation datasets...")
        train_dataset = self.prepare_training_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_validation_data(val_jsonl_path, images_dir)
        val_raw_data = self.load_jsonl(val_jsonl_path)
        print(
            f"Training: {len(train_dataset)} samples, "
            f"Validation (raw): {len(val_raw_data)} records"
        )

        FastVisionModel.for_training(self.model)

        # CER-based validation callback (like staircase script)
        best_ckpt_dir = os.path.join(output_dir, "best_model_cer")

        cer_callback = InventoryCERCallback(
            parent=self,
            val_data=val_raw_data,
            images_dir=images_dir,
            best_ckpt_dir=best_ckpt_dir,
            max_eval_samples=30,  # subset for speed & memory
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # No compute_metrics -> avoid logits RAM explosion
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=8,
                warmup_steps=50,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=10,

                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=False,   # we manage "best by CER" ourselves
                save_total_limit=1,
                metric_for_best_model="eval_cer",  # logged by callback
                greater_is_better=False,

                dataloader_num_workers=0,
                dataloader_pin_memory=False,

                weight_decay=0.06974561412485879,
                lr_scheduler_type="cosine",
                warmup_ratio=0.053387755131264064,
                optim="adamw_8bit",
                gradient_checkpointing=False,

                remove_unused_columns=False,
                dataset_text_field=None,
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=2048,

                report_to="tensorboard",
                logging_dir=f"{output_dir}/logs",
                seed=3407,
                output_dir=output_dir,
            ),
            callbacks=[cer_callback],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting training (best model tracked by CER callback)...")
        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(
            f"Peak reserved memory for training = {used_memory_for_lora} GB "
            f"({lora_percentage} % of max)."
        )

        # Load best-by-CER weights (filtered state_dict) same as staircase
        if cer_callback.best_state_path is not None:
            print(
                f"Loading best-by-CER weights from {cer_callback.best_state_path} ..."
            )
            loaded_state = torch.load(cer_callback.best_state_path, map_location="cpu")

            current_state = self.model.state_dict()
            filtered_state = {
                k: v for k, v in loaded_state.items() if k in current_state
            }

            print(
                f"Loaded state_dict has {len(loaded_state)} keys; "
                f"{len(filtered_state)} of them match the current model and will be loaded."
            )

            missing_keys, unexpected_keys = self.model.load_state_dict(
                filtered_state,
                strict=False,
            )

            if missing_keys:
                print(f"[WARN] Missing keys when loading best CER weights ({len(missing_keys)}):")
                print("  (showing first 10)", missing_keys[:10])
            if unexpected_keys:
                print(f"[WARN] Unexpected keys when loading best CER weights ({len(unexpected_keys)}):")
                print("  (showing first 10)", unexpected_keys[:10])

            trainer.model = self.model
            print(
                f"Restored best validation CER = {cer_callback.best_cer:.4f} "
                f"({cer_callback.best_cer*100:.2f}%)"
            )
        else:
            print(
                "WARNING: CER callback did not save any best model; using final epoch weights."
            )

        self.cer_history = cer_callback.cer_history

        torch.cuda.empty_cache()
        return trainer

    # --------------------- Evaluation on split (e.g. test) --------------------- #

    def evaluate_on_split(
        self, jsonl_path: str, images_dir: str, output_dir: str, split_name: str
    ) -> Dict:
        """
        Evaluation over a given split (e.g., test.jsonl) with greedy decoding,
        JSON extraction, and CER calculation.
        """
        split_name = split_name.lower()
        print(f"Starting evaluation on {split_name}.jsonl...")

        FastVisionModel.for_inference(self.model)
        self.model.eval()

        predictions = []
        all_cer_scores = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                test_item = json.loads(line)
                print(
                    f"Processing {split_name} image {len(predictions)+1}: {test_item['image_name']}"
                )

                image_path = self.find_image_path(
                    test_item["image_name"], images_dir
                )

                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    gt_obj = self.dict_without_image_meta(test_item)
                    prediction_entry = {
                        "image_name": test_item["image_name"],
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": gt_obj,
                        "target_text": self.json_to_string_no_sort(gt_obj),
                        "raw_response": "Error: Image not found",
                        "cer_score": 1.0,
                        "matched_image_path": image_path,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)
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
                    ).to("cuda")

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            use_cache=True,
                            temperature=0.0,
                            do_sample=False,       # deterministic greedy
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

                    predicted_json_raw = self.extract_json_from_response(
                        generated_text
                    )
                    predicted_json = self.dict_without_image_meta(
                        predicted_json_raw or {}
                    )

                    gt_obj = self.dict_without_image_meta(test_item)

                    gt_json_string = self.json_to_string_no_sort(gt_obj)
                    pred_json_string = self.json_to_string_no_sort(predicted_json)

                    cer_score = jiwer.cer(gt_json_string, pred_json_string)

                    prediction_entry = {
                        "image_name": test_item["image_name"],
                        "predicted_json": predicted_json,
                        "predicted_text": pred_json_string,
                        "target_json": gt_obj,
                        "target_text": gt_json_string,
                        "raw_response": generated_text,
                        "cer_score": cer_score,
                        "matched_image_path": image_path,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(cer_score)

                    print(f"  Processed successfully. CER: {cer_score:.3f}")

                except Exception as e:
                    print(f"Error processing {test_item['image_name']}: {str(e)}")
                    gt_obj = self.dict_without_image_meta(test_item)
                    prediction_entry = {
                        "image_name": test_item["image_name"],
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": gt_obj,
                        "target_text": self.json_to_string_no_sort(gt_obj),
                        "raw_response": f"Error: {str(e)}",
                        "cer_score": 1.0,
                        "matched_image_path": image_path,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)

                if (idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

        predictions_file = os.path.join(
            output_dir, f"{split_name}_predictions.jsonl"
        )
        self.save_jsonl(predictions, predictions_file)

        cer_stats = self.calculate_cer_statistics(all_cer_scores)

        cer_file = os.path.join(
            output_dir, f"cer_evaluation_{split_name}_results.txt"
        )
        self.save_cer_results(cer_stats, cer_file, len(predictions), split_name)

        torch.cuda.empty_cache()

        return {
            "predictions": predictions,
            "cer_stats": cer_stats,
            "predictions_file": predictions_file,
            "cer_file": cer_file,
        }

    # --------------------- Stats & saving --------------------- #

    def calculate_cer_statistics(self, all_cer_scores: List[float]) -> Dict:
        if not all_cer_scores:
            return {}
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        median_cer = float(np.median(all_cer_scores))
        std_cer = float(np.std(all_cer_scores))
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

    def save_cer_results(
        self, cer_stats: Dict, cer_file: str, num_predictions: int, split_name: str
    ):
        split_name = split_name.upper()
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CER EVALUATION RESULTS ON INVENTORY {split_name}.JSONL\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"Average CER: {cer_stats['average_cer']:.4f} "
                f"({cer_stats['average_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Median CER: {cer_stats['median_cer']:.4f} "
                f"({cer_stats['median_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Minimum CER: {cer_stats['minimum_cer']:.4f} "
                f"({cer_stats['minimum_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Maximum CER: {cer_stats['maximum_cer']:.4f} "
                f"({cer_stats['maximum_cer']*100:.2f}%)\n"
            )
            f.write(f"Standard Deviation: {cer_stats['std_cer']:.4f}\n\n")

            f.write(
                f"Perfect matches: {cer_stats['perfect_matches']}/"
                f"{cer_stats['total_images']} "
                f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n"
            )
            f.write(f"Total images processed: {num_predictions}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("=" * 60 + "\n")
            f.write("- Model: Qwen2.5-VL-7B-Instruct with LoRA (from local path)\n")
            f.write("- Dataset: Inventory German Historical Documents\n")
            f.write("- Model Selection: Best-by-CER on validation set\n")
            f.write("- LoRA rank: 32\n")
            f.write("- LoRA alpha: 32\n")
            f.write("- LoRA dropout: 0.1402367506273414\n")
            f.write("- Weight decay: 0.06974561412485879\n")
            f.write("- Output format: Pure JSON strings\n")
            f.write("- Multiprocessing: Disabled (dataloader_num_workers=0)\n")

            if hasattr(self, "cer_history"):
                f.write("\nCER HISTORY (validation subset):\n")
                for entry in self.cer_history:
                    f.write(
                        f"  Epoch {entry['epoch']}: {entry['cer']:.4f} "
                        f"({entry['cer']*100:.2f}%)\n"
                    )

        print(f"CER evaluation results saved to: {cer_file}")

    def save_model(self, trainer, output_dir: str):
        print(f"Saving best-by-CER model to {output_dir}...")

        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)

        print("Saving model in recommended formats...")
        trainer.model.save_pretrained_merged(
            f"{output_dir}_merged_16bit",
            trainer.tokenizer,
            save_method="merged_16bit",
        )
        trainer.model.save_pretrained_merged(
            f"{output_dir}_lora",
            trainer.tokenizer,
            save_method="lora",
        )
        print("Best model saved successfully in multiple formats!")


# ======================================================================
# CER Validation Callback (Inventory, same idea as QwenCERCallback)
# ======================================================================

class InventoryCERCallback(TrainerCallback):
    def __init__(
        self,
        parent: InventoryOCRFinetune,
        val_data: List[Dict],
        images_dir: str,
        best_ckpt_dir: str,
        max_eval_samples: int = 30,
    ):
        """
        parent: the InventoryOCRFinetune instance
        val_data: raw validation json records (with image_name)
        images_dir: where inventory images live
        best_ckpt_dir: directory to save best-by-CER state_dict
        max_eval_samples: evaluate on this many samples each epoch (subset for speed)
        """
        self.parent = parent
        self.images_dir = images_dir
        self.best_ckpt_dir = best_ckpt_dir
        self.max_eval_samples = max_eval_samples

        self.val_data = val_data[:max_eval_samples]
        self.best_cer = float("inf")
        self.best_state_path = None
        self.cer_history: List[Dict[str, float]] = []

    def on_evaluate(self, args, state, control, model, tokenizer=None, metrics=None, **kwargs):
        if state.epoch is None:
            return control

        print("\n" + "=" * 80)
        print(f"🔍 Inventory CER Callback: COMPUTING CER ON VALIDATION SUBSET (Epoch {int(state.epoch)})")
        print("=" * 80)

        model.eval()
        device = next(model.parameters()).device

        predictions = []
        targets = []

        for i, val_item in enumerate(self.val_data):
            image_name = val_item["image_name"]
            print(
                f"   Validating {i+1}/{len(self.val_data)}: {image_name}",
                end="\r",
            )

            image_path = self.parent.find_image_path(image_name, self.images_dir)
            if not os.path.exists(image_path):
                continue

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.parent.instruction},
                            {"type": "image", "image": image_path},
                        ],
                    }
                ]

                input_text = self.parent.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.parent.tokenizer(
                    text=[input_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_cache=True,
                        temperature=0.0,
                        do_sample=False,
                        repetition_penalty=1.0,
                        pad_token_id=self.parent.tokenizer.pad_token_id,
                        eos_token_id=self.parent.tokenizer.eos_token_id,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]

                generated_text = self.parent.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                predicted_json_raw = self.parent.extract_json_from_response(
                    generated_text
                )
                predicted_json = self.parent.dict_without_image_meta(
                    predicted_json_raw or {}
                )
                gt_obj = self.parent.dict_without_image_meta(val_item)

                gt_str = self.parent.json_to_string_no_sort(gt_obj)
                pred_str = self.parent.json_to_string_no_sort(predicted_json)

                predictions.append(pred_str)
                targets.append(gt_str)

                del inputs, outputs, generated_ids_trimmed
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n   ⚠️  Validation error on {image_name}: {e}")
                gt_obj = self.parent.dict_without_image_meta(val_item)
                gt_str = self.parent.json_to_string_no_sort(gt_obj)
                predictions.append("")
                targets.append(gt_str)
                continue

        # Calculate CER on subset
        val_cer = self.parent.calculate_cer(predictions, targets)
        self.cer_history.append(
            {"epoch": float(state.epoch), "cer": float(val_cer)}
        )

        if metrics is not None:
            metrics["eval_cer"] = val_cer

        print(
            f"\n   ✅ Validation CER (Epoch {int(state.epoch)}): "
            f"{val_cer:.4f} ({val_cer*100:.2f}%)"
        )

        # Save best model by CER: torch.save(state_dict)
        if val_cer < self.best_cer:
            improvement = self.best_cer - val_cer
            self.best_cer = val_cer
            print(
                f"   🎯 NEW BEST CER: {self.best_cer:.4f} "
                f"(improved by {improvement:.4f})"
            )

            os.makedirs(self.best_ckpt_dir, exist_ok=True)
            best_state_path = os.path.join(self.best_ckpt_dir, "pytorch_model_best_cer.bin")
            torch.save(model.state_dict(), best_state_path)
            self.best_state_path = best_state_path
            print(f"   💾 Saved best-by-CER state_dict to: {best_state_path}")
        else:
            print(f"   📊 No improvement. Best CER remains: {self.best_cer:.4f}")

        print("=" * 80 + "\n")

        model.train()
        return control


# ======================================================================
# main()
# ======================================================================

def main():
    base_checkpoint_dir = "/home/vault/iwi5/iwi5298h/models_image_text/qwen/inventory_dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    config = {
        "model_path": "/home/vault/iwi5/iwi5298h/models/qwen7b",
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",
        "output_dir": run_dir,
        "num_epochs": 20,
        "batch_size": 2,
        "learning_rate": 0.00021050710372623888,
    }

    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    finetuner = InventoryOCRFinetune(
        model_path=config["model_path"],
    )

    print("=" * 60)
    print("STARTING INVENTORY DATASET TRAINING (BEST MODEL BY VALIDATION CER)")
    print("=" * 60)

    trainer = finetuner.train_model(
        train_jsonl_path=config["train_jsonl_path"],
        val_jsonl_path=config["val_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
    )

    # Save best-by-CER model in HF format + merged/lora
    finetuner.save_model(trainer, config["output_dir"])

    print("\n" + "=" * 60)
    print("STARTING EVALUATION ON INVENTORY TEST.JSONL (BEST-BY-CER WEIGHTS)")
    print("=" * 60)

    test_results = finetuner.evaluate_on_split(
        jsonl_path=config["test_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
        split_name="test",
    )

    print("\nEvaluation completed!")
    print(f"TEST predictions saved to: {test_results['predictions_file']}")
    print(f"TEST CER results saved to: {test_results['cer_file']}")
    print(f"All files saved in: {run_dir}")

    cer_stats = test_results["cer_stats"]
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY - INVENTORY DATASET (TEST SPLIT)")
    print("=" * 60)
    print(
        f"Average CER: {cer_stats['average_cer']:.4f} "
        f"({cer_stats['average_cer']*100:.2f}%)"
    )
    print(
        f"Median CER: {cer_stats['median_cer']:.4f} "
        f"({cer_stats['median_cer']*100:.2f}%)"
    )
    print(
        f"Perfect matches: {cer_stats['perfect_matches']}/"
        f"{cer_stats['total_images']} "
        f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)"
    )
    print(f"Total images processed: {cer_stats['total_images']}")
    print("\nInventory dataset training and evaluation completed!")


if __name__ == "__main__":
    main()
