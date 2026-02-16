#!/usr/bin/env python3
# gemma3_schmuck_finetune_A100.py
# ✅ A100-OPTIMIZED with evaluation chunking + memory management
#
# UPDATED (only applicable params you provided):
# - learning_rate = 0.00019555461627661946
# - weight_decay  = 0.04326601357814654
# - lora_r        = 16
# - lora_alpha    = 64
# - lora_dropout  = 0.13005137922900745
# - use_rslora    = False
# - num_epochs    = 7
#
# Everything else is unchanged.

import sys
import os

# ✅ CRITICAL: Enable logits return for compute_metrics
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import json
import shutil
from typing import List, Dict
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
import re
import jiwer
from datetime import datetime
from transformers import EarlyStoppingCallback
import numpy as np
import gc
from PIL import Image


class SchmuckGemma3Finetune:
    def __init__(self, model_path: str):
        """Initialize with A100-optimized settings"""
        print("Loading Gemma-3 vision model with Unsloth...")

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

        # ✅ UPDATED LoRA config (from your provided params)
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=64,
            lora_dropout=0.13005137922900745,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
        )

        print("Gemma-3 vision model loaded with updated LoRA config")

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

    def dict_without_file_name(self, obj):
        return {k: v for k, v in obj.items() if k not in ["file_name", "image_path"]}

    def json_to_string_readable(self, obj):
        clean_obj = self.dict_without_file_name(obj)
        return json.dumps(clean_obj, ensure_ascii=False, indent=2)

    def extract_json_from_response(self, response: str) -> Dict:
        response = response.strip()

        # fenced
        if response.startswith("```"):
            response = response.split("```", 1)[1]
            if response.startswith("json"):
                response = response[4:].strip()
            if "```" in response:
                response = response.split("```")[0].strip()

        # greedy json pattern
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # last attempt
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        return {}

    def convert_to_conversation(self, sample):
        instruction = """Extract all information from this German jewelry catalog document image as a structured JSON object.

The JSON should contain these fields:
- Gegenstand: Object/item name
- Inv.Nr: Inventory number  
- Herkunft: Origin/provenance
- Foto Notes: Photo notes
- Standort: Location
- Material: Material description
- Datierung: Dating/time period
- Maße: Measurements
- Gewicht: Weight
- erworben von: Acquired from
- am: Acquired on (date)
- Preis: Price
- Vers.-Wert: Insurance value
- Beschreibung: Description
- Literatur: Literature references
- Ausstellungen: Exhibitions

Return ONLY the JSON object, properly formatted."""
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

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)

        for item in data:
            item["image_path"] = os.path.join(images_dir, item["file_name"])
            if not os.path.exists(item["image_path"]):
                print(f"Warning: Image not found: {item['image_path']}")

        # filter to only existing
        data = [item for item in data if os.path.exists(item["image_path"])]
        converted_dataset = [self.convert_to_conversation(sample) for sample in data]

        return converted_dataset

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
                cer = jiwer.cer(target_str, pred_str)
                total_cer += cer
                count += 1

        return total_cer / count if count > 0 else 1.0

    # ✅ A100-compatible compute_metrics with batch support
    def compute_metrics_for_trainer(self, eval_preds, compute_result: bool = True):
        """
        A100-optimized compute_metrics with eval_accumulation support.

        The key fix vs your original code:
        - we ALWAYS move predictions (and labels) to CPU numpy before np.argmax
        - we handle torch.Tensor, tuple of tensors, and Unsloth EmptyLogits
        """
        if not compute_result:
            return {}

        predictions, label_ids = eval_preds

        # Some trainers give (logits, ...) as tuple
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Handle EmptyLogits (Unsloth sometimes returns this)
        if hasattr(predictions, "__class__") and "EmptyLogits" in str(
            predictions.__class__
        ):
            return {"cer": 0.0, "cer_percentage": 0.0}

        # ✅ Move predictions to CPU + numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().float().numpy()

        # ✅ Move labels to CPU + numpy
        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.detach().cpu().numpy()

        # shape: (batch, seq_len, vocab)
        predicted_token_ids = np.argmax(predictions, axis=-1)

        pred_texts = []
        label_texts = []

        for pred_ids, label_ids_single in zip(predicted_token_ids, label_ids):
            # mask ignore index
            if isinstance(label_ids_single, np.ndarray):
                label_ids_single = label_ids_single[label_ids_single != -100]
            else:
                label_ids_single = np.array(
                    [x for x in label_ids_single if x != -100], dtype=np.int32
                )

            try:
                pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                label_text = self.tokenizer.decode(
                    label_ids_single, skip_special_tokens=True
                )
                pred_texts.append(pred_text)
                label_texts.append(label_text)
            except Exception:
                pred_texts.append("")
                label_texts.append("")

        cer = self.calculate_cer(pred_texts, label_texts)

        torch.cuda.empty_cache()
        gc.collect()

        return {
            "cer": cer,
            "cer_percentage": cer * 100.0,
        }

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 15,
        batch_size: int = 2,
        learning_rate: float = 3e-5,
    ):
        """Train with A100-optimized settings."""
        print("Preparing training and validation datasets...")
        train_dataset = self.prepare_training_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_training_data(val_jsonl_path, images_dir)
        print(
            f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples"
        )

        FastVisionModel.for_training(self.model)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics_for_trainer,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                eval_accumulation_steps=4,
                batch_eval_metrics=True,
                warmup_steps=100,
                num_train_epochs=num_epochs,
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

                # ✅ UPDATED (from your provided params)
                weight_decay=0.04326601357814654,

                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                optim="adamw_torch_fused",
                load_best_model_at_end=True,
                metric_for_best_model="cer",
                greater_is_better=False,
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=2048,
                report_to="tensorboard",
                logging_dir=f"{output_dir}/logs",
                logging_first_step=True,
                seed=3407,
                output_dir=output_dir,
                save_safetensors=True,
                prediction_loss_only=False,
                disable_tqdm=False,
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01,
                )
            ],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting training with CER-based evaluation (A100-optimized)...")
        trainer_stats = trainer.train()

        used_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

        return trainer

    def evaluate_on_test_set(
        self, test_jsonl_path: str, images_dir: str, output_dir: str
    ) -> Dict:
        """Evaluate with aggressive memory management and smaller chunks"""
        print("Starting evaluation on test.jsonl with aggressive chunking...")

        FastVisionModel.for_inference(self.model)

        test_data = self.load_jsonl(test_jsonl_path)
        print(f"Loaded {len(test_data)} test samples")

        predictions = []
        all_cer_scores = []

        instruction = """Extract all information from this German jewelry catalog document image as a structured JSON object.

The JSON should contain these fields:
- Gegenstand: Object/item name
- Inv.Nr: Inventory number
- Herkunft: Origin/provenance
- Foto Notes: Photo notes
- Standort: Location
- Material: Material description
- Datierung: Dating/time period
- Maße: Measurements
- Gewicht: Weight
- erworben von: Acquired from
- am: Acquired on (date)
- Preis: Price
- Vers.-Wert: Insurance value
- Beschreibung: Description
- Literatur: Literature references
- Ausstellungen: Exhibitions

Return ONLY the JSON object, properly formatted."""

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
                print(f"\n[{abs_idx}/{len(test_data)}] Processing: {test_item['file_name']}")

                image_path = os.path.join(images_dir, test_item["file_name"])

                if not os.path.exists(image_path):
                    print("  ⚠️  Image not found")
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
                    ).to("cuda")

                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1000,
                            do_sample=False,
                            temperature=0.1,
                            repetition_penalty=1.1,
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
                    pred_json_string = (
                        self.json_to_string_readable(predicted_json)
                        if predicted_json
                        else ""
                    )

                    cer_score = (
                        jiwer.cer(gt_json_string, pred_json_string)
                        if pred_json_string
                        else 1.0
                    )

                    prediction_entry = {
                        "file_name": test_item["file_name"],
                        "predicted_json": predicted_json,
                        "predicted_text": pred_json_string,
                        "target_json": self.dict_without_file_name(test_item),
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
                        "file_name": test_item["file_name"],
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": self.dict_without_file_name(test_item),
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

            print(f"\nChunk {chunk_idx+1} completed. Cleaning memory...")
            torch.cuda.empty_cache()
            gc.collect()

            if (chunk_idx + 1) % 3 == 0 or chunk_idx == num_chunks - 1:
                intermediate_file = os.path.join(
                    output_dir, f"test_predictions_chunk_{chunk_idx+1}.jsonl"
                )
                self.save_jsonl(predictions, intermediate_file)
                print(f"Saved intermediate results to: {intermediate_file}")

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
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        median_cer = np.median(all_cer_scores)
        std_cer = np.std(all_cer_scores)
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

    def save_cer_results(self, cer_stats: Dict, cer_file: str, num_predictions: int):
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("CER EVALUATION RESULTS - GEMMA-3 SCHMUCK (A100)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Median CER: {cer_stats['median_cer']:.4f} ({cer_stats['median_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Minimum CER: {cer_stats['minimum_cer']:.4f} ({cer_stats['minimum_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Maximum CER: {cer_stats['maximum_cer']:.4f} ({cer_stats['maximum_cer']*100:.2f}%)\n"
            )
            f.write(f"Standard Deviation: {cer_stats['std_cer']:.4f}\n\n")

            f.write(
                f"Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']} ({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n"
            )
            f.write(f"Total images processed: {num_predictions}\n")

        print(f"\nCER evaluation results saved to: {cer_file}")

    def save_model(self, trainer, output_dir: str):
        print(f"\nSaving model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("Model (LoRA adapters) saved successfully!")


def main():
    """Main function - A100 optimized with chunking."""

    base_checkpoint_dir = (
        "/home/vault/iwi5/iwi5298h/models_image_text/gemma/schmuck/finetune"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_A100_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    # ✅ UPDATED (only applicable params you provided)
    config = {
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",
        "output_dir": run_dir,
        "num_epochs": 7,
        "batch_size": 2,  # unchanged
        "learning_rate": 0.00019555461627661946,
    }

    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    local_model_path = "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"

    print("\n" + "=" * 60)
    print("GEMMA-3 SCHMUCK - A100 OPTIMIZED")
    print("=" * 60)

    finetuner = SchmuckGemma3Finetune(model_path=local_model_path)

    print("\n" + "=" * 60)
    print("STARTING TRAINING (A100-OPTIMIZED)")
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

    finetuner.save_model(trainer, config["output_dir"])

    print("\n" + "=" * 60)
    print("STARTING EVALUATION ON TEST.JSONL")
    print("=" * 60)

    test_results = finetuner.evaluate_on_test_set(
        test_jsonl_path=config["test_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
    )

    print(f"\n🎉 Gemma-3 Schmuck training completed!")
    print(f"All outputs saved to: {run_dir}")
    print(
        f"\nFinal CER: {test_results['cer_stats']['average_cer']:.4f} ({test_results['cer_stats']['average_cer']*100:.2f}%)"
    )


if __name__ == "__main__":
    main()
