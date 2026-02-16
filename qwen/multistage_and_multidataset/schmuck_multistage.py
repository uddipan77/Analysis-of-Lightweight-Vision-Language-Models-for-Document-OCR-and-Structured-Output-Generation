import torch
import json
import os
from typing import List, Dict, Any
from unsloth import FastVisionModel

from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

import re
import jiwer
from datetime import datetime
from transformers import EarlyStoppingCallback, TrainerCallback
import numpy as np
import glob
from PIL import Image, ImageEnhance, ImageFilter
import torchvision.transforms as transforms
import random
import tempfile
import gc
from torch.utils.data import DataLoader


class SchmuckOCRFinetune:
    def __init__(self, model_path: str = "/home/vault/iwi5/iwi5298h/models/qwen7b"):
        """Initialize with optimized Unsloth configuration using local model."""
        print(f"Loading Qwen2.5-VL model from {model_path} with Unsloth...")

        # Load model and tokenizer from local path
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,  # handle full JSON structure
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True
        )

        # Optimized LoRA configuration for Unsloth
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,  # rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,  # match rank
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )

        # Track best checkpoint by CER
        self.best_model_dir: str | None = None
        self.best_val_cer: float | None = None

        print("Model loaded successfully with optimized LoRA adapters!")

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from a JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def save_jsonl(self, data: List[Dict], file_path: str):
        """Save data to a JSONL file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def clean_dict_for_ground_truth(self, obj):
        """Remove both 'file_name' and 'image_path' keys for clean ground truth"""
        return {k: v for k, v in obj.items() if k not in ['file_name', 'image_path']}

    def json_to_string_no_sort(self, obj):
        """Convert dict to JSON string preserving original key order"""
        clean_obj = self.clean_dict_for_ground_truth(obj)
        return json.dumps(clean_obj, ensure_ascii=False, separators=(',', ':'))

    def safe_json_loads(self, s):
        """Safely parse JSON string"""
        try:
            return json.loads(s)
        except Exception:
            return None

    def canonical_json_string(self, obj):
        """Convert to canonical JSON string for comparison"""
        return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(',', ':'))

    def extract_json_from_response(self, response: str) -> Dict:
        """Extract JSON object from model response"""
        if isinstance(response, list):
            if len(response) > 0:
                response = str(response[0])
            else:
                response = ""

        response = str(response).strip()

        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            for match in sorted(matches, key=len, reverse=True):
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 3:
                        return parsed
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        return {}

    def find_image_path(self, file_name: str, images_dir: str) -> str:
        """Find image path using flexible matching."""
        exact_path = os.path.join(images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path

        base_name, _ = os.path.splitext(file_name)
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]

        return exact_path

    def convert_to_conversation(self, sample):
        """Convert sample to conversation format."""
        instruction = """Extract ALL the jewelry information from this German historical document image as a complete JSON object. 

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

Include ALL fields, even if empty (use empty string ""). Preserve exact German spelling and punctuation. Return ONLY the JSON object without any additional text or formatting."""

        gt_json_string = self.json_to_string_no_sort(sample)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image_path"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": gt_json_string}
                ]
            }
        ]
        return {"messages": conversation}


class ChunkedDataset:
    """Dataset that loads data in chunks to prevent OOM errors"""

    def __init__(self, jsonl_path: str, images_dir: str, chunk_size: int = 20):
        self.jsonl_path = jsonl_path
        self.images_dir = images_dir
        self.chunk_size = chunk_size
        self.total_samples = self._count_samples()
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size

        print(f"Dataset: {self.total_samples} samples, {self.num_chunks} chunks of size {chunk_size}")

    def _count_samples(self):
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

    def get_chunk(self, chunk_idx: int) -> List[Dict]:
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)

        chunk_data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if i >= end_idx:
                    break
                if line.strip():
                    item = json.loads(line.strip())
                    image_path = self.find_image_path(item["file_name"], self.images_dir)
                    if os.path.exists(image_path):
                        item["image_path"] = image_path
                        chunk_data.append(item)
                    else:
                        print(f"Warning: Image not found for {item['file_name']}, skipping...")

        return chunk_data

    def find_image_path(self, file_name: str, images_dir: str) -> str:
        exact_path = os.path.join(images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path

        base_name, _ = os.path.splitext(file_name)
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]

        return exact_path


def prepare_chunk_data(finetuner, chunk_data: List[Dict]) -> List[Dict]:
    converted_data = []
    for sample in chunk_data:
        conversation = finetuner.convert_to_conversation(sample)
        converted_data.append(conversation)
    return converted_data


class SchmuckOCRFinetune(SchmuckOCRFinetune):  # Extend the existing class

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)

        for item in data:
            item["image_path"] = self.find_image_path(item["file_name"], images_dir)

        valid_data = [item for item in data if os.path.exists(item["image_path"])]
        print(f"Found {len(valid_data)} valid samples out of {len(data)} total")

        converted_dataset = [self.convert_to_conversation(sample) for sample in valid_data]

        return converted_dataset

    def calculate_cer(self, predictions, targets):
        """Character Error Rate between strings using jiwer.cer only."""
        if not predictions or not targets or len(predictions) != len(targets):
            return 1.0

        total_cer = 0.0
        count = 0
        for pred, target in zip(predictions, targets):
            pred_str, target_str = str(pred), str(target)
            if len(target_str) == 0:
                continue
            cer_val = jiwer.cer(target_str, pred_str)
            total_cer += cer_val
            count += 1

        return total_cer / count if count > 0 else 1.0

    def calculate_cer_json(self, predictions, targets):
        preds_json = [self.safe_json_loads(p) for p in predictions]
        targets_json = [self.safe_json_loads(t) for t in targets]
        pred_strings = [self.canonical_json_string(j) if j is not None else "" for j in preds_json]
        target_strings = [self.canonical_json_string(j) if j is not None else "" for j in targets_json]
        return self.calculate_cer(pred_strings, target_strings)

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 15,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        stage1_epochs: int = 5,
    ):
        """
        Multi-stage training with teacher forcing:
        - Stage 1: warm-up (no eval, higher LR)
        - Stage 2: main training (eval + CER-based best model saving)
        """

        print("Preparing training and validation datasets...")
        train_dataset = self.prepare_training_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_training_data(val_jsonl_path, images_dir)
        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

        FastVisionModel.for_training(self.model)

        # ==========================
        # STAGE 1 – warm-up
        # ==========================
        print("\n===== STAGE 1: Warm-up training (no eval, teacher forcing) =====\n")

        stage1_args = SFTConfig(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            num_train_epochs=stage1_epochs,
            learning_rate=learning_rate * 3.0,
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            save_total_limit=1,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            weight_decay=0.05,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            load_best_model_at_end=False,
            optim="adamw_8bit",
            gradient_checkpointing=True,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/logs_stage1",
            seed=3407,
            output_dir=os.path.join(output_dir, "stage1"),
        )

        trainer_stage1 = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            args=stage1_args,
        )

        trainer_stage1.train()

        # ==========================
        # STAGE 2 – main training
        # ==========================
        print("\n===== STAGE 2: Main training (eval each epoch, CER-based best model) =====\n")

        remaining_epochs = max(1, num_epochs - stage1_epochs)

        cer_callback = ValidationCERCallback(
            finetuner=self,
            val_jsonl_path=val_jsonl_path,
            images_dir=images_dir,
            output_dir=output_dir,
            max_samples=30,
        )

        stage2_args = SFTConfig(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            num_train_epochs=remaining_epochs,
            learning_rate=learning_rate,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            weight_decay=0.05,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            load_best_model_at_end=False,          # we manage best model via CER callback
            metric_for_best_model="eval_cer",
            greater_is_better=False,
            optim="adamw_8bit",
            gradient_checkpointing=True,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048,
            report_to="tensorboard",
            logging_dir=f"{output_dir}/logs_stage2",
            seed=3407,
            output_dir=os.path.join(output_dir, "stage2"),
        )

        trainer_stage2 = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=stage2_args,
            callbacks=[
                cer_callback,
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001,
                ),
            ],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        trainer_stats = trainer_stage2.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        if cer_callback.best_model_dir is not None:
            print(f"\n[train_model] Best CER model saved at {cer_callback.best_model_dir} "
                  f"(CER={cer_callback.best_cer:.4f}).")
            self.best_model_dir = cer_callback.best_model_dir
            self.best_val_cer = cer_callback.best_cer
            self.model = trainer_stage2.model
        else:
            print("\n[train_model] No CER-based best model found; using final Stage 2 model.")
            self.best_model_dir = None
            self.best_val_cer = None
            self.model = trainer_stage2.model

        return trainer_stage2

    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict:
        """Evaluate the finetuned model on test.jsonl and calculate CER scores."""
        print("Starting evaluation on test.jsonl...")

        FastVisionModel.for_inference(self.model)

        test_chunked = ChunkedDataset(test_jsonl_path, images_dir, chunk_size=20)

        all_predictions = []
        all_cer_scores = []

        instruction = """Extract ALL the jewelry information from this German historical document image as a complete JSON object. 

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

Include ALL fields, even if empty (use empty string ""). Preserve exact German spelling and punctuation. Return ONLY the JSON object without any additional text or formatting."""

        from qwen_vl_utils import process_vision_info

        for chunk_idx in range(test_chunked.num_chunks):
            print(f"\nProcessing test chunk {chunk_idx + 1}/{test_chunked.num_chunks}")

            chunk_data = test_chunked.get_chunk(chunk_idx)

            for i, test_item in enumerate(chunk_data):
                print(f"Processing {i+1}/{len(chunk_data)} in chunk {chunk_idx + 1}", end='\r')

                try:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": test_item["image_path"]}
                        ]}
                    ]

                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
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
                            temperature=0.1,
                            do_sample=True,
                            repetition_penalty=1.1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]

                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )

                    generated_text = generated_texts[0] if generated_texts else ""

                    predicted_json = self.extract_json_from_response(generated_text)

                    gt_json_string = self.json_to_string_no_sort(test_item)
                    pred_json_string = self.json_to_string_no_sort(predicted_json) if predicted_json else ""

                    cer_score = jiwer.cer(gt_json_string, pred_json_string)

                    prediction_entry = {
                        "file_name": test_item['file_name'],
                        "predicted_json": predicted_json,
                        "predicted_text": pred_json_string,
                        "target_json": self.clean_dict_for_ground_truth(test_item),
                        "target_text": gt_json_string,
                        "raw_response": generated_text,
                        "cer_score": cer_score,
                        "matched_image_path": test_item["image_path"]
                    }
                    all_predictions.append(prediction_entry)
                    all_cer_scores.append(cer_score)

                except Exception as e:
                    print(f"Error processing {test_item['file_name']}: {str(e)}")
                    prediction_entry = {
                        "file_name": test_item['file_name'],
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": self.clean_dict_for_ground_truth(test_item),
                        "target_text": self.json_to_string_no_sort(test_item),
                        "raw_response": f"Error: {str(e)}",
                        "cer_score": 1.0,
                        "matched_image_path": test_item.get("image_path", "")
                    }
                    all_predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)
                    continue

            torch.cuda.empty_cache()
            gc.collect()

        predictions_file = os.path.join(output_dir, "test_predictions.jsonl")
        self.save_jsonl(all_predictions, predictions_file)

        cer_stats = self.calculate_cer_statistics(all_cer_scores)

        cer_file = os.path.join(output_dir, "cer_evaluation_results.txt")
        self.save_cer_results(cer_stats, cer_file, len(all_predictions))

        return {
            "predictions": all_predictions,
            "cer_stats": cer_stats,
            "predictions_file": predictions_file,
            "cer_file": cer_file
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
        with open(cer_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("CER EVALUATION RESULTS - SCHMUCK DATASET\n")
            f.write("="*60 + "\n\n")

            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n")
            f.write(f"Median CER: {cer_stats['median_cer']:.4f} ({cer_stats['median_cer']*100:.2f}%)\n")
            f.write(f"Minimum CER: {cer_stats['minimum_cer']:.4f} ({cer_stats['minimum_cer']*100:.2f}%)\n")
            f.write(f"Maximum CER: {cer_stats['maximum_cer']:.4f} ({cer_stats['maximum_cer']*100:.2f}%)\n")
            f.write(f"Standard Deviation: {cer_stats['std_cer']:.4f}\n\n")

            f.write(f"Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']} ({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n")
            f.write(f"Total images processed: {num_predictions}\n")

            f.write("\n" + "="*60 + "\n")
            f.write("FIXES APPLIED IN THIS VERSION\n")
            f.write("="*60 + "\n")
            f.write("✅ Fixed: 'list' object has no attribute 'strip' error\n")
            f.write("✅ Fixed: Proper handling of batch_decode output\n")
            f.write("✅ Enhanced: Detailed instructions for all required fields\n")
            f.write("✅ Enhanced: Better error handling and recovery\n")
            f.write("✅ Enhanced: Proper file_name to image_path mapping\n")

        print(f"CER evaluation results saved to: {cer_file}")

    def save_model(self, trainer, output_dir: str):
        """Save the finetuned model."""
        print(f"Saving model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")


class ValidationCERCallback(TrainerCallback):
    """Compute CER on val.jsonl after each eval and save best-CER model."""

    def __init__(
        self,
        finetuner: SchmuckOCRFinetune,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        max_samples: int = 30,
    ):
        self.finetuner = finetuner
        self.val_data = finetuner.load_jsonl(val_jsonl_path)
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.best_cer = float("inf")
        self.best_model_dir = None
        self.cer_history: List[Dict[str, Any]] = []

    def _compute_val_cer(self, model) -> float:
        from qwen_vl_utils import process_vision_info

        FastVisionModel.for_inference(model)

        predictions = []
        targets = []

        instruction = """Extract ALL the jewelry information from this German historical document image as a complete JSON object. 

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

Include ALL fields, even if empty (use empty string ""). Preserve exact German spelling and punctuation. Return ONLY the JSON object without any additional text or formatting."""

        device = "cuda" if torch.cuda.is_available() else "cpu"

        subset = self.val_data[: min(self.max_samples, len(self.val_data))]
        print(f"\n[ValidationCERCallback] Computing CER on {len(subset)} validation samples...")

        for idx, item in enumerate(subset):
            image_path = self.finetuner.find_image_path(item["file_name"], self.images_dir)
            if not os.path.exists(image_path):
                continue

            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": image_path},
                        ],
                    }
                ]

                input_text = self.finetuner.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.finetuner.tokenizer(
                    text=[input_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        use_cache=True,
                        temperature=0.1,
                        do_sample=True,
                        repetition_penalty=1.1,
                        pad_token_id=self.finetuner.tokenizer.pad_token_id,
                        eos_token_id=self.finetuner.tokenizer.eos_token_id,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]
                generated_text = self.finetuner.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                predicted_json = self.finetuner.extract_json_from_response(generated_text)
                pred_str = (
                    self.finetuner.json_to_string_no_sort(predicted_json)
                    if predicted_json
                    else ""
                )
                gt_str = self.finetuner.json_to_string_no_sort(item)

                predictions.append(pred_str)
                targets.append(gt_str)

            except Exception as e:
                print(f"  [Val CER] Error on {item.get('file_name', 'unknown')}: {e}")
                predictions.append("")
                targets.append(self.finetuner.json_to_string_no_sort(item))

        FastVisionModel.for_training(model)

        if not predictions:
            return 1.0

        cer = self.finetuner.calculate_cer_json(predictions, targets)
        return cer

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if model is None:
            return control

        epoch = state.epoch if state.epoch is not None else 0
        val_cer = self._compute_val_cer(model)

        if metrics is not None:
            metrics["eval_cer"] = val_cer

        self.cer_history.append({"epoch": epoch, "cer": val_cer})
        print(f"\n[ValidationCERCallback] Epoch {epoch:.2f} CER: {val_cer:.4f} ({val_cer*100:.2f}%)")

        if val_cer < self.best_cer:
            improvement = self.best_cer - val_cer
            self.best_cer = val_cer
            print(f"[ValidationCERCallback] New best CER: {self.best_cer:.4f} "
                  f"(improved by {improvement:.4f})")

            best_dir = os.path.join(self.output_dir, "best_model_cer")
            os.makedirs(best_dir, exist_ok=True)
            print(f"[ValidationCERCallback] Saving best CER model to {best_dir}")
            model.save_pretrained(best_dir)
            self.finetuner.tokenizer.save_pretrained(best_dir)
            self.best_model_dir = best_dir
            self.finetuner.best_model_dir = best_dir
            self.finetuner.best_val_cer = val_cer
        else:
            print(f"[ValidationCERCallback] No improvement. Best CER remains {self.best_cer:.4f}")

        return control


def main():
    """Main function to run the Schmuck dataset finetuning and evaluation process."""

    base_checkpoint_dir = "/home/vault/iwi5/iwi5298h/models_image_text/qwen/multi/schmuck"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    config = {
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",
        "output_dir": run_dir,
        "num_epochs": 15,
        "batch_size": 2,
        "learning_rate": 5e-5,
    }

    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    finetuner = SchmuckOCRFinetune(model_path="/home/vault/iwi5/iwi5298h/models/qwen7b")

    print("="*60)
    print("STARTING MULTI-STAGE TRAINING FOR SCHMUCK DATASET (TEACHER FORCING)")
    print("="*60)

    trainer = finetuner.train_model(
        train_jsonl_path=config["train_jsonl_path"],
        val_jsonl_path=config["val_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        stage1_epochs=5,
    )

    finetuner.save_model(trainer, config["output_dir"])

    print("\n" + "="*60)
    print("STARTING EVALUATION ON SCHMUCK TEST SET")
    print("="*60)

    test_results = finetuner.evaluate_on_test_set(
        test_jsonl_path=config["test_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"]
    )

    print(f"\nEvaluation completed!")
    print(f"Predictions saved to: {test_results['predictions_file']}")
    print(f"CER results saved to: {test_results['cer_file']}")
    print(f"All files saved in: {run_dir}")

    cer_stats = test_results['cer_stats']
    print(f"\n" + "="*60)
    print("FINAL RESULTS SUMMARY - SCHMUCK DATASET")
    print("="*60)
    print(f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)")
    print(f"Median CER: {cer_stats['median_cer']:.4f} ({cer_stats['median_cer']*100:.2f}%)")
    print(f"Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']} "
          f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)")
    print(f"Total images processed: {cer_stats['total_images']}")
    print("\nSchmuck dataset multi-stage training and evaluation completed successfully!")
    print(f"All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
