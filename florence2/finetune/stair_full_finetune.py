#!/usr/bin/env python3
# florence2_staircase_full_finetune_autoreg_valcer_tb_bestload_UPDATED.py
# ✅ FULL-PARAMETER finetuning of Florence-2-large on Staircase
# ✅ Train: teacher forcing (labels)
# ✅ Val/Test: autoregressive generate() CER + predictions
# ✅ Best model across epochs via lowest val CER (generate-based)
# ✅ Reload best model before test (guaranteed)
# ✅ Dedicated output folders + TensorBoard logging
# ✅ use_cache=False everywhere (model.config + generate kwargs)
# ✅ Do NOT flip train/eval inside low-level generate helper (mode controlled at higher level)

import sys
import os

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import json
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import torch
from PIL import Image
import jiwer

from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

# ---------------------------------------------------------------------
# CONFIG – UPDATE THESE PATHS FOR YOUR STAIRCASE DATA
# ---------------------------------------------------------------------
STAIR_IMAGES_DIR = "/home/woody/iwi5/iwi5298h/staircase_images"
STAIR_JSON_DIR = "/home/woody/iwi5/iwi5298h/json_staircase"

MODEL_PATH = "/home/vault/iwi5/iwi5298h/models/florence2_large"
OUTPUT_BASE_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/florence2/staircase"

TASK_PROMPT = "<OCR>"
DATASET_NAME = "Staircase Dataset"

# -----------------------
# IO helpers
# -----------------------
def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def append_jsonl(item: Dict, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

def create_label_string(json_data: Dict) -> str:
    """Create label string excluding image_name."""
    label_data = {k: v for k, v in json_data.items() if k != "image_name"}
    return json.dumps(label_data, ensure_ascii=False)

def json_to_string(json_obj: Dict) -> str:
    """Convert JSON object to compact string (without image_name)."""
    clean_obj = {k: v for k, v in json_obj.items() if k != "image_name"}
    return json.dumps(clean_obj, ensure_ascii=False, separators=(",", ":"))

# -----------------------
# Data Collator & Dataset
# -----------------------
class Florence2DataCollator:
    def __init__(self, processor, pad_token_id=None):
        self.processor = processor
        self.pad_token_id = pad_token_id if pad_token_id is not None else processor.tokenizer.pad_token_id

    def __call__(self, features):
        pixel_values = [f["pixel_values"] for f in features]
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        pixel_values = torch.stack(pixel_values)

        max_input_length = max(len(ids) for ids in input_ids)
        max_label_length = max(len(lab) for lab in labels)

        padded_input_ids, padded_attention_mask = [], []
        for i in range(len(input_ids)):
            cur_len = len(input_ids[i])
            pad_len = max_input_length - cur_len
            if pad_len > 0:
                padded_input_ids.append(
                    torch.cat([input_ids[i], torch.full((pad_len,), self.pad_token_id, dtype=input_ids[i].dtype)])
                )
                padded_attention_mask.append(
                    torch.cat([attention_mask[i], torch.zeros(pad_len, dtype=attention_mask[i].dtype)])
                )
            else:
                padded_input_ids.append(input_ids[i])
                padded_attention_mask.append(attention_mask[i])

        padded_labels = []
        for i in range(len(labels)):
            cur_len = len(labels[i])
            pad_len = max_label_length - cur_len
            if pad_len > 0:
                padded_labels.append(torch.cat([labels[i], torch.full((pad_len,), -100, dtype=labels[i].dtype)]))
            else:
                padded_labels.append(labels[i])

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_mask),
            "labels": torch.stack(padded_labels),
        }

class StaircaseOCRDataset(Dataset):
    def __init__(self, jsonl_data: List[Dict], images_dir: str, processor):
        self.data = jsonl_data
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.images_dir, item["image_name"])
        image = Image.open(image_path).convert("RGB")

        target_text = create_label_string(item)

        inputs = self.processor(
            text=TASK_PROMPT,
            images=image,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )

        target_inputs = self.processor.tokenizer(
            target_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0).to(torch.float32),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": target_inputs["input_ids"].squeeze(0),
        }

# -----------------------
# Trainer wrapper
# -----------------------
class Florence2StaircaseTrainer:
    def __init__(self, model_path: str, output_base_dir: str):
        self.model_path = model_path
        self.output_base_dir = output_base_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_base_dir, f"full_ft_run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Dedicated folders
        self.models_dir = os.path.join(self.output_dir, "models")
        self.best_model_dir = os.path.join(self.models_dir, "best_model")
        self.final_model_dir = os.path.join(self.models_dir, "final_model")
        self.pred_dir = os.path.join(self.output_dir, "predictions")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        self.logs_dir = os.path.join(self.output_dir, "logs")

        for d in [self.models_dir, self.pred_dir, self.metrics_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

        self.val_history_path = os.path.join(self.metrics_dir, "val_cer_history.jsonl")
        self.best_metric_path = os.path.join(self.metrics_dir, "best_val_metric.json")
        self.test_metrics_path = os.path.join(self.metrics_dir, "test_cer_metrics_full_ft.json")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32

        print("=" * 60)
        print("Florence-2 Staircase FULL-PARAMETER Fine-tuning")
        print("=" * 60)
        print(f"Dataset: {DATASET_NAME}")
        print(f"Device: {self.device}")
        print(f"Output dir: {self.output_dir}")
        print(f"TensorBoard logs: {self.logs_dir}")
        print("=" * 60 + "\n")

        self._load_model()

        self.best_eval_cer = float("inf")
        self.best_model_path: Optional[str] = None

        # generation settings identical for val & test
        self.gen_kwargs = dict(
            max_new_tokens=512,
            num_beams=3,
            do_sample=False,
            use_cache=False,  # generation side
            pad_token_id=self.processor.tokenizer.pad_token_id,
            early_stopping=False,
        )

    def _resolve_model_name(self) -> str:
        if os.path.exists(self.model_path):
            return self.model_path
        print("Model not found locally, using HF hub: microsoft/Florence-2-large")
        return "microsoft/Florence-2-large"

    def _load_model(self):
        model_name = self._resolve_model_name()
        print("⏳ Loading Florence-2-large base model (full fine-tuning)...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)

        # Full fine-tuning: ensure all params trainable
        for p in self.model.parameters():
            p.requires_grad_(True)

        # ✅ Make cache behavior consistent with gradient checkpointing
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded (FULL fine-tuning). Trainable params: {trainable_params:,} / {total_params:,} "
              f"({trainable_params/total_params*100:.2f}%)\n")

        self.model.train()

    def load_best_model_for_inference(self):
        if self.best_model_path and os.path.exists(self.best_model_path):
            load_path = self.best_model_path
        elif os.path.exists(self.best_model_dir):
            load_path = self.best_model_dir
        else:
            print("⚠️ Best model folder not found. Using current in-memory model.")
            return

        print(f"✅ Loading BEST model from: {load_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)

        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

        self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        self.gen_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
        self.model.eval()

    @torch.no_grad()
    def _generate_text_batch(self, input_ids, pixel_values) -> List[str]:
        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **self.gen_kwargs,
        )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _cer_from_strings(self, gt_json_string: str, pred_text: str) -> float:
        try:
            pred_json = json.loads(pred_text)
            pred_json_string = json_to_string(pred_json)
            return jiwer.cer(gt_json_string, pred_json_string)
        except Exception:
            return jiwer.cer(gt_json_string, pred_text)

    def evaluate_autoregressive_cer(self, eval_dataset: Dataset, batch_size: int) -> float:
        was_training = self.model.training
        self.model.eval()
        try:
            cer_scores = []
            n = len(eval_dataset)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_items = [eval_dataset[i] for i in range(start, end)]

                input_ids = torch.stack([x["input_ids"] for x in batch_items]).to(self.device)
                pixel_values = torch.stack([x["pixel_values"] for x in batch_items]).to(self.device, self.torch_dtype)

                pred_texts = self._generate_text_batch(input_ids=input_ids, pixel_values=pixel_values)

                for item, pred_text in zip(batch_items, pred_texts):
                    label_ids = item["labels"]
                    label_ids = label_ids[label_ids != -100]
                    gt_text = self.processor.tokenizer.decode(label_ids, skip_special_tokens=True).strip()

                    try:
                        gt_json = json.loads(gt_text)
                        gt_json_string = json_to_string(gt_json)
                    except Exception:
                        gt_json_string = gt_text

                    cer_scores.append(self._cer_from_strings(gt_json_string, pred_text))

            return float(np.mean(cer_scores)) if cer_scores else 1.0
        finally:
            if was_training:
                self.model.train()

    def train(self, train_data: List[Dict], val_data: List[Dict], images_dir: str):
        print("📂 Preparing datasets...")
        train_dataset = StaircaseOCRDataset(train_data, images_dir, self.processor)
        val_dataset = StaircaseOCRDataset(val_data, images_dir, self.processor)

        print(f"✅ Train dataset: {len(train_dataset)}")
        print(f"✅ Val dataset:   {len(val_dataset)}\n")

        data_collator = Florence2DataCollator(self.processor)

        # ✅ Params: stable full-FT defaults
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,

            learning_rate=5e-5,
            weight_decay=0.01,

            # Prefer ratio so it scales with dataset size
            warmup_ratio=0.05,

            logging_dir=self.logs_dir,
            logging_steps=10,

            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,

            report_to="tensorboard",
            dataloader_pin_memory=False,

            # If you want faster: set fp16=True on V100, bf16=True on A100
            fp16=False,
            bf16=False,

            gradient_checkpointing=True,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            optim="adamw_torch",
        )

        class CustomTrainer(Trainer):
            def __init__(self, parent_instance, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent = parent_instance

            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                _ = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

                eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
                bs = self.args.per_device_eval_batch_size

                try:
                    current_cer = self.parent.evaluate_autoregressive_cer(eval_ds, batch_size=bs)
                    current_epoch = float(self.state.epoch) if hasattr(self.state, "epoch") else float(self.state.global_step)

                    self.log({"eval_cer": current_cer})

                    print(
                        f"\n📊 Epoch {int(current_epoch)} - Validation CER (generate): "
                        f"{current_cer:.4f} ({current_cer*100:.2f}%)"
                    )

                    append_jsonl({"epoch": current_epoch, "eval_cer": current_cer}, self.parent.val_history_path)

                    if current_cer < self.parent.best_eval_cer:
                        improvement = self.parent.best_eval_cer - current_cer
                        self.parent.best_eval_cer = current_cer
                        self.parent.best_model_path = self.parent.best_model_dir

                        if os.path.exists(self.parent.best_model_dir):
                            import shutil
                            shutil.rmtree(self.parent.best_model_dir)

                        self.model.save_pretrained(self.parent.best_model_dir)
                        self.parent.processor.save_pretrained(self.parent.best_model_dir)

                        with open(self.parent.best_metric_path, "w") as f:
                            json.dump({"best_eval_cer": current_cer, "epoch": current_epoch}, f, indent=2)

                        print(f"🎯 New best CER: {current_cer:.4f} (improved by {improvement:.4f})")
                        print(f"✅ Best model saved to: {self.parent.best_model_dir}\n")
                    else:
                        print(f"   Best CER so far: {self.parent.best_eval_cer:.4f}\n")

                    return {"eval_cer": current_cer}
                except Exception as e:
                    print(f"❌ Error during generation-based evaluation: {e}")
                    return {"eval_cer": 1.0}

        trainer = CustomTrainer(
            self,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("=" * 60)
        print("STARTING FULL-PARAMETER TRAINING (AUTOREGRESSIVE VAL CER)")
        print("=" * 60 + "\n")

        trainer.train()

        self.model.save_pretrained(self.final_model_dir)
        self.processor.save_pretrained(self.final_model_dir)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best validation CER (generate): {self.best_eval_cer:.4f}")
        print(f"Best model dir:  {self.best_model_dir}")
        print(f"Final model dir: {self.final_model_dir}")
        print(f"TensorBoard logs: {self.logs_dir}")
        print("=" * 60 + "\n")

        return trainer

    def predict(self, test_data: List[Dict], images_dir: str) -> List[Dict]:
        print(f"Making predictions on {len(test_data)} test images...")
        self.model.eval()

        predictions = []
        json_parse_failures = 0

        for i, test_item in enumerate(test_data):
            print(f"[{i+1}/{len(test_data)}] {test_item['image_name']}")
            try:
                image_path = os.path.join(images_dir, test_item["image_name"])
                image = Image.open(image_path).convert("RGB")

                inputs = self.processor(text=TASK_PROMPT, images=image, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                pixel_values = inputs["pixel_values"].to(self.device, self.torch_dtype)

                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    **self.gen_kwargs,
                )

                prediction_string = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                predicted_json = None
                json_parse_success = False
                try:
                    predicted_json = json.loads(prediction_string)
                    json_parse_success = True
                except Exception:
                    json_parse_failures += 1

                gt_json_string = json_to_string(test_item)
                cer_score = self._cer_from_strings(gt_json_string, prediction_string)

                predictions.append({
                    "image_name": test_item["image_name"],
                    "prediction_string": prediction_string,
                    "prediction_json": predicted_json,
                    "json_parse_success": json_parse_success,
                    "ground_truth_json_string": gt_json_string,
                    "cer_score": cer_score,
                })
                print(f"  ✅ CER: {cer_score:.4f}")

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                gt_json_string = json_to_string(test_item)
                predictions.append({
                    "image_name": test_item["image_name"],
                    "prediction_string": f"Error: {str(e)}",
                    "prediction_json": None,
                    "json_parse_success": False,
                    "ground_truth_json_string": gt_json_string,
                    "cer_score": 1.0,
                })

        pred_path = os.path.join(self.pred_dir, "test_predictions_full_ft.jsonl")
        save_jsonl(predictions, pred_path)

        cer_scores = [p["cer_score"] for p in predictions]
        avg_cer = float(np.mean(cer_scores)) if cer_scores else 1.0
        perfect_matches = sum(1 for c in cer_scores if c == 0.0)
        json_success_rate = ((len(predictions) - json_parse_failures) / len(predictions) * 100) if predictions else 0.0

        metrics = {
            "average_cer": avg_cer,
            "min_cer": float(min(cer_scores)) if cer_scores else 1.0,
            "max_cer": float(max(cer_scores)) if cer_scores else 1.0,
            "perfect_matches": int(perfect_matches),
            "json_success_rate": float(json_success_rate),
            "total_samples": int(len(predictions)),
            "best_validation_cer": float(self.best_eval_cer),
            "predictions_path": pred_path,
        }

        with open(self.test_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n{'='*60}")
        print("TEST RESULTS (BEST MODEL LOADED - FULL FINE-TUNE)")
        print(f"{'='*60}")
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Perfect matches: {perfect_matches}/{len(predictions)}")
        print(f"JSON success rate: {json_success_rate:.1f}%")
        print(f"Predictions saved: {pred_path}")
        print(f"Metrics saved: {self.test_metrics_path}")
        print(f"{'='*60}\n")

        return predictions

def main():
    print(f"Dataset: {DATASET_NAME}")
    print("Loading datasets...")

    train_data = load_jsonl(os.path.join(STAIR_JSON_DIR, "train.jsonl"))
    val_data = load_jsonl(os.path.join(STAIR_JSON_DIR, "val.jsonl"))
    test_data = load_jsonl(os.path.join(STAIR_JSON_DIR, "test.jsonl"))

    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    print(f"Loaded {len(test_data)} test samples\n")

    trainer_obj = Florence2StaircaseTrainer(MODEL_PATH, OUTPUT_BASE_DIR)
    trainer_obj.train(train_data, val_data, STAIR_IMAGES_DIR)

    trainer_obj.load_best_model_for_inference()
    trainer_obj.predict(test_data, STAIR_IMAGES_DIR)

    print("Pipeline completed successfully!")
    print(f"Run directory: {trainer_obj.output_dir}")
    print(f"TensorBoard logs: {trainer_obj.logs_dir}")
    print("To launch TensorBoard:")
    print(f"  tensorboard --logdir {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()