#!/usr/bin/env python3
# florence2_schmuck_fullft_autoreg_valcer_chunked_tb.py
# ✅ Full-parameter finetuning (NO LoRA)
# ✅ Validation CER computed AUTOREGRESSIVELY via generate() (same as test)
# ✅ Best-model saving based on lowest val CER (generate-based)
# ✅ Reload best model before test (guaranteed)
# ✅ Chunked test evaluation to avoid OOM
# ✅ Dedicated output folders + TensorBoard logging
# ✅ use_cache=False everywhere
# ✅ Do NOT flip train/eval inside low-level generate helper (mode controlled at higher-level)

import sys
import os

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

import json
from typing import List, Dict, Optional
from datetime import datetime
import gc

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


# --------------------------
# Chunked Dataset (JSONL file backed)
# --------------------------
class ChunkedDataset:
    """Loads data in chunks to prevent OOM, also filters missing images."""

    def __init__(self, jsonl_path: str, images_dir: str, chunk_size: int = 20):
        self.jsonl_path = jsonl_path
        self.images_dir = images_dir
        self.chunk_size = chunk_size
        self.total_samples = self._count_samples()
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        print(f"📊 Dataset: {self.total_samples} samples, {self.num_chunks} chunks of size {chunk_size}")

    def _count_samples(self) -> int:
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def get_chunk(self, chunk_idx: int) -> List[Dict]:
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)

        chunk_data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if i >= end_idx:
                    break
                if line.strip():
                    item = json.loads(line.strip())
                    image_path = os.path.join(self.images_dir, item["file_name"])
                    if os.path.exists(image_path):
                        chunk_data.append(item)

        return chunk_data


# --------------------------
# IO helpers
# --------------------------
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
    """Create label string excluding file_name."""
    label_data = {k: v for k, v in json_data.items() if k != "file_name"}
    return json.dumps(label_data, ensure_ascii=False)


def json_to_string(json_obj: Dict) -> str:
    """Convert JSON object to compact string (without file_name)."""
    clean_obj = {k: v for k, v in json_obj.items() if k != "file_name"}
    return json.dumps(clean_obj, ensure_ascii=False, separators=(",", ":"))


# --------------------------
# Collator & Dataset
# --------------------------
class Florence2DataCollator:
    """Pads variable-length input_ids/labels; stacks pixel_values."""

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
                    torch.cat(
                        [input_ids[i], torch.full((pad_len,), self.pad_token_id, dtype=input_ids[i].dtype)]
                    )
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


class SchmuckOCRDataset(Dataset):
    """In-memory dataset for train/val (JSONL already loaded)."""

    def __init__(self, jsonl_data: List[Dict], images_dir: str, processor):
        self.data = jsonl_data
        self.images_dir = images_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_path = os.path.join(self.images_dir, item["file_name"])
        image = Image.open(image_path).convert("RGB")

        task_prompt = "<OCR>"
        target_text = create_label_string(item)

        inputs = self.processor(
            text=task_prompt,
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
            max_length=1024,
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0).to(torch.float32),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": target_inputs["input_ids"].squeeze(0),
        }


# --------------------------
# Trainer wrapper
# --------------------------
class Florence2SchmuckTrainer:
    """
    Full-parameter finetuning (NO LoRA) with:
    - generation-based validation CER for best model selection
    - best model reload for test
    - chunked test evaluation
    - TensorBoard logging
    """

    def __init__(self, model_path: str, output_base_dir: str):
        self.model_path = model_path
        self.output_base_dir = output_base_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_base_dir, f"fullft_run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Dedicated folders
        self.models_dir = os.path.join(self.output_dir, "models")
        self.best_model_dir = os.path.join(self.models_dir, "best_model")
        self.final_model_dir = os.path.join(self.models_dir, "final_model")
        self.pred_dir = os.path.join(self.output_dir, "predictions")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        self.logs_dir = os.path.join(self.output_dir, "logs")  # TensorBoard events

        for d in [self.models_dir, self.pred_dir, self.metrics_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)

        self.val_history_path = os.path.join(self.metrics_dir, "val_cer_history.jsonl")
        self.best_metric_path = os.path.join(self.metrics_dir, "best_val_metric.json")
        self.test_metrics_path = os.path.join(self.metrics_dir, "test_metrics.json")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32

        print("=" * 60)
        print("Florence-2 Schmuck Full-FT Fine-tuning (NO LoRA)")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"Task prompt: <OCR> only")
        print(f"TensorBoard logs: {self.logs_dir}")
        print("=" * 60 + "\n")

        self._load_model()

        self.best_eval_cer = float("inf")
        self.best_model_path: Optional[str] = None

        # Generation settings used for BOTH val and test (same behavior)
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
            use_cache=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            early_stopping=False,
        )

    def _resolve_model_name(self) -> str:
        if os.path.exists(self.model_path):
            return self.model_path
        print("Model not found locally, using HF hub: microsoft/Florence-2-large")
        return "microsoft/Florence-2-large"

    def _load_model(self):
        print("⏳ Loading Florence-2-large model (FULL FT)...")
        model_name = self._resolve_model_name()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)

        # Full FT => all params trainable
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.model.train()

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded successfully! Total params: {total_params:,}\n")

    def load_best_model_for_inference(self):
        """
        ✅ Guarantees test prediction uses best checkpoint (lowest val CER).
        """
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

        self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        # refresh pad id after reload
        self.gen_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id

        self.model.eval()

    @torch.no_grad()
    def _generate_text_batch(self, input_ids, pixel_values) -> List[str]:
        """
        ✅ No train/eval mode switching here.
        Caller controls the mode.
        """
        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            **self.gen_kwargs,
        )
        texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def _cer_from_strings(self, gt_json_string: str, pred_text: str) -> float:
        """
        CER between:
        - GT compact JSON string
        - Pred compact JSON string (if JSON parses), else raw prediction string
        """
        try:
            pred_json = json.loads(pred_text)
            pred_json_string = json_to_string(pred_json)
            return jiwer.cer(gt_json_string, pred_json_string)
        except Exception:
            return jiwer.cer(gt_json_string, pred_text)

    def evaluate_autoregressive_cer(self, eval_dataset: Dataset, batch_size: int) -> float:
        """
        Autoregressive validation CER (same style as test).
        """
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
                    # GT from labels -> decode -> normalize to compact JSON if possible
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

        train_dataset = SchmuckOCRDataset(train_data, images_dir, self.processor)
        val_dataset = SchmuckOCRDataset(val_data, images_dir, self.processor)

        print(f"✅ Train: {len(train_dataset)} samples")
        print(f"✅ Validation: {len(val_dataset)} samples\n")

        data_collator = Florence2DataCollator(self.processor)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,

            # Full FT → safer LR than LoRA
            learning_rate=5e-5,
            weight_decay=0.1,
            warmup_steps=50,

            logging_dir=self.logs_dir,
            logging_steps=10,

            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,

            metric_for_best_model="eval_cer",
            greater_is_better=False,

            report_to="tensorboard",
            dataloader_pin_memory=False,
            fp16=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            optim="adamw_torch",
        )

        class CustomTrainer(Trainer):
            def __init__(self, parent_instance, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent = parent_instance

            def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
                # Default HF eval for eval_loss logging
                _ = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

                eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
                bs = self.args.per_device_eval_batch_size

                try:
                    current_cer = self.parent.evaluate_autoregressive_cer(eval_ds, batch_size=bs)
                    current_epoch = float(self.state.epoch) if hasattr(self.state, "epoch") else float(self.state.global_step)

                    # log CER to TensorBoard
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
        print("STARTING TRAINING (FULL FT, NO LoRA)")
        print("=" * 60)
        print("Model selection: lowest validation CER (AUTOREGRESSIVE generate())")
        print("=" * 60 + "\n")

        trainer.train()

        # Always save final model
        self.model.save_pretrained(self.final_model_dir)
        self.processor.save_pretrained(self.final_model_dir)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Best validation CER: {self.best_eval_cer:.4f} ({self.best_eval_cer*100:.2f}%)")
        print(f"Best model dir: {self.best_model_dir}")
        print(f"Final model dir: {self.final_model_dir}")
        print(f"TensorBoard logs: {self.logs_dir}")
        print("=" * 60 + "\n")

        return trainer

    def predict_chunked(self, test_jsonl_path: str, images_dir: str, chunk_size: int = 20) -> List[Dict]:
        """
        Chunked autoregressive test evaluation (generate()).
        Assumes best model already loaded (call load_best_model_for_inference()).
        """
        print("\n" + "=" * 60)
        print("STARTING CHUNKED TEST EVALUATION (generate())")
        print("=" * 60)

        test_chunked = ChunkedDataset(test_jsonl_path, images_dir, chunk_size=chunk_size)

        self.model.eval()

        all_predictions = []
        all_cer_scores = []
        json_parse_failures = 0

        for chunk_idx in range(test_chunked.num_chunks):
            print(f"\n{'─'*60}")
            print(f"Processing chunk {chunk_idx + 1}/{test_chunked.num_chunks}")
            print(f"{'─'*60}")

            chunk_data = test_chunked.get_chunk(chunk_idx)

            for i, test_item in enumerate(chunk_data, 1):
                print(f"[Chunk {chunk_idx+1}] {i}/{len(chunk_data)}: {test_item['file_name']}")

                try:
                    image_path = os.path.join(images_dir, test_item["file_name"])
                    image = Image.open(image_path).convert("RGB")

                    inputs = self.processor(text="<OCR>", images=image, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(self.device)
                    pixel_values = inputs["pixel_values"].to(self.device, self.torch_dtype)

                    if pixel_values is None or pixel_values.numel() == 0:
                        raise ValueError("Empty pixel values")

                    with torch.no_grad():
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

                    entry = {
                        "file_name": test_item["file_name"],
                        "prediction_string": prediction_string,
                        "prediction_json": predicted_json,
                        "json_parse_success": json_parse_success,
                        "ground_truth_json_string": gt_json_string,
                        "cer_score": cer_score,
                    }

                    all_predictions.append(entry)
                    all_cer_scores.append(cer_score)

                    print(f"  ✅ CER: {cer_score:.4f} ({cer_score*100:.2f}%)")

                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()

                    gt_json_string = json_to_string(test_item)
                    entry = {
                        "file_name": test_item["file_name"],
                        "prediction_string": f"Error: {str(e)}",
                        "prediction_json": None,
                        "json_parse_success": False,
                        "ground_truth_json_string": gt_json_string,
                        "cer_score": 1.0,
                    }
                    all_predictions.append(entry)
                    all_cer_scores.append(1.0)

            # Cleanup between chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"✓ Chunk {chunk_idx + 1} completed\n")

        # Save predictions
        predictions_path = os.path.join(self.pred_dir, "test_predictions_chunked.jsonl")
        save_jsonl(all_predictions, predictions_path)

        avg_cer = float(np.mean(all_cer_scores)) if all_cer_scores else 1.0
        perfect_matches = sum(1 for c in all_cer_scores if c == 0.0)
        json_success_rate = ((len(all_predictions) - json_parse_failures) / len(all_predictions) * 100) if all_predictions else 0.0

        print("\n" + "=" * 60)
        print("TEST EVALUATION RESULTS")
        print("=" * 60)
        print(f"Total images: {len(all_predictions)}")
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Min CER: {float(min(all_cer_scores)):.4f}")
        print(f"Max CER: {float(max(all_cer_scores)):.4f}")
        print(f"Perfect matches: {perfect_matches}/{len(all_predictions)} ({perfect_matches/len(all_predictions)*100:.1f}%)")
        print(f"JSON success rate: {json_success_rate:.1f}%")
        print(f"\n✅ Predictions saved to: {predictions_path}")
        print("=" * 60 + "\n")

        summary = {
            "average_cer": avg_cer,
            "min_cer": float(min(all_cer_scores)) if all_cer_scores else 1.0,
            "max_cer": float(max(all_cer_scores)) if all_cer_scores else 1.0,
            "perfect_matches": int(perfect_matches),
            "json_success_rate": float(json_success_rate),
            "total_samples": int(len(all_predictions)),
            "best_validation_cer": float(self.best_eval_cer),
            "best_model_dir": self.best_model_dir,
            "final_model_dir": self.final_model_dir,
            "predictions_path": predictions_path,
        }

        with open(self.test_metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        return all_predictions


def main():
    images_dir = "/home/woody/iwi5/iwi5298h/schmuck_images"
    json_dir = "/home/woody/iwi5/iwi5298h/json_schmuck"
    model_path = "/home/vault/iwi5/iwi5298h/models/florence2_large"
    output_base_dir = "/home/vault/iwi5/iwi5298h/models_image_text/florence2/schmuck"

    print("=" * 60)
    print("FLORENCE-2 SCHMUCK FULL-FT (NO LoRA)")
    print("Autoregressive val CER + Chunked test + TensorBoard")
    print("=" * 60 + "\n")

    print("📂 Loading datasets...")
    train_data = load_jsonl(os.path.join(json_dir, "train.jsonl"))
    val_data = load_jsonl(os.path.join(json_dir, "val.jsonl"))
    test_jsonl_path = os.path.join(json_dir, "test.jsonl")

    print(f"✅ Train: {len(train_data)} samples")
    print(f"✅ Validation: {len(val_data)} samples\n")

    trainer_obj = Florence2SchmuckTrainer(model_path, output_base_dir)

    trainer_obj.train(train_data, val_data, images_dir)

    # ✅ Ensure best checkpoint is used for test
    trainer_obj.load_best_model_for_inference()

    trainer_obj.predict_chunked(test_jsonl_path, images_dir, chunk_size=20)

    print("=" * 60)
    print("FLORENCE-2 SCHMUCK PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"All outputs saved to: {trainer_obj.output_dir}")
    print("To launch TensorBoard:")
    print(f"  tensorboard --logdir {output_base_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
