#!/usr/bin/env python3
# florence2_staircase_finetune_gen_eval.py
# LoRA finetuning of Florence-2-large on the Staircase dataset
# ✅ Uses <OCR> as task token
# ✅ Uses AUTOREGRESSIVE generation for validation CER (same as test)
# ✅ CER-based best model selection on generation-based CER
# ✅ FIXED: training passes decoder_input_ids to Florence-2, avoiding
#          "input_ids cannot be None" error.

import sys
import os

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import json
from typing import List, Dict
import jiwer
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------
# CONFIG – UPDATE THESE PATHS FOR YOUR STAIRCASE DATA
# ---------------------------------------------------------------------
STAIR_IMAGES_DIR = "/home/woody/iwi5/iwi5298h/staircase_images"
STAIR_JSON_DIR = "/home/woody/iwi5/iwi5298h/json_staircase"

MODEL_PATH = "/home/vault/iwi5/iwi5298h/models/florence2_large"
OUTPUT_BASE_DIR = "/home/vault/iwi5/iwi5298h/models_image_text/florence2/staircase"

TASK_PROMPT = "<OCR>"
DATASET_NAME = "Staircase Dataset"


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_label_string(json_data: Dict) -> str:
    """
    Create label string from JSON data excluding image_name.
    (We don't want the model to learn the filename here.)
    """
    label_data = {k: v for k, v in json_data.items() if k != "image_name"}
    return json.dumps(label_data, ensure_ascii=False)


def json_to_string(json_obj: Dict) -> str:
    """Convert JSON object to string (without image_name)."""
    clean_obj = {k: v for k, v in json_obj.items() if k != "image_name"}
    return json.dumps(clean_obj, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------
# Data Collator & Dataset
# ---------------------------------------------------------------------
class Florence2DataCollator:
    """
    Custom data collator for Florence-2 vision-language tasks.

    We now train Florence-2 as a decoder-only model w.r.t. text:
    - decoder_input_ids: tokens for "<OCR> " + JSON
    - decoder_attention_mask: usual mask
    - labels: same as decoder_input_ids (no shift needed; model handles it)
    """

    def __init__(self, processor, pad_token_id=None):
        self.processor = processor
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else processor.tokenizer.pad_token_id
        )

    def __call__(self, features):
        # pixel_values: [B, C, H, W]
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        decoder_input_ids_list = [f["decoder_input_ids"] for f in features]
        decoder_attention_mask_list = [f["decoder_attention_mask"] for f in features]
        labels_list = [f["labels"] for f in features]

        max_len = max(len(ids) for ids in decoder_input_ids_list)

        padded_decoder_input_ids = []
        padded_decoder_attention_mask = []
        padded_labels = []

        for dec_ids, dec_mask, lab in zip(
            decoder_input_ids_list, decoder_attention_mask_list, labels_list
        ):
            cur_len = len(dec_ids)
            pad_len = max_len - cur_len

            if pad_len > 0:
                padded_decoder_input_ids.append(
                    torch.cat(
                        [
                            dec_ids,
                            torch.full(
                                (pad_len,),
                                self.pad_token_id,
                                dtype=dec_ids.dtype,
                            ),
                        ]
                    )
                )

                padded_decoder_attention_mask.append(
                    torch.cat(
                        [
                            dec_mask,
                            torch.zeros(pad_len, dtype=dec_mask.dtype),
                        ]
                    )
                )

                padded_labels.append(
                    torch.cat(
                        [
                            lab,
                            torch.full((pad_len,), -100, dtype=lab.dtype),
                        ]
                    )
                )
            else:
                padded_decoder_input_ids.append(dec_ids)
                padded_decoder_attention_mask.append(dec_mask)
                padded_labels.append(lab)

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": torch.stack(padded_decoder_input_ids),
            "decoder_attention_mask": torch.stack(padded_decoder_attention_mask),
            "labels": torch.stack(padded_labels),
        }


class StaircaseOCRDataset(Dataset):
    """
    Custom dataset for Staircase OCR finetuning.

    IMPORTANT CHANGE:
    - We train Florence-2 using decoder_input_ids only:

        target_text = "<OCR> " + JSON(label)

      so the task token is part of the decoder sequence.

    - During inference we still use the standard Florence-2 pattern:
        inputs = processor(text="<OCR>", images=image, ...)
        model.generate(input_ids=inputs["input_ids"], pixel_values=...)
    """

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

        # 1) Get pixel_values via processor (we ignore the text tokens here)
        pv_inputs = self.processor(
            text=TASK_PROMPT,  # ensures same preprocessing as inference
            images=image,
            return_tensors="pt",
        )
        pixel_values = pv_inputs["pixel_values"].squeeze(0).to(torch.float32)

        # 2) Build decoder_text = "<OCR> " + JSON(label)
        json_label = create_label_string(item)
        decoder_text = f"{TASK_PROMPT} {json_label}"

        target_inputs = self.processor.tokenizer(
            decoder_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )

        decoder_input_ids = target_inputs["input_ids"].squeeze(0)
        decoder_attention_mask = target_inputs["attention_mask"].squeeze(0)

        # Labels are the same as decoder_input_ids; Trainer/model will handle shifting.
        labels = decoder_input_ids.clone()

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }


# ---------------------------------------------------------------------
# Trainer Wrapper
# ---------------------------------------------------------------------
class Florence2StaircaseTrainer:
    """Main trainer class for Florence-2 staircase OCR finetuning with LoRA."""

    def __init__(self, model_path: str, output_base_dir: str):
        self.model_path = model_path
        self.output_base_dir = output_base_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_base_dir, f"lora_run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float32

        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")

        self._load_model()

        self.best_eval_cer = float("inf")
        self.best_model_path = None

        # will be set in train()
        self.val_records: List[Dict] = []
        self.images_dir: str = ""

    def _load_model(self):
        """Load Florence-2 model with LoRA and processor."""
        print("Loading Florence-2 model...")

        if not os.path.exists(self.model_path):
            print("Model not found locally, defaulting to HF hub...")
            model_name = "microsoft/Florence-2-large"
        else:
            model_name = self.model_path

        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)

        for param in self.base_model.parameters():
            param.requires_grad_(True)

        print("Applying LoRA...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
            bias="none",
        )

        self.model = get_peft_model(self.base_model, lora_config)
        self.model.train()

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        lora_modules = sum(
            1 for name, _ in self.model.named_parameters() if "lora_" in name
        )

        print("✅ Model with LoRA loaded!")
        print(
            f"Trainable params: {trainable_params:,} "
            f"({trainable_params/total_params*100:.2f}%)"
        )
        print(f"LoRA modules: {lora_modules}")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Optional: token-level metrics (teacher-forced logits).
        We don't use this CER for model selection, but keep for debugging.
        """
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_preds = []
        decoded_labels = []

        for pred, label in zip(predictions, labels):
            label = label[label != -100]

            if pred.ndim > 1:
                pred = np.argmax(pred, axis=-1)

            try:
                pred_text = self.processor.tokenizer.decode(
                    pred, skip_special_tokens=True
                )
                label_text = self.processor.tokenizer.decode(
                    label, skip_special_tokens=True
                )

                decoded_preds.append(pred_text.strip())
                decoded_labels.append(label_text.strip())
            except Exception:
                decoded_preds.append("")
                decoded_labels.append("")

        cer_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            try:
                if len(label) > 0:
                    cer = jiwer.cer(label, pred)
                else:
                    cer = 1.0 if len(pred) > 0 else 0.0
                cer_scores.append(cer)
            except Exception:
                cer_scores.append(1.0)

        avg_cer = np.mean(cer_scores) if cer_scores else 1.0

        return {
            "eval_cer_token_level": avg_cer,
        }

    def _evaluate_autoregressive_cer_on_val(self) -> float:
        """
        Run autoregressive generation on the validation JSON list (self.val_records),
        compute CER against json_to_string(item) – SAME logic as test.
        """
        print("\n🔎 Running autoregressive validation CER (generation)...")
        cer_scores = []

        self.model.eval()

        for item in self.val_records:
            image_name = item["image_name"]
            image_path = os.path.join(self.images_dir, image_name)

            if not os.path.exists(image_path):
                cer_scores.append(1.0)
                continue

            try:
                image = Image.open(image_path).convert("RGB")

                inputs = self.processor(
                    text=TASK_PROMPT,
                    images=image,
                    return_tensors="pt",
                )
                input_ids = inputs["input_ids"].to(self.device)
                pixel_values = inputs["pixel_values"].to(self.device, self.torch_dtype)

                if pixel_values is None or pixel_values.numel() == 0:
                    cer_scores.append(1.0)
                    continue

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        max_new_tokens=512,
                        num_beams=3,
                        do_sample=False,
                        use_cache=False,  # KV cache fix
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        early_stopping=False,
                    )

                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]
                prediction_string = generated_text.strip()

                gt_json_string = json_to_string(item)

                # Try JSON parse for fair comparison, but fall back to raw
                try:
                    predicted_json = json.loads(prediction_string)
                    pred_json_string = json_to_string({"dummy": "x", **predicted_json})
                    cer = jiwer.cer(gt_json_string, pred_json_string)
                except Exception:
                    cer = jiwer.cer(gt_json_string, prediction_string)

                cer_scores.append(cer)

            except Exception as e:
                print(f"  ⚠️  Val sample error ({image_name}): {e}")
                cer_scores.append(1.0)

        avg_cer = float(np.mean(cer_scores)) if cer_scores else 1.0
        print(
            f"🔚 Autoregressive validation CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)\n"
        )
        return avg_cer

    def train(self, train_data: List[Dict], val_data: List[Dict], images_dir: str):
        """Train the Florence-2 model with LoRA."""
        print("Preparing datasets...")

        self.val_records = val_data
        self.images_dir = images_dir

        train_dataset = StaircaseOCRDataset(train_data, images_dir, self.processor)
        val_dataset = StaircaseOCRDataset(val_data, images_dir, self.processor)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        data_collator = Florence2DataCollator(self.processor)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=20,                 # a bit more conservative
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,                  # slightly lower LR
            weight_decay=0.01,
            warmup_steps=50,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,
            metric_for_best_model="eval_cer",
            greater_is_better=False,
            report_to="none",
            dataloader_pin_memory=False,
            fp16=False,
            gradient_checkpointing=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            optim="adamw_torch",
            label_smoothing_factor=0.1,          # small regularization
            label_names=["labels"],              # avoids Trainer warning for PeftModel
        )

        class CustomTrainer(Trainer):
            def __init__(self, parent_instance, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.parent = parent_instance

            def evaluate(
                self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
            ):
                # First, get eval_loss and any token-level metrics
                eval_result = super().evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=metric_key_prefix,
                )

                # Now run real generation-based CER on val_records
                current_cer = self.parent._evaluate_autoregressive_cer_on_val()
                eval_result["eval_cer"] = current_cer

                current_epoch = (
                    int(self.state.epoch)
                    if hasattr(self.state, "epoch") and self.state.epoch is not None
                    else self.state.global_step
                )
                print(
                    f"📊 Epoch {current_epoch} - Generation-based Validation CER: "
                    f"{current_cer:.4f} ({current_cer*100:.2f}%)"
                )

                # Best-model tracking based on generation CER
                if current_cer < self.parent.best_eval_cer:
                    improvement = self.parent.best_eval_cer - current_cer
                    self.parent.best_eval_cer = current_cer

                    self.parent.best_model_path = os.path.join(
                        self.parent.output_dir, "best_model"
                    )

                    if os.path.exists(self.parent.best_model_path):
                        import shutil

                        shutil.rmtree(self.parent.best_model_path)

                    self.model.save_pretrained(self.parent.best_model_path)
                    self.parent.processor.save_pretrained(
                        self.parent.best_model_path
                    )
                    print(
                        f"🎯 New best CER: {current_cer:.4f} "
                        f"(improved by {improvement:.4f})"
                    )
                    print(f"✅ Model saved to {self.parent.best_model_path}\n")
                else:
                    print(
                        f"   Best CER so far: {self.parent.best_eval_cer:.4f}\n"
                    )

                return eval_result

        trainer = CustomTrainer(
            self,
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,  # token-level only, for logging
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        print("Starting LoRA training...")
        trainer.train()

        if self.best_model_path is None:
            final_model_path = os.path.join(self.output_dir, "final_model")
            self.model.save_pretrained(final_model_path)
            self.processor.save_pretrained(final_model_path)
            self.best_model_path = final_model_path

        print("\n✅ Training completed!")
        print(f"Best generation-based validation CER: {self.best_eval_cer:.4f}")
        print(f"Best model: {self.best_model_path}")

        return trainer

    def predict(self, test_data: List[Dict], images_dir: str) -> List[Dict]:
        """Make predictions using the LoRA finetuned model."""
        print(f"Making predictions on {len(test_data)} test images...")

        predictions = []
        json_parse_failures = 0

        self.model.eval()

        for i, test_item in enumerate(test_data):
            image_name = test_item["image_name"]
            print(f"[{i+1}/{len(test_data)}] {image_name}")

            try:
                image_path = os.path.join(images_dir, image_name)
                image = Image.open(image_path).convert("RGB")

                inputs = self.processor(
                    text=TASK_PROMPT,
                    images=image,
                    return_tensors="pt",
                )

                input_ids = inputs["input_ids"].to(self.device)
                pixel_values = inputs["pixel_values"].to(self.device, self.torch_dtype)

                if pixel_values is None or pixel_values.numel() == 0:
                    raise ValueError("Empty pixel values")

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        max_new_tokens=512,
                        num_beams=3,
                        do_sample=False,
                        use_cache=False,  # KV cache fix
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        early_stopping=False,
                    )

                generated_text = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0]

                prediction_string = generated_text.strip()

                # Try JSON parse
                predicted_json = None
                json_parse_success = False
                try:
                    predicted_json = json.loads(prediction_string)
                    json_parse_success = True
                except Exception:
                    json_parse_failures += 1

                gt_json_string = json_to_string(test_item)

                if json_parse_success and predicted_json:
                    pred_json_string = json_to_string({"dummy": "x", **predicted_json})
                    cer_score = jiwer.cer(gt_json_string, pred_json_string)
                else:
                    cer_score = jiwer.cer(gt_json_string, prediction_string)

                prediction_entry = {
                    "image_name": image_name,
                    "prediction_string": prediction_string,
                    "prediction_json": predicted_json,
                    "json_parse_success": json_parse_success,
                    "ground_truth_json_string": gt_json_string,
                    "cer_score": cer_score,
                }

                predictions.append(prediction_entry)
                print(f"  ✅ CER: {cer_score:.4f}")

            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                import traceback

                traceback.print_exc()

                gt_json_string = json_to_string(test_item)
                prediction_entry = {
                    "image_name": test_item.get("image_name", ""),
                    "prediction_string": f"Error: {str(e)}",
                    "prediction_json": None,
                    "json_parse_success": False,
                    "ground_truth_json_string": gt_json_string,
                    "cer_score": 1.0,
                }
                predictions.append(prediction_entry)

        predictions_path = os.path.join(self.output_dir, "test_predictions_lora.jsonl")
        save_jsonl(predictions, predictions_path)

        cer_scores = [p["cer_score"] for p in predictions]
        avg_cer = float(np.mean(cer_scores)) if cer_scores else 1.0
        perfect_matches = sum(1 for cer in cer_scores if cer == 0.0)
        json_success_rate = (
            (len(predictions) - json_parse_failures) / len(predictions) * 100
            if len(predictions) > 0
            else 0
        )

        print(f"\n{'='*60}")
        print("TEST RESULTS (Staircase)")
        print(f"{'='*60}")
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Perfect matches: {perfect_matches}/{len(predictions)}")
        print(f"JSON success rate: {json_success_rate:.1f}%")
        print(f"{'='*60}\n")

        cer_metrics = {
            "average_cer": avg_cer,
            "min_cer": float(min(cer_scores)) if cer_scores else 1.0,
            "max_cer": float(max(cer_scores)) if cer_scores else 1.0,
            "perfect_matches": int(perfect_matches),
            "json_success_rate": float(json_success_rate),
            "total_samples": len(predictions),
        }

        with open(
            os.path.join(self.output_dir, "test_cer_metrics_lora.json"), "w"
        ) as f:
            json.dump(cer_metrics, f, indent=2)

        return predictions


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    """Main function to run the LoRA finetuning pipeline on Staircase dataset."""

    images_dir = STAIR_IMAGES_DIR
    json_dir = STAIR_JSON_DIR
    model_path = MODEL_PATH
    output_base_dir = OUTPUT_BASE_DIR

    print(f"Dataset: {DATASET_NAME}")
    print("Loading datasets...")

    train_data = load_jsonl(os.path.join(json_dir, "train.jsonl"))
    val_data = load_jsonl(os.path.join(json_dir, "val.jsonl"))
    test_data = load_jsonl(os.path.join(json_dir, "test.jsonl"))

    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    print(f"Loaded {len(test_data)} test samples")

    trainer_obj = Florence2StaircaseTrainer(model_path, output_base_dir)

    trainer_obj.train(train_data, val_data, images_dir)

    _ = trainer_obj.predict(test_data, images_dir)

    print("LoRA finetuning pipeline completed successfully!")
    print(f"Results saved in: {trainer_obj.output_dir}")


if __name__ == "__main__":
    main()
