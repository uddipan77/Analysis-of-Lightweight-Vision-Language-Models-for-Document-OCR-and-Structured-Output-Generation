#!/usr/bin/env python3
# donut_staircase_fullft_autoreg_valcer_tb_cord.py
# ✅ Donut FULL finetuning (NO LoRA)
# ✅ Uses CORD-v2 finetuned base: naver-clova-ix/donut-base-finetuned-cord-v2 (loaded from your local dir)
# ✅ Autoregressive (generate-based) validation CER for best-model selection (LIKE your inventory script)
# ✅ No argparse; config is inside the code
# ✅ TensorBoard logging enabled
# ✅ Keeps memory usage safe for high-res images by relying on DonutProcessor resizing + conservative batch sizes
# ✅ No Phi-style JSON extraction in evaluation; compare generated text directly (optionally JSON-normalized for structured CER)


import os
import json
import unicodedata
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
import editdistance

from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    EarlyStoppingCallback,
)


# -----------------------
# CONFIG (no argparse)
# -----------------------
CONFIG = {
    # Data
    "data_dir": "/home/woody/iwi5/iwi5298h/json_staircase",
    "image_dir": "/home/woody/iwi5/iwi5298h/staircase_images",

    # UPDATED: Model/processor loading - use donut-base-finetuned-cord-v2
    #"model_name": "naver-clova-ix/donut-base-finetuned-cord-v2",
    #"local_model_dir": "/home/vault/iwi5/iwi5298h/models/donut_cord",

    # Model:  donut base (local)
    "model_name": "naver-clova-ix/donut-base",
    "local_model_dir": "/home/vault/iwi5/iwi5298h/models/donut_base",

    # Training (keep conservative for high-res)
    "epochs": 15,
    "train_batch_size": 1,         # 👈 safest
    "eval_batch_size": 1,
    "lr": 2e-5,
    "max_length": 768,
    "gradient_checkpointing": True,
    "weight_decay": 0.01,
    "num_beams": 3,

    # Mixed precision (saves VRAM). If your cluster doesn't like fp16, set False.
    "fp16": True,

    # Optional: accumulate grads to simulate bigger batch without OOM
    "gradient_accumulation_steps": 1,

    # Early stopping
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.01,

    # Augmentation (train only)
    "use_augmentation": False,
    "augmentation_factor": 0,

    # Autoregressive eval
    "autoreg_eval_batch_size": 1,   # 👈 safest
    "autoreg_eval_subset": None,    # e.g. 200 for faster eval, or None for full val

    # Prediction after training
    "predict_on_test": True,
    "prediction_batch_size": 1,     # 👈 safest

    # Output root
    "output_root": "/home/vault/iwi5/iwi5298h/models_image_text/donut/staircase",
}


# -----------------------
# Text helpers
# -----------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def canonical_json_string(obj: Any) -> str:
    # No sort_keys -> preserves original order if dict already ordered
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse_json_string(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def calculate_cer_percent(predictions: List[str], targets: List[str], json_mode: bool = False) -> float:
    """
    Returns CER in percentage using editdistance.
    If json_mode=True: tries to parse as JSON and canonicalize before CER.
    """
    total_chars = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        if json_mode:
            try:
                pred_json = parse_json_string(pred)
                target_json = parse_json_string(target)
                pred_str = canonical_json_string(pred_json) if isinstance(pred_json, dict) else str(pred_json)
                target_str = canonical_json_string(target_json) if isinstance(target_json, dict) else str(target_json)
            except Exception:
                pred_str, target_str = pred, target
        else:
            pred_str, target_str = pred, target

        total_chars += len(target_str)
        total_errors += editdistance.eval(pred_str, target_str)

    return (total_errors / total_chars) * 100.0 if total_chars > 0 else 0.0


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def download_and_cache_model(model_name: str, local_cache_dir: str) -> Tuple[DonutProcessor, VisionEncoderDecoderModel]:
    """
    Load from local_cache_dir if config.json exists; else download and cache.
    """
    os.makedirs(local_cache_dir, exist_ok=True)
    config_path = os.path.join(local_cache_dir, "config.json")

    if os.path.exists(config_path):
        print(f"Loading model from local cache: {local_cache_dir}")
        processor = DonutProcessor.from_pretrained(local_cache_dir)
        model = VisionEncoderDecoderModel.from_pretrained(local_cache_dir)
    else:
        print(f"Downloading model: {model_name}")
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        print(f"Saving to local cache: {local_cache_dir}")
        processor.save_pretrained(local_cache_dir)
        model.save_pretrained(local_cache_dir)

    return processor, model


# -----------------------
# Data augmentation
# -----------------------
def apply_augmentation(image: Image.Image, aug_type: Optional[str] = None) -> Image.Image:
    """
    Mild document-friendly augmentation.
    """
    if aug_type is None:
        aug_types = ["brightness", "contrast", "rotate", "blur", "noise", "original"]
        aug_type = random.choice(aug_types)

    if aug_type == "original":
        return image

    img_array = np.array(image)

    if aug_type == "brightness":
        enhancer = ImageEnhance.Brightness(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)

    if aug_type == "contrast":
        enhancer = ImageEnhance.Contrast(image)
        factor = random.uniform(0.7, 1.3)
        return enhancer.enhance(factor)

    if aug_type == "rotate":
        angle = random.uniform(-5, 5)
        return image.rotate(angle, fillcolor=(255, 255, 255))

    if aug_type == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    if aug_type == "noise":
        noise = np.random.normal(0, 10, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    return image


# -----------------------
# Dataset + Collator (aligned to inventory logic)
# -----------------------
class DonutOCRDatasetWithAugmentation(Dataset):
    """
    Returns:
      pixel_values, decoder_input_ids (prompt+target), labels (masked prompt), sample_index
    This matches your inventory dataset return contract so autoreg eval can fetch GT by sample_index.
    """
    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        processor: DonutProcessor,
        max_length: int = 768,
        use_augmentation: bool = False,
        augmentation_factor: int = 0,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_dir = Path(image_dir)

        self.original_data = load_jsonl(jsonl_path)

        # Expand data with augmentation copies (train only)
        self.data = []
        for item in self.original_data:
            self.data.append({**item, "aug_type": "original"})
            if use_augmentation:
                for i in range(augmentation_factor):
                    self.data.append({**item, "aug_type": f"aug_{i+1}"})

        print(f"Loaded {len(self.original_data)} original samples from {jsonl_path}")
        if use_augmentation:
            print(f"Using augmentation: total samples = {len(self.data)} (factor={augmentation_factor})")
        else:
            print(f"No augmentation: total samples = {len(self.data)}")

        # keep for GT formatting (exclude image_name, aug_type)
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.data)

    def format_target_text(self, item: Dict) -> str:
        target_dict = {k: v for k, v in item.items() if k not in ["image_name", "aug_type"]}
        return canonical_json_string(target_dict)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path = self.image_dir / item["image_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        if self.use_augmentation and item.get("aug_type", "original") != "original":
            image = apply_augmentation(image)

        target_text = self.format_target_text(item)

        # IMPORTANT for high-res: DonutProcessor will resize internally to the model's expected size.
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        # Prompt token (CORD-v2)
        decoder_start = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            decoder_start, add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0)

        # Tokenize target
        target_tokenized = self.processor.tokenizer(
            target_text,
            add_special_tokens=False,
            padding=False,
            truncation=True,
            max_length=self.max_length - len(decoder_input_ids) - 1,
            return_tensors="pt",
        )

        eos_token_id = self.processor.tokenizer.eos_token_id
        target_ids = target_tokenized.input_ids.squeeze(0)
        target_ids = torch.cat([target_ids, torch.tensor([eos_token_id], dtype=torch.long)])

        full_target_ids = torch.cat([decoder_input_ids, target_ids])

        if len(full_target_ids) > self.max_length:
            full_target_ids = full_target_ids[: self.max_length]
            full_target_ids[-1] = eos_token_id

        vocab_size = len(self.processor.tokenizer)
        if torch.any(full_target_ids >= vocab_size) or torch.any(full_target_ids < 0):
            full_target_ids = torch.clamp(full_target_ids, 0, vocab_size - 1).long()

        labels = full_target_ids.clone()
        labels[: len(decoder_input_ids)] = -100

        return {
            "pixel_values": pixel_values.to(torch.float32),
            "decoder_input_ids": full_target_ids.long(),
            "labels": labels.long(),
            "sample_index": idx,
        }


class ImprovedDataCollator:
    def __init__(self, processor: DonutProcessor, max_length: int = 768):
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.vocab_size = len(processor.tokenizer)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        max_len = min(max(len(item["decoder_input_ids"]) for item in batch), self.max_length)

        decoder_input_ids, labels, sample_index = [], [], []
        for item in batch:
            input_ids = item["decoder_input_ids"]
            item_labels = item["labels"]

            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                item_labels = item_labels[:max_len]
                if input_ids[-1] != self.processor.tokenizer.eos_token_id:
                    input_ids[-1] = self.processor.tokenizer.eos_token_id

            if torch.any(input_ids >= self.vocab_size) or torch.any(input_ids < 0):
                input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1).long()

            seq_len = len(input_ids)
            pad_len = max_len - seq_len
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
                item_labels = torch.cat([item_labels, torch.full((pad_len,), -100, dtype=torch.long)])

            decoder_input_ids.append(input_ids.long())
            labels.append(item_labels.long())
            if "sample_index" in item:
                sample_index.append(item["sample_index"])

        result = {
            "pixel_values": pixel_values,
            "decoder_input_ids": torch.stack(decoder_input_ids),
            "labels": torch.stack(labels),
        }
        if sample_index:
            result["sample_index"] = torch.tensor(sample_index, dtype=torch.long)

        return result


# -----------------------
# Autoregressive validation CER (generate-based) — same as inventory
# -----------------------
@torch.no_grad()
def autoreg_eval_cer(
    model: VisionEncoderDecoderModel,
    dataset: DonutOCRDatasetWithAugmentation,
    processor: DonutProcessor,
    device: torch.device,
    max_length: int,
    num_beams: int,
    batch_size: int,
    subset: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()

    if subset is not None and subset < len(dataset):
        indices = list(range(subset))
    else:
        indices = list(range(len(dataset)))

    class _Sub(Dataset):
        def __init__(self, base, idxs):
            self.base = base
            self.idxs = idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i): return self.base[self.idxs[i]]

    sub_ds = _Sub(dataset, indices)

    data_collator = ImprovedDataCollator(processor, max_length=max_length)
    loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=data_collator)

    start_token = "<s_cord-v2>"
    start_ids = processor.tokenizer(start_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device).long()

    preds, gts = [], []

    for batch in tqdm(loader, desc="Autoreg val CER (generate)"):
        pixel_values = batch["pixel_values"].to(device)
        sample_index = batch["sample_index"].cpu().numpy().tolist()

        prompt_ids = start_ids.repeat(pixel_values.size(0), 1)

        generated_ids = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=prompt_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        for i in range(len(generated_ids)):
            pred_text = processor.tokenizer.decode(
                generated_ids[i][len(prompt_ids[0]) :],
                skip_special_tokens=True,
            ).strip()

            pred_text = pred_text.replace(start_token, "").strip()

            # Ground truth from dataset by sample_index (aligned with inventory logic)
            gt_text = dataset.format_target_text(dataset.data[sample_index[i]])
            gt_text = gt_text.replace(start_token, "").strip()

            preds.append(normalize_text(pred_text))
            gts.append(normalize_text(gt_text))

    cer_string = calculate_cer_percent(preds, gts, json_mode=False)
    cer_structured = calculate_cer_percent(preds, gts, json_mode=True)

    return {
        "eval_gen_cer_string": float(cer_string),
        "eval_gen_cer_structured": float(cer_structured),
    }


# -----------------------
# Prediction (test) — aligned with inventory logic
# -----------------------
@torch.no_grad()
def generate_predictions(
    model: VisionEncoderDecoderModel,
    dataset: DonutOCRDatasetWithAugmentation,
    processor: DonutProcessor,
    device: torch.device,
    max_length: int,
    num_beams: int,
    batch_size: int,
) -> List[Dict]:
    model.eval()

    collator = ImprovedDataCollator(processor, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)

    start_token = "<s_cord-v2>"
    start_ids = processor.tokenizer(start_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device).long()

    out = []

    for _, batch in enumerate(tqdm(loader, desc="Generating predictions")):
        pixel_values = batch["pixel_values"].to(device)
        sample_index = batch["sample_index"].cpu().numpy().tolist()

        prompt_ids = start_ids.repeat(pixel_values.size(0), 1)

        generated_ids = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=prompt_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        for i in range(len(generated_ids)):
            idx = sample_index[i]
            pred_text = processor.tokenizer.decode(
                generated_ids[i][len(prompt_ids[0]) :],
                skip_special_tokens=True,
            ).strip()

            pred_text = pred_text.replace(start_token, "").strip()

            gt_text = dataset.format_target_text(dataset.data[idx])
            gt_text = gt_text.replace(start_token, "").strip()

            out.append({
                "image_name": dataset.data[idx]["image_name"],
                "prediction": pred_text,
                "ground_truth": gt_text,
                "sample_index": idx,
                "original_data": dataset.data[idx].copy(),
            })

    return out


# -----------------------
# Main
# -----------------------
def main():
    # Good defaults for performance; does not increase VRAM much
    torch.backends.cuda.matmul.allow_tf32 = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(CONFIG["output_root"], f"staircase_data_cord_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tb_log_dir = os.path.join(OUTPUT_DIR, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"TensorBoard logs: {tb_log_dir}")

    # Load model/processor (CORD-v2 finetuned base)
    processor, model = download_and_cache_model(CONFIG["model_name"], CONFIG["local_model_dir"])

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Configure model
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_cord-v2>")
    model.config.max_length = CONFIG["max_length"]

    generation_config = GenerationConfig(
        max_length=CONFIG["max_length"],
        num_beams=CONFIG["num_beams"],
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<s_cord-v2>"),
    )
    model.generation_config = generation_config
    model.to(device)

    # Datasets
    dataset_max_length = min(CONFIG["max_length"], 768)

    train_dataset = DonutOCRDatasetWithAugmentation(
        jsonl_path=os.path.join(CONFIG["data_dir"], "train.jsonl"),
        image_dir=CONFIG["image_dir"],
        processor=processor,
        max_length=dataset_max_length,
        use_augmentation=CONFIG["use_augmentation"],
        augmentation_factor=CONFIG["augmentation_factor"],
    )

    val_dataset = DonutOCRDatasetWithAugmentation(
        jsonl_path=os.path.join(CONFIG["data_dir"], "val.jsonl"),
        image_dir=CONFIG["image_dir"],
        processor=processor,
        max_length=dataset_max_length,
        use_augmentation=False,
    )

    test_dataset = None
    test_path = os.path.join(CONFIG["data_dir"], "test.jsonl")
    if os.path.exists(test_path):
        test_dataset = DonutOCRDatasetWithAugmentation(
            jsonl_path=test_path,
            image_dir=CONFIG["image_dir"],
            processor=processor,
            max_length=dataset_max_length,
            use_augmentation=False,
        )

    data_collator = ImprovedDataCollator(processor, dataset_max_length)

    # Training args — aligned to inventory logic
    use_fp16 = bool(CONFIG["fp16"] and torch.cuda.is_available())
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["lr"],
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,

        # keep generate enabled so Trainer eval loop is consistent
        predict_with_generate=True,
        generation_max_length=dataset_max_length,
        generation_num_beams=CONFIG["num_beams"],

        logging_steps=50,
        remove_unused_columns=False,

        fp16=use_fp16,              # helps VRAM a lot
        dataloader_num_workers=0,
        report_to="tensorboard",
        logging_dir=tb_log_dir,
        dataloader_pin_memory=False,
        gradient_checkpointing=CONFIG["gradient_checkpointing"],
        weight_decay=CONFIG["weight_decay"],

        load_best_model_at_end=True,
        metric_for_best_model="eval_gen_cer_string",
        greater_is_better=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
    )

    # Custom trainer — same pattern as inventory (strip sample_index + autoreg evaluate)
    class CustomTrainer(Seq2SeqTrainer):
        def _strip_extra_keys(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            if "sample_index" in inputs:
                inputs = dict(inputs)
                inputs.pop("sample_index", None)
            return inputs

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = self._strip_extra_keys(inputs)
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
            inputs = self._strip_extra_keys(inputs)
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            base_metrics = super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            ds = eval_dataset if eval_dataset is not None else self.eval_dataset

            gen_metrics = autoreg_eval_cer(
                model=self.model,
                dataset=ds,
                processor=processor,
                device=device,
                max_length=dataset_max_length,
                num_beams=CONFIG["num_beams"],
                batch_size=CONFIG["autoreg_eval_batch_size"],
                subset=CONFIG["autoreg_eval_subset"],
            )

            self.log(gen_metrics)
            base_metrics.update(gen_metrics)

            print(
                f"\n[Autoreg Val CER] "
                f"cer_string={gen_metrics['eval_gen_cer_string']:.4f}% | "
                f"cer_structured={gen_metrics['eval_gen_cer_structured']:.4f}%\n"
            )
            return base_metrics

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=CONFIG["early_stopping_patience"],
                early_stopping_threshold=CONFIG["early_stopping_threshold"],
            ),
        ],
    )

    print("\nStarting training...")
    print("Best model selection based on: eval_gen_cer_string (AUTOREGRESSIVE, lower is better)\n")
    trainer.train()

    # Save final model (best is already restored in memory by load_best_model_at_end)
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    trainer.model.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    generation_config.save_pretrained(final_model_dir)

    print(f"\nSaved final model to: {final_model_dir}")
    print("✅ Note: trainer.model is the BEST model (restored by load_best_model_at_end).")

    # Predict on test with BEST model in memory
    if CONFIG["predict_on_test"] and test_dataset is not None:
        print("\nGenerating predictions on test set using BEST model...")
        best_model = trainer.model

        test_predictions = generate_predictions(
            model=best_model,
            dataset=test_dataset,
            processor=processor,
            device=device,
            max_length=dataset_max_length,
            num_beams=CONFIG["num_beams"],
            batch_size=CONFIG["prediction_batch_size"],
        )

        pred_path = os.path.join(OUTPUT_DIR, "test_predictions.jsonl")
        save_jsonl(test_predictions, pred_path)

        preds_text = [normalize_text(d["prediction"]) for d in test_predictions]
        gt_text = [normalize_text(d["ground_truth"]) for d in test_predictions]

        test_cer_str = calculate_cer_percent(preds_text, gt_text, json_mode=False)
        test_cer_struct = calculate_cer_percent(preds_text, gt_text, json_mode=True)

        scores_path = os.path.join(OUTPUT_DIR, "final_CER_scores.txt")
        with open(scores_path, "w", encoding="utf-8") as f:
            f.write("DONUT MODEL TRAINING RESULTS (Staircase)\n")
            f.write("=" * 60 + "\n")
            f.write("Base init: donut-base-finetuned-cord-v2\n")
            f.write("Model Selection: eval_gen_cer_string (AUTOREGRESSIVE)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Test CER (string): {test_cer_str:.4f}\n")
            f.write(f"Test CER (structured): {test_cer_struct:.4f}\n")

        print(f"\nTEST CER (string): {test_cer_str:.4f}%")
        print(f"TEST CER (structured): {test_cer_struct:.4f}%")
        print(f"Predictions saved: {pred_path}")
        print(f"Scores saved: {scores_path}")

    print(f"\nAll outputs saved in: {OUTPUT_DIR}")
    print(f"To view TensorBoard:\n  tensorboard --logdir {CONFIG['output_root']}")


if __name__ == "__main__":
    main()
