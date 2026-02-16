#!/usr/bin/env python3
# donut_inventory_fullft_autoreg_valcer_tb.py
# ✅ Donut FULL finetuning (NO LoRA)
# ✅ Autoregressive (generate-based) validation CER for best-model selection
# ✅ No argparse; config is inside the code
# ✅ TensorBoard logging enabled
# ✅ No curly-brace extraction; compare generated text (optionally JSON-normalized if parseable)


import os
import json
import glob
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple


import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
    "data_dir": "/home/woody/iwi5/iwi5298h/updated_json_inven",
    "image_dir": "/home/woody/iwi5/iwi5298h/inventory_images",


    "model_name": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "local_model_dir": "/home/vault/iwi5/iwi5298h/models/donut_cord",


    # Training
    "epochs": 3,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "lr": 1e-5,
    "max_length": 512,
    "gradient_checkpointing": True,
    "weight_decay": 0.01,
    "num_beams": 3,


    # Early stopping
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0,


    # Autoregressive eval
    "autoreg_eval_batch_size": 2,   # generation eval batch size
    "autoreg_eval_subset": None,    # e.g. 200 for faster eval, or None for full val


    # Prediction after training
    "predict_on_test": True,
    "prediction_batch_size": 2,


    # Output root
    "output_root": "/home/vault/iwi5/iwi5298h/models_image_text/donut/inventory",
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
    Returns CER in percentage.
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


# -----------------------
# IO helpers
# -----------------------
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



def find_image_path(image_name: str, images_dir: str) -> str:
    exact_path = os.path.join(images_dir, image_name)
    if os.path.exists(exact_path):
        return exact_path


    name_without_ext = os.path.splitext(image_name)[0]
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        p = os.path.join(images_dir, f"{name_without_ext}{ext}")
        if os.path.exists(p):
            return p


    search_pattern = os.path.join(images_dir, f"*{name_without_ext}*")
    matches = glob.glob(search_pattern)
    return matches[0] if matches else exact_path


# -----------------------
# Dataset + Collator
# -----------------------
class DonutOCRDataset(Dataset):
    def __init__(self, jsonl_path: str, image_dir: str, processor: DonutProcessor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
        self.image_dir = Path(image_dir)
        self.data = load_jsonl(jsonl_path)
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")


    def __len__(self):
        return len(self.data)


    def format_target_text(self, item: Dict) -> str:
        target_dict = {k: v for k, v in item.items() if k != "image_name"}
        return json.dumps(target_dict, ensure_ascii=False, separators=(",", ":"))


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        image_path_str = find_image_path(item["image_name"], str(self.image_dir))
        image_path = Path(image_path_str)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")


        image = Image.open(image_path).convert("RGB")
        target_text = self.format_target_text(item)


        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)


        decoder_start = "<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            decoder_start, add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0)


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
        if torch.any(full_target_ids >= vocab_size):
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
    def __init__(self, processor: DonutProcessor, max_length: int = 512):
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
            # ✅ FIX: Only append sample_index if it exists in the item
            if "sample_index" in item:
                sample_index.append(item["sample_index"])


        result = {
            "pixel_values": pixel_values,
            "decoder_input_ids": torch.stack(decoder_input_ids),
            "labels": torch.stack(labels),
        }

        # ✅ FIX: Only add sample_index to result if we collected any
        if sample_index:
            result["sample_index"] = torch.tensor(sample_index, dtype=torch.long)

        return result


# -----------------------
# Autoregressive validation CER (generate-based)
# -----------------------
@torch.no_grad()
def autoreg_eval_cer(
    model: VisionEncoderDecoderModel,
    dataset: DonutOCRDataset,
    processor: DonutProcessor,
    device: torch.device,
    max_length: int,
    num_beams: int,
    batch_size: int,
    subset: int | None = None,
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
# Prediction (test)
# -----------------------
@torch.no_grad()
def generate_predictions(
    model: VisionEncoderDecoderModel,
    dataset: DonutOCRDataset,
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


    for _, batch in enumerate(tqdm(loader, desc="Generating test predictions")):
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
    torch.backends.cuda.matmul.allow_tf32 = True


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(CONFIG["output_root"], f"inventory_data_cord_{timestamp}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    tb_log_dir = os.path.join(OUTPUT_DIR, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"TensorBoard logs: {tb_log_dir}")


    # Load model/processor
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
    dataset_max_length = min(CONFIG["max_length"], 512)
    train_dataset = DonutOCRDataset(
        os.path.join(CONFIG["data_dir"], "train.jsonl"),
        CONFIG["image_dir"],
        processor,
        dataset_max_length,
    )
    val_dataset = DonutOCRDataset(
        os.path.join(CONFIG["data_dir"], "val.jsonl"),
        CONFIG["image_dir"],
        processor,
        dataset_max_length,
    )
    test_path = os.path.join(CONFIG["data_dir"], "test.jsonl")
    test_dataset = DonutOCRDataset(test_path, CONFIG["image_dir"], processor, dataset_max_length) if os.path.exists(test_path) else None


    data_collator = ImprovedDataCollator(processor, dataset_max_length)


    # Training args
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        learning_rate=CONFIG["lr"],
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,


        # keep generate enabled so Trainer eval loop is consistent
        predict_with_generate=True,
        generation_max_length=dataset_max_length,
        generation_num_beams=CONFIG["num_beams"],


        logging_steps=50,


        # ✅ FIX: Set to False since we manually handle extra keys
        remove_unused_columns=False,


        fp16=False,
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


    # Custom trainer: override evaluate to compute autoregressive CER via generate()
    # Also strip sample_index before model(**inputs) so encoder doesn't see it.
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
            f.write("DONUT MODEL TRAINING RESULTS\n")
            f.write("=" * 60 + "\n")
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