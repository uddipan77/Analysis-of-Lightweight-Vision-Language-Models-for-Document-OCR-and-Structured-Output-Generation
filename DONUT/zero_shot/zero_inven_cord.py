#!/usr/bin/env python3
# donut_inventory_cord_zeroshot_infer.py
# Inference-only on test.jsonl using Donut CORD model
# - Model: naver-clova-ix/donut-base-finetuned-cord-v2
# - Saves predictions + CER (string + structured) in a dedicated output folder

import os
import json
import glob
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import editdistance

from transformers import DonutProcessor, VisionEncoderDecoderModel, GenerationConfig


# -----------------------
# CONFIG (edit paths)
# -----------------------
CONFIG = {
    "data_dir": "/home/woody/iwi5/iwi5298h/updated_json_inven",
    "image_dir": "/home/woody/iwi5/iwi5298h/inventory_images",

    "model_name": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "local_model_dir": "/home/vault/iwi5/iwi5298h/models/donut_cord",

    "max_length": 512,
    "num_beams": 3,
    "batch_size": 2,

    # CORD prompt token (keep as in your training code)
    "start_token": "<s_cord-v2>",

    # Dedicated output root for inference runs
    "output_root": "/home/vault/iwi5/iwi5298h/models_image_text/donut/inventory_cord_infer",
}


# -----------------------
# Text helpers
# -----------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text


def canonical_json_string(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse_json_string(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def calculate_cer_percent(predictions: List[str], targets: List[str], json_mode: bool = False) -> float:
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
# Dataset + Collator (inference-only)
# -----------------------
class DonutTestDataset(Dataset):
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
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        return {
            "pixel_values": pixel_values.to(torch.float32),
            "sample_index": idx,
        }


class InferCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        sample_index = torch.tensor([b["sample_index"] for b in batch], dtype=torch.long)
        return {"pixel_values": pixel_values, "sample_index": sample_index}


# -----------------------
# Inference
# -----------------------
@torch.no_grad()
def generate_predictions(
    model: VisionEncoderDecoderModel,
    dataset: DonutTestDataset,
    processor: DonutProcessor,
    device: torch.device,
    max_length: int,
    num_beams: int,
    batch_size: int,
    start_token: str,
) -> List[Dict]:
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=InferCollator())

    start_ids = processor.tokenizer(start_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device).long()

    out = []

    for batch in tqdm(loader, desc="Generating test predictions (CORD model)"):
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
                generated_ids[i][len(prompt_ids[0]):],
                skip_special_tokens=True,
            ).strip()

            pred_text = pred_text.replace(start_token, "").strip()
            gt_text = dataset.format_target_text(dataset.data[idx]).replace(start_token, "").strip()

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

    test_jsonl = os.path.join(CONFIG["data_dir"], "test.jsonl")
    if not os.path.exists(test_jsonl):
        raise FileNotFoundError(f"Missing test.jsonl at: {test_jsonl}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(CONFIG["output_root"], f"cord_infer_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Load model/processor
    processor, model = download_and_cache_model(CONFIG["model_name"], CONFIG["local_model_dir"])

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Configure model + generation
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(CONFIG["start_token"])
    model.config.max_length = CONFIG["max_length"]

    generation_config = GenerationConfig(
        max_length=CONFIG["max_length"],
        num_beams=CONFIG["num_beams"],
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids(CONFIG["start_token"]),
    )
    model.generation_config = generation_config
    model.to(device)

    # Dataset
    dataset_max_length = min(CONFIG["max_length"], 512)
    test_dataset = DonutTestDataset(test_jsonl, CONFIG["image_dir"], processor, dataset_max_length)

    # Predict
    test_predictions = generate_predictions(
        model=model,
        dataset=test_dataset,
        processor=processor,
        device=device,
        max_length=dataset_max_length,
        num_beams=CONFIG["num_beams"],
        batch_size=CONFIG["batch_size"],
        start_token=CONFIG["start_token"],
    )

    pred_path = os.path.join(OUTPUT_DIR, "predictions_test.jsonl")
    save_jsonl(test_predictions, pred_path)

    # CER
    preds_text = [normalize_text(d["prediction"]) for d in test_predictions]
    gt_text = [normalize_text(d["ground_truth"]) for d in test_predictions]

    test_cer_str = calculate_cer_percent(preds_text, gt_text, json_mode=False)
    test_cer_struct = calculate_cer_percent(preds_text, gt_text, json_mode=True)

    scores_path = os.path.join(OUTPUT_DIR, "cer_scores.txt")
    with open(scores_path, "w", encoding="utf-8") as f:
        f.write("DONUT CORD INFERENCE RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {CONFIG['model_name']}\n")
        f.write(f"Start token: {CONFIG['start_token']}\n")
        f.write(f"Test samples: {len(test_predictions)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test CER (string): {test_cer_str:.4f}\n")
        f.write(f"Test CER (structured): {test_cer_struct:.4f}\n")

    print(f"\nTEST CER (string): {test_cer_str:.4f}%")
    print(f"TEST CER (structured): {test_cer_struct:.4f}%")
    print(f"Predictions saved: {pred_path}")
    print(f"Scores saved     : {scores_path}")
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
