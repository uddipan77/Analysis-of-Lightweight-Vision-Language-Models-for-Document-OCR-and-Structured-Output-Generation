#!/usr/bin/env python3
# zero_schmuck_base_fixed.py

import os, json, glob, unicodedata
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import editdistance

from transformers import DonutProcessor, VisionEncoderDecoderModel, GenerationConfig


CONFIG = {
    "test_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",

    "model_name": "naver-clova-ix/donut-base",
    "local_model_dir": "/home/vault/iwi5/iwi5298h/models/donut_base",

    "start_token": "<s_cord-v2>",
    "max_length": 512,
    "num_beams": 3,
    "batch_size": 2,

    "output_root": "/home/vault/iwi5/iwi5298h/models_image_text/donut/schmuck_infer_base",
}


# -----------------------
# Text helpers
# -----------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def canonical_json_string(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse_json_string(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def calculate_cer_percent(predictions: List[str], targets: List[str], json_mode: bool = False) -> float:
    total_chars, total_errors = 0, 0
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
def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(data: List[Dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def download_and_cache_model(model_name: str, local_cache_dir: str) -> Tuple[DonutProcessor, VisionEncoderDecoderModel]:
    os.makedirs(local_cache_dir, exist_ok=True)
    if os.path.exists(os.path.join(local_cache_dir, "config.json")):
        print(f"Loading from cache: {local_cache_dir}")
        processor = DonutProcessor.from_pretrained(local_cache_dir)
        model = VisionEncoderDecoderModel.from_pretrained(local_cache_dir)
    else:
        print(f"Downloading: {model_name}")
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        print(f"Saving to cache: {local_cache_dir}")
        processor.save_pretrained(local_cache_dir)
        model.save_pretrained(local_cache_dir)
    return processor, model


def find_image_path(image_name: str, images_dir: str) -> str:
    exact = os.path.join(images_dir, image_name)
    if os.path.exists(exact):
        return exact

    stem = os.path.splitext(image_name)[0]
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        p = os.path.join(images_dir, f"{stem}{ext}")
        if os.path.exists(p):
            return p

    matches = glob.glob(os.path.join(images_dir, f"*{stem}*"))
    return matches[0] if matches else exact


def get_image_key(item: Dict) -> str:
    # robust: supports both file_name and image_name
    if "file_name" in item:
        return "file_name"
    if "image_name" in item:
        return "image_name"
    raise KeyError("Neither 'file_name' nor 'image_name' exists in the JSONL item.")


# -----------------------
# Dataset + Collator
# -----------------------
class TestDataset(Dataset):
    def __init__(self, jsonl_path: str, image_dir: str, processor: DonutProcessor):
        self.processor = processor
        self.image_dir = Path(image_dir)
        self.data = load_jsonl(jsonl_path)
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self): return len(self.data)

    def format_gt(self, item: Dict) -> str:
        key = get_image_key(item)
        target = {k: v for k, v in item.items() if k != key}
        return json.dumps(target, ensure_ascii=False, separators=(",", ":"))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        key = get_image_key(item)
        image_path = Path(find_image_path(item[key], str(self.image_dir)))
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        return {"pixel_values": pixel_values.to(torch.float32), "sample_index": idx}


class Collator:
    def __call__(self, batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "sample_index": torch.tensor([b["sample_index"] for b in batch], dtype=torch.long),
        }


# -----------------------
# Inference
# -----------------------
@torch.no_grad()
def infer(model, dataset, processor, device, start_token, max_length, num_beams, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=Collator())
    start_ids = processor.tokenizer(start_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device).long()

    out = []
    for batch in tqdm(loader, desc="Infer (Donut base)"):
        pixel_values = batch["pixel_values"].to(device)
        idxs = batch["sample_index"].cpu().numpy().tolist()

        prompt_ids = start_ids.repeat(pixel_values.size(0), 1)
        gen_ids = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=prompt_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        for i in range(len(gen_ids)):
            idx = idxs[i]
            item = dataset.data[idx]
            key = get_image_key(item)

            pred = processor.tokenizer.decode(gen_ids[i][len(prompt_ids[0]):], skip_special_tokens=True).strip()
            pred = pred.replace(start_token, "").strip()

            gt = dataset.format_gt(item).replace(start_token, "").strip()

            out.append({
                key: item[key],
                "prediction": pred,
                "ground_truth": gt,
                "sample_index": idx,
                "original_data": item.copy(),
            })
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(CONFIG["output_root"], f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    processor, model = download_and_cache_model(CONFIG["model_name"], CONFIG["local_model_dir"])
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(CONFIG["start_token"])
    model.generation_config = GenerationConfig(
        max_length=CONFIG["max_length"],
        num_beams=CONFIG["num_beams"],
        do_sample=False,
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids(CONFIG["start_token"]),
    )
    model.to(device).eval()

    ds = TestDataset(CONFIG["test_jsonl_path"], CONFIG["images_dir"], processor)
    preds = infer(
        model=model,
        dataset=ds,
        processor=processor,
        device=device,
        start_token=CONFIG["start_token"],
        max_length=min(CONFIG["max_length"], 512),
        num_beams=CONFIG["num_beams"],
        batch_size=CONFIG["batch_size"],
    )

    pred_path = os.path.join(out_dir, "predictions_test.jsonl")
    save_jsonl(preds, pred_path)

    p_txt = [normalize_text(x["prediction"]) for x in preds]
    g_txt = [normalize_text(x["ground_truth"]) for x in preds]
    cer_str = calculate_cer_percent(p_txt, g_txt, json_mode=False)
    cer_struct = calculate_cer_percent(p_txt, g_txt, json_mode=True)

    score_path = os.path.join(out_dir, "cer_scores.txt")
    with open(score_path, "w", encoding="utf-8") as f:
        f.write("DONUT BASE (SCHMUCK) INFERENCE RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {CONFIG['model_name']}\n")
        f.write(f"Test samples: {len(preds)}\n")
        f.write(f"Test CER (string): {cer_str:.4f}\n")
        f.write(f"Test CER (structured): {cer_struct:.4f}\n")

    print(f"\nSaved: {pred_path}")
    print(f"Saved: {score_path}")
    print(f"CER string: {cer_str:.4f}% | CER structured: {cer_struct:.4f}%")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
