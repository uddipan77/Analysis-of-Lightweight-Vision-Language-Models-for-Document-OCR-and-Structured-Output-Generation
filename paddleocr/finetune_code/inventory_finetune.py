#!/usr/bin/env python3
"""
Finetune PaddleOCR-VL on the Inventory dataset with CER-based model selection.

UPDATED METRIC (recommended):
- Best model selection uses CER between:
    GT (canonical JSON string) vs prediction_stripped_for_cer
- Test evaluation uses the same CER.
- prediction_canonical is kept only for debugging/logging.

Why:
- phi_style_first_json() can "hide" failures by extracting a tiny JSON fragment.
- Using prediction_stripped aligns selection with the real task: emit full JSON.

Memory-safety:
- Load model in bfloat16
- Enable gradient checkpointing
- Disable use_cache during training
- Batch size = 1
- No manual truncation of token sequences
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from PIL import Image
import jiwer
import editdistance

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)
from torch.utils.tensorboard import SummaryWriter


# -----------------------------
# Constants
# -----------------------------

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

TASK = "ocr"


# -----------------------------
# Helpers: text / JSON / CER
# -----------------------------

def normalize_unicode(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.split())
    return text


def strip_chat_prefix(text: str) -> str:
    """
    PaddleOCR-VL sometimes returns:
        "User: OCR: Assistant: <content>"
    Remove that prefix robustly.
    """
    s = str(text).strip()
    lowered = s.lower()
    if lowered.startswith("user:") and "assistant:" in lowered:
        idx = lowered.rfind("assistant:")
        s = s[idx + len("assistant:") :].strip()
    return s


def maybe_unescape_once(s: str) -> str:
    """
    Some predictions look like they contain JSON-escaped quotes, e.g.:
        Anthropologisch...\\",\\"Inventarnummer\\": ...
    That means the model output is "string-escaped" once.

    We do a conservative one-time unescape ONLY if it looks heavily escaped.
    """
    s = str(s)
    # Heuristic: many occurrences of \" suggests it's escaped content.
    if s.count('\\"') >= 4 or s.count('\\\\\"') >= 2:
        try:
            # json.loads on a quoted string will unescape once safely
            return json.loads(f'"{s}"')
        except Exception:
            return s
    return s


def prediction_stripped_for_cer(raw_text: str) -> str:
    """
    Final text used for CER:
      raw decode -> normalize -> strip chat prefix -> maybe unescape once -> strip
    """
    s = normalize_unicode(raw_text)
    s = strip_chat_prefix(s)
    s = maybe_unescape_once(s)
    return s.strip()


def phi_style_first_json(response: str) -> str:
    """
    Debug-only: try to extract the first valid JSON object from response.
    (NOT used for model selection CER anymore.)
    """
    response = str(response).strip()
    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]

        try:
            brace_count = 0
            first_end = -1
            for i, ch in enumerate(json_str):
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        first_end = i + 1
                        break
            if first_end > 0:
                first_json = json_str[:first_end]
                parsed = json.loads(first_json)
                return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            pass

        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            return response

    return response


def dict_without_image_name(obj: Dict) -> Dict:
    return {k: v for k, v in obj.items() if k != "image_name"}


def json_to_string_no_sort(obj: Dict) -> str:
    clean = dict_without_image_name(obj)
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def calculate_structured_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Debug metric: parse JSON if possible, compute editdistance error rate (%).
    """
    def _parse_json(s):
        try:
            return json.loads(s)
        except Exception:
            return s

    total_chars = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        pred_json = _parse_json(pred)
        target_json = _parse_json(target)

        if isinstance(pred_json, dict):
            pred_str = json.dumps(pred_json, ensure_ascii=False, separators=(",", ":"))
        else:
            pred_str = str(pred_json)

        if isinstance(target_json, dict):
            target_str = json.dumps(target_json, ensure_ascii=False, separators=(",", ":"))
        else:
            target_str = str(target_json)

        total_chars += len(target_str)
        total_errors += editdistance.eval(pred_str, target_str)

    return (total_errors / total_chars * 100.0) if total_chars > 0 else 0.0


# -----------------------------
# Model cache (bf16 + checkpointing)
# -----------------------------

def download_and_cache_model(
    model_name: str,
    local_cache_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    os.makedirs(local_cache_dir, exist_ok=True)
    config_path = os.path.join(local_cache_dir, "config.json")

    if os.path.exists(config_path):
        print(f"Loading PaddleOCR-VL from local cache: {local_cache_dir}")
        processor = AutoProcessor.from_pretrained(local_cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            local_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
    else:
        print(f"Downloading {model_name} to {local_cache_dir} ...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        print("Saving to local cache...")
        processor.save_pretrained(local_cache_dir)
        model.save_pretrained(local_cache_dir)
        print("✅ Model cached.")

    model.to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing on model.")
        except Exception as e:
            print(f"Warning: could not enable gradient checkpointing: {e}")

    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
            print("Disabled use_cache for training.")
        except Exception:
            pass

    return processor, model


# -----------------------------
# Dataset (Inventory)
# -----------------------------

class InventoryPaddleOCRDataset(Dataset):
    def __init__(self, jsonl_path: str, images_dir: str):
        self.images_dir = Path(images_dir)
        self.data: List[Dict] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        image_name = item["image_name"]
        img_path = self._find_image_path(image_name)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        target_text = json_to_string_no_sort(item)

        return {"image": image, "target_text": target_text, "image_name": image_name}

    def _find_image_path(self, image_name: str) -> Path:
        exact = self.images_dir / image_name
        if exact.exists():
            return exact
        base = Path(image_name).stem
        candidates = list(self.images_dir.glob(f"*{base}*"))
        if candidates:
            return candidates[0]
        return exact


# -----------------------------
# Collate function
# -----------------------------

def make_collate_fn(processor, task_prompt: str):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_encodings = []
        for sample in batch:
            image = sample["image"]
            target_text = sample["target_text"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": task_prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]

            enc = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            batch_encodings.append(enc)

        merged: Dict[str, torch.Tensor] = {}
        for k in batch_encodings[0].keys():
            merged[k] = torch.cat([be[k] for be in batch_encodings], dim=0)

        merged["labels"] = merged["input_ids"].clone()
        return merged

    return collate


# -----------------------------
# Evaluation: generate + CER (selection uses stripped)
# -----------------------------

def evaluate_cer(
    model,
    processor,
    dataset: InventoryPaddleOCRDataset,
    device: torch.device,
    max_samples: Optional[int] = None,
    deterministic: bool = True,
) -> Tuple[float, float]:
    """
    Returns:
      - val_avg_cer: CER(gt_json, prediction_stripped_for_cer)
      - val_struct_cer_debug: structured CER (%) computed on extracted canonical JSON (debug only)
    """
    model.eval()
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    cer_sum = 0.0
    cer_count = 0

    # Debug structured CER (still computed, but not used for selection)
    preds_canon = []
    tgts = []

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Validation / CER"):
            sample = dataset[i]
            image = sample["image"]
            gt_text = sample["target_text"]  # canonical GT JSON string

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPTS[TASK]},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            gen_kwargs = {"max_new_tokens": 512, "use_cache": True}
            if deterministic:
                gen_kwargs.update(dict(temperature=0.0, do_sample=False, repetition_penalty=1.0))
            else:
                gen_kwargs.update(dict(temperature=0.2, do_sample=True, top_p=0.9, repetition_penalty=1.05))

            outputs = model.generate(**inputs, **gen_kwargs)

            raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            pred_for_cer = prediction_stripped_for_cer(raw_text)

            # Primary selection metric
            if len(gt_text) > 0:
                cer_sum += jiwer.cer(gt_text, pred_for_cer)
                cer_count += 1

            # Debug-only canonical extraction
            stripped_dbg = strip_chat_prefix(normalize_unicode(raw_text))
            canon_dbg = phi_style_first_json(stripped_dbg)
            preds_canon.append(canon_dbg)
            tgts.append(gt_text)

            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()

    avg_cer = cer_sum / cer_count if cer_count > 0 else 1.0
    struct_cer_debug = calculate_structured_cer(preds_canon, tgts)

    return avg_cer, struct_cer_debug


def save_test_predictions(
    model,
    processor,
    dataset: InventoryPaddleOCRDataset,
    device: torch.device,
    output_path: str,
):
    """
    Save JSONL:
      - prediction_raw
      - prediction_stripped (postprocessed used for CER)
      - prediction_canonical (debug only)
      - CER uses: jiwer.cer(ground_truth, prediction_stripped)
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    cer_list = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Test predictions"):
            sample = dataset[i]
            image = sample["image"]
            gt_text = sample["target_text"]
            image_name = sample["image_name"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": PROMPTS[TASK]},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.0,
                use_cache=True,
            )

            raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            pred_for_cer = prediction_stripped_for_cer(raw_text)

            # Debug canonical
            stripped_dbg = strip_chat_prefix(normalize_unicode(raw_text))
            canon_dbg = phi_style_first_json(stripped_dbg)

            cer_score = jiwer.cer(gt_text, pred_for_cer) if len(gt_text) > 0 else 0.0
            cer_list.append(float(cer_score))

            rec = {
                "image_name": image_name,
                "prediction_raw": normalize_unicode(raw_text),
                "prediction_stripped": pred_for_cer,          # USED FOR CER
                "prediction_canonical": canon_dbg,            # DEBUG ONLY
                "ground_truth": gt_text,
                "cer_score": float(cer_score),                # CER(gt, prediction_stripped)
            }
            records.append(rec)

            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_cer = float(np.mean(cer_list)) if cer_list else 1.0
    med_cer = float(np.median(cer_list)) if cer_list else 1.0

    print(f"Saved {len(records)} test predictions to {output_path}")
    return avg_cer, med_cer


# -----------------------------
# Training loop
# -----------------------------

def train(
    model,
    processor,
    train_dataset: InventoryPaddleOCRDataset,
    val_dataset: InventoryPaddleOCRDataset,
    device: torch.device,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 1,
    lr: float = 5e-5,
    grad_accum_steps: int = 4,
    max_val_samples: int = 30,
    tb_writer: SummaryWriter = None,
):
    os.makedirs(output_dir, exist_ok=True)

    collate_fn = make_collate_fn(processor, PROMPTS[TASK])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    total_train_steps = num_epochs * len(train_loader) // max(1, grad_accum_steps)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_train_steps),
        num_training_steps=total_train_steps,
    )

    model.to(device)

    best_val_cer = float("inf")
    best_epoch = -1
    history = []

    global_step = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        model.train()
        running_loss = 0.0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            running_loss += loss.item() * grad_accum_steps

            if (step + 1) % grad_accum_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} - train loss: {avg_train_loss:.4f}")

        print("\nRunning validation (CER-based model selection)...")
        val_avg_cer, val_struct_cer_debug = evaluate_cer(
            model,
            processor,
            val_dataset,
            device,
            max_samples=max_val_samples,
            deterministic=True,
        )

        print(f"Validation CER (PRIMARY, stripped vs GT): {val_avg_cer:.4f} ({val_avg_cer*100:.2f}%)")
        print(f"Validation structured CER (DEBUG, canonical extraction): {val_struct_cer_debug:.4f}%")

        if tb_writer is not None:
            tb_writer.add_scalar("train_loss", avg_train_loss, epoch)
            tb_writer.add_scalar("eval_cer_primary", val_avg_cer, epoch)
            tb_writer.add_scalar("eval_cer_primary_pct", val_avg_cer * 100, epoch)
            tb_writer.add_scalar("eval_structured_cer_debug", val_struct_cer_debug, epoch)
            tb_writer.flush()

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_cer_primary_stripped": float(val_avg_cer),
                "val_structured_cer_debug": float(val_struct_cer_debug),
            }
        )

        last_dir = os.path.join(output_dir, "last_model")
        os.makedirs(last_dir, exist_ok=True)
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        if val_avg_cer < best_val_cer:
            improvement = best_val_cer - val_avg_cer if best_val_cer < float("inf") else 0.0
            best_val_cer = val_avg_cer
            best_epoch = epoch

            print(
                f"🎯 New BEST validation CER (primary): {best_val_cer:.4f} "
                f"({best_val_cer*100:.2f}%), improvement {improvement:.4f}"
            )
            best_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
        else:
            print(
                f"No improvement. Best CER remains {best_val_cer:.4f} "
                f"({best_val_cer*100:.2f}%) from epoch {best_epoch}"
            )

        with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}, best val CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)")
    return best_val_cer


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Finetune PaddleOCR-VL on Inventory dataset with CER-based model selection"
    )
    parser.add_argument("--train_jsonl", default="/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl")
    parser.add_argument("--val_jsonl", default="/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl")
    parser.add_argument("--test_jsonl", default="/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl")
    parser.add_argument("--image_dir", default="/home/woody/iwi5/iwi5298h/inventory_images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_val_samples", type=int, default=30)
    parser.add_argument("--model_name", default="PaddlePaddle/PaddleOCR-VL")
    parser.add_argument("--local_model_dir", default="/home/vault/iwi5/iwi5298h/models/PaddleOCR-VL")
    parser.add_argument("--output_root", default="/home/vault/iwi5/iwi5298h/models_image_text/paddleocr/inven")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"run_{timestamp}_cer")
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "model_name": args.model_name,
        "local_model_dir": args.local_model_dir,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "test_jsonl": args.test_jsonl,
        "image_dir": args.image_dir,
        "output_dir": run_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_accum_steps": args.grad_accum_steps,
        "max_val_samples": args.max_val_samples,
        "task": TASK,
        "prompt": PROMPTS[TASK],
        "selection_metric": "jiwer.cer(GT_json_string, prediction_stripped_for_cer)",
        "prediction_postprocess_for_cer": "strip_chat_prefix + optional one-time unescape",
        "canonical_extraction": "debug_only_not_for_selection",
    }
    with open(os.path.join(run_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("PADDLEOCR-VL FINETUNING - INVENTORY DATASET")
    print("BEST MODEL SELECTION: CER(GT JSON string vs prediction_stripped_for_cer)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Local cache: {args.local_model_dir}")
    print(f"Output run dir: {run_dir}")
    print(f"Prompt task: {TASK} -> '{PROMPTS[TASK]}'")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor, model = download_and_cache_model(
        args.model_name,
        args.local_model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
    )

    train_dataset = InventoryPaddleOCRDataset(args.train_jsonl, args.image_dir)
    val_dataset = InventoryPaddleOCRDataset(args.val_jsonl, args.image_dir)
    test_dataset = InventoryPaddleOCRDataset(args.test_jsonl, args.image_dir)

    tb_log_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logging to: {tb_log_dir}")

    best_val_cer = train(
        model,
        processor,
        train_dataset,
        val_dataset,
        device,
        output_dir=run_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        max_val_samples=args.max_val_samples,
        tb_writer=tb_writer,
    )
    tb_writer.close()

    best_model_dir = os.path.join(run_dir, "best_model")
    if os.path.exists(best_model_dir):
        print(f"\nLoading best model from {best_model_dir} for test evaluation...")
        processor = AutoProcessor.from_pretrained(best_model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            best_model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device)
        if hasattr(model, "config"):
            model.config.use_cache = True
    else:
        print("\nBest model directory not found, using last trained model.")

    test_pred_path = os.path.join(run_dir, "test_predictions.jsonl")
    test_avg_cer, test_med_cer = save_test_predictions(
        model, processor, test_dataset, device, test_pred_path
    )

    summary_path = os.path.join(run_dir, "final_CER_scores.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PaddleOCR-VL FINETUNING - INVENTORY DATASET\n")
        f.write("=" * 60 + "\n")
        f.write("Selection metric: CER(GT JSON string vs prediction_stripped_for_cer)\n")
        f.write("Test metric:      CER(GT JSON string vs prediction_stripped_for_cer)\n")
        f.write("Canonical extraction: debug only\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best validation CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)\n")
        f.write(f"Test average CER:    {test_avg_cer:.4f} ({test_avg_cer*100:.2f}%)\n")
        f.write(f"Test median CER:     {test_med_cer:.4f} ({test_med_cer*100:.2f}%)\n")

    print("\n" + "=" * 60)
    print("FINAL RESULTS - INVENTORY")
    print("=" * 60)
    print(f"Best validation CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)")
    print(f"Test average CER:    {test_avg_cer:.4f} ({test_avg_cer*100:.2f}%)")
    print(f"Test median CER:     {test_med_cer:.4f} ({test_med_cer*100:.2f}%)")
    print(f"\nRun directory: {run_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
