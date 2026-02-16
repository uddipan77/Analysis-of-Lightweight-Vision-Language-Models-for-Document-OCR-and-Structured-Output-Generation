#!/usr/bin/env python3
"""
Finetune PaddleOCR-VL on the Staircase dataset with CER-based model selection.

ALIGNED WITH INVENTORY "CODE 2" LOGIC:
- Primary selection/test metric: CER(GT_json_string, prediction_stripped_for_cer)
- prediction_stripped_for_cer:
    normalize_unicode -> strip_chat_prefix -> maybe_unescape_once -> strip
- prediction_canonical (phi_style_first_json) is DEBUG ONLY.

OOM HARDENING (NO LoRA):
✅ ONE-TIME IMAGE DOWNSCALING (main lever): --max_image_side
✅ GRID-ALIGNED RESIZE to prevent "tokens/features mismatch": --size_multiple (default 14)
✅ Disable KV cache during generation + cap generation length: --max_new_tokens, use_cache=False
✅ Prefer memory-efficient SDPA kernels if available
✅ gradient_checkpointing_enable(use_reentrant=False)
✅ Optional attention implementation override: --attn_impl {sdpa,eager}
✅ Set PYTORCH_ALLOC_CONF=expandable_segments:True to reduce fragmentation
✅ Aggressive memory cleanup during validation/test
✅ bf16 autocast on forward (safe & helps memory)

CRITICAL BUGFIX (your error):
- Avoid post-tokenization truncation of encodings (can break image token/feature alignment).
  So we DO NOT slice enc tensors by max_seq_length anymore.
  If you need to limit length, do it via --max_new_tokens and (optionally) truncating GT string safely.
"""

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import gc
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageOps
import jiwer
import editdistance

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)


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
# Helpers: normalization / stripping / unescape / JSON / CER
# -----------------------------
def normalize_unicode(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.split())
    return text


def strip_chat_prefix(text: str) -> str:
    """
    Robustly remove any "User: ... Assistant:" wrappers.
    Also remove a leading 'OCR:' if it remains.
    """
    s = str(text).strip()
    low = s.lower()

    if "assistant:" in low:
        idx = low.rfind("assistant:")
        s = s[idx + len("assistant:"):].strip()

    if s.lower().startswith("ocr:"):
        s = s[len("ocr:"):].strip()

    return s


def maybe_unescape_once(s: str) -> str:
    """
    Some outputs are string-escaped once, e.g. lots of \\".
    Unescape ONCE conservatively if it looks heavily escaped.
    """
    s = str(s)
    if s.count('\\"') >= 4 or s.count('\\\\\"') >= 2:
        try:
            return json.loads(f'"{s}"')
        except Exception:
            return s
    return s


def prediction_stripped_for_cer(raw_text: str) -> str:
    """
    EXACTLY like code-2 pipeline:
      raw -> normalize -> strip_chat_prefix -> maybe_unescape_once -> strip
    """
    s = normalize_unicode(raw_text)
    s = strip_chat_prefix(s)
    s = maybe_unescape_once(s)
    return s.strip()


def phi_style_first_json(response: str) -> str:
    """
    DEBUG ONLY: Take FIRST complete JSON object if present.
    Not used for primary CER/selection.
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


def calculate_cer_macro_micro(
    predictions: List[str], targets: List[str]
) -> Tuple[float, float, Dict[str, int]]:
    """
    Macro CER = average of per-sample CER
    Micro CER = (S + D + I) / reference_chars aggregated
    """
    if not predictions or not targets or len(predictions) != len(targets):
        return 1.0, 1.0, {"substitutions": 0, "deletions": 0, "insertions": 0, "ref_chars": 0}

    per_sample = []
    total_sub = total_del = total_ins = 0
    total_ref_chars = 0

    for pred, target in zip(predictions, targets):
        target = "" if target is None else str(target)
        pred = "" if pred is None else str(pred)
        if len(target) == 0:
            continue

        try:
            per_sample.append(float(jiwer.cer(target, pred)))
        except Exception:
            per_sample.append(1.0)

        try:
            m = jiwer.process_characters(target, pred)
            sub = int(m.substitutions)
            dele = int(m.deletions)
            ins = int(m.insertions)
            hits = int(m.hits)

            total_sub += sub
            total_del += dele
            total_ins += ins
            total_ref_chars += (hits + sub + dele)
        except Exception:
            total_del += len(target)
            total_ins += len(pred)
            total_ref_chars += len(target)

    macro_cer = float(sum(per_sample) / len(per_sample)) if per_sample else 1.0
    micro_cer = float((total_sub + total_del + total_ins) / total_ref_chars) if total_ref_chars > 0 else 1.0

    return macro_cer, micro_cer, {
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "ref_chars": total_ref_chars,
    }


def calculate_structured_cer(predictions: List[str], targets: List[str]) -> float:
    """
    Optional debug metric: parse JSON and compute editdistance error rate (%).
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
# SDPA kernel preference
# -----------------------------
def configure_sdp_kernels():
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
        print("Configured torch.backends.cuda.sdp_kernel(mem_efficient=True, math=True, flash=False)")
        return
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print("Configured legacy SDP kernel toggles (mem_efficient=True, math=True, flash=False)")
    except Exception as e:
        print(f"Warning: could not configure SDP kernels: {e}")


# -----------------------------
# Model cache (bf16 + checkpointing)
# -----------------------------
def download_and_cache_model(
    model_name: str,
    local_cache_dir: str,
    device: torch.device,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_impl: Optional[str] = None,
):
    os.makedirs(local_cache_dir, exist_ok=True)
    config_path = os.path.join(local_cache_dir, "config.json")

    kwargs = dict(trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl

    if os.path.exists(config_path):
        print(f"Loading PaddleOCR-VL from local cache: {local_cache_dir}")
        processor = AutoProcessor.from_pretrained(local_cache_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(local_cache_dir, **kwargs)
    else:
        print(f"Downloading {model_name} to {local_cache_dir} ...")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        print("Saving to local cache...")
        processor.save_pretrained(local_cache_dir)
        model.save_pretrained(local_cache_dir)
        print("✅ Model cached.")

    model.to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("Enabled gradient checkpointing (use_reentrant=False).")
        except TypeError:
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing.")
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
# Dataset with GRID-ALIGNED ONE-TIME DOWNSCALING (fixes tokens/features mismatch)
# -----------------------------
class StaircasePaddleOCRDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        max_image_side: int = 768,
        resample: str = "bicubic",
        size_multiple: int = 14,
        max_gt_chars: int = 0,  # 0 disables; optional safety
    ):
        self.images_dir = Path(images_dir)
        self.max_image_side = int(max_image_side)
        self.resample = resample
        self.size_multiple = int(size_multiple)
        self.max_gt_chars = int(max_gt_chars)

        self.data: List[Dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        print(
            f"Loaded {len(self.data)} samples from {jsonl_path} | "
            f"max_image_side={self.max_image_side} size_multiple={self.size_multiple} max_gt_chars={self.max_gt_chars}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def _find_image_path(self, image_name: str) -> Path:
        exact = self.images_dir / image_name
        if exact.exists():
            return exact
        base = Path(image_name).stem
        candidates = list(self.images_dir.glob(f"*{base}*"))
        return candidates[0] if candidates else exact

    def _pil_resample(self):
        if self.resample == "nearest":
            return Image.NEAREST
        if self.resample == "bilinear":
            return Image.BILINEAR
        if self.resample == "lanczos":
            return Image.LANCZOS
        return Image.BICUBIC

    @staticmethod
    def _snap_down_to_multiple(x: int, m: int) -> int:
        if m <= 1:
            return max(1, x)
        return max(m, (x // m) * m)

    def _resize_grid_aligned(self, img: Image.Image) -> Image.Image:
        if self.max_image_side <= 0:
            return img

        w, h = img.size
        mx = max(w, h)

        # scale down (floor) to max_image_side
        if mx > self.max_image_side:
            scale = self.max_image_side / float(mx)
            new_w = max(1, int(w * scale))  # floor
            new_h = max(1, int(h * scale))
        else:
            new_w, new_h = w, h

        # snap DOWN to a multiple to avoid patch-grid rounding mismatch
        new_w = self._snap_down_to_multiple(new_w, self.size_multiple)
        new_h = self._snap_down_to_multiple(new_h, self.size_multiple)

        if (new_w, new_h) == (w, h):
            return img

        return img.resize((new_w, new_h), resample=self._pil_resample())

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        image_name = item["image_name"]
        img_path = self._find_image_path(image_name)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)          # IMPORTANT for consistent dimensions
        image = self._resize_grid_aligned(image)        # ✅ fix mismatch

        target_text = json_to_string_no_sort(item)

        # OPTIONAL: safer than token slicing (does NOT touch image tokens)
        if self.max_gt_chars and self.max_gt_chars > 0 and len(target_text) > self.max_gt_chars:
            target_text = target_text[: self.max_gt_chars]

        return {"image": image, "target_text": target_text, "image_name": image_name}


# -----------------------------
# Collate function (DO NOT TRUNCATE TOKEN TENSORS)
# -----------------------------
def make_collate_fn(processor, task_prompt: str):
    """
    IMPORTANT: DO NOT slice enc tensors after processor.apply_chat_template.
    That can break image token/feature alignment.
    """
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_encodings = []
        for sample in batch:
            image = sample["image"]
            target_text = sample["target_text"]

            messages = [
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": task_prompt}]},
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
# Evaluation: generate + CER (use_cache=False to avoid spikes)
# -----------------------------
def evaluate_cer(
    model,
    processor,
    dataset: StaircasePaddleOCRDataset,
    device: torch.device,
    max_samples: Optional[int] = None,
    deterministic: bool = True,
    max_new_tokens: int = 256,
) -> Tuple[float, float, float]:
    model.eval()

    preds_primary = []
    tgts = []

    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Validation / CER"):
            gt_text = ""
            try:
                sample = dataset[i]
                image = sample["image"]
                gt_text = sample["target_text"]

                messages = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPTS[TASK]}]}
                ]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
                ).to(device)

                gen_kwargs = {"max_new_tokens": max_new_tokens, "use_cache": False}
                if deterministic:
                    gen_kwargs.update(dict(temperature=0.0, do_sample=False, repetition_penalty=1.0))
                else:
                    gen_kwargs.update(dict(temperature=0.2, do_sample=True, top_p=0.9, repetition_penalty=1.05))

                outputs = model.generate(**inputs, **gen_kwargs)
                raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                pred_for_cer = prediction_stripped_for_cer(raw_text)

                preds_primary.append(pred_for_cer)
                tgts.append(gt_text)

                del inputs, outputs

                if device.type == "cuda" and (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                gc.collect()

            except torch.cuda.OutOfMemoryError:
                print(f"\n⚠️ CUDA OOM at validation sample {i}. Skipping and clearing cache.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                preds_primary.append("")
                tgts.append(gt_text)
                continue

            except ValueError as e:
                # Catch the specific mismatch too, so training continues
                if "Image features and image tokens do not match" in str(e):
                    print(f"\n⚠️ Token/feature mismatch at validation sample {i}. Skipping sample. Error: {e}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    preds_primary.append("")
                    tgts.append(gt_text)
                    continue
                raise

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    macro_cer, micro_cer, _counts = calculate_cer_macro_micro(preds_primary, tgts)
    structured_cer = calculate_structured_cer(preds_primary, tgts)
    return macro_cer, micro_cer, structured_cer


def save_test_predictions(
    model,
    processor,
    dataset: StaircasePaddleOCRDataset,
    device: torch.device,
    output_path: str,
    max_new_tokens: int = 256,
):
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    records = []
    cer_list = []

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Test predictions"):
            gt_text = ""
            image_name = ""
            try:
                sample = dataset[i]
                image = sample["image"]
                gt_text = sample["target_text"]
                image_name = sample["image_name"]

                messages = [
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPTS[TASK]}]}
                ]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
                ).to(device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    repetition_penalty=1.0,
                    use_cache=False,   # ✅ avoid KV cache VRAM spike
                )

                raw_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                raw_text_norm = normalize_unicode(raw_text)
                pred_for_cer = prediction_stripped_for_cer(raw_text_norm)

                stripped_dbg = strip_chat_prefix(raw_text_norm)
                canon_dbg = phi_style_first_json(stripped_dbg)

                cer_score = jiwer.cer(gt_text, pred_for_cer) if len(gt_text) > 0 else 0.0
                cer_list.append(float(cer_score))

                records.append(
                    {
                        "image_name": image_name,
                        "prediction_raw": raw_text_norm,
                        "prediction_stripped": pred_for_cer,
                        "prediction_canonical": canon_dbg,
                        "ground_truth": gt_text,
                        "cer_score": float(cer_score),
                    }
                )

                del inputs, outputs

                if device.type == "cuda" and (i + 1) % 5 == 0:
                    torch.cuda.empty_cache()
                gc.collect()

            except torch.cuda.OutOfMemoryError:
                print(f"\n⚠️ CUDA OOM at test sample {i}. Skipping and clearing cache.")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                records.append(
                    {
                        "image_name": image_name,
                        "prediction_raw": "OOM_ERROR",
                        "prediction_stripped": "",
                        "prediction_canonical": "",
                        "ground_truth": gt_text,
                        "cer_score": 1.0,
                    }
                )
                cer_list.append(1.0)
                continue

            except ValueError as e:
                if "Image features and image tokens do not match" in str(e):
                    print(f"\n⚠️ Token/feature mismatch at test sample {i}. Skipping sample. Error: {e}")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    records.append(
                        {
                            "image_name": image_name,
                            "prediction_raw": "TOKEN_FEATURE_MISMATCH",
                            "prediction_stripped": "",
                            "prediction_canonical": "",
                            "ground_truth": gt_text,
                            "cer_score": 1.0,
                        }
                    )
                    cer_list.append(1.0)
                    continue
                raise

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    preds = [r["prediction_stripped"] for r in records]
    gts = [r["ground_truth"] for r in records]
    test_macro_cer, test_micro_cer, counts = calculate_cer_macro_micro(preds, gts)
    test_med_cer = float(np.median(cer_list)) if cer_list else 1.0

    print(f"Saved {len(records)} test predictions to {output_path}")
    print(
        f"Test CER macro: {test_macro_cer:.6f} ({test_macro_cer*100:.2f}%) | "
        f"micro: {test_micro_cer:.6f} ({test_micro_cer*100:.2f}%) | "
        f"median(per-sample): {test_med_cer:.6f} ({test_med_cer*100:.2f}%)"
    )
    print(
        f"Micro counts: S={counts['substitutions']} D={counts['deletions']} "
        f"I={counts['insertions']} ref_chars={counts['ref_chars']}"
    )
    return test_macro_cer, test_micro_cer, test_med_cer


# -----------------------------
# Training loop (bf16 autocast + safer)
# -----------------------------
def train(
    model,
    processor,
    train_dataset: StaircasePaddleOCRDataset,
    val_dataset: StaircasePaddleOCRDataset,
    device: torch.device,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 1,
    lr: float = 5e-5,
    grad_accum_steps: int = 4,
    max_val_samples: int = 30,
    max_new_tokens: int = 256,
):
    os.makedirs(output_dir, exist_ok=True)

    collate_fn = make_collate_fn(processor, PROMPTS[TASK])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    )

    total_train_steps = num_epochs * len(train_loader) // max(1, grad_accum_steps)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_train_steps),
        num_training_steps=total_train_steps,
    )

    model.to(device)

    best_val_macro_cer = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'=' * 60}")

        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
            try:
                batch = {k: v.to(device) for k, v in batch.items()}

                # keep pixel_values bf16 if present
                if "pixel_values" in batch and isinstance(batch["pixel_values"], torch.Tensor):
                    batch["pixel_values"] = batch["pixel_values"].to(dtype=torch.bfloat16)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    outputs = model(**batch, use_cache=False)
                    loss = outputs.loss / grad_accum_steps

                loss.backward()
                running_loss += loss.item() * grad_accum_steps

                del outputs, loss

            except torch.cuda.OutOfMemoryError:
                print("\n⚠️ CUDA OOM during backward. Skipping this batch and clearing cache.")
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                continue

            if (step + 1) % grad_accum_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if device.type == "cuda" and (step + 1) % (grad_accum_steps * 8) == 0:
                    torch.cuda.empty_cache()
                gc.collect()

        avg_train_loss = running_loss / max(1, len(train_loader))
        print(f"Epoch {epoch} - train loss: {avg_train_loss:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        print("\nRunning validation (CER-based model selection)...")
        val_macro_cer, val_micro_cer, val_struct_cer = evaluate_cer(
            model,
            processor,
            val_dataset,
            device,
            max_samples=max_val_samples,
            deterministic=True,
            max_new_tokens=max_new_tokens,
        )

        print(f"Validation CER macro (PRIMARY): {val_macro_cer:.4f} ({val_macro_cer*100:.2f}%)")
        print(f"Validation CER micro:          {val_micro_cer:.4f} ({val_micro_cer*100:.2f}%)")
        print(f"Validation structured CER (debug): {val_struct_cer:.4f}%")

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(avg_train_loss),
                "val_macro_cer_primary": float(val_macro_cer),
                "val_micro_cer": float(val_micro_cer),
                "val_structured_cer_debug": float(val_struct_cer),
            }
        )

        last_dir = os.path.join(output_dir, "last_model")
        os.makedirs(last_dir, exist_ok=True)
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        if val_macro_cer < best_val_macro_cer:
            improvement = best_val_macro_cer - val_macro_cer if best_val_macro_cer < float("inf") else 0.0
            best_val_macro_cer = val_macro_cer
            best_epoch = epoch

            print(
                f"🎯 New BEST validation macro CER: {best_val_macro_cer:.4f} "
                f"({best_val_macro_cer*100:.2f}%), improvement {improvement:.4f}"
            )
            best_dir = os.path.join(output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
        else:
            print(
                f"No improvement. Best macro CER remains {best_val_macro_cer:.4f} "
                f"({best_val_macro_cer*100:.2f}%) from epoch {best_epoch}"
            )

        with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}, best val macro CER: {best_val_macro_cer:.4f} ({best_val_macro_cer*100:.2f}%)")
    return best_val_macro_cer


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Finetune PaddleOCR-VL on Staircase with CER-based selection")
    parser.add_argument("--data_dir", default="/home/woody/iwi5/iwi5298h/json_staircase")
    parser.add_argument("--image_dir", default="/home/woody/iwi5/iwi5298h/staircase_images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_val_samples", type=int, default=30)

    # memory safety knobs (no LoRA)
    parser.add_argument("--max_image_side", type=int, default=768, help="Downscale images by longest side.")
    parser.add_argument("--resample", default="bicubic", choices=["nearest", "bilinear", "bicubic", "lanczos"])
    parser.add_argument("--size_multiple", type=int, default=14, help="Snap resized W/H DOWN to a multiple (try 14 or 16).")

    # Generation cap
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Cap generation length for val/test.")

    # Optional safe GT truncation (char-based), does NOT touch image tokens
    parser.add_argument("--max_gt_chars", type=int, default=0, help="If >0, truncate GT JSON string to this many chars.")

    parser.add_argument("--model_name", default="PaddlePaddle/PaddleOCR-VL")
    parser.add_argument("--local_model_dir", default="/home/vault/iwi5/iwi5298h/models/PaddleOCR-VL")
    parser.add_argument("--output_root", default="/home/vault/iwi5/iwi5298h/models_image_text/paddleocr/stair")
    parser.add_argument(
        "--attn_impl",
        default=None,
        choices=[None, "sdpa", "eager"],
        help="Optional Transformers attention implementation override (try 'sdpa' or 'eager')",
    )
    args = parser.parse_args()

    configure_sdp_kernels()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"run_{timestamp}_cer")
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "model_name": args.model_name,
        "local_model_dir": args.local_model_dir,
        "data_dir": args.data_dir,
        "image_dir": args.image_dir,
        "output_dir": run_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_accum_steps": args.grad_accum_steps,
        "max_val_samples": args.max_val_samples,
        "task": TASK,
        "prompt": PROMPTS[TASK],
        "selection_metric": "val_macro_cer_primary = jiwer.cer(GT_json_string, prediction_stripped_for_cer)",
        "prediction_postprocess_for_cer": "normalize_unicode -> strip_chat_prefix -> maybe_unescape_once",
        "canonical_extraction": "debug_only",
        "attn_impl": args.attn_impl,
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "max_image_side": args.max_image_side,
        "resample": args.resample,
        "size_multiple": args.size_multiple,
        "max_new_tokens": args.max_new_tokens,
        "max_gt_chars": args.max_gt_chars,
        "critical_fix": "grid-aligned resize + remove post-tokenization truncation",
    }
    with open(os.path.join(run_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("PADDLEOCR-VL FINETUNING - STAIRCASE DATASET")
    print("BEST MODEL SELECTION: CER(GT JSON string vs prediction_stripped_for_cer)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Local cache: {args.local_model_dir}")
    print(f"Output run dir: {run_dir}")
    print(f"Prompt task: {TASK} -> '{PROMPTS[TASK]}'")
    print(f"Attention impl override: {args.attn_impl}")
    print(f"Image downscale max side: {args.max_image_side}")
    print(f"Resize snap multiple: {args.size_multiple}")
    print(f"Max new tokens (val/test): {args.max_new_tokens}")
    if args.max_gt_chars and args.max_gt_chars > 0:
        print(f"GT char cap enabled: {args.max_gt_chars}")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    processor, model = download_and_cache_model(
        args.model_name,
        args.local_model_dir,
        device=device,
        torch_dtype=torch.bfloat16,
        attn_impl=args.attn_impl,
    )

    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    val_jsonl = os.path.join(args.data_dir, "val.jsonl")
    test_jsonl = os.path.join(args.data_dir, "test.jsonl")

    train_dataset = StaircasePaddleOCRDataset(
        train_jsonl, args.image_dir, args.max_image_side, args.resample, args.size_multiple, args.max_gt_chars
    )
    val_dataset = StaircasePaddleOCRDataset(
        val_jsonl, args.image_dir, args.max_image_side, args.resample, args.size_multiple, args.max_gt_chars
    )
    test_dataset = StaircasePaddleOCRDataset(
        test_jsonl, args.image_dir, args.max_image_side, args.resample, args.size_multiple, args.max_gt_chars
    )

    best_val_macro_cer = train(
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
        max_new_tokens=args.max_new_tokens,
    )

    best_model_dir = os.path.join(run_dir, "best_model")
    if os.path.exists(best_model_dir):
        print(f"\nLoading best model from {best_model_dir} for test evaluation...")
        processor = AutoProcessor.from_pretrained(best_model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            best_model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(device)
        if hasattr(model, "config"):
            model.config.use_cache = False  # keep safe
    else:
        print("\nBest model directory not found, using last trained model.")

    test_pred_path = os.path.join(run_dir, "test_predictions.jsonl")
    test_macro_cer, test_micro_cer, test_med_cer = save_test_predictions(
        model, processor, test_dataset, device, test_pred_path, max_new_tokens=args.max_new_tokens
    )

    summary_path = os.path.join(run_dir, "final_CER_scores.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PaddleOCR-VL FINETUNING - STAIRCASE DATASET\n")
        f.write("=" * 60 + "\n")
        f.write("Selection metric: CER(GT JSON string vs prediction_stripped_for_cer)\n")
        f.write("Test metric:      CER(GT JSON string vs prediction_stripped_for_cer)\n")
        f.write("Canonical extraction: debug only\n")
        f.write("Fixes: grid-aligned resize + no post-tokenization truncation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best validation macro CER: {best_val_macro_cer:.6f} ({best_val_macro_cer*100:.2f}%)\n")
        f.write(f"Test macro CER:           {test_macro_cer:.6f} ({test_macro_cer*100:.2f}%)\n")
        f.write(f"Test micro CER:           {test_micro_cer:.6f} ({test_micro_cer*100:.2f}%)\n")
        f.write(f"Test median CER:          {test_med_cer:.6f} ({test_med_cer*100:.2f}%)\n")

    print("\n" + "=" * 60)
    print("FINAL RESULTS - STAIRCASE")
    print("=" * 60)
    print(f"Best validation macro CER: {best_val_macro_cer:.6f} ({best_val_macro_cer*100:.2f}%)")
    print(f"Test macro CER:           {test_macro_cer:.6f} ({test_macro_cer*100:.2f}%)")
    print(f"Test micro CER:           {test_micro_cer:.6f} ({test_micro_cer*100:.2f}%)")
    print(f"Test median per-sample CER:{test_med_cer:.6f} ({test_med_cer*100:.2f}%)")
    print(f"\nRun directory: {run_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
