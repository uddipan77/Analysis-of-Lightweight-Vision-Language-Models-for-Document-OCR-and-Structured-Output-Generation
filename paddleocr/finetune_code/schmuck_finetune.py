#!/usr/bin/env python3
"""
Finetune PaddleOCR-VL on the Schmuck dataset with CER-based model selection
(ALIGNED to Inventory Code-2 logic + CUDA OOM hardening).

Primary logic (aligned to Inventory Code-2):
- Selection metric (validation): CER( GT canonical JSON string  vs  prediction_stripped_for_cer(raw_decode) )
- Test evaluation uses the SAME CER.
- prediction_canonical (phi-style first JSON) is DEBUG ONLY (not used for selection).

Memory efficiency:
- JSONL is NOT fully loaded into RAM.
  Instead we build a lightweight line-offset index (or load a cached .idx file).
- Validation/test do not accumulate large lists in memory:
  compute running CER stats; test predictions are streamed to disk line-by-line.

CUDA OOM hardening (keeps the same finetune/CER concept):
- bf16 weights
- gradient checkpointing
- disable use_cache during training
- bf16 autocast during training forward (reduces activation memory)
- optimizer.zero_grad(set_to_none=True) (reduces allocator pressure)
- free tensors promptly; periodic empty_cache in eval/test
- try/except torch.cuda.OutOfMemoryError in training to skip offending batch safely
- prefer memory-efficient SDPA kernels + allocator config to reduce fragmentation

IMPORTANT FIX ADDED (your request):
- One-time image downscale in Dataset (train/val/test):
  max(width,height) <= --max_image_side (default 768; try 512 if still OOM)
"""

import os
# Reduce fragmentation (must be set before significant CUDA allocations)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import gc
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
from PIL import Image, ImageOps
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
# SDP kernel preference (OOM hardening)
# -----------------------------

def configure_sdp_kernels():
    """
    Prefer memory-efficient SDPA kernels to reduce backward spikes.
    Works across torch versions.
    """
    try:
        # Newer API
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
        print("Configured torch.backends.cuda.sdp_kernel(mem_efficient=True, math=True, flash=False)")
        return
    except Exception:
        pass

    try:
        # Older API
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        print("Configured legacy SDP kernel toggles (mem_efficient=True, math=True, flash=False)")
    except Exception as e:
        print(f"Warning: could not configure SDP kernels: {e}")


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
    Robust stripping aligned to Inventory Code-2 / aligned Staircase script.

    PaddleOCR-VL may return:
      - "User: OCR: Assistant: <content>"
      - "User: ... Assistant: OCR: Assistant: <content>"
      - "OCR: Assistant: <content>"
    We strip everything up to the LAST 'assistant:' and then remove a leading 'OCR:' if present.
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
    Conservative one-time unescape if output appears "string-escaped" once.
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
    Final text used for CER (ALIGNED to Inventory Code-2):
      raw decode -> normalize -> strip chat prefix -> maybe unescape once -> strip
    """
    s = normalize_unicode(raw_text)
    s = strip_chat_prefix(s)
    s = maybe_unescape_once(s)
    return s.strip()


def phi_style_first_json(response: str) -> str:
    """
    Debug-only: take the first valid JSON object from response and return as JSON string.
    NOT used for model selection/test CER.
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
    """
    Remove image key from GT JSON.
    Works for both 'image_name' and 'file_name'.
    """
    return {k: v for k, v in obj.items() if k not in ("image_name", "file_name")}


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
    attn_impl: Optional[str] = None,  # try "sdpa" or "eager" if your transformers supports it
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

    # Gradient checkpointing (try use_reentrant=False if supported)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("Enabled gradient checkpointing (use_reentrant=False).")
        except TypeError:
            model.gradient_checkpointing_enable()
            print("Enabled gradient checkpointing.")
        except Exception as e:
            print(f"Warning: could not enable gradient checkpointing: {e}")

    # Disable cache for training
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
            print("Disabled use_cache for training.")
        except Exception:
            pass

    return processor, model


# -----------------------------
# Memory-efficient JSONL dataset (offset index) + ONE-TIME IMAGE RESIZE
# -----------------------------

class SchmuckPaddleOCRJsonlIndexedDataset(Dataset):
    """
    Memory-efficient dataset:
    - builds/loads a file-offset index to allow random access without storing all JSON lines.
    - keeps only offsets + a tiny header object (for image_key detection).

    IMPORTANT FIX:
    - resize each image once in __getitem__ so high-res images don't explode VRAM.
      max(width,height) <= max_image_side
    """

    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        index_path: Optional[str] = None,
        build_index: bool = True,
        force_rebuild: bool = False,
        # NEW: resize control
        max_image_side: int = 768,
        resize_resample: str = "bicubic",  # "bicubic" or "bilinear"
    ):
        self.jsonl_path = str(jsonl_path)
        self.images_dir = Path(images_dir)
        self.index_path = index_path

        self.max_image_side = int(max_image_side)
        self.resize_resample = resize_resample.lower().strip()

        if self.index_path and os.path.exists(self.index_path) and not force_rebuild:
            self.offsets = self._load_index(self.index_path)
            print(f"Loaded index with {len(self.offsets)} offsets: {self.index_path}")
        else:
            if not build_index:
                raise ValueError(
                    "Index file not found and build_index=False. "
                    "Either provide an existing --index_path or allow building."
                )
            self.offsets = self._build_offsets(self.jsonl_path)
            if self.index_path:
                self._save_index(self.index_path, self.offsets)
                print(f"Saved index with {len(self.offsets)} offsets: {self.index_path}")
            else:
                print(f"Built index with {len(self.offsets)} offsets (not saved).")

        if len(self.offsets) == 0:
            raise ValueError(f"No data found in {self.jsonl_path}")

        # Detect image key from first record (read once)
        first = self._read_json_at(0)
        if "file_name" in first:
            self.image_key = "file_name"
        elif "image_name" in first:
            self.image_key = "image_name"
        else:
            raise KeyError(
                f"Expected 'file_name' or 'image_name' in JSON, got keys: {list(first.keys())}"
            )

        print(
            f"Dataset ready: {self.jsonl_path} | samples={len(self.offsets)} "
            f"| image_key='{self.image_key}' | max_image_side={self.max_image_side}"
        )

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict:
        item = self._read_json_at(idx)

        image_name = item[self.image_key]
        img_path = self._find_image_path(image_name)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        # ---- load + EXIF fix + one-time resize ----
        image = Image.open(img_path).convert("RGB")
        image = ImageOps.exif_transpose(image)  # safe orientation fix

        if self.max_image_side and self.max_image_side > 0:
            image = self._resize_max_side(image, self.max_image_side)

        target_text = json_to_string_no_sort(item)
        return {"image": image, "target_text": target_text, "image_name": image_name}

    def _resize_max_side(self, img: Image.Image, max_side: int) -> Image.Image:
        w, h = img.size
        if max(w, h) <= max_side:
            return img

        scale = max_side / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if self.resize_resample == "bilinear":
            resample = Image.BILINEAR
        else:
            resample = Image.BICUBIC

        return img.resize((new_w, new_h), resample=resample)

    def _read_json_at(self, idx: int) -> Dict:
        off = self.offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(off)
            line = f.readline()
        line = line.decode("utf-8").strip()
        return json.loads(line)

    def _find_image_path(self, image_name: str) -> Path:
        exact = self.images_dir / image_name
        if exact.exists():
            return exact
        base = Path(image_name).stem
        candidates = list(self.images_dir.glob(f"*{base}*"))
        if candidates:
            return candidates[0]
        return exact

    @staticmethod
    def _build_offsets(jsonl_path: str) -> List[int]:
        offsets: List[int] = []
        with open(jsonl_path, "rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(pos)
        return offsets

    @staticmethod
    def _save_index(index_path: str, offsets: List[int]) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            for off in offsets:
                f.write(str(off) + "\n")

    @staticmethod
    def _load_index(index_path: str) -> List[int]:
        offsets: List[int] = []
        with open(index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    offsets.append(int(line))
        return offsets


# -----------------------------
# Collate function (no truncation)
# -----------------------------

def make_collate_fn(processor, task_prompt: str):
    """
    IMPORTANT: Do NOT slice input_ids/attention_mask after processor,
    to avoid <image> token mismatch errors.
    """
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
# Evaluation: generate + CER (PRIMARY uses stripped)
# -----------------------------

def evaluate_cer(
    model,
    processor,
    dataset: SchmuckPaddleOCRJsonlIndexedDataset,
    device: torch.device,
    max_samples: Optional[int] = None,
    deterministic: bool = True,
    empty_cache_every: int = 20,
) -> Tuple[float, float]:
    """
    Returns:
      - val_avg_cer: CER(gt_json, prediction_stripped_for_cer)
      - val_struct_cer_debug: structured CER (%) on canonical JSON extraction (debug only)
    """
    model.eval()
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    cer_sum = 0.0
    cer_count = 0

    preds_canon_dbg: List[str] = []
    tgts: List[str] = []

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Validation / CER"):
            sample = dataset[i]
            image = sample["image"]
            gt_text = sample["target_text"]

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

            if len(gt_text) > 0:
                cer_sum += jiwer.cer(gt_text, pred_for_cer)
                cer_count += 1

            stripped_dbg = strip_chat_prefix(normalize_unicode(raw_text))
            canon_dbg = phi_style_first_json(stripped_dbg)
            preds_canon_dbg.append(canon_dbg)
            tgts.append(gt_text)

            del inputs, outputs

            if torch.cuda.is_available() and empty_cache_every and (i + 1) % empty_cache_every == 0:
                torch.cuda.empty_cache()

    avg_cer = cer_sum / cer_count if cer_count > 0 else 1.0
    struct_cer_debug = calculate_structured_cer(preds_canon_dbg, tgts)
    return avg_cer, struct_cer_debug


def save_test_predictions(
    model,
    processor,
    dataset: SchmuckPaddleOCRJsonlIndexedDataset,
    device: torch.device,
    output_path: str,
    empty_cache_every: int = 20,
) -> Tuple[float, float]:
    """
    Stream predictions to JSONL (no big in-RAM list).
    """
    model.eval()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cer_vals: List[float] = []

    with open(output_path, "w", encoding="utf-8") as f_out:
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

                stripped_dbg = strip_chat_prefix(normalize_unicode(raw_text))
                canon_dbg = phi_style_first_json(stripped_dbg)

                cer_score = jiwer.cer(gt_text, pred_for_cer) if len(gt_text) > 0 else 0.0
                cer_vals.append(float(cer_score))

                rec = {
                    "image_name": image_name,
                    "prediction_raw": normalize_unicode(raw_text),
                    "prediction_stripped": pred_for_cer,   # USED FOR CER
                    "prediction_canonical": canon_dbg,     # DEBUG ONLY
                    "ground_truth": gt_text,
                    "cer_score": float(cer_score),
                }
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

                del inputs, outputs

                if torch.cuda.is_available() and empty_cache_every and (i + 1) % empty_cache_every == 0:
                    torch.cuda.empty_cache()

    avg_cer = float(np.mean(cer_vals)) if cer_vals else 1.0
    med_cer = float(np.median(cer_vals)) if cer_vals else 1.0
    print(f"Saved {len(dataset)} test predictions to {output_path}")
    return avg_cer, med_cer


# -----------------------------
# Training loop (UNCHANGED)
# -----------------------------

def train(
    model,
    processor,
    train_dataset: SchmuckPaddleOCRJsonlIndexedDataset,
    val_dataset: SchmuckPaddleOCRJsonlIndexedDataset,
    device: torch.device,
    output_dir: str,
    num_epochs: int = 10,
    batch_size: int = 1,
    lr: float = 5e-5,
    grad_accum_steps: int = 4,
    max_val_samples: int = 30,
    tb_writer: Optional[SummaryWriter] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    collate_fn = make_collate_fn(processor, PROMPTS[TASK])

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
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
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training epoch {epoch}")):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            try:
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(**batch, use_cache=False)
                        loss = outputs.loss / grad_accum_steps
                else:
                    outputs = model(**batch, use_cache=False)
                    loss = outputs.loss / grad_accum_steps

                loss.backward()
                running_loss += loss.item() * grad_accum_steps

                del outputs, loss

            except torch.cuda.OutOfMemoryError:
                print("\n⚠️ CUDA OOM during training step. Skipping batch and clearing cache.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                gc.collect()
                continue

            if (step + 1) % grad_accum_steps == 0:
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if torch.cuda.is_available() and (step + 1) % 50 == 0:
                torch.cuda.empty_cache()

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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}, best val CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)")
    return best_val_cer


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Finetune PaddleOCR-VL on Schmuck dataset with CER-based model selection (Code-2 aligned, memory-efficient)"
    )
    parser.add_argument("--data_dir", default="/home/woody/iwi5/iwi5298h/json_schmuck")
    parser.add_argument("--image_dir", default="/home/woody/iwi5/iwi5298h/schmuck_images")

    # indexing for large JSONL
    parser.add_argument("--index_dir", default=None, help="Where to store/load .idx files. If None, uses data_dir.")
    parser.add_argument("--rebuild_index", action="store_true", help="Force rebuilding JSONL offset indices even if .idx exists.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_val_samples", type=int, default=30)

    parser.add_argument("--model_name", default="PaddlePaddle/PaddleOCR-VL")
    parser.add_argument("--local_model_dir", default="/home/vault/iwi5/iwi5298h/models/PaddleOCR-VL")
    parser.add_argument("--output_root", default="/home/vault/iwi5/iwi5298h/models_image_text/paddleocr/schmuck")

    # optional attention override (if transformers supports it)
    parser.add_argument("--attn_impl", default=None, choices=[None, "sdpa", "eager"],
                        help="Optional Transformers attention implementation override (try 'sdpa' or 'eager')")

    # NEW: one-time image resize
    parser.add_argument("--max_image_side", type=int, default=768,
                        help="Downscale images so max(width,height) <= this. Try 512 if still OOM.")
    parser.add_argument("--resize_resample", default="bicubic", choices=["bicubic", "bilinear"],
                        help="Resample method for resizing.")

    args = parser.parse_args()

    configure_sdp_kernels()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, f"run_{timestamp}_cer")
    os.makedirs(run_dir, exist_ok=True)

    # Resolve index paths
    index_dir = args.index_dir or args.data_dir
    os.makedirs(index_dir, exist_ok=True)

    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    val_jsonl = os.path.join(args.data_dir, "val.jsonl")
    test_jsonl = os.path.join(args.data_dir, "test.jsonl")

    train_idx = os.path.join(index_dir, "train.idx")
    val_idx = os.path.join(index_dir, "val.idx")
    test_idx = os.path.join(index_dir, "test.idx")

    # Save config
    config = {
        "model_name": args.model_name,
        "local_model_dir": args.local_model_dir,
        "data_dir": args.data_dir,
        "image_dir": args.image_dir,
        "index_dir": index_dir,
        "output_dir": run_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_accum_steps": args.grad_accum_steps,
        "max_val_samples": args.max_val_samples,
        "task": TASK,
        "prompt": PROMPTS[TASK],
        "selection_metric": "jiwer.cer(GT_json_string, prediction_stripped_for_cer)",
        "prediction_postprocess_for_cer": "normalize_unicode -> strip_chat_prefix(last assistant) -> maybe_unescape_once",
        "canonical_extraction": "debug_only_not_for_selection",
        "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
        "attn_impl": args.attn_impl,
        "max_image_side": args.max_image_side,
        "resize_resample": args.resize_resample,
    }
    with open(os.path.join(run_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("PADDLEOCR-VL FINETUNING - SCHMUCK DATASET")
    print("BEST MODEL SELECTION: CER(GT JSON string vs prediction_stripped_for_cer)")
    print("Memory-efficient JSONL indexing + streamed test writing + OOM hardening")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Local cache: {args.local_model_dir}")
    print(f"Output run dir: {run_dir}")
    print(f"Prompt task: {TASK} -> '{PROMPTS[TASK]}'")
    print(f"Attention impl override: {args.attn_impl}")
    print(f"Image resize: max_image_side={args.max_image_side} resample={args.resize_resample}")
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

    # TensorBoard
    tb_log_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logging to: {tb_log_dir}")

    # Datasets (offset-indexed) + resize
    train_dataset = SchmuckPaddleOCRJsonlIndexedDataset(
        train_jsonl,
        args.image_dir,
        index_path=train_idx,
        build_index=True,
        force_rebuild=args.rebuild_index,
        max_image_side=args.max_image_side,
        resize_resample=args.resize_resample,
    )
    val_dataset = SchmuckPaddleOCRJsonlIndexedDataset(
        val_jsonl,
        args.image_dir,
        index_path=val_idx,
        build_index=True,
        force_rebuild=args.rebuild_index,
        max_image_side=args.max_image_side,
        resize_resample=args.resize_resample,
    )
    test_dataset = SchmuckPaddleOCRJsonlIndexedDataset(
        test_jsonl,
        args.image_dir,
        index_path=test_idx,
        build_index=True,
        force_rebuild=args.rebuild_index,
        max_image_side=args.max_image_side,
        resize_resample=args.resize_resample,
    )

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

    # Load best model for test evaluation
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
            model.config.use_cache = True
    else:
        print("\nBest model directory not found, using last trained model.")

    test_pred_path = os.path.join(run_dir, "test_predictions.jsonl")
    test_avg_cer, test_med_cer = save_test_predictions(
        model, processor, test_dataset, device, test_pred_path
    )

    summary_path = os.path.join(run_dir, "final_CER_scores.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("PaddleOCR-VL FINETUNING - SCHMUCK DATASET\n")
        f.write("=" * 60 + "\n")
        f.write("Selection metric: CER(GT JSON string vs prediction_stripped_for_cer)\n")
        f.write("Test metric:      CER(GT JSON string vs prediction_stripped_for_cer)\n")
        f.write("Canonical extraction: debug only\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best validation CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)\n")
        f.write(f"Test average CER:    {test_avg_cer:.4f} ({test_avg_cer*100:.2f}%)\n")
        f.write(f"Test median CER:     {test_med_cer:.4f} ({test_med_cer*100:.2f}%)\n")

    print("\n" + "=" * 60)
    print("FINAL RESULTS - SCHMUCK")
    print("=" * 60)
    print(f"Best validation CER: {best_val_cer:.4f} ({best_val_cer*100:.2f}%)")
    print(f"Test average CER:    {test_avg_cer:.4f} ({test_avg_cer*100:.2f}%)")
    print(f"Test median CER:     {test_med_cer:.4f} ({test_med_cer*100:.2f}%)")
    print(f"\nRun directory: {run_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
