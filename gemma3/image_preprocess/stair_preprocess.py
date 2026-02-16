#!/usr/bin/env python3
# gemma3_staircase_finetune_A100_aug1_fixed_better_cer.py
# ✅ A100-OPTIMIZED with evaluation chunking + memory management
# ✅ Staircase dataset + data augmentation (factor=1 on train)
# ✅ More robust JSON extraction (recovers longest valid prefix)
# ✅ Canonical JSON (sorted keys) before CER → order-invariant
# ✅ Better CER metric (ignores prompt tokens)

import sys
import os

# ✅ CRITICAL: Enable logits return for compute_metrics
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import json
import shutil
from typing import List, Dict
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
import re
import jiwer
from datetime import datetime
from transformers import EarlyStoppingCallback
import numpy as np
import gc
from PIL import Image

# 🔁 Augmentation
import torchvision.transforms as transforms
import random
import tempfile

# ============================================================================
# ADDED: 4-step preprocessing (on-the-fly, no saving)
# ============================================================================

import cv2


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def letterbox_resize(img, target_size=(1024, 1024), pad_value=255):
    th, tw = target_size
    h, w = img.shape[:2]

    scale = min(tw / w, th / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left
    pad_top = (th - new_h) // 2
    pad_bot = th - new_h - pad_top

    border_val = pad_value if resized.ndim == 2 else (pad_value, pad_value, pad_value)

    return cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=border_val,
    )


def illumination_flatten(gray, bg_kernel_size=51):
    # EXACTLY like Code 2
    if bg_kernel_size % 2 == 0:
        bg_kernel_size += 1

    bg = cv2.GaussianBlur(gray, (bg_kernel_size, bg_kernel_size), 0)

    gray_f = gray.astype(np.float32)
    bg_f = np.clip(bg.astype(np.float32), 1.0, 255.0)

    norm = (gray_f / bg_f) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def clahe_on_luminance(bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def preprocess_exact_4steps(
    image_path,
    output_path=None,
    target_size=(1024, 1024),
    bg_kernel_size=51,
    denoise_method="bilateral",
):
    # Unicode-safe read (NOT a transformation)
    bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # 1) Illumination flattening (grayscale)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flat = illumination_flatten(gray, bg_kernel_size=bg_kernel_size)

    # 2) Gentle denoising
    if denoise_method == "bilateral":
        den = cv2.bilateralFilter(flat, d=7, sigmaColor=50, sigmaSpace=50)
    elif denoise_method == "nlm":
        den = cv2.fastNlMeansDenoising(
            flat, None, h=7, templateWindowSize=7, searchWindowSize=21
        )
    else:
        raise ValueError("denoise_method must be 'bilateral' or 'nlm'")

    # 3) CLAHE (luminance)
    den_bgr = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    clahe_bgr = clahe_on_luminance(den_bgr, clip_limit=2.0, tile_grid_size=(8, 8))
    out_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # 4) Letterbox resize
    out = letterbox_resize(out_gray, target_size=target_size, pad_value=255)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not imwrite_unicode(output_path, out):
            raise IOError(f"Could not write output image: {output_path}")

    return out


def preprocess_exact_4steps_pil(
    pil_rgb: Image.Image,
    target_size=(1024, 1024),
    bg_kernel_size=51,
    denoise_method="bilateral",
) -> Image.Image:
    """
    Same EXACT 4 processing steps as preprocess_exact_4steps(),
    but applied on-the-fly to a PIL image (no disk I/O, no saving).
    Output is a PIL RGB image ready to be consumed by Unsloth tokenizer/template.
    """
    pil_rgb = pil_rgb.convert("RGB")
    rgb = np.array(pil_rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1) Illumination flattening (grayscale)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flat = illumination_flatten(gray, bg_kernel_size=bg_kernel_size)

    # 2) Gentle denoising
    if denoise_method == "bilateral":
        den = cv2.bilateralFilter(flat, d=7, sigmaColor=50, sigmaSpace=50)
    elif denoise_method == "nlm":
        den = cv2.fastNlMeansDenoising(
            flat, None, h=7, templateWindowSize=7, searchWindowSize=21
        )
    else:
        raise ValueError("denoise_method must be 'bilateral' or 'nlm'")

    # 3) CLAHE (luminance)
    den_bgr = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    clahe_bgr = clahe_on_luminance(den_bgr, clip_limit=2.0, tile_grid_size=(8, 8))
    out_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # 4) Letterbox resize
    out = letterbox_resize(out_gray, target_size=target_size, pad_value=255)

    return Image.fromarray(out).convert("RGB")


# =====================================================================
# Lazy dataset: applies 4-step preprocessing on-the-fly in __getitem__
# =====================================================================

class StaircaseConversationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_items: List[Dict],
        make_conversation_fn,
        preprocess_fn,
        preprocess_cfg: Dict,
    ):
        self.items = raw_items
        self.make_conversation_fn = make_conversation_fn
        self.preprocess_fn = preprocess_fn
        self.preprocess_cfg = preprocess_cfg
        print(f"   \U0001f4ca Loaded {len(self.items)} valid samples")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        img = self.preprocess_fn(
            img,
            target_size=self.preprocess_cfg.get("target_size", (1024, 1024)),
            bg_kernel_size=self.preprocess_cfg.get("bg_kernel_size", 51),
            denoise_method=self.preprocess_cfg.get("denoise_method", "bilateral"),
        )
        return self.make_conversation_fn(sample, preprocessed_image=img)


class StaircaseGemma3Finetune:
    def __init__(self, model_path: str, augment_factor: int = 1):
        """Initialize with A100-optimized settings + augmentation."""
        print("Loading Gemma-3 vision model with Unsloth for STAIRCASE...")

        self.augment_factor = augment_factor

        # Temp dir for augmented images
        self.temp_dir = tempfile.mkdtemp(prefix="stair_gemma_aug_")
        print(f"Temporary directory for augmented images: {self.temp_dir}")
        print(f"Augmentation factor (per original image): {self.augment_factor}")

        cache_dir = "/home/hpc/iwi5/iwi5298h/Uddipan-Thesis/gemma3/unsloth_compiled_cache"
        if os.path.exists(cache_dir):
            print(f"Clearing Unsloth compiled cache: {cache_dir}")
            shutil.rmtree(cache_dir)

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
            local_files_only=True,
        )

        print("Model loaded - Unsloth auto-configured dtype and attention")

        # ✅ A100-optimized LoRA config (slightly more regularized)
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )

        print("Gemma-3 vision model loaded with A100-optimized LoRA config")

        # Preprocess config used everywhere
        self.preprocess_cfg = {
            "target_size": (1024, 1024),
            "bg_kernel_size": 51,
            "denoise_method": "bilateral",
        }

    # ----------------------- I/O HELPERS -----------------------

    def load_jsonl(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def save_jsonl(self, data: List[Dict], file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def dict_without_image_name(self, obj):
        # staircase JSON uses "image_name"; we also drop "image_path" helper.
        return {k: v for k, v in obj.items() if k not in ["image_name", "image_path"]}

    # 🔧 NEW: Canonicalize JSON so key order doesn't affect CER
    def _canonicalize_json(self, obj):
        """
        Recursively sort dict keys so that semantically identical JSON
        with different key ordering produces the same string.
        """
        if isinstance(obj, dict):
            return {k: self._canonicalize_json(obj[k]) for k in sorted(obj.keys())}
        elif isinstance(obj, list):
            return [self._canonicalize_json(x) for x in obj]
        else:
            return obj

    def json_to_string_readable(self, obj):
        clean_obj = self.dict_without_image_name(obj)
        canonical = self._canonicalize_json(clean_obj)
        # deterministic formatting (indent only for readability)
        return json.dumps(canonical, ensure_ascii=False, indent=2)

    # ----------------------- AUGMENTATION -----------------------

    def create_augmentation_transforms(self):
        """Document-friendly augmentation (NO rotation)."""
        augmentation_transforms = [
            transforms.ColorJitter(brightness=(0.9, 1.1)),
            transforms.ColorJitter(contrast=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
        ]
        return augmentation_transforms

    def augment_image(self, image_path: str, aug_id: int) -> str:
        """Create augmented version of image and save to temp_dir."""
        try:
            image = Image.open(image_path).convert("RGB")
            transforms_list = self.create_augmentation_transforms()

            # Randomly select 2–3 transforms
            num_transforms = random.randint(2, min(3, len(transforms_list)))
            selected_transforms = random.sample(transforms_list, num_transforms)

            augmented_image = image
            for t in selected_transforms:
                augmented_image = t(augmented_image)

            base_name = os.path.basename(image_path)
            name_wo_ext, ext = os.path.splitext(base_name)
            aug_name = f"{name_wo_ext}_aug{aug_id}{ext}"
            aug_path = os.path.join(self.temp_dir, aug_name)

            augmented_image.save(aug_path, quality=95)
            return aug_path
        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return None

    # ----------------------- JSON EXTRACTION -----------------------

    # 🔧 NEW: more robust JSON extraction that can recover the longest
    #         valid prefix when the model truncates the tail.
    def extract_json_from_response(self, response: str) -> Dict:
        """
        More robust JSON extraction for slightly truncated outputs.

        Strategy:
        1. Strip markdown fences.
        2. Try to parse the outermost {...} span.
        3. If that fails, walk backwards and find the longest prefix
           ending with '}' that is valid JSON.
        4. Fallback to nested-brace heuristic.
        5. Finally, try parsing the whole response.
        """
        if isinstance(response, list):
            response = response[0] if response else ""
        if response is None:
            return {}

        text = str(response).strip()
        if not text:
            return {}

        # Strip Markdown fences if present
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 3:
                body = parts[1]
                if body.lstrip().startswith("json"):
                    body = body[4:].strip()
                text = body.strip()

        def try_parse(candidate: str):
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None

        # Focus on the main JSON-ish core
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            core = text[start : end + 1]
        else:
            core = text

        # 1) Try full core
        parsed = try_parse(core)
        if parsed is not None:
            return parsed

        # 2) Walk backwards over '}' positions to find a valid prefix
        brace_positions = [m.start() for m in re.finditer(r"\}", core)]
        for pos in reversed(brace_positions):
            candidate = core[: pos + 1]
            parsed = try_parse(candidate)
            if parsed is not None:
                return parsed

        # 3) Fallback: nested balanced candidates, longest first
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, core, re.DOTALL)
        if matches:
            for match in sorted(matches, key=len, reverse=True):
                parsed = try_parse(match)
                if parsed is not None:
                    return parsed

        # 4) Last resort: whole text
        parsed = try_parse(text)
        if parsed is not None:
            return parsed

        return {}

    # ----------------------- CONVERSIONS -----------------------

    def convert_to_conversation(self, sample, preprocessed_image: Image.Image):
        # Simple English prompt, with explicit “do not drop keys”.
        instruction = (
            "You are an OCR model for historical staircase survey forms.\n\n"
            "Task:\n"
            "Given ONE image of a filled-in staircase form, read all printed text, "
            "handwritten notes and all checked/unchecked boxes and output a single JSON object "
            "that represents the complete form.\n\n"
            "Rules:\n"
            "- Return ONLY one valid JSON object, with no extra text before or after it.\n"
            "- Use exactly the same field names, nesting, accents, and capitalization as in the "
            "training JSON for this form type (e.g. keys like \"stair_type\", \"Name des Hauses\", "
            "\"Adresse\", \"LÄUFE\", \"GELÄNDER\", etc.).\n"
            "- Never drop a key that appears in the form’s JSON structure. If a field is empty on "
            "the form, still include it with an empty string \"\" (or false for an unchecked box).\n"
            "- Use booleans for checkbox options: true if the box is checked, false if it is empty.\n"
            "- Use strings for numbers and free-text fields (measurements, dates, names, notes).\n"
            "- Do NOT invent new fields."
        )

        gt_json_string = self.json_to_string_readable(sample)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": preprocessed_image},
                    {"type": "text", "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": gt_json_string},
                ],
            },
        ]
        return {"messages": conversation}

    # ----------------------- DATA PREP (WITH AUG) -----------------------

    def prepare_training_data_with_augmentation(
        self, jsonl_path: str, images_dir: str
    ) -> List[Dict]:
        """Prepare TRAIN data: original + augment_factor augmented copies."""
        data = self.load_jsonl(jsonl_path)

        processed = []
        aug_count = 0

        print(
            f"Preparing staircase TRAIN data with augmentation factor = {self.augment_factor}"
        )

        for i, item in enumerate(data):
            if "image_name" not in item:
                print(f"Warning: 'image_name' missing in item index {i}, skipping.")
                continue

            image_path = os.path.join(images_dir, item["image_name"])

            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}, skipping...")
                continue

            # Original
            orig_item = item.copy()
            orig_item["image_path"] = image_path
            processed.append(orig_item)

            # Augmented copies
            for aug_id in range(1, self.augment_factor + 1):
                print(
                    f"  Augmenting [{i+1}/{len(data)}] {item['image_name']} "
                    f"({aug_id}/{self.augment_factor})"
                )
                aug_path = self.augment_image(image_path, aug_id)
                if aug_path and os.path.exists(aug_path):
                    aug_item = item.copy()
                    aug_item["image_path"] = aug_path
                    aug_item["image_name"] = os.path.basename(aug_path)
                    processed.append(aug_item)
                    aug_count += 1
                else:
                    print(f"    Failed to create augmentation {aug_id} for {image_path}")

        print(
            f"Training: {len(processed)} samples "
            f"(original: {len(data)}, augmented: {aug_count})"
        )

        return processed

    def prepare_data_no_augmentation(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        """Prepare data WITHOUT augmentation (for validation / test)."""
        data = self.load_jsonl(jsonl_path)

        processed = []
        for i, item in enumerate(data):
            if "image_name" not in item:
                print(f"Warning: 'image_name' missing in item index {i}, skipping.")
                continue

            image_path = os.path.join(images_dir, item["image_name"])
            if os.path.exists(image_path):
                item_copy = item.copy()
                item_copy["image_path"] = image_path
                processed.append(item_copy)
            else:
                print(f"Warning: Image not found: {image_path}, skipping...")

        print(
            f"Validation/Test: {len(processed)} valid samples out of {len(data)} total (no augmentation)"
        )

        return processed

    # ----------------------- METRICS -----------------------

    def calculate_cer(self, predictions, targets):
        if (
            predictions is None
            or targets is None
            or len(predictions) == 0
            or len(targets) == 0
            or len(predictions) != len(targets)
        ):
            return 1.0

        total_cer = 0.0
        count = 0
        for pred, target in zip(predictions, targets):
            pred_str, target_str = str(pred), str(target)
            if len(target_str) > 0:
                cer = jiwer.cer(target_str, pred_str)
                total_cer += cer
                count += 1

        return total_cer / count if count > 0 else 1.0

    def compute_metrics_for_trainer(self, eval_preds, compute_result: bool = True):
        """
        A100-optimized compute_metrics with eval_accumulation support.

        ✅ Important fix:
        - We ONLY decode predicted tokens at positions where label_ids != -100
          (i.e. assistant output), so CER isn't polluted by the prompt.
        """
        if not compute_result:
            return {}

        predictions, label_ids = eval_preds

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Handle Unsloth EmptyLogits case
        if hasattr(predictions, "__class__") and "EmptyLogits" in str(
            predictions.__class__
        ):
            return {"cer": 0.0, "cer_percentage": 0.0}

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().float().numpy()

        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.detach().cpu().numpy()

        predicted_token_ids = np.argmax(predictions, axis=-1)

        pred_texts = []
        label_texts = []

        for pred_ids, label_ids_single in zip(predicted_token_ids, label_ids):
            label_ids_single = np.array(label_ids_single, dtype=np.int32)

            # mask for assistant tokens
            mask = label_ids_single != -100
            if not np.any(mask):
                continue

            label_ids_clean = label_ids_single[mask]
            pred_ids_clean = np.array(pred_ids, dtype=np.int32)[mask]

            try:
                pred_text = self.tokenizer.decode(
                    pred_ids_clean, skip_special_tokens=True
                )
                label_text = self.tokenizer.decode(
                    label_ids_clean, skip_special_tokens=True
                )

                pred_texts.append(pred_text)
                label_texts.append(label_text)
            except Exception:
                pred_texts.append("")
                label_texts.append("")

        cer = self.calculate_cer(pred_texts, label_texts)

        torch.cuda.empty_cache()
        gc.collect()

        return {
            "cer": cer,
            "cer_percentage": cer * 100.0,
        }

    # ----------------------- TRAINING -----------------------

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 12,
        batch_size: int = 2,
        learning_rate: float = 3e-5,
    ):
        """Train with A100-optimized settings + augmentation on train."""
        print("Preparing training and validation datasets for STAIRCASE...")

        train_raw = self.prepare_training_data_with_augmentation(
            train_jsonl_path, images_dir
        )
        val_raw = self.prepare_data_no_augmentation(val_jsonl_path, images_dir)

        # ✅ Lazy datasets: preprocessing applied on-the-fly in __getitem__
        train_dataset = StaircaseConversationDataset(
            raw_items=train_raw,
            make_conversation_fn=self.convert_to_conversation,
            preprocess_fn=preprocess_exact_4steps_pil,
            preprocess_cfg=self.preprocess_cfg,
        )
        val_dataset = StaircaseConversationDataset(
            raw_items=val_raw,
            make_conversation_fn=self.convert_to_conversation,
            preprocess_fn=preprocess_exact_4steps_pil,
            preprocess_cfg=self.preprocess_cfg,
        )

        print(
            f"Training: {len(train_dataset)} samples (with augmentation), "
            f"Validation: {len(val_dataset)} samples"
        )

        FastVisionModel.for_training(self.model)

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics_for_trainer,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                eval_accumulation_steps=4,
                batch_eval_metrics=True,
                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,
                max_grad_norm=1.0,
                dataloader_num_workers=0,
                dataloader_pin_memory=False,
                bf16=True,
                fp16=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                optim="adamw_torch_fused",
                load_best_model_at_end=True,
                metric_for_best_model="cer",
                greater_is_better=False,
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=2048,
                report_to="tensorboard",
                logging_dir=f"{output_dir}/logs",
                logging_first_step=True,
                seed=3407,
                output_dir=output_dir,
                save_safetensors=True,
                prediction_loss_only=False,
                disable_tqdm=False,
                # 🔧 NEW (optional but helpful): a bit of label smoothing
                label_smoothing_factor=0.05,
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01,
                )
            ],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting training with CER-based evaluation (A100-optimized, STAIRCASE)...")
        trainer_stats = trainer.train()

        used_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(
            f"Peak reserved memory for training % of max memory = {lora_percentage} %."
        )

        return trainer

    # ----------------------- TEST EVAL -----------------------

    def evaluate_on_test_set(
        self, test_jsonl_path: str, images_dir: str, output_dir: str
    ) -> Dict:
        """Evaluate with aggressive memory management and smaller chunks (no augmentation)."""
        print("Starting evaluation on STAIRCASE test.jsonl with aggressive chunking...")

        FastVisionModel.for_inference(self.model)

        test_data = self.load_jsonl(test_jsonl_path)
        print(f"Loaded {len(test_data)} test samples")

        predictions = []
        all_cer_scores = []

        instruction = (
            "You are an OCR model for historical staircase survey forms.\n\n"
            "Given the image of a filled-in staircase form, read everything and return ONLY one valid JSON object:\n"
            "- Use the same field names, nesting and capitalization as in the training JSONs (do not rename keys).\n"
            "- Use booleans for checkboxes (true = checked, false = unchecked).\n"
            "- Use strings for other fields.\n"
            "- If a field exists but is empty on the form, still include it with \"\".\n"
            "Do not add any extra text before or after the JSON."
        )

        chunk_size = 5
        num_chunks = (len(test_data) + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(test_data))
            chunk_data = test_data[start_idx:end_idx]

            print("\n" + "=" * 60)
            print(f"Chunk {chunk_idx+1}/{num_chunks} ({len(chunk_data)} images)")
            print("=" * 60)

            for i, test_item in enumerate(chunk_data):
                abs_idx = start_idx + i + 1
                img_name = test_item.get("image_name", f"idx_{abs_idx}")
                print(f"\n[{abs_idx}/{len(test_data)}] Processing: {img_name}")

                image_path = os.path.join(images_dir, img_name)

                if not os.path.exists(image_path):
                    print("  ⚠️  Image not found")
                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": self.json_to_string_readable(test_item),
                        "raw_response": "Error: Image not found",
                        "cer_score": 1.0,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)
                    continue

                image = None
                try:
                    image = Image.open(image_path)

                    # ADDED: 4-step preprocessing before giving image to the model (on-the-fly)
                    image = preprocess_exact_4steps_pil(
                        image,
                        target_size=self.preprocess_cfg["target_size"],
                        bg_kernel_size=self.preprocess_cfg["bg_kernel_size"],
                        denoise_method=self.preprocess_cfg["denoise_method"],
                    )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ]

                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    ).to("cuda")

                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            # 🔧 NEW: give the model more room to finish the JSON
                            max_new_tokens=1024,
                            do_sample=False,
                            repetition_penalty=1.0,  # 🔧 NEW: make decoding more neutral
                            use_cache=True,
                        )

                    input_len = inputs["input_ids"].shape[-1]
                    generated_ids_trimmed = outputs[0][input_len:]
                    generated_text = self.tokenizer.decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    predicted_json = self.extract_json_from_response(generated_text)
                    gt_json_string = self.json_to_string_readable(test_item)
                    pred_json_string = (
                        self.json_to_string_readable(predicted_json)
                        if predicted_json
                        else ""
                    )

                    cer_score = (
                        jiwer.cer(gt_json_string, pred_json_string)
                        if pred_json_string
                        else 1.0
                    )

                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": predicted_json,
                        "predicted_text": pred_json_string,
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": gt_json_string,
                        "raw_response": generated_text,
                        "cer_score": cer_score,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(cer_score)

                    print(f"  ✅ CER: {cer_score:.4f} ({cer_score*100:.2f}%)")

                except Exception as e:
                    print(f"  ❌ Error: {str(e)}")
                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": self.json_to_string_readable(test_item),
                        "raw_response": f"Error: {str(e)}",
                        "cer_score": 1.0,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)

                finally:
                    if image is not None:
                        try:
                            image.close()
                        except Exception:
                            pass
                        del image
                    if "inputs" in locals():
                        del inputs
                    if "outputs" in locals():
                        del outputs
                    if "generated_ids_trimmed" in locals():
                        del generated_ids_trimmed

                    torch.cuda.empty_cache()
                    gc.collect()

            print(f"\nChunk {chunk_idx+1} completed. Cleaning memory...")
            torch.cuda.empty_cache()
            gc.collect()

            if (chunk_idx + 1) % 3 == 0 or chunk_idx == num_chunks - 1:
                intermediate_file = os.path.join(
                    output_dir, f"test_predictions_chunk_{chunk_idx+1}.jsonl"
                )
                self.save_jsonl(predictions, intermediate_file)
                print(f"Saved intermediate results to: {intermediate_file}")

        predictions_file = os.path.join(output_dir, "test_predictions.jsonl")
        self.save_jsonl(predictions, predictions_file)

        cer_stats = self.calculate_cer_statistics(all_cer_scores)
        cer_file = os.path.join(output_dir, "cer_evaluation_results.txt")
        self.save_cer_results(cer_stats, cer_file, len(predictions))

        return {
            "predictions": predictions,
            "cer_stats": cer_stats,
            "predictions_file": predictions_file,
            "cer_file": cer_file,
        }

    def calculate_cer_statistics(self, all_cer_scores: List[float]) -> Dict:
        if not all_cer_scores:
            return {}

        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        median_cer = np.median(all_cer_scores)
        std_cer = np.std(all_cer_scores)
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)

        return {
            "total_images": len(all_cer_scores),
            "average_cer": avg_cer,
            "median_cer": median_cer,
            "minimum_cer": min_cer,
            "maximum_cer": max_cer,
            "std_cer": std_cer,
            "perfect_matches": perfect_matches,
        }

    def save_cer_results(self, cer_stats: Dict, cer_file: str, num_predictions: int):
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("CER EVALUATION RESULTS - GEMMA-3 STAIRCASE (A100)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Median CER: {cer_stats['median_cer']:.4f} ({cer_stats['median_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Minimum CER: {cer_stats['minimum_cer']:.4f} ({cer_stats['minimum_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Maximum CER: {cer_stats['maximum_cer']:.4f} ({cer_stats['maximum_cer']*100:.2f}%)\n"
            )
            f.write(f"Standard Deviation: {cer_stats['std_cer']:.4f}\n\n")

            f.write(
                f"Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']} "
                f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n"
            )
            f.write(f"Total images processed: {num_predictions}\n")

        print(f"\nCER evaluation results saved to: {cer_file}")

    def save_model(self, trainer, output_dir: str):
        print(f"\nSaving model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("Model (LoRA adapters) saved successfully!")

    def cleanup_temp_files(self):
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {self.temp_dir}: {e}")


def main():
    """Main function - A100 optimized with chunking + augmentation (factor=1)."""

    base_checkpoint_dir = (
        "/home/vault/iwi5/iwi5298h/models_image_text/gemma/stair/finetune"
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_A100_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    config = {
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/staircase_images",
        "output_dir": run_dir,
        "num_epochs": 12,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "augment_factor": 1,  # 🔁 original + 1 augmented copy
    }

    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    local_model_path = (
        "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/"
        "models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/"
        "snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"
    )

    print("\n" + "=" * 60)
    print("GEMMA-3 STAIRCASE - A100 OPTIMIZED + AUGMENTATION (factor=1)")
    print("=" * 60)

    finetuner = StaircaseGemma3Finetune(
        model_path=local_model_path,
        augment_factor=config["augment_factor"],
    )

    try:
        print("\n" + "=" * 60)
        print("STARTING TRAINING (A100-OPTIMIZED, STAIRCASE)")
        print("=" * 60)

        trainer = finetuner.train_model(
            train_jsonl_path=config["train_jsonl_path"],
            val_jsonl_path=config["val_jsonl_path"],
            images_dir=config["images_dir"],
            output_dir=config["output_dir"],
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
        )

        finetuner.save_model(trainer, config["output_dir"])

        print("\n" + "=" * 60)
        print("STARTING EVALUATION ON STAIRCASE TEST.JSONL")
        print("=" * 60)

        test_results = finetuner.evaluate_on_test_set(
            test_jsonl_path=config["test_jsonl_path"],
            images_dir=config["images_dir"],
            output_dir=config["output_dir"],
        )

        print(f"\n🎉 Gemma-3 Staircase training completed!")
        print(f"All outputs saved to: {run_dir}")
        print(
            f"\nFinal CER: {test_results['cer_stats']['average_cer']:.4f} "
            f"({test_results['cer_stats']['average_cer']*100:.2f}%)"
        )
    finally:
        finetuner.cleanup_temp_files()


if __name__ == "__main__":
    main()
