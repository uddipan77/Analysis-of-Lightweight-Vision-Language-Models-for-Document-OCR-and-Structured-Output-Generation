#!/usr/bin/env python3
# staircase_qwen_finetune_schema_prompt_cer.py
# Qwen2.5-VL-7B Staircase OCR fine-tuning with Unsloth
# - Compact schema-style prompt (structure is in labels)
# - Best model selected by lowest CER on validation set (like Phi script)
# - CER is computed via autoregressive generation on a small val subset each epoch
# - No compute_metrics -> avoids huge logits-in-RAM
# - No second FastVisionModel.from_pretrained call -> avoids inner_training_loop patch error

import os

# ----------------------------------------------------------------------
# IMPORTANT: set Unsloth / PyTorch env BEFORE importing unsloth/torch
# ----------------------------------------------------------------------
os.environ["UNSLOTH_OFFLOAD_GRADIENTS"] = "0"          # keep grads on GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import re
import glob
import random
import tempfile
from datetime import datetime
from typing import List, Dict, Any

import torch
import numpy as np
import jiwer
from PIL import Image
import torchvision.transforms as transforms

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

from transformers import TrainerCallback

from qwen_vl_utils import process_vision_info

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

    pad_left  = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left
    pad_top   = (th - new_h) // 2
    pad_bot   = th - new_h - pad_top

    border_val = pad_value if resized.ndim == 2 else (pad_value, pad_value, pad_value)

    return cv2.copyMakeBorder(
        resized, pad_top, pad_bot, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=border_val
    )


def illumination_flatten(gray, bg_kernel_size=51):
    # EXACTLY like Code 2
    if bg_kernel_size % 2 == 0:
        bg_kernel_size += 1

    bg = cv2.GaussianBlur(gray, (bg_kernel_size, bg_kernel_size), 0)

    gray_f = gray.astype(np.float32)
    bg_f   = np.clip(bg.astype(np.float32), 1.0, 255.0)

    norm = (gray_f / bg_f) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


def clahe_on_luminance(bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


def preprocess_exact_4steps(image_path,
                            output_path=None,
                            target_size=(1024, 1024),
                            bg_kernel_size=51,
                            denoise_method="bilateral"):

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
        den = cv2.fastNlMeansDenoising(flat, None, h=7, templateWindowSize=7, searchWindowSize=21)
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
    Output is a PIL RGB image; original files are untouched.
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
        den = cv2.fastNlMeansDenoising(flat, None, h=7, templateWindowSize=7, searchWindowSize=21)
    else:
        raise ValueError("denoise_method must be 'bilateral' or 'nlm'")

    # 3) CLAHE (luminance)
    den_bgr = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    clahe_bgr = clahe_on_luminance(den_bgr, clip_limit=2.0, tile_grid_size=(8, 8))
    out_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # 4) Letterbox resize
    out = letterbox_resize(out_gray, target_size=target_size, pad_value=255)

    return Image.fromarray(out).convert("RGB")


# ======================================================================
# Compact JSON-schema prompt (same spirit as Phi's INSTRUCTION)
# ======================================================================

STAIRCASE_SCHEMA_PROMPT = """You are an OCR model for historical German staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this dataset (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
- Never drop a key that appears in the form’s JSON structure. If a field is empty on the form, still include it with an empty string "" (or false for an unchecked box).
- Use booleans for checkbox options: true if the box is checked, false if it is empty.
- Use strings for numbers and free-text fields (measurements, dates, names, notes).
- Do NOT invent new fields.
"""


# Normalize minor key variants to a canonical form (optional but helps robustness)
KEY_NORMALIZATION = {
    "Gesamt Ø cm": "Gesamt Durchmesser cm",
    "Gesamt Durchmesser cm": "Gesamt Durchmesser cm",
    "Gehlinie": "GEHLINIE",
    "Hohe": "Hohe",  # keep spelling as used in schema / labels
}


# ======================================================================
# Main Finetuner Class
# ======================================================================

class StaircaseOCRFinetune:
    def __init__(
        self,
        model_path: str = "/home/vault/iwi5/iwi5298h/models/qwen7b",
        augment_factor: int = 1,
    ):
        """Initialize Unsloth Qwen2.5-VL with LoRA and CER-based selection."""
        print(f"Loading Qwen2.5-VL model from {model_path} with Unsloth...")

        self.augment_factor = augment_factor
        self.instruction = STAIRCASE_SCHEMA_PROMPT

        # Base Unsloth model (4-bit) for training
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=1024,    # shorter to save memory
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        # LoRA: medium-light, no rslora, no gradient checkpointing (safer with Unsloth)
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=3407,
            use_rslora=False,
        )

        # Temp dir for augmented images
        self.temp_dir = tempfile.mkdtemp(prefix="augmented_images_")
        print(f"Created temporary directory for augmented images: {self.temp_dir}")
        print("Model loaded successfully with LoRA (r=16, no offload)!")

    # --------------------- Utility & IO --------------------- #

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

    def dict_without_image_meta(self, obj: Dict) -> Dict:
        """Drop image-specific keys (image_name, image_path) from labels/targets."""
        return {k: v for k, v in obj.items() if k not in ("image_name", "image_path")}

    def json_to_string_no_sort(self, obj: Dict) -> str:
        return json.dumps(
            self.dict_without_image_meta(obj),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def safe_json_loads(self, s: str):
        try:
            return json.loads(s)
        except Exception:
            return None

    def normalize_keys(self, d: Any) -> Any:
        if isinstance(d, dict):
            out = {}
            for k, v in d.items():
                nk = KEY_NORMALIZATION.get(k, k)
                out[nk] = self.normalize_keys(v)
            return out
        elif isinstance(d, list):
            return [self.normalize_keys(x) for x in d]
        else:
            return d

    # --------------------- JSON extraction --------------------- #

    def extract_json_from_response(self, response: str) -> Dict:
        response = response.strip()

        # Try to find JSON block
        matches = re.findall(r"\{.*\}", response, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Fallback: whole string
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    # --------------------- Image handling --------------------- #

    def create_augmentation_transforms(self):
        # Slightly lighter and 512x512 to match common vision settings
        return [
            transforms.RandomRotation(degrees=(-2, 2), fill=255),
            transforms.ColorJitter(brightness=(0.9, 1.1)),
            transforms.ColorJitter(contrast=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
            transforms.RandomResizedCrop(
                size=(512, 512),
                scale=(0.95, 1.0),
                ratio=(0.9, 1.1),
            ),
        ]

    def augment_image(self, image_path: str, aug_id: int) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            transforms_list = self.create_augmentation_transforms()
            num_transforms = random.randint(2, min(3, len(transforms_list)))
            selected_transforms = random.sample(transforms_list, num_transforms)

            augmented_image = image
            for transform in selected_transforms:
                augmented_image = transform(augmented_image)

            original_name = os.path.basename(image_path)
            name_without_ext, ext = os.path.splitext(original_name)
            augmented_name = f"{name_without_ext}_aug{aug_id}{ext}"
            augmented_path = os.path.join(self.temp_dir, augmented_name)
            augmented_image.save(augmented_path, quality=95)
            return augmented_path
        except Exception as e:
            print(f"Error augmenting image {image_path}: {str(e)}")
            return None

    def find_image_path(self, image_name: str, images_dir: str) -> str:
        """Flexible matching for '... (123).jpg' etc."""
        exact_path = os.path.join(images_dir, image_name)
        if os.path.exists(exact_path):
            return exact_path

        # Match "(number).jpg"
        pattern_match = re.search(r"\((\d+)\)\.jpg$", image_name)
        if pattern_match:
            pattern = f"({pattern_match.group(1)}).jpg"
            search_pattern = os.path.join(images_dir, f"*{pattern}")
            matching_files = glob.glob(search_pattern)
            if matching_files:
                return matching_files[0]

        # Fallback: by basename
        base_name = os.path.splitext(image_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]

        return exact_path

    # --------------------- Conversation conversion --------------------- #

    def convert_to_conversation(self, sample: Dict) -> Dict:
        """Build chat-style training example: instruction + image -> pure JSON text."""
        label_obj = self.dict_without_image_meta(sample)
        gt_json_string = self.json_to_string_no_sort(label_obj)

        image_path = sample.get("image_path", None)
        contents = [{"type": "text", "text": self.instruction}]

        # Only add image if path exists
        if isinstance(image_path, str) and image_path and os.path.exists(image_path):
            # ADDED: on-the-fly preprocessing; provide a PIL image to the template
            try:
                pil_img = Image.open(image_path).convert("RGB")
                pil_img = preprocess_exact_4steps_pil(
                    pil_img,
                    target_size=(1024, 1024),
                    bg_kernel_size=51,
                    denoise_method="bilateral",
                )
                contents.append({"type": "image", "image": pil_img})
            except Exception as e:
                print(f"[WARN] preprocess failed for {image_path}: {e} (using original path)")
                contents.append({"type": "image", "image": image_path})
        else:
            print(
                f"[WARN] convert_to_conversation: missing/invalid image_path, using text-only."
            )

        conversation = [
            {
                "role": "user",
                "content": contents,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gt_json_string}],
            },
        ]
        return {"messages": conversation}

    # --------------------- Dataset prep --------------------- #

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        """
        Prepare training data with augmentation, in conversation format.
        """
        data = self.load_jsonl(jsonl_path)

        converted_dataset = []
        augmented_count = 0

        print(f"Preparing training data with augmentation factor: {self.augment_factor}")

        for i, item in enumerate(data):
            image_path = self.find_image_path(item["image_name"], images_dir)
            if os.path.exists(image_path):
                # Original
                original_item = item.copy()
                original_item["image_path"] = image_path
                converted_dataset.append(self.convert_to_conversation(original_item))

                # Augmented
                for aug_id in range(1, self.augment_factor + 1):
                    print(
                        f"Creating augmentation {aug_id}/{self.augment_factor} "
                        f"for image {i+1}/{len(data)}: {item['image_name']}"
                    )
                    augmented_path = self.augment_image(image_path, aug_id)
                    if augmented_path and os.path.exists(augmented_path):
                        augmented_item = item.copy()
                        augmented_item["image_path"] = augmented_path
                        augmented_item["image_name"] = f"{item['image_name']}_aug{aug_id}"
                        converted_dataset.append(
                            self.convert_to_conversation(augmented_item)
                        )
                        augmented_count += 1
                    else:
                        print(
                            f"Warning: Failed to create augmentation {aug_id} for {item['image_name']}"
                        )
            else:
                print(f"Warning: Image not found for {item['image_name']}, skipping...")

        print(f"Successfully created {augmented_count} augmented images")
        print(
            f"Total training samples: {len(converted_dataset)} "
            f"(original: {len(data)}, augmented: {augmented_count})"
        )

        return converted_dataset

    def prepare_validation_data(
        self, jsonl_path: str, images_dir: str
    ) -> List[Dict]:
        """Validation data without augmentation."""
        data = self.load_jsonl(jsonl_path)
        converted_dataset = []

        for item in data:
            image_path = self.find_image_path(item["image_name"], images_dir)
            if os.path.exists(image_path):
                item = item.copy()
                item["image_path"] = image_path
                converted_dataset.append(self.convert_to_conversation(item))
            else:
                print(f"Warning: Image not found for {item['image_name']}, skipping...")

        print(f"Validation samples: {len(converted_dataset)}")
        return converted_dataset

    # --------------------- CER helpers --------------------- #

    def calculate_cer(self, predictions, targets):
        """
        Compute average CER over pairs of prediction/target strings using jiwer.cer.
        No use of jiwer.compute_measures (for compatibility with older jiwer versions).
        """
        if not predictions or not targets or len(predictions) != len(targets):
            return 1.0

        total_cer = 0.0
        valid_pairs = 0

        for pred, target in zip(predictions, targets):
            pred_str, target_str = str(pred), str(target)

            if len(target_str) == 0:
                continue

            try:
                cer_val = jiwer.cer(target_str, pred_str)
            except Exception as e:
                print(f"[WARN] CER computation failed for one sample: {e}")
                continue

            total_cer += cer_val
            valid_pairs += 1

        return (total_cer / valid_pairs) if valid_pairs > 0 else 1.0

    def calculate_cer_json(self, predictions, targets):
        preds_json = [self.safe_json_loads(p) for p in predictions]
        targets_json = [self.safe_json_loads(t) for t in targets]

        pred_strings = []
        target_strings = []
        for pj, tj in zip(preds_json, targets_json):
            if pj is None:
                pred_strings.append("")
            else:
                pj = self.normalize_keys(self.dict_without_image_meta(pj))
                pred_strings.append(self.json_to_string_no_sort(pj))

            if tj is None:
                target_strings.append("")
            else:
                tj = self.normalize_keys(self.dict_without_image_meta(tj))
                target_strings.append(self.json_to_string_no_sort(tj))

        return self.calculate_cer(pred_strings, target_strings)

    # --------------------- Training (best model = lowest CER) --------------------- #

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 20,
        batch_size: int = 1,
        learning_rate: float = 5e-5,
    ):
        print("Preparing training and validation datasets...")
        train_dataset = self.prepare_training_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_validation_data(val_jsonl_path, images_dir)
        val_raw_data = self.load_jsonl(val_jsonl_path)
        print(
            f"Training: {len(train_dataset)} samples, "
            f"Validation (raw): {len(val_raw_data)} records"
        )

        FastVisionModel.for_training(self.model)

        # CER-based validation callback (like Phi)
        best_ckpt_dir = os.path.join(output_dir, "best_model_cer")

        cer_callback = QwenCERCallback(
            parent=self,
            val_data=val_raw_data,
            images_dir=images_dir,
            best_ckpt_dir=best_ckpt_dir,
            max_eval_samples=30,  # subset for speed & memory
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # No compute_metrics -> avoid logits RAM explosion
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=4,
                warmup_steps=50,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                logging_steps=10,

                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=False,   # we manage "best by CER" ourselves
                save_total_limit=1,
                metric_for_best_model="eval_cer",  # logged by callback
                greater_is_better=False,

                dataloader_num_workers=0,
                dataloader_pin_memory=False,

                weight_decay=0.05,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,
                optim="adamw_8bit",
                gradient_checkpointing=False,

                remove_unused_columns=False,
                dataset_text_field=None,
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=1024,

                report_to="tensorboard",
                logging_dir=f"{output_dir}/logs",
                seed=3407,
                output_dir=output_dir,
            ),
            callbacks=[cer_callback],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        max_memory = round(gpu_stats.total_memory / 1024**3, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting training (best model tracked by CER callback)...")
        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(
            f"Peak reserved memory for training = {used_memory_for_lora} GB "
            f"({lora_percentage} % of max)."
        )

        # ------------------------------------------------------------------
        # >>> FIX: safe best-by-CER state_dict loading <<<
        #  - Filter out unexpected NF4 / quantization keys
        #  - Use strict=False to ignore extras / minor mismatches
        # ------------------------------------------------------------------
        if cer_callback.best_state_path is not None:
            print(
                f"Loading best-by-CER weights from {cer_callback.best_state_path} ..."
            )
            loaded_state = torch.load(cer_callback.best_state_path, map_location="cpu")

            current_state = self.model.state_dict()
            filtered_state = {
                k: v for k, v in loaded_state.items() if k in current_state
            }

            print(
                f"Loaded state_dict has {len(loaded_state)} keys; "
                f"{len(filtered_state)} of them match the current model and will be loaded."
            )

            missing_keys, unexpected_keys = self.model.load_state_dict(
                filtered_state,
                strict=False,  # ignore any remaining issues (e.g. buffers)
            )

            if missing_keys:
                print(f"[WARN] Missing keys when loading best CER weights ({len(missing_keys)}):")
                print("  (showing first 10)", missing_keys[:10])
            if unexpected_keys:
                print(f"[WARN] Unexpected keys when loading best CER weights ({len(unexpected_keys)}):")
                print("  (showing first 10)", unexpected_keys[:10])

            trainer.model = self.model  # keep Trainer reference in sync
            print(
                f"Restored best validation CER = {cer_callback.best_cer:.4f} "
                f"({cer_callback.best_cer*100:.2f}%)"
            )
        else:
            print(
                "WARNING: CER callback did not save any best model; using final epoch weights."
            )

        # store CER history for later logging if needed
        self.cer_history = cer_callback.cer_history

        torch.cuda.empty_cache()
        return trainer

    # --------------------- Evaluation on a split (e.g. test) --------------------- #

    def evaluate_on_split(
        self, jsonl_path: str, images_dir: str, output_dir: str, split_name: str
    ) -> Dict:
        """
        Evaluation over a given split (e.g., test.jsonl) with greedy decoding,
        JSON extraction, key normalization, and CER calculation.
        """
        split_name = split_name.lower()
        print(f"Starting evaluation on {split_name}.jsonl...")

        FastVisionModel.for_inference(self.model)
        self.model.eval()

        predictions = []
        all_cer_scores = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                test_item = json.loads(line)
                print(
                    f"Processing {split_name} image {len(predictions)+1}: {test_item['image_name']}"
                )

                image_path = self.find_image_path(
                    test_item["image_name"], images_dir
                )

                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    gt_obj = self.normalize_keys(
                        self.dict_without_image_meta(test_item)
                    )
                    prediction_entry = {
                        "image_name": test_item["image_name"],
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": gt_obj,
                        "target_text": self.json_to_string_no_sort(gt_obj),
                        "raw_response": "Error: Image not found",
                        "cer_score": 1.0,
                        "matched_image_path": image_path,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)
                    continue

                try:
                    # ADDED: preprocess image into a PIL object, then pass it to the message
                    pil_img = Image.open(image_path).convert("RGB")
                    pil_img = preprocess_exact_4steps_pil(
                        pil_img,
                        target_size=(1024, 1024),
                        bg_kernel_size=51,
                        denoise_method="bilateral",
                    )

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.instruction},
                                {"type": "image", "image": pil_img},
                            ],
                        }
                    ]

                    input_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    image_inputs, video_inputs = process_vision_info(messages)

                    inputs = self.tokenizer(
                        text=[input_text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")

                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=1024,
                            use_cache=True,
                            temperature=0.0,
                            do_sample=False,       # deterministic greedy
                            repetition_penalty=1.0,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]

                    generated_text = self.tokenizer.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    predicted_json_raw = self.extract_json_from_response(
                        generated_text
                    )
                    predicted_json = self.normalize_keys(
                        self.dict_without_image_meta(predicted_json_raw or {})
                    )

                    gt_obj = self.normalize_keys(
                        self.dict_without_image_meta(test_item)
                    )

                    gt_json_string = self.json_to_string_no_sort(gt_obj)
                    pred_json_string = self.json_to_string_no_sort(predicted_json)

                    cer_score = jiwer.cer(gt_json_string, pred_json_string)

                    prediction_entry = {
                        "image_name": test_item["image_name"],
                        "predicted_json": predicted_json,
                        "predicted_text": pred_json_string,
                        "target_json": gt_obj,
                        "target_text": gt_json_string,
                        "raw_response": generated_text,
                        "cer_score": cer_score,
                        "matched_image_path": image_path,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(cer_score)

                    print(f"  Processed successfully. CER: {cer_score:.3f}")

                except Exception as e:
                    print(f"Error processing {test_item['image_name']}: {str(e)}")
                    gt_obj = self.normalize_keys(
                        self.dict_without_image_meta(test_item)
                    )
                    prediction_entry = {
                        "image_name": test_item["image_name"],
                        "predicted_json": {},
                        "predicted_text": "",
                        "target_json": gt_obj,
                        "target_text": self.json_to_string_no_sort(gt_obj),
                        "raw_response": f"Error: {str(e)}",
                        "cer_score": 1.0,
                        "matched_image_path": image_path,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)

                if (idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

        predictions_file = os.path.join(
            output_dir, f"{split_name}_predictions.jsonl"
        )
        self.save_jsonl(predictions, predictions_file)

        cer_stats = self.calculate_cer_statistics(all_cer_scores)

        cer_file = os.path.join(
            output_dir, f"cer_evaluation_{split_name}_results.txt"
        )
        self.save_cer_results(cer_stats, cer_file, len(predictions), split_name)

        torch.cuda.empty_cache()

        return {
            "predictions": predictions,
            "cer_stats": cer_stats,
            "predictions_file": predictions_file,
            "cer_file": cer_file,
        }

    # --------------------- Stats & saving --------------------- #

    def calculate_cer_statistics(self, all_cer_scores: List[float]) -> Dict:
        if not all_cer_scores:
            return {}
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        median_cer = float(np.median(all_cer_scores))
        std_cer = float(np.std(all_cer_scores))
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

    def save_cer_results(
        self, cer_stats: Dict, cer_file: str, num_predictions: int, split_name: str
    ):
        split_name = split_name.upper()
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"CER EVALUATION RESULTS ON STAIRCASE {split_name}.JSONL\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(
                f"Average CER: {cer_stats['average_cer']:.4f} "
                f"({cer_stats['average_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Median CER: {cer_stats['median_cer']:.4f} "
                f"({cer_stats['median_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Minimum CER: {cer_stats['minimum_cer']:.4f} "
                f"({cer_stats['minimum_cer']*100:.2f}%)\n"
            )
            f.write(
                f"Maximum CER: {cer_stats['maximum_cer']:.4f} "
                f"({cer_stats['maximum_cer']*100:.2f}%)\n"
            )
            f.write(f"Standard Deviation: {cer_stats['std_cer']:.4f}\n\n")

            f.write(
                f"Perfect matches: {cer_stats['perfect_matches']}/"
                f"{cer_stats['total_images']} "
                f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n"
            )
            f.write(f"Total images processed: {num_predictions}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("=" * 60 + "\n")
            f.write("- Model: Qwen2.5-VL-7B-Instruct with LoRA (from local path)\n")
            f.write("- Dataset: Staircase German Historical Documents\n")
            f.write("- Model Selection: Best-by-CER on validation set\n")
            f.write("- LoRA rank: 16\n")
            f.write("- LoRA alpha: 32\n")
            f.write("- LoRA dropout: 0.0\n")
            f.write("- Weight decay: 0.05\n")
            f.write("- Output format: Pure JSON strings\n")
            f.write("- Multiprocessing: Disabled (dataloader_num_workers=0)\n")
            f.write(
                f"- Data augmentation: Enabled (augment_factor={self.augment_factor})\n"
            )
            f.write("- UNSLOTH_OFFLOAD_GRADIENTS=0 (no CPU offload)\n")
            f.write("- max_seq_length=1024\n")

            if hasattr(self, "cer_history"):
                f.write("\nCER HISTORY (validation subset):\n")
                for entry in self.cer_history:
                    f.write(
                        f"  Epoch {entry['epoch']}: {entry['cer']:.4f} "
                        f"({entry['cer']*100:.2f}%)\n"
                    )

        print(f"CER evaluation results saved to: {cer_file}")

    def save_model(self, trainer, output_dir: str):
        print(f"Saving best-by-CER model to {output_dir}...")

        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)

        print("Saving model in recommended formats...")
        trainer.model.save_pretrained_merged(
            f"{output_dir}_merged_16bit",
            trainer.tokenizer,
            save_method="merged_16bit",
        )
        trainer.model.save_pretrained_merged(
            f"{output_dir}_lora",
            trainer.tokenizer,
            save_method="lora",
        )
        print("Best model saved successfully in multiple formats!")

    def cleanup_temp_files(self):
        try:
            import shutil

            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(
                f"Warning: Could not clean up temporary directory {self.temp_dir}: {str(e)}"
            )


# ======================================================================
# CER Validation Callback (like Phi, but for Qwen + Unsloth)
# ======================================================================

class QwenCERCallback(TrainerCallback):
    def __init__(
        self,
        parent: StaircaseOCRFinetune,
        val_data: List[Dict],
        images_dir: str,
        best_ckpt_dir: str,
        max_eval_samples: int = 30,
    ):
        """
        parent: the StaircaseOCRFinetune instance (for helpers, tokenizer, prompt)
        val_data: raw validation json records (with image_name)
        images_dir: where staircase images live
        best_ckpt_dir: directory to save best-by-CER state_dict
        max_eval_samples: evaluate on this many samples each epoch (subset for speed)
        """
        self.parent = parent
        self.images_dir = images_dir
        self.best_ckpt_dir = best_ckpt_dir
        self.max_eval_samples = max_eval_samples

        self.val_data = val_data[:max_eval_samples]
        self.best_cer = float("inf")
        self.best_state_path = None
        self.cer_history: List[Dict[str, float]] = []

    def on_evaluate(self, args, state, control, model, tokenizer=None, metrics=None, **kwargs):
        if state.epoch is None:
            return control

        print("\n" + "=" * 80)
        print(f"🔍 Qwen CER Callback: COMPUTING CER ON VALIDATION SUBSET (Epoch {int(state.epoch)})")
        print("=" * 80)

        model.eval()
        device = next(model.parameters()).device

        predictions = []
        targets = []

        for i, val_item in enumerate(self.val_data):
            image_name = val_item["image_name"]
            print(
                f"   Validating {i+1}/{len(self.val_data)}: {image_name}",
                end="\r",
            )

            image_path = self.parent.find_image_path(image_name, self.images_dir)
            if not os.path.exists(image_path):
                continue

            try:
                # ADDED: preprocess image into a PIL object, then pass it to the message
                pil_img = Image.open(image_path).convert("RGB")
                pil_img = preprocess_exact_4steps_pil(
                    pil_img,
                    target_size=(1024, 1024),
                    bg_kernel_size=51,
                    denoise_method="bilateral",
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.parent.instruction},
                            {"type": "image", "image": pil_img},
                        ],
                    }
                ]

                input_text = self.parent.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs = process_vision_info(messages)

                inputs = self.parent.tokenizer(
                    text=[input_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_cache=True,
                        temperature=0.0,
                        do_sample=False,
                        repetition_penalty=1.0,
                        pad_token_id=self.parent.tokenizer.pad_token_id,
                        eos_token_id=self.parent.tokenizer.eos_token_id,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]

                generated_text = self.parent.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                predicted_json_raw = self.parent.extract_json_from_response(
                    generated_text
                )
                predicted_json = self.parent.normalize_keys(
                    self.parent.dict_without_image_meta(predicted_json_raw or {})
                )
                gt_obj = self.parent.normalize_keys(
                    self.parent.dict_without_image_meta(val_item)
                )

                gt_str = self.parent.json_to_string_no_sort(gt_obj)
                pred_str = self.parent.json_to_string_no_sort(predicted_json)

                predictions.append(pred_str)
                targets.append(gt_str)

                del inputs, outputs, generated_ids_trimmed
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n   ⚠️  Validation error on {image_name}: {e}")
                gt_obj = self.parent.normalize_keys(
                    self.parent.dict_without_image_meta(val_item)
                )
                gt_str = self.parent.json_to_string_no_sort(gt_obj)
                predictions.append("")
                targets.append(gt_str)
                continue

        # Calculate CER on subset
        val_cer = self.parent.calculate_cer(predictions, targets)
        self.cer_history.append(
            {"epoch": float(state.epoch), "cer": float(val_cer)}
        )

        if metrics is not None:
            metrics["eval_cer"] = val_cer

        print(
            f"\n   ✅ Validation CER (Epoch {int(state.epoch)}): "
            f"{val_cer:.4f} ({val_cer*100:.2f}%)"
        )

        # Save best model by CER: torch.save(state_dict) and reload later with load_state_dict
        if val_cer < self.best_cer:
            improvement = self.best_cer - val_cer
            self.best_cer = val_cer
            print(
                f"   🎯 NEW BEST CER: {self.best_cer:.4f} "
                f"(improved by {improvement:.4f})"
            )

            os.makedirs(self.best_ckpt_dir, exist_ok=True)
            best_state_path = os.path.join(self.best_ckpt_dir, "pytorch_model_best_cer.bin")
            torch.save(model.state_dict(), best_state_path)
            self.best_state_path = best_state_path
            print(f"   💾 Saved best-by-CER state_dict to: {best_state_path}")
        else:
            print(f"   📊 No improvement. Best CER remains: {self.best_cer:.4f}")

        print("=" * 80 + "\n")

        model.train()
        return control


# ======================================================================
# main()
# ======================================================================

def main():
    base_checkpoint_dir = "/home/vault/iwi5/iwi5298h/models_image_text/qwen/stair"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    config = {
        "model_path": "/home/vault/iwi5/iwi5298h/models/qwen7b",
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/staircase_images",
        "output_dir": run_dir,
        "num_epochs": 20,
        "batch_size": 1,
        "learning_rate": 5e-5,
        "augment_factor": 1,
    }

    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    finetuner = StaircaseOCRFinetune(
        model_path=config["model_path"],
        augment_factor=config["augment_factor"],
    )

    try:
        print("=" * 60)
        print("STARTING STAIRCASE DATASET TRAINING (BEST MODEL BY VALIDATION CER)")
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

        # Save best-by-CER model in HF format + merged/lora
        finetuner.save_model(trainer, config["output_dir"])

        print("\n" + "=" * 60)
        print("STARTING EVALUATION ON STAIRCASE TEST.JSONL (BEST-BY-CER WEIGHTS)")
        print("=" * 60)

        test_results = finetuner.evaluate_on_split(
            jsonl_path=config["test_jsonl_path"],
            images_dir=config["images_dir"],
            output_dir=config["output_dir"],
            split_name="test",
        )

        print("\nEvaluation completed!")
        print(f"TEST predictions saved to: {test_results['predictions_file']}")
        print(f"TEST CER results saved to: {test_results['cer_file']}")
        print(f"All files saved in: {run_dir}")

        cer_stats = test_results["cer_stats"]
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY - STAIRCASE DATASET (TEST SPLIT)")
        print("=" * 60)
        print(
            f"Average CER: {cer_stats['average_cer']:.4f} "
            f"({cer_stats['average_cer']*100:.2f}%)"
        )
        print(
            f"Median CER: {cer_stats['median_cer']:.4f} "
            f"({cer_stats['median_cer']*100:.2f}%)"
        )
        print(
            f"Perfect matches: {cer_stats['perfect_matches']}/"
            f"{cer_stats['total_images']} "
            f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)"
        )
        print(f"Total images processed: {cer_stats['total_images']}")
        print("\nStaircase dataset training and evaluation completed!")

    finally:
        finetuner.cleanup_temp_files()


if __name__ == "__main__":
    main()
