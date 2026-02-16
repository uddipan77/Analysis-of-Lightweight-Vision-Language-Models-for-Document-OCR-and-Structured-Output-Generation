#!/usr/bin/env python3
# phi_3_5_vision_staircase_finetune_qlora.py
# Memory-optimized fine-tuning with BitsAndBytes QLoRA 4-bit for STAIRCASE dataset

import torch
import json
import os
from typing import List, Dict
from PIL import Image, ImageEnhance
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import jiwer
from datetime import datetime
import numpy as np
import unicodedata
from dataclasses import dataclass
import random

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
    clahe_bgr = clahe_on_luminance(
        den_bgr, clip_limit=2.0, tile_grid_size=(8, 8)
    )
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
    but applied on-the-fly to a PIL RGB image (no disk I/O, no saving).
    Output is a PIL RGB image ready for the processor.
    """
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
    clahe_bgr = clahe_on_luminance(
        den_bgr, clip_limit=2.0, tile_grid_size=(8, 8)
    )
    out_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # 4) Letterbox resize
    out = letterbox_resize(out_gray, target_size=target_size, pad_value=255)

    return Image.fromarray(out).convert("RGB")


# ============================================================================
# Configuration - Staircase dataset
# ============================================================================

CONFIG = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision",
    "train_jsonl": "/home/woody/iwi5/iwi5298h/json_staircase/train.jsonl",
    "val_jsonl": "/home/woody/iwi5/iwi5298h/json_staircase/val.jsonl",
    "test_jsonl": "/home/woody/iwi5/iwi5298h/json_staircase/test.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/staircase_images",
    # UPDATED: base output dir + timestamp subfolder
    "output_dir": os.path.join(
        "/home/vault/iwi5/iwi5298h/models_image_text/phi/stair",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ),
    "num_epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "max_seq_length": 2048,
    "weight_decay": 0.05,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "data_augmentation": True,
    "use_4bit": True,
    "use_nested_quant": True,
}

# ============================================================================
# Instruction Prompt for STAIRCASE Dataset
# ============================================================================

INSTRUCTION = """You are an OCR model for historical staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this dataset (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
- Never drop a key that appears in the form’s JSON structure. If a field is empty on the form, still include it with an empty string "" (or false for an unchecked box).
- Use booleans for checkbox options: true if the box is checked, false if it is empty.
- Use strings for numbers and free-text fields (measurements, dates, names, notes).
- Do NOT invent new fields."""

# ============================================================================
# Data Utilities
# ============================================================================


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def load_jsonl(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_label_string(json_data: Dict) -> str:
    """Ground truth without image_name (metadata)."""
    label_data = {k: v for k, v in json_data.items() if k != "image_name"}
    return json.dumps(label_data, ensure_ascii=False, sort_keys=False)


def extract_json_from_response(response: str) -> str:
    """Extract FIRST JSON from response to prevent repetition issues."""
    response = response.strip()

    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]

        # Only take the first complete JSON object
        try:
            brace_count = 0
            first_end = -1
            for i, char in enumerate(json_str):
                if char == "{":
                    brace_count += 1
                elif char == "}":
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

        # Fallback: try parsing the whole thing
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            return json_str

    return response


# ============================================================================
# Data Augmentation
# ============================================================================


class DocumentImageAugmenter:
    """Augmentation for historical staircase forms."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def augment(self, image: Image.Image) -> Image.Image:
        if not self.enabled or random.random() > 0.7:
            return image

        # Random brightness (±15%)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))

        # Random contrast (±15%)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))

        # Random sharpness
        if random.random() > 0.5:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))

        # Small rotation (±2 degrees)
        if random.random() > 0.7:
            angle = random.uniform(-2, 2)
            image = image.rotate(angle, fillcolor=(255, 255, 255), expand=False)

        return image


# ============================================================================
# Custom Dataset for STAIRCASE
# ============================================================================


class StaircaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        processor,
        instruction: str,
        augment: bool = False,
    ):
        self.data = load_jsonl(jsonl_path)
        self.images_dir = images_dir
        self.processor = processor
        self.instruction = instruction
        self.augmenter = DocumentImageAugmenter(enabled=augment)

        # Filter valid samples
        self.valid_samples = []
        for item in self.data:
            image_name = item["image_name"]
            image_path = os.path.join(self.images_dir, image_name)
            if os.path.exists(image_path):
                self.valid_samples.append(item)

        print(
            f"   📊 Loaded {len(self.valid_samples)} valid samples "
            f"(out of {len(self.data)} total)"
        )
        if augment:
            print("   🔄 Data augmentation ENABLED")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        image_name = item["image_name"]
        image_path = os.path.join(self.images_dir, image_name)

        # Load and augment image
        image = Image.open(image_path).convert("RGB")
        image = self.augmenter.augment(image)

        # ADDED: 4-step preprocessing before giving image to the model (on-the-fly)
        image = preprocess_exact_4steps_pil(
            image,
            target_size=(1024, 1024),
            bg_kernel_size=51,
            denoise_method="bilateral",
        )

        # Ground truth JSON string (excluding image_name)
        gt_json_str = create_label_string(item)

        # Format as chat messages
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{self.instruction}",
            },
            {
                "role": "assistant",
                "content": gt_json_str,
            },
        ]

        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Process inputs
        inputs = self.processor(prompt, [image], return_tensors="pt")

        # Teacher forcing: labels = input_ids, with instruction masked as -100
        labels = inputs["input_ids"].clone()

        # Find where assistant response starts to mask instruction
        assistant_token = self.processor.tokenizer.encode(
            "<|assistant|>", add_special_tokens=False
        )
        input_ids_list = inputs["input_ids"][0].tolist()

        try:
            for i in range(len(input_ids_list) - len(assistant_token)):
                if input_ids_list[i : i + len(assistant_token)] == assistant_token:
                    labels[0, : i + len(assistant_token)] = -100
                    break
        except Exception:
            pass

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0)
            if "pixel_values" in inputs
            else None,
            "image_sizes": inputs["image_sizes"].squeeze(0)
            if "image_sizes" in inputs
            else None,
            "labels": labels.squeeze(0),
            "image_name": image_name,
            "ground_truth": gt_json_str,
        }


# ============================================================================
# Data Collator
# ============================================================================


@dataclass
class DataCollatorForPhi3Vision:
    processor: AutoProcessor

    def __call__(self, features):
        batch = {}

        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])

        if features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])

        if features[0]["image_sizes"] is not None:
            batch["image_sizes"] = torch.stack([f["image_sizes"] for f in features])

        return batch


# ============================================================================
# CER Validation Callback (best model by CER)
# ============================================================================


class CERValidationCallback(TrainerCallback):
    def __init__(self, model, processor, val_data, images_dir, instruction, output_dir):
        self.model = model
        self.processor = processor
        self.val_data = val_data[:30]  # subset for speed
        self.images_dir = images_dir
        self.instruction = instruction
        self.output_dir = output_dir
        self.best_cer = float("inf")
        self.best_checkpoint_path = None
        self.cer_history = []

    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        print("\n" + "=" * 80)
        print(f"🔍 COMPUTING CER ON VALIDATION SET (Epoch {int(state.epoch)})")
        print("=" * 80)

        model.eval()
        predictions = []
        targets = []

        device = next(model.parameters()).device

        for i, val_item in enumerate(self.val_data):
            image_name = val_item["image_name"]
            print(
                f"   Validating {i+1}/{len(self.val_data)}: {image_name}",
                end="\r",
            )

            image_path = os.path.join(self.images_dir, image_name)
            if not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path).convert("RGB")

                # ADDED: 4-step preprocessing before giving image to the model (on-the-fly)
                image = preprocess_exact_4steps_pil(
                    image,
                    target_size=(1024, 1024),
                    bg_kernel_size=51,
                    denoise_method="bilateral",
                )

                messages = [
                    {
                        "role": "user",
                        "content": f"<|image_1|>\n{self.instruction}",
                    }
                ]

                prompt = self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                inputs = self.processor(prompt, [image], return_tensors="pt").to(device)

                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.0,
                        do_sample=False,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )

                generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

                raw_output = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                raw_output = normalize_unicode(raw_output)
                prediction = extract_json_from_response(raw_output)
                ground_truth = create_label_string(val_item)

                predictions.append(prediction)
                targets.append(ground_truth)

                del inputs, generate_ids
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"\n   ⚠️  Validation error on {image_name}: {e}")
                predictions.append("")
                targets.append(create_label_string(val_item))
                continue

        # Calculate CER
        val_cer = self.calculate_cer(predictions, targets)
        self.cer_history.append({"epoch": int(state.epoch), "cer": val_cer})

        if metrics is not None:
            metrics["eval_cer"] = val_cer

        print(
            f"\n   ✅ Validation CER (Epoch {int(state.epoch)}): "
            f"{val_cer:.4f} ({val_cer*100:.2f}%)"
        )

        # Save best model by CER
        if val_cer < self.best_cer:
            improvement = self.best_cer - val_cer
            self.best_cer = val_cer
            print(
                f"   🎯 NEW BEST CER: {self.best_cer:.4f} "
                f"(improved by {improvement:.4f})"
            )

            best_model_path = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)

            print(f"   💾 Saving best model to: {best_model_path}")
            model.save_pretrained(best_model_path)

            try:
                self.processor.tokenizer.save_pretrained(best_model_path)
                print("   ✅ Model and tokenizer saved successfully")
            except Exception as e:
                print(f"   ⚠️  Tokenizer save warning: {e}")
                print("   ✅ Model saved (tokenizer will be loaded from original path)")

            self.best_checkpoint_path = best_model_path
        else:
            print(f"   📊 No improvement. Best CER remains: {self.best_cer:.4f}")

        print("=" * 80 + "\n")

        model.train()
        return control

    def calculate_cer(self, predictions, targets):
        if not predictions or not targets:
            return 1.0

        total_cer = 0.0
        valid_pairs = 0

        for pred, target in zip(predictions, targets):
            if len(target) > 0:
                cer_score = jiwer.cer(target, pred)
                total_cer += cer_score
                valid_pairs += 1

        return total_cer / valid_pairs if valid_pairs > 0 else 1.0


# ============================================================================
# Main Fine-tuning Function
# ============================================================================


def main():
    print("\n" + "=" * 80)
    print("PHI-3.5-VISION FINE-TUNING - STAIRCASE DATASET")
    print("BitsAndBytes QLoRA 4-bit (Memory-Optimized)")
    print("=" * 80)

    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Save config
    config_file = os.path.join(CONFIG["output_dir"], "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print(f"\n📂 Configuration:")
    print(f"   • Output directory: {CONFIG['output_dir']}")
    print(f"   • Training epochs: {CONFIG['num_epochs']}")
    print(f"   • Batch size: {CONFIG['batch_size']}")
    print(f"   • Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
    print(f"   • Learning rate: {CONFIG['learning_rate']}")

    print(f"\n🛡️  Memory Optimization:")
    print(f"   • 4-bit NF4 Quantization: {CONFIG['use_4bit']}")
    print(f"   • Nested Quantization: {CONFIG['use_nested_quant']}")
    print(
        f"   • LoRA r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, "
        f"dropout={CONFIG['lora_dropout']}"
    )

    print(f"\n🔧 Anti-Repetition:")
    print(f"   • First-JSON extraction: enabled")
    print(f"   • Using temperature=0.0 for deterministic generation")

    # Configure 4-bit quantization
    print(f"\n⏳ Loading Phi-3.5-Vision with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["use_4bit"],
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=CONFIG["use_nested_quant"],
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        num_crops=16,
    )
    print("   ✅ Processor loaded")

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )

    print("   ✅ Model loaded with 4-bit quantization")

    # Show memory after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(
            f"   📊 GPU Memory - Allocated: {allocated:.2f} GB, "
            f"Reserved: {reserved:.2f} GB"
        )

    # Prepare for k-bit training
    print("\n📝 Preparing model for LoRA training...")
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Apply LoRA
    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print("   ✅ LoRA applied successfully")
    print(
        f"   📊 Trainable params: {trainable_params:,} / {all_params:,} "
        f"({100 * trainable_params / all_params:.2f}%)"
    )

    # Load datasets
    print("\n📊 Loading STAIRCASE datasets...")

    print("   📁 Training dataset:")
    train_dataset = StaircaseDataset(
        CONFIG["train_jsonl"],
        CONFIG["images_dir"],
        processor,
        INSTRUCTION,
        augment=CONFIG["data_augmentation"],
    )

    print("   📁 Validation dataset:")
    eval_dataset = StaircaseDataset(
        CONFIG["val_jsonl"],
        CONFIG["images_dir"],
        processor,
        INSTRUCTION,
        augment=False,
    )

    # Load validation data for CER callback
    val_data = load_jsonl(CONFIG["val_jsonl"])
    print(f"   📊 Validation samples for CER computation: {len(val_data)}")

    # Data collator
    data_collator = DataCollatorForPhi3Vision(processor=processor)

    # CER callback (best model by CER)
    cer_callback = CERValidationCallback(
        model=model,
        processor=processor,
        val_data=val_data,
        images_dir=CONFIG["images_dir"],
        instruction=INSTRUCTION,
        output_dir=CONFIG["output_dir"],
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        max_grad_norm=CONFIG["max_grad_norm"],
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,  # we manage "best" via CER callback
        metric_for_best_model="eval_cer",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir=os.path.join(CONFIG["output_dir"], "logs"),
        optim="paged_adamw_8bit",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[cer_callback],
    )

    # GPU info
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"\n{'='*80}")
        print("GPU INFORMATION")
        print(f"{'='*80}")
        print(f"   GPU: {gpu_stats.name}")
        print(f"   Max memory: {round(gpu_stats.total_memory / 1024**3, 3)} GB")
        print(f"{'='*80}\n")

    # Train
    print("🚀 Starting QLoRA training on STAIRCASE dataset...\n")
    print("=" * 80)
    trainer.train()

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED!")
    print("=" * 80)
    print(
        f"   🎯 Best Validation CER: {cer_callback.best_cer:.4f} "
        f"({cer_callback.best_cer*100:.2f}%)"
    )
    print(f"   💾 Best model saved at: {cer_callback.best_checkpoint_path}")
    print("=" * 80)

    # Test inference
    print("\n" + "=" * 80)
    print("PHASE 2: EVALUATION ON TEST SET")
    print("=" * 80)

    print(f"\n⏳ Loading best model from: {cer_callback.best_checkpoint_path}")

    # Load best model (LoRA+base in 4-bit again)
    best_model = AutoModelForCausalLM.from_pretrained(
        cer_callback.best_checkpoint_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation="eager",
    )

    best_processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        num_crops=16,
    )

    print("   ✅ Best model and processor loaded")

    best_model.eval()

    # Run test
    test_data = load_jsonl(CONFIG["test_jsonl"])
    print(f"\n📊 Running inference on {len(test_data)} test samples...\n")

    results = []
    cer_scores = []

    device = next(best_model.parameters()).device

    for idx, item in enumerate(test_data):
        image_name = item["image_name"]
        print(f"[{idx+1}/{len(test_data)}] Processing {image_name}")

        image_path = os.path.join(CONFIG["images_dir"], image_name)

        if not os.path.exists(image_path):
            print("   ❌ Image not found")
            results.append(
                {
                    "image_name": image_name,
                    "predicted_text": "",
                    "ground_truth_text": create_label_string(item),
                    "cer_score": 1.0,
                    "error": "Image not found",
                }
            )
            cer_scores.append(1.0)
            continue

        try:
            image = Image.open(image_path).convert("RGB")

            # ADDED: 4-step preprocessing before giving image to the model (on-the-fly)
            image = preprocess_exact_4steps_pil(
                image,
                target_size=(1024, 1024),
                bg_kernel_size=51,
                denoise_method="bilateral",
            )

            messages = [
                {
                    "role": "user",
                    "content": f"<|image_1|>\n{INSTRUCTION}",
                }
            ]

            prompt = best_processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = best_processor(prompt, [image], return_tensors="pt").to(device)

            with torch.no_grad():
                generate_ids = best_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=best_processor.tokenizer.eos_token_id,
                )

            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

            raw_output = best_processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            raw_output = normalize_unicode(raw_output)
            prediction = extract_json_from_response(raw_output)
            ground_truth = create_label_string(item)

            cer = (
                jiwer.cer(ground_truth, prediction)
                if len(ground_truth) > 0
                else (1.0 if len(prediction) > 0 else 0.0)
            )
            cer_scores.append(cer)

            status = (
                "✨"
                if cer == 0.0
                else "✅"
                if cer < 0.1
                else "⚠️"
                if cer < 0.3
                else "❌"
            )
            print(f"   {status} CER: {cer:.4f} ({cer*100:.2f}%)")

            results.append(
                {
                    "image_name": image_name,
                    "predicted_text": prediction,
                    "ground_truth_text": ground_truth,
                    "raw_output": raw_output,
                    "cer_score": cer,
                }
            )

            del inputs, generate_ids

            if (idx + 1) % 3 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append(
                {
                    "image_name": image_name,
                    "predicted_text": "",
                    "ground_truth_text": create_label_string(item),
                    "cer_score": 1.0,
                    "error": str(e),
                }
            )
            cer_scores.append(1.0)

    # Save results
    results_file = os.path.join(CONFIG["output_dir"], "test_predictions.jsonl")
    save_jsonl(results, results_file)

    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    perfect_matches = sum(1 for cer in cer_scores if cer == 0.0)
    good_matches = sum(1 for cer in cer_scores if cer < 0.1)

    print(f"\n{'='*80}")
    print("📊 FINAL TEST RESULTS - STAIRCASE DATASET")
    print(f"{'='*80}")
    print(f"   Total Samples: {len(test_data)}")
    print(f"   Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(
        f"   Perfect Matches (CER=0): {perfect_matches}/{len(cer_scores)} "
        f"({perfect_matches/len(cer_scores)*100:.1f}%)"
    )
    print(
        f"   Good Matches (CER<0.1): {good_matches}/{len(cer_scores)} "
        f"({good_matches/len(cer_scores)*100:.1f}%)"
    )
    print(f"   Results saved to: {results_file}")
    print(f"{'='*80}")

    # Save summary with CER history
    summary_file = os.path.join(CONFIG["output_dir"], "training_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("PHI-3.5-VISION STAIRCASE DATASET FINE-TUNING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(
            f"Best Validation CER: {cer_callback.best_cer:.4f} "
            f"({cer_callback.best_cer*100:.2f}%)\n"
        )
        f.write(f"Test CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)\n")
        f.write(
            f"Perfect Matches: {perfect_matches}/{len(cer_scores)} "
            f"({perfect_matches/len(cer_scores)*100:.1f}%)\n"
        )
        f.write(
            f"Good Matches (CER<0.1): {good_matches}/{len(cer_scores)} "
            f"({good_matches/len(cer_scores)*100:.1f}%)\n\n"
        )
        f.write("Anti-Repetition Measures:\n")
        f.write("  - First-JSON extraction enabled\n")
        f.write("  - Using temperature=0.0 for deterministic generation\n\n")
        f.write("CER History:\n")
        for entry in cer_callback.cer_history:
            f.write(
                f"  Epoch {entry['epoch']}: {entry['cer']:.4f} "
                f"({entry['cer']*100:.2f}%)\n"
            )

    print(f"\n✅ Summary saved to: {summary_file}")
    print("\n🎉 Fine-tuning and evaluation complete!\n")


if __name__ == "__main__":
    main()
