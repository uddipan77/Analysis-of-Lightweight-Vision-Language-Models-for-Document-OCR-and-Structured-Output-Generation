#!/usr/bin/env python3
# gemma3_inventory_finetune_A100_simple.py
# ✅ A100-OPTIMIZED with evaluation chunking + memory management
# ✅ Simplified structure (like Schmuck) - trainer-based compute_metrics
# ✅ NO key sorting - preserves original JSON key order from JSONL
# ✅ Inventory dataset with German field names
# ✅ Added: 4-step image preprocessing applied in TRAIN + VAL(gen-CER) + TEST

import sys
import os

# ✅ CRITICAL: Enable logits return for compute_metrics
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
import json
import shutil
from typing import List, Dict, Any
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
import re
import jiwer
from datetime import datetime
from transformers import EarlyStoppingCallback, TrainerCallback
import numpy as np
import gc
from PIL import Image
import random
import cv2


# =====================================================================
# 4-step image preprocessing (illumination flatten + denoise + CLAHE + letterbox)
# =====================================================================

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


def preprocess_exact_4steps_pil(
    pil_rgb: Image.Image,
    target_size=(1024, 1024),
    bg_kernel_size=51,
    denoise_method="bilateral",
) -> Image.Image:
    """
    4 steps:
      1) Illumination flattening (grayscale)
      2) Gentle denoising
      3) CLAHE (luminance)
      4) Letterbox resize
    Returns PIL RGB.
    """
    rgb = np.array(pil_rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1) illumination flatten (gray)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flat = illumination_flatten(gray, bg_kernel_size=bg_kernel_size)

    # 2) denoise
    if denoise_method == "bilateral":
        den = cv2.bilateralFilter(flat, d=7, sigmaColor=50, sigmaSpace=50)
    elif denoise_method == "nlm":
        den = cv2.fastNlMeansDenoising(flat, None, h=7, templateWindowSize=7, searchWindowSize=21)
    else:
        raise ValueError("denoise_method must be 'bilateral' or 'nlm'")

    # 3) CLAHE
    den_bgr = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    clahe_bgr = clahe_on_luminance(den_bgr, clip_limit=2.0, tile_grid_size=(8, 8))
    out_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # 4) letterbox
    out = letterbox_resize(out_gray, target_size=target_size, pad_value=255)

    return Image.fromarray(out).convert("RGB")


# =====================================================================
# Generation CER callback (AUTOREGRESSIVE, for best model selection)
# =====================================================================

class GenerationCEREvalCallback(TrainerCallback):
    """
    Computes autoregressive generation CER on a fixed validation subset during evaluation.
    Injects:
      metrics["eval_gen_cer"]
      metrics["eval_gen_cer_percentage"]
    """

    def __init__(
        self,
        finetuner,
        val_raw_items: List[Dict[str, Any]],
        images_dir: str,
        subset_size: int = 50,
        subset_seed: int = 3407,
        max_new_tokens: int = 512,
    ):
        self.finetuner = finetuner
        self.images_dir = images_dir
        self.max_new_tokens = max_new_tokens

        # stable ordering + deterministic subset selection (NO sorting of JSON keys)
        def stable_key(x: Dict[str, Any]) -> str:
            name = x.get("image_name", "")
            return name if isinstance(name, str) and name else str(hash(json.dumps(x, ensure_ascii=False)))

        sorted_val = sorted(val_raw_items, key=stable_key)

        rng = random.Random(int(subset_seed))
        if len(sorted_val) > subset_size:
            idxs = list(range(len(sorted_val)))
            rng.shuffle(idxs)
            self.val_subset = [sorted_val[i] for i in idxs[:subset_size]]
        else:
            self.val_subset = sorted_val

        print(f"[GenCER Callback] Fixed val subset size = {len(self.val_subset)}")
        self.cer_history: List[Dict[str, float]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}

        model = kwargs.get("model", None)
        if model is None:
            return control

        epoch = state.epoch if state.epoch is not None else 0.0

        gen_cer = self.finetuner.evaluate_generation_cer_on_items(
            model=model,
            val_items=self.val_subset,
            images_dir=self.images_dir,
            max_new_tokens=self.max_new_tokens,
        )

        metrics["eval_gen_cer"] = float(gen_cer)
        metrics["eval_gen_cer_percentage"] = float(gen_cer) * 100.0

        # optional non-prefixed copies
        metrics["gen_cer"] = float(gen_cer)
        metrics["gen_cer_percentage"] = float(gen_cer) * 100.0

        self.cer_history.append({"epoch": float(epoch), "cer": float(gen_cer)})

        print(f"\n[GenCER Callback] Epoch {epoch:.2f} gen_cer = {gen_cer:.4f} ({gen_cer*100:.2f}%)")
        return control


# =====================================================================
# Lazy dataset to avoid pre-loading all images into RAM
# (applies preprocess at __getitem__, so TRAIN/VAL both see it)
# =====================================================================

class InventoryConversationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_items: List[Dict[str, Any]],
        images_dir: str,
        make_conversation_fn,
        preprocess_fn,
        preprocess_cfg: Dict[str, Any],
        strict_existing: bool = True,
    ):
        self.images_dir = images_dir
        self.make_conversation_fn = make_conversation_fn
        self.preprocess_fn = preprocess_fn
        self.preprocess_cfg = preprocess_cfg

        filtered = []
        for item in raw_items:
            img_name = item.get("image_name", "")
            img_path = os.path.join(images_dir, img_name)
            if strict_existing and not os.path.exists(img_path):
                continue
            item2 = dict(item)
            item2["image_path"] = img_path
            filtered.append(item2)

        self.items = filtered
        print(f"   📊 Loaded {len(self.items)} valid samples (out of {len(raw_items)} total)")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]

        # Load + preprocess image here (TRAIN/VAL use the same)
        img = Image.open(sample["image_path"]).convert("RGB")
        img = self.preprocess_fn(
            img,
            target_size=self.preprocess_cfg.get("target_size", (1024, 1024)),
            bg_kernel_size=self.preprocess_cfg.get("bg_kernel_size", 51),
            denoise_method=self.preprocess_cfg.get("denoise_method", "bilateral"),
        )

        # Build conversation with PIL image (NOT path)
        return self.make_conversation_fn(sample, preprocessed_image=img)


class InventoryGemma3Finetune:
    def __init__(self, model_path: str):
        print("Loading Gemma-3 vision model with Unsloth for INVENTORY...")

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

        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
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
        return {k: v for k, v in obj.items() if k not in ["image_name", "image_path"]}

    def json_to_string_readable(self, obj):
        clean_obj = self.dict_without_image_name(obj)
        return json.dumps(clean_obj, ensure_ascii=False, indent=2)

    def extract_json_from_response(self, response: str) -> Dict:
        response = response.strip()

        if response.startswith("```"):
            response = response.split("```", 1)[1]
            if response.startswith("json"):
                response = response[4:].strip()
            if "```" in response:
                response = response.split("```")[0].strip()

        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    def get_instruction(self) -> str:
        return """Du bist ein OCR- und Information-Extraction-Modell für deutsche historische Inventardokumente.

Aufgabe:
Lies ALLE Informationen aus dem Bild dieses Inventarblatts und gib GENAU EIN JSON-Objekt zurück.

Das JSON MUSS folgende Felder enthalten:
- Überschrift: Dokumenttitel
- Inventarnummer: Inventar- oder Katalognummer
- Maße: Objekt mit L, B, D (Länge, Breite, Tiefe)
- Objektbezeichnung: Beschreibung/Name des Objekts
- Fundort: Fundort des Objekts
- Fundzeit: Zeit der Auffindung
- Beschreibungstext: Ausführlicher Beschreibungstext

Regeln:
- Gib NUR ein gültiges JSON-Objekt zurück (kein extra Text davor oder danach).
- Verwende GENAU diese Feldnamen und Groß-/Kleinschreibung.
- Wenn ein Feld leer ist oder nicht sichtbar, gib einen leeren String "" zurück.
- Das Feld "Maße" MUSS immer ein Objekt mit den Schlüsseln "L", "B", "D" sein, auch wenn leer.
- Erfinde keine zusätzlichen Felder."""

    # ✅ Updated: accept preprocessed_image so dataset can pass PIL directly
    def convert_to_conversation(self, sample, preprocessed_image: Image.Image):
        instruction = self.get_instruction()
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
                "content": [{"type": "text", "text": gt_json_string}],
            },
        ]
        return {"messages": conversation}

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

    # =====================================================================
    # ✅ Autoregressive Generation CER evaluation (uses SAME preprocess)
    # =====================================================================

    def evaluate_generation_cer_on_items(
        self,
        model,
        val_items: List[Dict[str, Any]],
        images_dir: str,
        max_new_tokens: int = 512,
    ) -> float:
        FastVisionModel.for_inference(model)
        model.eval()
        device = next(model.parameters()).device

        instruction = self.get_instruction()
        preds, tgts = [], []

        for idx, item in enumerate(val_items):
            img_name = item.get("image_name", f"val_{idx}")
            image_path = os.path.join(images_dir, img_name)

            gt_str = self.json_to_string_readable(item)

            if not os.path.exists(image_path):
                preds.append("")
                tgts.append(gt_str)
                continue

            image = None
            try:
                image = Image.open(image_path).convert("RGB")

                # ✅ PREPROCESS HERE (VAL gen-CER)
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
                ).to(device)

                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        repetition_penalty=1.0,
                        use_cache=True,
                    )

                input_len = inputs["input_ids"].shape[-1]
                gen_ids = outputs[0][input_len:]
                gen_text = self.tokenizer.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                pred_json = self.extract_json_from_response(gen_text)
                pred_str = self.json_to_string_readable(pred_json) if pred_json else ""

                preds.append(pred_str)
                tgts.append(gt_str)

            except Exception:
                preds.append("")
                tgts.append(gt_str)
            finally:
                if image is not None:
                    image.close()
                    del image
                if "inputs" in locals():
                    del inputs
                if "outputs" in locals():
                    del outputs
                if "gen_ids" in locals():
                    del gen_ids
                torch.cuda.empty_cache()
                gc.collect()

        avg_cer = self.calculate_cer(preds, tgts)

        FastVisionModel.for_training(model)
        model.train()

        return avg_cer

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 12,
        batch_size: int = 2,
        learning_rate: float = 3e-5,
        gen_val_subset_size: int = 50,
        gen_val_max_new_tokens: int = 1024,
    ):
        print("Preparing training and validation datasets...")

        train_raw = self.load_jsonl(train_jsonl_path)
        val_raw = self.load_jsonl(val_jsonl_path)

        # ✅ Lazy datasets (preprocess applied inside __getitem__)
        train_dataset = InventoryConversationDataset(
            raw_items=train_raw,
            images_dir=images_dir,
            make_conversation_fn=self.convert_to_conversation,
            preprocess_fn=preprocess_exact_4steps_pil,
            preprocess_cfg=self.preprocess_cfg,
            strict_existing=True,
        )

        val_dataset = InventoryConversationDataset(
            raw_items=val_raw,
            images_dir=images_dir,
            make_conversation_fn=self.convert_to_conversation,
            preprocess_fn=preprocess_exact_4steps_pil,
            preprocess_cfg=self.preprocess_cfg,
            strict_existing=True,
        )

        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

        FastVisionModel.for_training(self.model)

        gen_cer_callback = GenerationCEREvalCallback(
            finetuner=self,
            val_raw_items=val_raw,   # raw items for callback subset selection
            images_dir=images_dir,
            subset_size=gen_val_subset_size,
            subset_seed=3407,
            max_new_tokens=gen_val_max_new_tokens,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=None,  # using callback for AUTOREGRESSIVE CER
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
                metric_for_best_model="gen_cer",
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
                label_smoothing_factor=0.05,
            ),
            callbacks=[
                gen_cer_callback,
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.01,
                )
            ],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting training with AUTOREGRESSIVE CER-based evaluation (A100-optimized)...")
        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        return trainer

    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict:
        print("Starting evaluation on test.jsonl with aggressive chunking...")

        FastVisionModel.for_inference(self.model)

        test_data = self.load_jsonl(test_jsonl_path)
        print(f"Loaded {len(test_data)} test samples")

        predictions = []
        all_cer_scores = []

        instruction = self.get_instruction()

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
                print(f"\n[{abs_idx}/{len(test_data)}] Processing: {test_item['image_name']}")

                image_path = os.path.join(images_dir, test_item["image_name"])

                if not os.path.exists(image_path):
                    print("  ⚠️  Image not found")
                    continue

                image = None
                try:
                    image = Image.open(image_path).convert("RGB")

                    # ✅ PREPROCESS HERE (TEST)
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
                            max_new_tokens=1024,
                            do_sample=False,
                            repetition_penalty=1.0,
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
                    pred_json_string = self.json_to_string_readable(predicted_json) if predicted_json else ""

                    cer_score = jiwer.cer(gt_json_string, pred_json_string) if pred_json_string else 1.0

                    prediction_entry = {
                        "image_name": test_item["image_name"],
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
                        "image_name": test_item["image_name"],
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
                        image.close()
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
                intermediate_file = os.path.join(output_dir, f"test_predictions_chunk_{chunk_idx+1}.jsonl")
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
            f.write("CER EVALUATION RESULTS - GEMMA-3 INVENTORY (A100)\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"CER Statistics across {cer_stats['total_images']} images:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n")
            f.write(f"Median CER: {cer_stats['median_cer']:.4f} ({cer_stats['median_cer']*100:.2f}%)\n")
            f.write(f"Minimum CER: {cer_stats['minimum_cer']:.4f} ({cer_stats['minimum_cer']*100:.2f}%)\n")
            f.write(f"Maximum CER: {cer_stats['maximum_cer']:.4f} ({cer_stats['maximum_cer']*100:.2f}%)\n")
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


def main():
    base_checkpoint_dir = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/inventory_dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_A100_genCER_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Created run directory: {run_dir}")

    config = {
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",
        "output_dir": run_dir,
        "num_epochs": 12,
        "batch_size": 2,
        "learning_rate": 3e-5,
        "gen_val_subset_size": 50,
        "gen_val_max_new_tokens": 1024,
    }

    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    local_model_path = "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"

    print("\n" + "=" * 60)
    print("GEMMA-3 INVENTORY - A100 OPTIMIZED + AUTOREGRESSIVE GEN-CER")
    print("=" * 60)

    finetuner = InventoryGemma3Finetune(model_path=local_model_path)

    print("\n" + "=" * 60)
    print("STARTING TRAINING (BEST MODEL BY AUTOREGRESSIVE GEN-CER)")
    print("=" * 60)

    trainer = finetuner.train_model(
        train_jsonl_path=config["train_jsonl_path"],
        val_jsonl_path=config["val_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        gen_val_subset_size=config["gen_val_subset_size"],
        gen_val_max_new_tokens=config["gen_val_max_new_tokens"],
    )

    finetuner.save_model(trainer, config["output_dir"])

    print("\n" + "=" * 60)
    print("STARTING EVALUATION ON TEST.JSONL (USING BEST MODEL)")
    print("=" * 60)

    test_results = finetuner.evaluate_on_test_set(
        test_jsonl_path=config["test_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
    )

    print(f"\n🎉 Gemma-3 Inventory training completed!")
    print(f"All outputs saved to: {run_dir}")
    print(
        f"\nFinal TEST CER: {test_results['cer_stats']['average_cer']:.4f} "
        f"({test_results['cer_stats']['average_cer']*100:.2f}%)"
    )


if __name__ == "__main__":
    main()
