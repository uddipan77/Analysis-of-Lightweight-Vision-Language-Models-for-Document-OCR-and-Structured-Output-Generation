#!/usr/bin/env python3
# gemma3_staircase_finetune_A100_aug1_genCER_best.py
#
# ✅ Full updated script to reliably beat ~17% CER by:
#    1) Using the "14% style" training regime (LoRA r=32/alpha=32/dropout=0.05, lr=3e-5, grad_accum=8, etc.)
#    2) Selecting BEST checkpoint by **generation-based CER** on a fixed validation subset (NOT teacher-forced CER)
#    3) Using compact canonical JSON targets to reduce sequence length + truncation risk
#    4) Making augmentation deterministic (seeded)
#
# ✅ FIX APPLIED:
#    Hugging Face Trainer expects best-metric keys as "eval_<metric>".
#    So the callback now injects:
#       metrics["eval_gen_cer"]
#       metrics["eval_gen_cer_percentage"]
#    (and also keeps the non-prefixed versions for convenience)

import sys
import os

# ✅ Required for compute_metrics logits
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import json
import shutil
import gc
import re
import random
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import jiwer
from PIL import Image
import torchvision.transforms as transforms

from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from transformers import EarlyStoppingCallback, TrainerCallback


# ----------------------- REPRO SEEDING -----------------------

def seed_everything(seed: int = 3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Determinism knobs (can slightly reduce speed, but helps stability)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------- CALLBACK: GENERATION CER ON VAL SUBSET -----------------------

class GenerationCEREvalCallback(TrainerCallback):
    """
    Computes generation-based CER on a fixed validation subset at each evaluation event (epoch),
    injects it into metrics as `eval_gen_cer`, and enables:
      load_best_model_at_end=True + metric_for_best_model="gen_cer"
    """
    def __init__(
        self,
        finetuner,
        val_raw_items: List[Dict[str, Any]],
        images_dir: str,
        subset_size: int = 50,
        subset_seed: int = 3407,
        max_new_tokens: int = 1024,
    ):
        self.finetuner = finetuner
        self.images_dir = images_dir
        self.max_new_tokens = max_new_tokens

        # Stable ordering + deterministic subset
        def stable_key(x: Dict[str, Any]) -> str:
            name = x.get("image_name", "")
            return name if isinstance(name, str) and name else json.dumps(
                x, ensure_ascii=False, sort_keys=True
            )

        sorted_val = sorted(val_raw_items, key=stable_key)

        rng = random.Random(int(subset_seed))
        if len(sorted_val) > subset_size:
            idxs = list(range(len(sorted_val)))
            rng.shuffle(idxs)
            self.val_subset = [sorted_val[i] for i in idxs[:subset_size]]
        else:
            self.val_subset = sorted_val

        print(f"[GenCER Callback] Fixed val subset size = {len(self.val_subset)}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        IMPORTANT:
        Trainer expects evaluation metrics keys to be prefixed with "eval_".
        If metric_for_best_model="gen_cer", Trainer will look for "eval_gen_cer".
        """
        if metrics is None:
            metrics = {}

        model = kwargs.get("model", None)
        if model is None:
            return control

        gen_cer = self.finetuner.evaluate_generation_cer_on_items(
            model=model,
            val_items=self.val_subset,
            images_dir=self.images_dir,
            max_new_tokens=self.max_new_tokens,
        )

        # ✅ MUST provide eval_* for Trainer best-metric / early stopping
        metrics["eval_gen_cer"] = float(gen_cer)
        metrics["eval_gen_cer_percentage"] = float(gen_cer) * 100.0

        # Optional: also store non-prefixed versions for readability
        metrics["gen_cer"] = float(gen_cer)
        metrics["gen_cer_percentage"] = float(gen_cer) * 100.0

        print(f"\n[GenCER Callback] gen_cer = {gen_cer:.4f} ({gen_cer*100:.2f}%)")
        return control


# ----------------------- MAIN FINETUNE CLASS -----------------------

class StaircaseGemma3Finetune:
    def __init__(self, model_path: str, augment_factor: int = 1, seed: int = 3407):
        print("Loading Gemma-3 vision model with Unsloth for STAIRCASE...")

        self.seed = int(seed)
        self.augment_factor = augment_factor

        # Seed early so augmentation becomes deterministic
        seed_everything(self.seed)

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

        # ✅ Match the "14% style" LoRA regime
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
            random_state=self.seed,
            use_rslora=True,
        )

        print("Gemma-3 vision model loaded with LoRA config: r=32, alpha=32, dropout=0.05, rslora=True")

    # ----------------------- PROMPT -----------------------

    def staircase_instruction(self) -> str:
        return (
            "You are an OCR model for historical staircase survey forms.\n\n"
            "Task:\n"
            "Given ONE image of a filled-in staircase form, read all printed text, "
            "handwritten notes and all checked/unchecked boxes and output a single JSON object "
            "that represents the complete form.\n\n"
            "Rules:\n"
            "- Return ONLY one valid JSON object, with no extra text before or after it.\n"
            "- Use exactly the same field names, nesting, accents, and capitalization as in the "
            "training JSON for this form type.\n"
            "- Never drop a key that appears in the form’s JSON structure. If a field is empty on "
            "the form, still include it with an empty string \"\" (or false for an unchecked box).\n"
            "- Use booleans for checkbox options: true if the box is checked, false if it is empty.\n"
            "- Use strings for numbers and free-text fields.\n"
            "- Do NOT invent new fields."
        )

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
        return {k: v for k, v in obj.items() if k not in ["image_name", "image_path"]}

    # ----------------------- JSON CANONICALIZATION -----------------------

    def _canonicalize_json(self, obj):
        if isinstance(obj, dict):
            return {k: self._canonicalize_json(obj[k]) for k in sorted(obj.keys())}
        elif isinstance(obj, list):
            return [self._canonicalize_json(x) for x in obj]
        else:
            return obj

    def json_to_string_target(self, obj) -> str:
        """
        Compact canonical JSON string for training targets and CER stability.
        Shorter than indent=2 => fewer tokens => less truncation.
        """
        clean_obj = self.dict_without_image_name(obj)
        canonical = self._canonicalize_json(clean_obj)
        return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))

    def json_to_string_readable(self, obj) -> str:
        """Readable version for saving outputs (not used for CER)."""
        clean_obj = self.dict_without_image_name(obj)
        canonical = self._canonicalize_json(clean_obj)
        return json.dumps(canonical, ensure_ascii=False, indent=2)

    # ----------------------- AUGMENTATION -----------------------

    def create_augmentation_transforms(self):
        return [
            transforms.ColorJitter(brightness=(0.9, 1.1)),
            transforms.ColorJitter(contrast=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
        ]

    def augment_image(self, image_path: str, aug_id: int) -> Optional[str]:
        try:
            image = Image.open(image_path).convert("RGB")
            transforms_list = self.create_augmentation_transforms()

            # deterministic randomness (because we seeded global random)
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

    def extract_json_from_response(self, response: str) -> Dict:
        if isinstance(response, list):
            response = response[0] if response else ""
        if response is None:
            return {}

        text = str(response).strip()
        if not text:
            return {}

        # Strip Markdown fences
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

        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            core = text[start:end + 1]
        else:
            core = text

        parsed = try_parse(core)
        if parsed is not None:
            return parsed

        brace_positions = [m.start() for m in re.finditer(r"\}", core)]
        for pos in reversed(brace_positions):
            candidate = core[:pos + 1]
            parsed = try_parse(candidate)
            if parsed is not None:
                return parsed

        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, core, re.DOTALL)
        if matches:
            for match in sorted(matches, key=len, reverse=True):
                parsed = try_parse(match)
                if parsed is not None:
                    return parsed

        parsed = try_parse(text)
        if parsed is not None:
            return parsed

        return {}

    # ----------------------- CONVERSIONS -----------------------

    def convert_to_conversation(self, sample):
        instruction = self.staircase_instruction()
        gt_json_string = self.json_to_string_target(sample)  # ✅ compact targets

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": instruction},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": gt_json_string}]},
        ]
        return {"messages": conversation}

    # ----------------------- DATA PREP -----------------------

    def prepare_training_data_with_augmentation(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)

        processed = []
        aug_count = 0

        print(f"Preparing TRAIN data with augmentation factor = {self.augment_factor}")

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
                aug_path = self.augment_image(image_path, aug_id)
                if aug_path and os.path.exists(aug_path):
                    aug_item = item.copy()
                    aug_item["image_path"] = aug_path
                    aug_item["image_name"] = os.path.basename(aug_path)
                    processed.append(aug_item)
                    aug_count += 1

        print(f"Training: {len(processed)} samples (original: {len(data)}, augmented: {aug_count})")
        return [self.convert_to_conversation(sample) for sample in processed]

    def prepare_data_no_augmentation(self, jsonl_path: str, images_dir: str) -> List[Dict]:
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

        print(f"Val/Test: {len(processed)} valid samples out of {len(data)} total (no augmentation)")
        return [self.convert_to_conversation(sample) for sample in processed]

    # ----------------------- METRICS (Teacher-forced, for logging only) -----------------------

    def calculate_cer(self, predictions, targets):
        if (
            predictions is None
            or targets is None
            or len(predictions) == 0
            or len(targets) == 0
            or len(predictions) != len(targets)
        ):
            return 1.0

        total = 0.0
        count = 0
        for p, t in zip(predictions, targets):
            t = str(t)
            p = str(p)
            if t:
                try:
                    total += jiwer.cer(t, p)
                except Exception:
                    total += 1.0
                count += 1
        return total / count if count > 0 else 1.0

    def compute_metrics_for_trainer(self, eval_preds, compute_result: bool = True):
        """
        Teacher-forced argmax CER on assistant tokens only.
        Kept for logging; BEST checkpoint is selected by gen_cer callback.
        """
        if not compute_result:
            return {}

        predictions, label_ids = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if hasattr(predictions, "__class__") and "EmptyLogits" in str(predictions.__class__):
            return {"cer": 0.0, "cer_percentage": 0.0}

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().float().numpy()
        if isinstance(label_ids, torch.Tensor):
            label_ids = label_ids.detach().cpu().numpy()

        pred_token_ids = np.argmax(predictions, axis=-1)

        pred_texts, label_texts = [], []
        for pred_ids, label_ids_single in zip(pred_token_ids, label_ids):
            label_ids_single = np.array(label_ids_single, dtype=np.int32)
            mask = label_ids_single != -100
            if not np.any(mask):
                continue
            label_clean = label_ids_single[mask]
            pred_clean = np.array(pred_ids, dtype=np.int32)[mask]

            try:
                pred_text = self.tokenizer.decode(pred_clean, skip_special_tokens=True)
                label_text = self.tokenizer.decode(label_clean, skip_special_tokens=True)
            except Exception:
                pred_text, label_text = "", ""

            pred_texts.append(pred_text)
            label_texts.append(label_text)

        cer = self.calculate_cer(pred_texts, label_texts)
        torch.cuda.empty_cache()
        gc.collect()

        return {"cer": cer, "cer_percentage": cer * 100.0}

    # ----------------------- GENERATION-BASED VALIDATION CER (for best checkpoint) -----------------------

    def evaluate_generation_cer_on_items(
        self,
        model,
        val_items: List[Dict[str, Any]],
        images_dir: str,
        max_new_tokens: int = 1024,
    ) -> float:
        FastVisionModel.for_inference(model)
        model.eval()
        device = next(model.parameters()).device

        instruction = self.staircase_instruction()

        preds, tgts = [], []

        for item in val_items:
            img_name = item.get("image_name", "")
            image_path = os.path.join(images_dir, img_name)

            gt_item = dict(item)
            gt_item["image_path"] = image_path
            gt_str = self.json_to_string_target(gt_item)

            if not os.path.exists(image_path):
                preds.append("")
                tgts.append(gt_str)
                continue

            image = None
            try:
                image = Image.open(image_path).convert("RGB")

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
                pred_str = self.json_to_string_target(pred_json) if pred_json else ""

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

        # restore training mode (important for Trainer)
        FastVisionModel.for_training(model)
        model.train()

        return avg_cer

    # ----------------------- TRAINING -----------------------

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 12,
        batch_size: int = 2,
        learning_rate: float = 3e-5,  # ✅ match better regime
        gen_val_subset_size: int = 50,  # increase for stability if you can (e.g., 100)
        gen_val_max_new_tokens: int = 1024,
    ):
        print("Preparing training and validation datasets for STAIRCASE...")

        seed_everything(self.seed)

        train_dataset = self.prepare_training_data_with_augmentation(train_jsonl_path, images_dir)
        val_dataset = self.prepare_data_no_augmentation(val_jsonl_path, images_dir)

        # raw val (for gen CER callback)
        val_raw = self.load_jsonl(val_jsonl_path)

        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

        FastVisionModel.for_training(self.model)

        gen_cer_callback = GenerationCEREvalCallback(
            finetuner=self,
            val_raw_items=val_raw,
            images_dir=images_dir,
            subset_size=gen_val_subset_size,
            subset_seed=self.seed,
            max_new_tokens=gen_val_max_new_tokens,
        )

        # IMPORTANT: put gen_cer_callback BEFORE EarlyStoppingCallback
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics_for_trainer,  # logging only
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=1,

                # ✅ match better regime
                gradient_accumulation_steps=8,
                eval_accumulation_steps=4,
                batch_eval_metrics=True,

                warmup_steps=100,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                weight_decay=0.01,
                warmup_ratio=0.1,
                lr_scheduler_type="cosine",
                max_grad_norm=1.0,

                optim="adamw_torch_fused",

                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,

                load_best_model_at_end=True,
                metric_for_best_model="gen_cer",
                greater_is_better=False,

                dataloader_num_workers=0,
                dataloader_pin_memory=False,

                bf16=True,
                fp16=False,

                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},

                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=2048,

                report_to="tensorboard",
                logging_dir=f"{output_dir}/logs",
                logging_first_step=True,
                seed=self.seed,
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
                ),
            ],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting training (best checkpoint selected by gen_cer)...")
        trainer_stats = trainer.train()

        # Helpful debug
        print("\n=== BEST CHECKPOINT INFO ===")
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best metric (gen_cer): {trainer.state.best_metric}")

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_pct = round(used_memory / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"Peak reserved memory = {used_memory} GB ({used_pct}%).")

        return trainer

    # ----------------------- TEST EVAL -----------------------

    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict:
        print("Starting evaluation on STAIRCASE test.jsonl...")

        FastVisionModel.for_inference(self.model)
        self.model.eval()

        test_data = self.load_jsonl(test_jsonl_path)
        print(f"Loaded {len(test_data)} test samples")

        predictions = []
        all_cer_scores = []

        instruction = self.staircase_instruction()

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

                gt_str_compact = self.json_to_string_target({**test_item, "image_path": image_path})

                if not os.path.exists(image_path):
                    print("  ⚠️  Image not found")
                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": {},
                        "predicted_text_readable": "",
                        "target_text_readable": self.json_to_string_readable(test_item),
                        "raw_response": "Error: Image not found",
                        "cer_score": 1.0,
                    }
                    predictions.append(prediction_entry)
                    all_cer_scores.append(1.0)
                    continue

                image = None
                try:
                    image = Image.open(image_path).convert("RGB")

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
                    gen_ids = outputs[0][input_len:]
                    gen_text = self.tokenizer.decode(
                        gen_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    predicted_json = self.extract_json_from_response(gen_text)
                    pred_str_compact = self.json_to_string_target(predicted_json) if predicted_json else ""

                    cer_score = jiwer.cer(gt_str_compact, pred_str_compact) if pred_str_compact else 1.0

                    prediction_entry = {
                        "image_name": img_name,
                        "predicted_json": predicted_json,
                        "predicted_text_readable": self.json_to_string_readable(predicted_json) if predicted_json else "",
                        "target_text_readable": self.json_to_string_readable(test_item),
                        "raw_response": gen_text,
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
                        "predicted_text_readable": "",
                        "target_text_readable": self.json_to_string_readable(test_item),
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
                    if "gen_ids" in locals():
                        del gen_ids

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
        avg_cer = float(sum(all_cer_scores) / len(all_cer_scores))
        return {
            "total_images": len(all_cer_scores),
            "average_cer": avg_cer,
            "median_cer": float(np.median(all_cer_scores)),
            "minimum_cer": float(min(all_cer_scores)),
            "maximum_cer": float(max(all_cer_scores)),
            "std_cer": float(np.std(all_cer_scores)),
            "perfect_matches": int(sum(1 for cer in all_cer_scores if cer == 0.0)),
        }

    def save_cer_results(self, cer_stats: Dict, cer_file: str, num_predictions: int):
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("CER EVALUATION RESULTS - GEMMA-3 STAIRCASE (A100)\n")
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

    def cleanup_temp_files(self):
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {self.temp_dir}: {e}")


# ----------------------- MAIN -----------------------

def main():
    base_checkpoint_dir = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/stair/finetune"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_A100_genCER_{timestamp}")
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
        "learning_rate": 3e-5,     # ✅ match better regime
        "augment_factor": 1,
        "seed": 3407,
        "gen_val_subset_size": 50,  # If you can afford it, use 100 for more stable selection
        "gen_val_max_new_tokens": 1024,
    }

    with open(os.path.join(run_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    local_model_path = (
        "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/"
        "models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/"
        "snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"
    )

    print("\n" + "=" * 70)
    print("GEMMA-3 STAIRCASE - GEN-CER BEST CHECKPOINT TRAINING")
    print("=" * 70)

    finetuner = StaircaseGemma3Finetune(
        model_path=local_model_path,
        augment_factor=config["augment_factor"],
        seed=config["seed"],
    )

    try:
        print("\n" + "=" * 70)
        print("STARTING TRAINING (best checkpoint selected by generation CER)")
        print("=" * 70)

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

        print("\n" + "=" * 70)
        print("STARTING EVALUATION ON TEST.JSONL")
        print("=" * 70)

        test_results = finetuner.evaluate_on_test_set(
            test_jsonl_path=config["test_jsonl_path"],
            images_dir=config["images_dir"],
            output_dir=config["output_dir"],
        )

        print(f"\n🎉 Training completed!")
        print(f"All outputs saved to: {run_dir}")
        print(
            f"\nFinal TEST CER: {test_results['cer_stats']['average_cer']:.4f} "
            f"({test_results['cer_stats']['average_cer']*100:.2f}%)"
        )

    finally:
        finetuner.cleanup_temp_files()


if __name__ == "__main__":
    main()
