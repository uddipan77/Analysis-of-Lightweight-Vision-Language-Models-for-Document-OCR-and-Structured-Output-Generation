import torch
import json
import os
from typing import List, Dict
from unsloth import FastVisionModel

from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator

import re
import jiwer
from datetime import datetime
from transformers import EarlyStoppingCallback
import numpy as np
import glob
from PIL import Image
import random
import gc

# ============================
# OpenCV preprocessing
# ============================
import cv2


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
    Exact 4 processing steps:
      1) Illumination flattening (grayscale)
      2) Gentle denoising (bilateral or NLM)
      3) CLAHE (luminance)
      4) Letterbox resize to target_size
    Output is a PIL RGB image.
    """
    rgb = np.array(pil_rgb, dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 1) Illumination flattening
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    flat = illumination_flatten(gray, bg_kernel_size=bg_kernel_size)

    # 2) Denoising
    if denoise_method == "bilateral":
        den = cv2.bilateralFilter(flat, d=7, sigmaColor=50, sigmaSpace=50)
    elif denoise_method == "nlm":
        den = cv2.fastNlMeansDenoising(flat, None, h=7, templateWindowSize=7, searchWindowSize=21)
    else:
        raise ValueError("denoise_method must be 'bilateral' or 'nlm'")

    # 3) CLAHE on luminance
    den_bgr = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    clahe_bgr = clahe_on_luminance(den_bgr, clip_limit=2.0, tile_grid_size=(8, 8))
    out_gray = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)

    # 4) Letterbox resize
    out = letterbox_resize(out_gray, target_size=target_size, pad_value=255)

    return Image.fromarray(out).convert("RGB")


# ==========================================================
# Safe Unsloth collator applying preprocessing runtime
# ==========================================================
class PreprocessUnslothVisionDataCollator(UnslothVisionDataCollator):
    """
    Runtime-only preprocessing (NO saving, NO overwriting original files).
    GUARANTEE: never pass image=None to Unsloth.
    """

    def __init__(
        self,
        model,
        tokenizer,
        target_size=(1024, 1024),
        bg_kernel_size=51,
        denoise_method="bilateral",
        blank_fallback=True,
    ):
        super().__init__(model, tokenizer)
        self.target_size = target_size
        self.bg_kernel_size = bg_kernel_size
        self.denoise_method = denoise_method
        self.blank_fallback = blank_fallback

    def _blank_image(self):
        w, h = self.target_size[1], self.target_size[0]
        return Image.new("RGB", (w, h), (255, 255, 255))

    def _preprocess_any_image_obj(self, img_obj):
        """
        Accepts: str path, PIL.Image, list[str]
        Returns: PIL.Image (preprocessed) or None
        """
        if img_obj is None:
            return None

        if isinstance(img_obj, list):
            img_obj = next((x for x in img_obj if isinstance(x, str)), None)

        if isinstance(img_obj, str):
            if not os.path.exists(img_obj):
                return None
            pil_img = Image.open(img_obj).convert("RGB")
            return preprocess_exact_4steps_pil(
                pil_img,
                target_size=self.target_size,
                bg_kernel_size=self.bg_kernel_size,
                denoise_method=self.denoise_method,
            )

        if isinstance(img_obj, Image.Image):
            pil_img = img_obj.convert("RGB")
            return preprocess_exact_4steps_pil(
                pil_img,
                target_size=self.target_size,
                bg_kernel_size=self.bg_kernel_size,
                denoise_method=self.denoise_method,
            )

        return None

    def __call__(self, features):
        # 1) preprocess pass
        for ex_idx, ex in enumerate(features):
            msgs = ex.get("messages")
            if not isinstance(msgs, list):
                continue

            for msg in msgs:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue

                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        new_img = self._preprocess_any_image_obj(part.get("image", None))
                        part["image"] = new_img

        # 2) safety pass - never allow None
        for ex_idx, ex in enumerate(features):
            msgs = ex.get("messages")
            if not isinstance(msgs, list):
                continue

            for msg in msgs:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue

                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        if part.get("image", None) is None:
                            if self.blank_fallback:
                                part["image"] = self._blank_image()
                                print(
                                    f"[WARN] image=None in batch example {ex_idx}. "
                                    f"Replaced with blank image to avoid crash."
                                )
                            else:
                                raise ValueError(
                                    f"Found image=None in batch example {ex_idx}. "
                                    f"Set blank_fallback=True to avoid crash."
                                )

        return super().__call__(features)


# ============================
# Chunked dataset for test eval
# ============================
class ChunkedDataset:
    def __init__(self, jsonl_path: str, images_dir: str, chunk_size: int = 20):
        self.jsonl_path = jsonl_path
        self.images_dir = images_dir
        self.chunk_size = chunk_size
        self.total_samples = self._count_samples()
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        print(f"Dataset: {self.total_samples} samples, {self.num_chunks} chunks of size {chunk_size}")

    def _count_samples(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    def find_image_path(self, file_name: str, images_dir: str) -> str:
        exact_path = os.path.join(images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path

        base_name = os.path.splitext(file_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)

        if matching_files:
            matching_files.sort()
            return matching_files[0]

        return exact_path

    def get_chunk(self, chunk_idx: int) -> List[Dict]:
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)

        chunk_data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    continue
                if i >= end_idx:
                    break
                if line.strip():
                    item = json.loads(line.strip())
                    image_path = self.find_image_path(item["file_name"], self.images_dir)
                    if os.path.exists(image_path):
                        item["image_path"] = image_path
                        chunk_data.append(item)
                    else:
                        print(f"Warning: Image not found for {item['file_name']}, skipping...")

        return chunk_data


# ============================
# Main finetuner class
# ============================
class SchmuckOCRFinetune:
    def __init__(self, model_path: str = "/home/vault/iwi5/iwi5298h/models/qwen7b"):
        print(f"Loading Qwen2.5-VL model from {model_path} with Unsloth...")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )

        print("Model loaded successfully with optimized LoRA adapters!")

    @staticmethod
    def instruction_text():
        return """Extract ALL the jewelry information from this German historical document image as a complete JSON object. 

The JSON must include ALL of these fields with their exact German names:
- Gegenstand (type of jewelry - REQUIRED, never skip this)
- Inv.Nr (inventory number)
- Herkunft (origin/provenance) 
- Foto Notes (photo information)
- Standort (location/storage)
- Material (materials used)
- Datierung (dating information)
- Maße (measurements/dimensions)
- Gewicht (weight)
- erworben von (acquired from)
- am (date acquired)
- Preis (price)
- Vers.-Wert (insurance value)
- Beschreibung (description)
- Literatur (literature references)
- Ausstellungen (exhibitions)

Include ALL fields, even if empty (use empty string ""). Preserve exact German spelling and punctuation.
Return ONLY the JSON object without any additional text or formatting."""

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

    def clean_dict_for_ground_truth(self, obj):
        return {k: v for k, v in obj.items() if k not in ["file_name", "image_path"]}

    def json_to_string_no_sort(self, obj):
        clean_obj = self.clean_dict_for_ground_truth(obj)
        return json.dumps(clean_obj, ensure_ascii=False, separators=(",", ":"))

    def safe_json_loads(self, s):
        try:
            return json.loads(s)
        except Exception:
            return None

    def canonical_json_string(self, obj):
        return json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"))

    def extract_json_from_response(self, response: str) -> Dict:
        if isinstance(response, list):
            response = str(response[0]) if len(response) > 0 else ""
        response = str(response).strip()

        json_pattern = r"\{.*\}"
        matches = re.findall(json_pattern, response, re.DOTALL)

        if matches:
            for match in sorted(matches, key=len, reverse=True):
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 3:
                        return parsed
                except json.JSONDecodeError:
                    continue

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}

    # IMPORTANT: always return a STRING path, never a list
    def find_image_path(self, file_name: str, images_dir: str) -> str:
        exact_path = os.path.join(images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path

        base_name = os.path.splitext(file_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)

        if matching_files:
            matching_files.sort()
            return matching_files[0]

        return exact_path

    def convert_to_conversation(self, sample):
        instruction = self.instruction_text()
        gt_json_string = self.json_to_string_no_sort(sample)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image_path"]},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": gt_json_string}]},
        ]
        return {"messages": conversation}

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        data = self.load_jsonl(jsonl_path)
        for item in data:
            item["image_path"] = self.find_image_path(item["file_name"], images_dir)

        valid_data = [item for item in data if os.path.exists(item["image_path"])]
        print(f"Found {len(valid_data)} valid samples out of {len(data)} total")
        return [self.convert_to_conversation(sample) for sample in valid_data]

    def _micro_cer_from_strings(self, preds: List[str], gts: List[str]) -> Dict[str, float]:
        """
        MICRO CER = total edits / total GT chars
        Uses jiwer.process_characters for insert/sub/del, aggregated.
        """
        total_sub = total_del = total_ins = 0
        total_chars = 0

        for gt, pr in zip(gts, preds):
            gt = "" if gt is None else str(gt)
            pr = "" if pr is None else str(pr)
            total_chars += len(gt)
            # Use process_characters for character-level metrics (newer jiwer API)
            output = jiwer.process_characters(gt, pr)
            total_sub += output.substitutions
            total_del += output.deletions
            total_ins += output.insertions

        total_edits = total_sub + total_del + total_ins
        micro = (total_edits / total_chars) if total_chars > 0 else 1.0

        return {
            "micro_cer": micro,
            "total_chars": float(total_chars),
            "total_substitutions": float(total_sub),
            "total_deletions": float(total_del),
            "total_insertions": float(total_ins),
            "total_edits": float(total_edits),
        }

    def calculate_cer_statistics(self, per_sample_cer: List[float]) -> Dict:
        if not per_sample_cer:
            return {}

        avg_cer = float(np.mean(per_sample_cer))
        min_cer = float(np.min(per_sample_cer))
        max_cer = float(np.max(per_sample_cer))
        median_cer = float(np.median(per_sample_cer))
        std_cer = float(np.std(per_sample_cer))
        perfect_matches = int(sum(1 for x in per_sample_cer if x == 0.0))

        return {
            "total_images": int(len(per_sample_cer)),
            "average_cer": avg_cer,
            "median_cer": median_cer,
            "minimum_cer": min_cer,
            "maximum_cer": max_cer,
            "std_cer": std_cer,
            "perfect_matches": perfect_matches,
        }

    def save_cer_results(
        self,
        cer_stats: Dict,
        micro_stats: Dict,
        cer_file: str,
        num_predictions: int,
    ):
        with open(cer_file, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("CER EVALUATION RESULTS - SCHMUCK DATASET\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Images evaluated: {num_predictions}\n\n")

            f.write("PER-SAMPLE CER STATS (mean of per-sample CER)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Average CER: {cer_stats['average_cer']:.6f} ({cer_stats['average_cer']*100:.3f}%)\n")
            f.write(f"Median CER : {cer_stats['median_cer']:.6f} ({cer_stats['median_cer']*100:.3f}%)\n")
            f.write(f"Min CER    : {cer_stats['minimum_cer']:.6f} ({cer_stats['minimum_cer']*100:.3f}%)\n")
            f.write(f"Max CER    : {cer_stats['maximum_cer']:.6f} ({cer_stats['maximum_cer']*100:.3f}%)\n")
            f.write(f"Std CER    : {cer_stats['std_cer']:.6f}\n")
            f.write(f"Perfect    : {cer_stats['perfect_matches']}/{cer_stats['total_images']} "
                    f"({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n\n")

            f.write("MICRO CER (global edits / global GT chars)\n")
            f.write("-" * 60 + "\n")
            f.write(f"Micro CER: {micro_stats['micro_cer']:.6f} ({micro_stats['micro_cer']*100:.3f}%)\n")
            f.write(f"Total GT chars: {int(micro_stats['total_chars'])}\n")
            f.write(f"Total edits   : {int(micro_stats['total_edits'])}\n")
            f.write(f"  substitutions: {int(micro_stats['total_substitutions'])}\n")
            f.write(f"  deletions    : {int(micro_stats['total_deletions'])}\n")
            f.write(f"  insertions   : {int(micro_stats['total_insertions'])}\n\n")

            f.write("NOTES\n")
            f.write("-" * 60 + "\n")
            f.write("✅ Runtime-only preprocessing (no images saved/overwritten)\n")
            f.write("✅ Training uses safe collator to prevent image=None crash\n")
            f.write("✅ Test evaluation uses same 4-step preprocessing for consistency\n")

        print(f"CER evaluation results saved to: {cer_file}")

    def train_model(
        self,
        train_jsonl_path: str,
        val_jsonl_path: str,
        images_dir: str,
        output_dir: str,
        num_epochs: int = 15,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
    ):
        print("Preparing training and validation datasets...")
        train_dataset = self.prepare_training_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_training_data(val_jsonl_path, images_dir)
        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

        FastVisionModel.for_training(self.model)

        data_collator = PreprocessUnslothVisionDataCollator(
            model=self.model,
            tokenizer=self.tokenizer,
            target_size=(1024, 1024),
            bg_kernel_size=51,
            denoise_method="bilateral",
            blank_fallback=True,
        )

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
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
                save_total_limit=1,

                dataloader_num_workers=0,
                dataloader_pin_memory=False,

                weight_decay=0.05,
                lr_scheduler_type="cosine",
                warmup_ratio=0.1,

                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,

                optim="adamw_8bit",
                gradient_checkpointing=True,

                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=2048,

                report_to="tensorboard",
                logging_dir=f"{output_dir}/logs",
                seed=3407,
                output_dir=output_dir,
            ),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001,
                )
            ],
        )

        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        print("Starting training with early stopping...")
        trainer_stats = trainer.train()

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")  # noqa
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")  # noqa

        return trainer

    def save_model(self, trainer, output_dir: str):
        print(f"Saving model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully!")

    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict:
        """
        Generates predictions for ENTIRE test set, saves:
          - test_predictions.jsonl
          - cer_evaluation_results.txt
          - test_summary.json
        Computes:
          - per-sample CER stats
          - MICRO CER over entire test set
        """
        print("Starting evaluation on test.jsonl...")

        FastVisionModel.for_inference(self.model)

        test_chunked = ChunkedDataset(test_jsonl_path, images_dir, chunk_size=20)

        instruction = self.instruction_text()

        # Import here to match your working setup
        from qwen_vl_utils import process_vision_info

        all_predictions = []
        per_sample_cer = []

        # For micro CER from strings, we need the raw strings lists
        gt_texts = []
        pred_texts = []

        for chunk_idx in range(test_chunked.num_chunks):
            print(f"\nProcessing test chunk {chunk_idx + 1}/{test_chunked.num_chunks}")

            chunk_data = test_chunked.get_chunk(chunk_idx)

            for i, test_item in enumerate(chunk_data):
                print(f"Processing {i+1}/{len(chunk_data)} in chunk {chunk_idx + 1}", end="\r")
                try:
                    # preprocess image (runtime only)
                    pil_img = Image.open(test_item["image_path"]).convert("RGB")
                    pil_img = preprocess_exact_4steps_pil(pil_img)

                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": instruction},
                                {"type": "image", "image": pil_img},
                            ],
                        }
                    ]

                    input_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
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
                            temperature=0.1,
                            do_sample=True,
                            repetition_penalty=1.1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]

                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    generated_text = generated_texts[0] if generated_texts else ""

                    predicted_json = self.extract_json_from_response(generated_text)
                    pred_json_string = self.json_to_string_no_sort(predicted_json) if predicted_json else ""

                    gt_json_string = self.json_to_string_no_sort(test_item)

                    # per-sample CER (jiwer.cer)
                    cer_i = float(jiwer.cer(gt_json_string, pred_json_string))

                    per_sample_cer.append(cer_i)
                    gt_texts.append(gt_json_string)
                    pred_texts.append(pred_json_string)

                    prediction_entry = {
                        "file_name": test_item["file_name"],
                        "matched_image_path": test_item["image_path"],
                        "predicted_json": predicted_json,
                        "predicted_text": pred_json_string,
                        "target_json": self.clean_dict_for_ground_truth(test_item),
                        "target_text": gt_json_string,
                        "raw_response": generated_text,
                        "cer_score": cer_i,
                    }
                    all_predictions.append(prediction_entry)

                except Exception as e:
                    # hard failure => store a blank prediction
                    gt_json_string = self.json_to_string_no_sort(test_item)
                    pred_json_string = ""

                    cer_i = 1.0
                    per_sample_cer.append(cer_i)
                    gt_texts.append(gt_json_string)
                    pred_texts.append(pred_json_string)

                    all_predictions.append(
                        {
                            "file_name": test_item.get("file_name", ""),
                            "matched_image_path": test_item.get("image_path", ""),
                            "predicted_json": {},
                            "predicted_text": "",
                            "target_json": self.clean_dict_for_ground_truth(test_item),
                            "target_text": gt_json_string,
                            "raw_response": f"Error: {str(e)}",
                            "cer_score": cer_i,
                        }
                    )

            torch.cuda.empty_cache()
            gc.collect()

        # Save predictions JSONL
        predictions_file = os.path.join(output_dir, "test_predictions.jsonl")
        self.save_jsonl(all_predictions, predictions_file)

        # Stats: per-sample CER distribution
        cer_stats = self.calculate_cer_statistics(per_sample_cer)

        # Micro CER over entire test set
        micro_stats = self._micro_cer_from_strings(pred_texts, gt_texts)

        # Save CER text report
        cer_file = os.path.join(output_dir, "cer_evaluation_results.txt")
        self.save_cer_results(cer_stats, micro_stats, cer_file, len(all_predictions))

        # Save summary json too (handy)
        summary = {
            "num_predictions": len(all_predictions),
            "predictions_file": predictions_file,
            "cer_file": cer_file,
            "cer_stats": cer_stats,
            "micro_stats": micro_stats,
        }
        summary_file = os.path.join(output_dir, "test_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("TEST EVALUATION COMPLETED")
        print("=" * 60)
        print(f"Predictions JSONL: {predictions_file}")
        print(f"CER report TXT   : {cer_file}")
        print(f"Summary JSON     : {summary_file}")
        print(f"Micro CER        : {micro_stats['micro_cer']:.6f} ({micro_stats['micro_cer']*100:.3f}%)")
        print("=" * 60)

        # Switch back to training mode if needed later
        FastVisionModel.for_training(self.model)

        return summary


def main():
    base_checkpoint_dir = "/home/vault/iwi5/iwi5298h/models_image_text/qwen/schmuck_dataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_checkpoint_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Created run directory: {run_dir}")

    config = {
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",
        "output_dir": run_dir,
        "num_epochs": 15,
        "batch_size": 2,
        "learning_rate": 5e-5,
    }

    # save config
    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    finetuner = SchmuckOCRFinetune(model_path="/home/vault/iwi5/iwi5298h/models/qwen7b")

    print("=" * 60)
    print("STARTING TRAINING FOR SCHMUCK DATASET")
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

    # save best model (trainer already holds best weights due to load_best_model_at_end=True)
    finetuner.save_model(trainer, config["output_dir"])

    print("\n" + "=" * 60)
    print("STARTING TEST EVALUATION (best model)")
    print("=" * 60)

    summary = finetuner.evaluate_on_test_set(
        test_jsonl_path=config["test_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=config["output_dir"],
    )

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)
    print(f"All outputs saved to: {run_dir}")
    print(f"Micro CER: {summary['micro_stats']['micro_cer']:.6f} ({summary['micro_stats']['micro_cer']*100:.3f}%)")


if __name__ == "__main__":
    main()
