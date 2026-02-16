# nanonets_inventory_finetune_eval_cer.py
# Finetune Nanonets-OCR-s on Inventory dataset with CER-based model selection

import torch
import json
import os
from typing import List, Dict, Tuple
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
import re
import jiwer
from datetime import datetime
from transformers import EarlyStoppingCallback, TrainerCallback
import numpy as np
import glob
import gc
from PIL import Image
import torchvision.transforms as transforms
import random
import tempfile
import shutil


class CERValidationCallback(TrainerCallback):
    """Custom callback to compute CER on validation set during training"""
    
    def __init__(self, finetuner, val_data, images_dir, instruction):
        self.finetuner = finetuner
        self.val_data = val_data[:30]  # Use subset for speed (30 samples)
        self.images_dir = images_dir
        self.instruction = instruction
        self.best_cer = float('inf')
        self.cer_history = []  # Track CER over epochs for TensorBoard
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Compute CER after each evaluation"""
        print("\n" + "="*60)
        print(f"Computing CER on validation set (Epoch {state.epoch})...")
        print("="*60)
        
        # Switch to inference mode
        FastVisionModel.for_inference(model)
        
        predictions = []
        targets = []
        
        for i, val_item in enumerate(self.val_data):
            print(f"Validating {i+1}/{len(self.val_data)}", end='\r')
            
            try:
                image_path = self.finetuner.find_image_path(val_item['image_name'], self.images_dir)
                if not os.path.exists(image_path):
                    continue
                
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": self.instruction},
                        {"type": "image", "image": image_path}
                    ]}
                ]
                
                input_text = self.finetuner.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.finetuner.tokenizer(
                    text=[input_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        use_cache=True,
                        temperature=0.1,
                        do_sample=True,
                    )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                ]
                
                generated_texts = self.finetuner.tokenizer.batch_decode(
                    generated_ids_trimmed, 
                    skip_special_tokens=True, 
                    clean_up_tokenization_spaces=False
                )
                
                generated_text = generated_texts[0] if generated_texts else ""
                
                _, pred_str = self.finetuner.extract_json_from_response(generated_text)
                gt_str = self.finetuner.json_to_string_no_sort(val_item)
                
                predictions.append(pred_str if pred_str else "")
                targets.append(gt_str)
                
                del inputs, outputs, generated_ids_trimmed, generated_texts
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nValidation error on {val_item['image_name']}: {e}")
                predictions.append("")
                targets.append(self.finetuner.json_to_string_no_sort(val_item))
                continue
        
        # Calculate CER
        val_cer = self.calculate_cer(predictions, targets)
        
        # Track CER history for later saving and TensorBoard
        current_epoch = state.epoch if state.epoch is not None else 0
        self.cer_history.append({
            "epoch": float(current_epoch),
            "cer": float(val_cer),
            "cer_percentage": float(val_cer * 100)
        })
        
        # Add to metrics - these will be logged to TensorBoard automatically
        if metrics is not None:
            metrics['eval_cer'] = val_cer
            metrics['eval_cer_percentage'] = val_cer * 100
        
        print(f"\n✅ Validation CER (Epoch {state.epoch}): {val_cer:.4f} ({val_cer*100:.2f}%)")
        
        # Track best CER
        if val_cer < self.best_cer:
            self.best_cer = val_cer
            print(f"🎯 New best CER: {self.best_cer:.4f}")
        
        # Switch back to training mode
        FastVisionModel.for_training(model)
        
        return control
    
    def calculate_cer(self, predictions, targets):
        """Calculate average CER"""
        if not predictions or not targets or len(predictions) != len(targets):
            return 1.0
        
        total_cer = 0.0
        valid_pairs = 0
        
        for pred, target in zip(predictions, targets):
            if len(target) > 0:
                cer_score = jiwer.cer(target, pred)
                total_cer += cer_score
                valid_pairs += 1
        
        return total_cer / valid_pairs if valid_pairs > 0 else 1.0


class NanonetsInventoryFinetune:
    def __init__(self, model_path: str = "/home/vault/iwi5/iwi5298h/models/Nanonets-OCR-s", augment_factor: int = 1):
        """Initialize with Nanonets-OCR-s and data augmentation"""
        print("="*60)
        print("LOADING NANONETS-OCR-S WITH UNSLOTH FOR INVENTORY DATASET")
        print("="*60)
        print(f"Model path: {model_path}")
        print(f"Augmentation factor: {augment_factor}")
        
        # Store augmentation factor
        self.augment_factor = augment_factor
        
        # Create temporary directory for augmented images
        self.temp_dir = tempfile.mkdtemp(prefix="augmented_inventory_")
        print(f"Created temporary directory: {self.temp_dir}")
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        print("✓ Model directory verified")
        
        # Load from local path
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        
        print("\n✅ Model loaded successfully!")
        print("Applying LoRA adaptors...")
        
        # Apply LoRA with higher rank for more trainable parameters
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )
        
        print("✅ LoRA applied successfully!")
        print("="*60 + "\n")
    
    def create_augmentation_transforms(self):
        """Create document-friendly augmentation transforms (NO rotation)"""
        augmentation_transforms = [
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=255),
            transforms.RandomResizedCrop(size=(1024, 1024), scale=(0.95, 1.0), ratio=(0.9, 1.1)),
        ]
        return augmentation_transforms
    
    def augment_image(self, image_path: str, aug_id: int) -> str:
        """Create augmented version of image"""
        try:
            image = Image.open(image_path).convert('RGB')
            transforms_list = self.create_augmentation_transforms()
            num_transforms = random.randint(2, min(3, len(transforms_list)))
            selected_transforms = random.sample(transforms_list, num_transforms)
            
            augmented_image = image
            for transform in selected_transforms:
                augmented_image = transform(augmented_image)
            
            original_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(original_name)[0]
            ext = os.path.splitext(original_name)[1]
            augmented_name = f"{name_without_ext}_aug{aug_id}{ext}"
            augmented_path = os.path.join(self.temp_dir, augmented_name)
            
            augmented_image.save(augmented_path, quality=95)
            return augmented_path
        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            return None
    
    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data
    
    def save_jsonl(self, data: List[Dict], file_path: str):
        """Save data to JSONL file"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def dict_without_image_name(self, obj):
        """Remove image_name key"""
        return {k: v for k, v in obj.items() if k != "image_name"}
    
    def json_to_string_no_sort(self, obj):
        """Convert dict to JSON string"""
        clean_obj = self.dict_without_image_name(obj)
        return json.dumps(clean_obj, ensure_ascii=False, separators=(',', ':'))
    
    def extract_json_from_response(self, response: str) -> Tuple[Dict, str]:
        """Extract JSON from model response"""
        if isinstance(response, list):
            response = str(response[0]) if response else ""
        
        original_response = str(response).strip()
        response = original_response
        
        if not response:
            return {}, ""
        
        if response.startswith("```"):
            parts = response.split("```")
            if len(parts) >= 3:
                body = parts[1]
                if body.lstrip().startswith("json"):
                    body = body[4:].strip()
                response = body.strip()
        
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            for match in sorted(matches, key=len, reverse=True):
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed, match
                except json.JSONDecodeError:
                    continue
        
        try:
            parsed = json.loads(response)
            return parsed, response
        except json.JSONDecodeError:
            pass
        
        return {}, original_response
    
    def find_image_path(self, image_name: str, images_dir: str) -> str:
        """Find image path with flexible matching"""
        exact_path = os.path.join(images_dir, image_name)
        if os.path.exists(exact_path):
            return exact_path
        
        base_name = os.path.splitext(image_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]
        
        return exact_path
    
    def convert_to_conversation(self, sample):
        """Convert sample to conversation format with detailed inventory prompt"""
        instruction = """Extract the inventory information from this German historical document image as a complete JSON object.

The JSON must include ALL of the following fields with their exact German names:
- Überschrift: Document heading/title
- Inventarnummer: Inventory number
- Maße: Object with "L", "B", "D" (length, breadth, depth as strings)
- Objektbezeichnung: Object description/name
- Fundort: Find location/origin
- Fundzeit: Find time/dating
- Beschreibungstext: Main descriptive text (full transcription)

IMPORTANT:
- Include ALL fields in every response, even if empty.
- If a field has no visible text, use an empty string "".
- For Maße, ALWAYS include L, B, D keys even if empty (e.g. {"L": "", "B": "", "D": ""}).
- Preserve exact German text with umlauts (ä, ö, ü, ß) and special characters.
- Return ONLY the JSON object without any additional text or formatting."""
        
        gt_json_string = self.json_to_string_no_sort(sample)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image_path"]}
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": gt_json_string}
                ]
            }
        ]
        return {"messages": conversation}
    
    def prepare_training_data_with_augmentation(self, jsonl_path: str, images_dir: str) -> Tuple[List[Dict], List[Dict]]:
        """Prepare training data WITH augmentation, return both converted and raw data"""
        data = self.load_jsonl(jsonl_path)
        
        processed_data = []
        augmented_count = 0
        
        print(f"Preparing training data with augmentation factor: {self.augment_factor}")
        
        for i, item in enumerate(data):
            image_path = self.find_image_path(item["image_name"], images_dir)
            if os.path.exists(image_path):
                original_item = item.copy()
                original_item["image_path"] = image_path
                processed_data.append(original_item)
                
                for aug_id in range(1, self.augment_factor + 1):
                    print(f"Creating augmentation {aug_id}/{self.augment_factor} for image {i+1}/{len(data)}: {item['image_name']}")
                    
                    augmented_path = self.augment_image(image_path, aug_id)
                    if augmented_path and os.path.exists(augmented_path):
                        augmented_item = item.copy()
                        augmented_item["image_path"] = augmented_path
                        augmented_item["image_name"] = f"{item['image_name']}_aug{aug_id}"
                        processed_data.append(augmented_item)
                        augmented_count += 1
            else:
                print(f"Warning: Image not found for {item['image_name']}, skipping...")
        
        print(f"Successfully created {augmented_count} augmented images")
        print(f"Total training samples: {len(processed_data)} (original: {len(data)}, augmented: {augmented_count})")
        
        converted_dataset = [self.convert_to_conversation(sample) for sample in processed_data]
        
        return converted_dataset, processed_data  # Return both
    
    def prepare_training_data_no_augmentation(self, jsonl_path: str, images_dir: str) -> Tuple[List[Dict], List[Dict]]:
        """Prepare data WITHOUT augmentation (for validation), return both converted and raw data"""
        data = self.load_jsonl(jsonl_path)
        
        processed_data = []
        for item in data:
            image_path = self.find_image_path(item["image_name"], images_dir)
            if os.path.exists(image_path):
                item["image_path"] = image_path
                processed_data.append(item)
            else:
                print(f"Warning: Image not found for {item['image_name']}, skipping...")
        
        print(f"Found {len(processed_data)} valid samples out of {len(data)} total")
        
        converted_dataset = [self.convert_to_conversation(sample) for sample in processed_data]
        
        return converted_dataset, processed_data  # Return both
    
    def train_model(self, 
                   train_jsonl_path: str,
                   val_jsonl_path: str,
                   images_dir: str,
                   output_dir: str,
                   num_epochs: int = 15,
                   batch_size: int = 2,
                   learning_rate: float = 5e-5):
        """Train with CER-based model selection"""
        
        print("Preparing datasets...")
        train_dataset, _ = self.prepare_training_data_with_augmentation(train_jsonl_path, images_dir)
        val_dataset, val_raw_data = self.prepare_training_data_no_augmentation(val_jsonl_path, images_dir)
        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples\n")
        
        # Get instruction for callback
        instruction = """Extract the inventory information from this German historical document image as a complete JSON object.

The JSON must include ALL of the following fields with their exact German names:
- Überschrift: Document heading/title
- Inventarnummer: Inventory number
- Maße: Object with "L", "B", "D" (length, breadth, depth as strings)
- Objektbezeichnung: Object description/name
- Fundort: Find location/origin
- Fundzeit: Find time/dating
- Beschreibungstext: Main descriptive text (full transcription)

IMPORTANT:
- Include ALL fields in every response, even if empty.
- If a field has no visible text, use an empty string "".
- For Maße, ALWAYS include L, B, D keys even if empty (e.g. {"L": "", "B": "", "D": ""}).
- Preserve exact German text with umlauts (ä, ö, ü, ß) and special characters.
- Return ONLY the JSON object without any additional text or formatting."""
        
        # Enable training mode
        FastVisionModel.for_training(self.model)
        
        # Create CER validation callback
        cer_callback = CERValidationCallback(self, val_raw_data, images_dir, instruction)
        
        # Unsloth's SFTTrainer with CER-based model selection
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
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
                
                # CER-based model selection
                load_best_model_at_end=True,
                metric_for_best_model="eval_cer",  # Use CER for model selection
                greater_is_better=False,  # Lower CER is better
                
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
                cer_callback,  # Add CER validation callback
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001,
                )
            ],
        )
        
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            max_memory = round(gpu_stats.total_memory / 1024**3, 3)
            print(f"\n{'='*60}")
            print("GPU INFORMATION")
            print(f"{'='*60}")
            print(f"GPU: {gpu_stats.name}")
            print(f"Max memory: {max_memory} GB")
            print(f"Reserved: {start_memory} GB")
            print(f"\n✅ UNSLOTH OPTIMIZATIONS:")
            print(f"   • 4-bit quantization")
            print(f"   • LoRA rank 16 (RSLoRA)")
            print(f"   • Gradient checkpointing")
            print(f"   • Data augmentation: {self.augment_factor}x")
            print(f"   • Model selection: eval_cer_jiwer (best CER)")
            print(f"   • Batch size: {batch_size}, Grad accum: 4")
            print(f"{'='*60}\n")
        
        print("Starting training with CER-based model selection...")
        trainer_stats = trainer.train()
        
        if torch.cuda.is_available():
            used_memory = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
            used_for_training = round(used_memory - start_memory, 3)
            runtime_min = round(trainer_stats.metrics['train_runtime']/60, 2)
            print(f"\n{'='*60}")
            print("TRAINING COMPLETED")
            print(f"{'='*60}")
            print(f"Runtime: {runtime_min} min")
            print(f"Peak memory: {used_memory} GB")
            print(f"Training memory: {used_for_training} GB")
            print(f"Best validation CER: {cer_callback.best_cer:.4f} ({cer_callback.best_cer*100:.2f}%)")
            print(f"{'='*60}\n")
        
        # Store CER history for later use
        self.cer_history = cer_callback.cer_history
        self.best_val_cer = cer_callback.best_cer
        
        return trainer
    
    # [Rest of the methods stay the same as before: evaluate_on_test_set, calculate_cer_statistics, save_cer_results, save_model, cleanup_temp_files]
    
    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict:
        """Evaluate on test set with chunking"""
        print("Starting evaluation on test set...")
        
        FastVisionModel.for_inference(self.model)
        
        test_data = self.load_jsonl(test_jsonl_path)
        print(f"Loaded {len(test_data)} test samples")
        
        all_predictions = []
        all_cer_scores = []
        
        instruction = """Extract the inventory information from this German historical document image as a complete JSON object.

The JSON must include ALL of the following fields with their exact German names:
- Überschrift: Document heading/title
- Inventarnummer: Inventory number
- Maße: Object with "L", "B", "D" (length, breadth, depth as strings)
- Objektbezeichnung: Object description/name
- Fundort: Find location/origin
- Fundzeit: Find time/dating
- Beschreibungstext: Main descriptive text (full transcription)

IMPORTANT:
- Include ALL fields in every response, even if empty.
- If a field has no visible text, use an empty string "".
- For Maße, ALWAYS include L, B, D keys even if empty (e.g. {"L": "", "B": "", "D": ""}).
- Preserve exact German text with umlauts (ä, ö, ü, ß) and special characters.
- Return ONLY the JSON object without any additional text or formatting."""
        
        chunk_size = 20
        num_chunks = (len(test_data) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(test_data))
            chunk_data = test_data[start_idx:end_idx]
            
            print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_data)} images)")
            
            for i, test_item in enumerate(chunk_data):
                local_idx = start_idx + i + 1
                print(f"Processing test image {local_idx}/{len(test_data)}: {test_item['image_name']}")
                
                image_path = self.find_image_path(test_item['image_name'], images_dir)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    all_predictions.append({
                        "image_name": test_item['image_name'],
                        "predicted_json": {},
                        "prediction_string": "Error: Image not found",
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": self.json_to_string_no_sort(test_item),
                        "raw_response": "Error: Image not found",
                        "cer_score": 1.0,
                        "matched_image_path": image_path,
                        "parsing_success": False
                    })
                    all_cer_scores.append(1.0)
                    continue
                
                try:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": image_path}
                        ]}
                    ]
                    
                    input_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    from qwen_vl_utils import process_vision_info
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
                        )
                    
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
                    ]
                    
                    generated_texts = self.tokenizer.batch_decode(
                        generated_ids_trimmed, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=False
                    )
                    
                    generated_text = generated_texts[0] if generated_texts else ""
                    
                    predicted_json, pred_str = self.extract_json_from_response(generated_text)
                    gt_str = self.json_to_string_no_sort(test_item)
                    cer_score = jiwer.cer(gt_str, pred_str) if pred_str else 1.0
                    
                    prediction_entry = {
                        "image_name": test_item['image_name'],
                        "predicted_json": predicted_json,
                        "prediction_string": pred_str,
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": gt_str,
                        "raw_response": generated_text,
                        "cer_score": cer_score,
                        "matched_image_path": image_path,
                        "parsing_success": bool(predicted_json and pred_str and pred_str != generated_text)
                    }
                    all_predictions.append(prediction_entry)
                    all_cer_scores.append(cer_score)
                    
                    print(f"  Processed successfully. CER: {cer_score:.3f}")
                    
                    del inputs, outputs, generated_ids_trimmed, generated_texts
                    
                except Exception as e:
                    print(f"Error processing {test_item['image_name']}: {e}")
                    error_msg = f"Error: {str(e)}"
                    all_predictions.append({
                        "image_name": test_item['image_name'],
                        "predicted_json": {},
                        "prediction_string": error_msg,
                        "target_json": self.dict_without_image_name(test_item),
                        "target_text": self.json_to_string_no_sort(test_item),
                        "raw_response": error_msg,
                        "cer_score": 1.0,
                        "matched_image_path": image_path,
                        "parsing_success": False
                    })
                    all_cer_scores.append(1.0)
            
            torch.cuda.empty_cache()
            gc.collect()
        
        predictions_file = os.path.join(output_dir, "test_predictions.jsonl")
        self.save_jsonl(all_predictions, predictions_file)
        
        cer_stats = self.calculate_cer_statistics(all_cer_scores, all_predictions)
        
        cer_file = os.path.join(output_dir, "cer_evaluation_results.txt")
        self.save_cer_results(cer_stats, cer_file, len(all_predictions))
        
        return {
            "predictions": all_predictions,
            "cer_stats": cer_stats,
            "predictions_file": predictions_file,
            "cer_file": cer_file
        }
    
    def calculate_cer_statistics(self, all_cer_scores: List[float], all_predictions: List[Dict]) -> Dict:
        """Calculate CER statistics"""
        if not all_cer_scores:
            return {}
        
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        median_cer = np.median(all_cer_scores)
        std_cer = np.std(all_cer_scores)
        
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)
        parsing_successes = sum(1 for pred in all_predictions if pred.get('parsing_success', False))
        
        total_chars = sum(len(pred['target_text']) for pred in all_predictions)
        total_errors = sum(int(round(pred['cer_score'] * len(pred['target_text']))) 
                          for pred in all_predictions)
        weighted_cer = total_errors / total_chars if total_chars > 0 else 0.0
        
        return {
            "total_images": len(all_cer_scores),
            "average_cer": avg_cer,
            "median_cer": median_cer,
            "minimum_cer": min_cer,
            "maximum_cer": max_cer,
            "std_cer": std_cer,
            "weighted_cer": weighted_cer,
            "perfect_matches": perfect_matches,
            "parsing_success_rate": parsing_successes / len(all_predictions) if all_predictions else 0.0,
            "parsing_successes": parsing_successes,
            "total_characters": total_chars,
            "total_errors": total_errors,
        }
    
    def save_cer_results(self, cer_stats: Dict, cer_file: str, num_predictions: int):
        """Save CER results"""
        with open(cer_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("NANONETS-OCR-S FINETUNING - INVENTORY DATASET RESULTS\n")
            f.write("CER-BASED MODEL SELECTION\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"CER Statistics ({cer_stats['total_images']} images):\n")
            f.write("-" * 50 + "\n")
            f.write(f"Average CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)\n")
            f.write(f"Median CER: {cer_stats['median_cer']:.4f} ({cer_stats['median_cer']*100:.2f}%)\n")
            f.write(f"Min CER: {cer_stats['minimum_cer']:.4f} | Max CER: {cer_stats['maximum_cer']:.4f}\n")
            f.write(f"Std Dev: {cer_stats['std_cer']:.4f}\n\n")
            
            f.write(f"Weighted CER: {cer_stats['weighted_cer']:.4f} ({cer_stats['weighted_cer']*100:.2f}%)\n")
            f.write(f"Total characters: {cer_stats['total_characters']}\n")
            f.write(f"Total errors: {cer_stats['total_errors']}\n\n")
            
            f.write(f"Perfect matches: {cer_stats['perfect_matches']}/{cer_stats['total_images']} ({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)\n")
            f.write(f"JSON parsing: {cer_stats['parsing_successes']}/{num_predictions} ({cer_stats['parsing_success_rate']*100:.2f}%)\n")
            
            # Add validation CER history
            if hasattr(self, 'cer_history') and self.cer_history:
                f.write("\n" + "="*60 + "\n")
                f.write("VALIDATION CER HISTORY (logged to TensorBoard)\n")
                f.write("="*60 + "\n")
                for entry in self.cer_history:
                    f.write(f"  Epoch {entry['epoch']:.1f}: {entry['cer']:.4f} ({entry['cer_percentage']:.2f}%)\n")
                if hasattr(self, 'best_val_cer'):
                    f.write(f"\nBest Validation CER: {self.best_val_cer:.4f} ({self.best_val_cer*100:.2f}%)\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("MODEL DETAILS\n")
            f.write("="*60 + "\n")
            f.write("Model: nanonets/Nanonets-OCR-s\n")
            f.write("Dataset: Anthropological-Prehistoric Inventory\n")
            f.write("Framework: Unsloth\n")
            f.write("Quantization: 4-bit\n")
            f.write("LoRA: Rank 16, RSLoRA\n")
            f.write(f"Data Augmentation: {self.augment_factor}x\n")
            f.write("Augmentation: brightness, contrast, blur, perspective, crop (NO rotation)\n")
            f.write("Model Selection: eval_cer (CER-based, best checkpoint)\n")
            f.write("TensorBoard: train_loss, eval_loss, eval_cer, eval_cer_percentage\n")
        
        print(f"\nResults saved to: {cer_file}")
    
    def save_model(self, trainer, output_dir: str):
        """Save finetuned model"""
        print(f"\nSaving best model (selected by CER) to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("✅ Best model saved (CER-based selection)!")
    
    def cleanup_temp_files(self):
        """Clean up temporary augmented images"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {self.temp_dir}: {e}")


def main():
    """Main training pipeline for inventory dataset"""
    base_dir = "/home/vault/iwi5/iwi5298h/models_image_text/nanonets/inven"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}_unsloth_cer")
    os.makedirs(run_dir, exist_ok=True)
    
    config = {
        "model_path": "/home/vault/iwi5/iwi5298h/models/Nanonets-OCR-s",
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/inventory_images",
        "output_dir": run_dir,
        "num_epochs": 15,
        "batch_size": 2,
        "learning_rate": 5e-5,
        "augment_factor": 1,
    }
    
    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("NANONETS-OCR-S FINETUNING - INVENTORY DATASET")
    print("CER-BASED MODEL SELECTION")
    print("="*60)
    print(f"Model: {config['model_path']}")
    print(f"Output: {run_dir}")
    print(f"Augmentation: {config['augment_factor']}x")
    print("="*60 + "\n")
    
    finetuner = NanonetsInventoryFinetune(
        model_path=config["model_path"],
        augment_factor=config["augment_factor"]
    )
    
    try:
        print("\n" + "="*60)
        print("PHASE 1: TRAINING WITH CER-BASED MODEL SELECTION")
        print("="*60 + "\n")
        
        trainer = finetuner.train_model(
            train_jsonl_path=config["train_jsonl_path"],
            val_jsonl_path=config["val_jsonl_path"],
            images_dir=config["images_dir"],
            output_dir=run_dir,
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"]
        )
        
        finetuner.save_model(trainer, run_dir)
        
        print("\n" + "="*60)
        print("PHASE 2: EVALUATION")
        print("="*60 + "\n")
        
        results = finetuner.evaluate_on_test_set(
            test_jsonl_path=config["test_jsonl_path"],
            images_dir=config["images_dir"],
            output_dir=run_dir
        )
        
        cer_stats = results['cer_stats']
        print(f"\n{'='*60}")
        print("FINAL RESULTS - CER-BASED MODEL SELECTION")
        print(f"{'='*60}")
        print(f"Model: Nanonets-OCR-s (finetuned, best CER checkpoint)")
        print(f"Dataset: Inventory (Anthropological-Prehistoric)\n")
        print(f"Metrics:")
        print(f"  Avg CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)")
        print(f"  Weighted CER: {cer_stats['weighted_cer']:.4f} ({cer_stats['weighted_cer']*100:.2f}%)")
        print(f"  Median: {cer_stats['median_cer']:.4f} | Std: {cer_stats['std_cer']:.4f}")
        print(f"\nAccuracy:")
        print(f"  Perfect: {cer_stats['perfect_matches']}/{cer_stats['total_images']} ({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)")
        print(f"  Parsing: {cer_stats['parsing_successes']}/{cer_stats['total_images']} ({cer_stats['parsing_success_rate']*100:.2f}%)")
        print(f"\nSaved to: {run_dir}")
        print("="*60 + "\n")
        
    finally:
        finetuner.cleanup_temp_files()


if __name__ == "__main__":
    main()
