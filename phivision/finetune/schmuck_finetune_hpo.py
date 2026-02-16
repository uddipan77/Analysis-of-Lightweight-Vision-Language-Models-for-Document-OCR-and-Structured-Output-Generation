#!/usr/bin/env python3
# phi_3_5_vision_schmuck_finetune_qlora.py
# Memory-optimized fine-tuning with BitsAndBytes QLoRA 4-bit for Schmuck dataset

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
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import jiwer
from datetime import datetime
import numpy as np
import unicodedata
from dataclasses import dataclass
import gc
import random


# ============================================================================
# Configuration - Memory Optimized with 4-bit QLoRA
# ============================================================================

CONFIG = {
    "model_path": "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision",
    "train_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
    "val_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
    "test_jsonl": "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
    "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",
    "output_dir": "/home/vault/iwi5/iwi5298h/models_image_text/phi/schmuck/finetune",
    "num_epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 0.0001248726459555529,
    "max_seq_length": 2048,
    "weight_decay": 0.012203823484477884,
    "lora_r": 24,
    "lora_alpha": 16,
    "lora_dropout": 0.0684854455525527,
    "max_grad_norm": 1.0,
    "early_stopping_patience": 3,
    "data_augmentation": True,
    "use_4bit": True,
    "use_nested_quant": True,
}


# ============================================================================
# Instruction Prompt for Schmuck Dataset
# ============================================================================

INSTRUCTION = """Extract all information from this German historical jewelry/schmuck catalog image and return ONLY a JSON object with exactly these keys:

{
  "Gegenstand": "Object/item type",
  "Inv.Nr": "Inventory number (e.g., Sch 3051)",
  "Herkunft": "Origin/provenance",
  "Foto Notes": "Photo notes/number",
  "Standort": "Location/storage",
  "Material": "Material description",
  "Datierung": "Dating/time period",
  "Maße": "Measurements/dimensions",
  "Gewicht": "Weight",
  "erworben von": "Acquired from",
  "am": "Acquisition date",
  "Preis": "Price",
  "Vers.-Wert": "Insurance value",
  "Beschreibung": "Description",
  "Literatur": "Literature references",
  "Ausstellungen": "Exhibitions"
}

Return ONLY the JSON object with these exact keys. Use empty string "" for missing values. No additional commentary."""


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
    """Ground truth without file_name (metadata)."""
    label_data = {k: v for k, v in json_data.items() if k != "file_name"}
    return json.dumps(label_data, ensure_ascii=False, sort_keys=False)


def extract_json_from_response(response: str) -> str:
    """Extract FIRST JSON from response to prevent repetition issues."""
    response = response.strip()
    
    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]
        
        # IMPORTANT: Only take the first complete JSON object
        try:
            # Try to find just the first complete JSON
            brace_count = 0
            first_end = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        first_end = i + 1
                        break
            
            if first_end > 0:
                first_json = json_str[:first_end]
                parsed = json.loads(first_json)
                return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except:
            pass
        
        # Fallback: try parsing the whole thing
        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except:
            return json_str
    
    return response


# ============================================================================
# Data Augmentation
# ============================================================================

class DocumentImageAugmenter:
    """Augmentation for historical documents/catalog images"""
    
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
# Custom Dataset for Schmuck
# ============================================================================

class SchmuckDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, images_dir: str, processor, instruction: str, 
                 augment: bool = False):
        self.data = load_jsonl(jsonl_path)
        self.images_dir = images_dir
        self.processor = processor
        self.instruction = instruction
        self.augmenter = DocumentImageAugmenter(enabled=augment)
        
        # Filter valid samples
        self.valid_samples = []
        for item in self.data:
            image_path = os.path.join(self.images_dir, item["file_name"])
            if os.path.exists(image_path):
                self.valid_samples.append(item)
        
        print(f"   📊 Loaded {len(self.valid_samples)} valid samples (out of {len(self.data)} total)")
        if augment:
            print("   🔄 Data augmentation ENABLED")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def __getitem__(self, idx):
        item = self.valid_samples[idx]
        image_path = os.path.join(self.images_dir, item["file_name"])
        
        # Load and augment image
        image = Image.open(image_path).convert("RGB")
        image = self.augmenter.augment(image)
        
        # Create ground truth JSON string (excluding file_name)
        gt_json_str = create_label_string(item)
        
        # Format as chat messages
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{self.instruction}"
            },
            {
                "role": "assistant",
                "content": gt_json_str
            }
        ]
        
        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Process inputs
        inputs = self.processor(prompt, [image], return_tensors="pt")
        
        # Teacher forcing: labels = input_ids shifted
        labels = inputs["input_ids"].clone()
        
        # Find where assistant response starts to mask instruction
        assistant_token = self.processor.tokenizer.encode("<|assistant|>", add_special_tokens=False)
        input_ids_list = inputs["input_ids"][0].tolist()
        
        # Mask all tokens before assistant response
        try:
            for i in range(len(input_ids_list) - len(assistant_token)):
                if input_ids_list[i:i+len(assistant_token)] == assistant_token:
                    labels[0, :i+len(assistant_token)] = -100
                    break
        except:
            pass
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
            "image_sizes": inputs["image_sizes"].squeeze(0) if "image_sizes" in inputs else None,
            "labels": labels.squeeze(0),
            "file_name": item["file_name"],
            "ground_truth": gt_json_str
        }


# ============================================================================
# Data Collator
# ============================================================================

@dataclass
class DataCollatorForPhi3Vision:
    processor: AutoProcessor
    
    def __call__(self, features):
        batch = {}
        
        # Stack input_ids and attention_mask
        batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
        batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
        batch["labels"] = torch.stack([f["labels"] for f in features])
        
        # Stack pixel_values if present
        if features[0]["pixel_values"] is not None:
            batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
        
        if features[0]["image_sizes"] is not None:
            batch["image_sizes"] = torch.stack([f["image_sizes"] for f in features])
        
        return batch


# ============================================================================
# CER Validation Callback - FIXED (No CUDA-breaking parameters)
# ============================================================================

class CERValidationCallback(TrainerCallback):
    def __init__(self, model, processor, val_data, images_dir, instruction, output_dir):
        self.model = model
        self.processor = processor
        self.val_data = val_data[:30]  # Use subset for speed
        self.images_dir = images_dir
        self.instruction = instruction
        self.output_dir = output_dir
        self.best_cer = float('inf')
        self.best_checkpoint_path = None
        self.cer_history = []
        
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        print("\n" + "="*80)
        print(f"🔍 COMPUTING CER ON VALIDATION SET (Epoch {int(state.epoch)})")
        print("="*80)
        
        model.eval()
        predictions = []
        targets = []
        
        device = next(model.parameters()).device
        
        for i, val_item in enumerate(self.val_data):
            print(f"   Validating {i+1}/{len(self.val_data)}: {val_item['file_name']}", end='\r')
            
            image_path = os.path.join(self.images_dir, val_item["file_name"])
            if not os.path.exists(image_path):
                continue
            
            try:
                image = Image.open(image_path).convert("RGB")
                
                messages = [
                    {
                        "role": "user",
                        "content": f"<|image_1|>\n{self.instruction}"
                    }
                ]
                
                prompt = self.processor.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.processor(prompt, [image], return_tensors="pt").to(device)
                
                # FIXED: Removed repetition_penalty and no_repeat_ngram_size to avoid CUDA errors
                with torch.no_grad():
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.0,
                        do_sample=False,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                    )
                
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                
                raw_output = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0].strip()
                
                raw_output = normalize_unicode(raw_output)
                prediction = extract_json_from_response(raw_output)  # Uses improved extraction
                ground_truth = create_label_string(val_item)
                
                predictions.append(prediction)
                targets.append(ground_truth)
                
                del inputs, generate_ids
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\n   ⚠️  Validation error on {val_item['file_name']}: {e}")
                predictions.append("")
                targets.append(create_label_string(val_item))
                continue
        
        # Calculate CER
        val_cer = self.calculate_cer(predictions, targets)
        self.cer_history.append({"epoch": int(state.epoch), "cer": val_cer})
        
        if metrics is not None:
            metrics['eval_cer'] = val_cer
        
        print(f"\n   ✅ Validation CER (Epoch {int(state.epoch)}): {val_cer:.4f} ({val_cer*100:.2f}%)")
        
        # Save best model
        if val_cer < self.best_cer:
            improvement = self.best_cer - val_cer
            self.best_cer = val_cer
            print(f"   🎯 NEW BEST CER: {self.best_cer:.4f} (improved by {improvement:.4f})")
            
            best_model_path = os.path.join(self.output_dir, "best_model")
            os.makedirs(best_model_path, exist_ok=True)
            
            print(f"   💾 Saving best model to: {best_model_path}")
            model.save_pretrained(best_model_path)
            
            try:
                self.processor.tokenizer.save_pretrained(best_model_path)
                print(f"   ✅ Model and tokenizer saved successfully")
            except Exception as e:
                print(f"   ⚠️  Tokenizer save warning: {e}")
                print(f"   ✅ Model saved (tokenizer will be loaded from original path)")
            
            self.best_checkpoint_path = best_model_path
        else:
            print(f"   📊 No improvement. Best CER remains: {self.best_cer:.4f}")
        
        print("="*80 + "\n")
        
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
    print("\n" + "="*80)
    print("PHI-3.5-VISION FINE-TUNING - SCHMUCK DATASET")
    print("BitsAndBytes QLoRA 4-bit (Memory-Optimized)")
    print("="*80)
    
    # Create output directory
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Save config
    config_file = os.path.join(CONFIG["output_dir"], "training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
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
    print(f"   • LoRA r={CONFIG['lora_r']}, alpha={CONFIG['lora_alpha']}, dropout={CONFIG['lora_dropout']}")
    
    print(f"\n🔧 Anti-Repetition:")
    print(f"   • First-JSON extraction: enabled")
    print(f"   • Note: Using improved JSON parsing instead of generation penalties")
    
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
        num_crops=16
    )
    print("   ✅ Processor loaded")
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation='eager'
    )
    
    print("   ✅ Model loaded with 4-bit quantization")
    
    # Show memory after loading
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"   📊 GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
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
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    print(f"   ✅ LoRA applied successfully")
    print(f"   📊 Trainable params: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")
    
    # Load datasets
    print("\n📊 Loading Schmuck datasets...")
    
    print("   📁 Training dataset:")
    train_dataset = SchmuckDataset(
        CONFIG["train_jsonl"],
        CONFIG["images_dir"],
        processor,
        INSTRUCTION,
        augment=CONFIG["data_augmentation"]
    )
    
    print("   📁 Validation dataset:")
    eval_dataset = SchmuckDataset(
        CONFIG["val_jsonl"],
        CONFIG["images_dir"],
        processor,
        INSTRUCTION,
        augment=False
    )
    
    # Load validation data for CER callback
    val_data = load_jsonl(CONFIG["val_jsonl"])
    print(f"   📊 Validation samples for CER computation: {len(val_data)}")
    
    # Data collator
    data_collator = DataCollatorForPhi3Vision(processor=processor)
    
    # CER callback
    cer_callback = CERValidationCallback(
        model=model,
        processor=processor,
        val_data=val_data,
        images_dir=CONFIG["images_dir"],
        instruction=INSTRUCTION,
        output_dir=CONFIG["output_dir"]
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
        load_best_model_at_end=False,
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
    print("🚀 Starting QLoRA training on Schmuck dataset...\n")
    print("="*80)
    trainer.train()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"   🎯 Best Validation CER: {cer_callback.best_cer:.4f} ({cer_callback.best_cer*100:.2f}%)")
    print(f"   💾 Best model saved at: {cer_callback.best_checkpoint_path}")
    print("="*80)
    
    # Test inference
    print("\n" + "="*80)
    print("PHASE 2: EVALUATION ON TEST SET")
    print("="*80)
    
    print(f"\n⏳ Loading best model from: {cer_callback.best_checkpoint_path}")
    
    # Load best model
    best_model = AutoModelForCausalLM.from_pretrained(
        cer_callback.best_checkpoint_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation='eager'
    )
    
    best_processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        num_crops=16
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
        print(f"[{idx+1}/{len(test_data)}] Processing {item['file_name']}")
        
        image_path = os.path.join(CONFIG["images_dir"], item["file_name"])
        
        if not os.path.exists(image_path):
            print(f"   ❌ Image not found")
            results.append({
                "file_name": item["file_name"],
                "predicted_text": "",
                "ground_truth_text": create_label_string(item),
                "cer_score": 1.0,
                "error": "Image not found"
            })
            cer_scores.append(1.0)
            continue
        
        try:
            image = Image.open(image_path).convert("RGB")
            
            messages = [
                {
                    "role": "user",
                    "content": f"<|image_1|>\n{INSTRUCTION}"
                }
            ]
            
            prompt = best_processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = best_processor(prompt, [image], return_tensors="pt").to(device)
            
            # FIXED: Removed repetition_penalty and no_repeat_ngram_size
            with torch.no_grad():
                generate_ids = best_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.0,
                    do_sample=False,
                    eos_token_id=best_processor.tokenizer.eos_token_id,
                )
            
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            
            raw_output = best_processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
            
            raw_output = normalize_unicode(raw_output)
            prediction = extract_json_from_response(raw_output)  # Uses improved extraction
            ground_truth = create_label_string(item)
            
            cer = jiwer.cer(ground_truth, prediction) if len(ground_truth) > 0 else (1.0 if len(prediction) > 0 else 0.0)
            cer_scores.append(cer)
            
            status = "✨" if cer == 0.0 else "✅" if cer < 0.1 else "⚠️" if cer < 0.3 else "❌"
            print(f"   {status} CER: {cer:.4f} ({cer*100:.2f}%)")
            
            results.append({
                "file_name": item["file_name"],
                "predicted_text": prediction,
                "ground_truth_text": ground_truth,
                "raw_output": raw_output,
                "cer_score": cer,
            })
            
            del inputs, generate_ids
            
            if (idx + 1) % 3 == 0:
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                "file_name": item["file_name"],
                "predicted_text": "",
                "ground_truth_text": create_label_string(item),
                "cer_score": 1.0,
                "error": str(e)
            })
            cer_scores.append(1.0)
    
    # Save results
    results_file = os.path.join(CONFIG["output_dir"], "test_predictions.jsonl")
    save_jsonl(results, results_file)
    
    avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 1.0
    perfect_matches = sum(1 for cer in cer_scores if cer == 0.0)
    good_matches = sum(1 for cer in cer_scores if cer < 0.1)
    
    print(f"\n{'='*80}")
    print("📊 FINAL TEST RESULTS - SCHMUCK DATASET")
    print(f"{'='*80}")
    print(f"   Total Samples: {len(test_data)}")
    print(f"   Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"   Perfect Matches (CER=0): {perfect_matches}/{len(cer_scores)} ({perfect_matches/len(cer_scores)*100:.1f}%)")
    print(f"   Good Matches (CER<0.1): {good_matches}/{len(cer_scores)} ({good_matches/len(cer_scores)*100:.1f}%)")
    print(f"   Results saved to: {results_file}")
    print(f"{'='*80}")
    
    # Save summary with CER history
    summary_file = os.path.join(CONFIG["output_dir"], "training_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PHI-3.5-VISION SCHMUCK DATASET FINE-TUNING RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best Validation CER: {cer_callback.best_cer:.4f} ({cer_callback.best_cer*100:.2f}%)\n")
        f.write(f"Test CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)\n")
        f.write(f"Perfect Matches: {perfect_matches}/{len(cer_scores)} ({perfect_matches/len(cer_scores)*100:.1f}%)\n")
        f.write(f"Good Matches (CER<0.1): {good_matches}/{len(cer_scores)} ({good_matches/len(cer_scores)*100:.1f}%)\n\n")
        f.write("Anti-Repetition Measures:\n")
        f.write("  - First-JSON extraction enabled\n")
        f.write("  - Using temperature=0.0 for deterministic generation\n\n")
        f.write("CER History:\n")
        for entry in cer_callback.cer_history:
            f.write(f"  Epoch {entry['epoch']}: {entry['cer']:.4f} ({entry['cer']*100:.2f}%)\n")
    
    print(f"\n✅ Summary saved to: {summary_file}")
    print("\n🎉 Fine-tuning and evaluation complete!\n")


if __name__ == "__main__":
    main()
