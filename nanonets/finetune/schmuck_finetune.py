# nanonets_schmuck_finetune_unsloth.py
# Final version - Finetune Nanonets-OCR-s with Unsloth
# Uses ORIGINAL model: /home/vault/iwi5/iwi5298h/models/Nanonets-OCR-s

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
from transformers import EarlyStoppingCallback
import numpy as np
import glob
import gc


class ChunkedDataset:
    """Dataset that loads data in chunks to prevent OOM errors"""
    
    def __init__(self, jsonl_path: str, images_dir: str, chunk_size: int = 20):
        self.jsonl_path = jsonl_path
        self.images_dir = images_dir
        self.chunk_size = chunk_size
        self.total_samples = self._count_samples()
        self.num_chunks = (self.total_samples + chunk_size - 1) // chunk_size
        
        print(f"Dataset: {self.total_samples} samples, {self.num_chunks} chunks of size {chunk_size}")
    
    def _count_samples(self):
        """Count total samples without loading them"""
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def get_chunk(self, chunk_idx: int) -> List[Dict]:
        """Load a specific chunk from disk"""
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.total_samples)
        
        chunk_data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
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
    
    def find_image_path(self, file_name: str, images_dir: str) -> str:
        """Find image path using flexible matching"""
        exact_path = os.path.join(images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path
        
        base_name = os.path.splitext(file_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]
        
        return exact_path


class NanonetsSchmuckFinetune:
    def __init__(self, model_path: str = "/home/vault/iwi5/iwi5298h/models/Nanonets-OCR-s"):
        """Initialize with ORIGINAL Nanonets-OCR-s from local path"""
        print("="*60)
        print("LOADING NANONETS-OCR-S WITH UNSLOTH")
        print("="*60)
        print(f"Model path: {model_path}")
        
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
        
        # Apply LoRA
        self.model = FastVisionModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=True,
        )
        
        print("✅ LoRA applied successfully!")
        print("="*60 + "\n")
    
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
    
    def clean_dict_for_ground_truth(self, obj):
        """Remove file_name and image_path keys"""
        return {k: v for k, v in obj.items() if k not in ['file_name', 'image_path']}
    
    def json_to_string_no_sort(self, obj):
        """Convert dict to JSON string"""
        clean_obj = self.clean_dict_for_ground_truth(obj)
        return json.dumps(clean_obj, ensure_ascii=False, separators=(',', ':'))
    
    def extract_json_from_response(self, response: str) -> Tuple[Dict, str]:
        """Extract JSON from model response"""
        if isinstance(response, list):
            response = str(response[0]) if response else ""
        
        original_response = str(response).strip()
        response = original_response
        
        if not response:
            return {}, ""
        
        # Remove markdown code blocks
        if response.startswith("```"):
            parts = response.split("```")
            if len(parts) >= 3:
                body = parts[1]
                if body.lstrip().startswith("json"):
                    body = body[4:].strip()
                response = body.strip()
        
        # Try to find JSON object
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        if matches:
            for match in sorted(matches, key=len, reverse=True):
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 3:
                        return parsed, match
                except json.JSONDecodeError:
                    continue
        
        # Try parsing entire response
        try:
            parsed = json.loads(response)
            return parsed, response
        except json.JSONDecodeError:
            pass
        
        return {}, original_response
    
    def find_image_path(self, file_name: str, images_dir: str) -> str:
        """Find image path using flexible matching"""
        exact_path = os.path.join(images_dir, file_name)
        if os.path.exists(exact_path):
            return exact_path
        
        base_name = os.path.splitext(file_name)[0]
        search_pattern = os.path.join(images_dir, f"*{base_name}*")
        matching_files = glob.glob(search_pattern)
        if matching_files:
            return matching_files[0]
        
        return exact_path

    def convert_to_conversation(self, sample):
        """Convert sample to conversation format"""
        instruction = """Extract the jewelry catalog information from this German museum document as a JSON object.

Return a JSON object with ALL fields: Gegenstand, Inv.Nr, Herkunft, Foto Notes, Standort, Material, Datierung, Maße, Gewicht, erworben von, am, Preis, Vers.-Wert, Beschreibung, Literatur, Ausstellungen

Use empty strings "" for missing fields. Preserve exact German text with umlauts (ä, ö, ü, ß). Return ONLY JSON."""
        
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

    def prepare_training_data(self, jsonl_path: str, images_dir: str) -> List[Dict]:
        """Prepare training data in Unsloth format"""
        data = self.load_jsonl(jsonl_path)
        
        for item in data:
            item["image_path"] = self.find_image_path(item["file_name"], images_dir)
        
        valid_data = [item for item in data if os.path.exists(item["image_path"])]
        print(f"Found {len(valid_data)} valid samples out of {len(data)} total")
        
        converted_dataset = [self.convert_to_conversation(sample) for sample in valid_data]
        
        return converted_dataset
    
    def train_model(self, 
                   train_jsonl_path: str,
                   val_jsonl_path: str,
                   images_dir: str,
                   output_dir: str,
                   num_epochs: int = 15,
                   batch_size: int = 2,
                   learning_rate: float = 5e-5):
        """Train with Unsloth optimizations"""
        
        print("Preparing datasets...")
        train_dataset = self.prepare_training_data(train_jsonl_path, images_dir)
        val_dataset = self.prepare_training_data(val_jsonl_path, images_dir)
        print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples\n")
        
        # Enable training mode
        FastVisionModel.for_training(self.model)
        
        # Unsloth's SFTTrainer
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
            print(f"   • Batch size: {batch_size}, Grad accum: 4")
            print(f"{'='*60}\n")
        
        print("Starting training...")
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
            print(f"{'='*60}\n")
        
        return trainer
    
    def evaluate_on_test_set(self, test_jsonl_path: str, images_dir: str, output_dir: str) -> Dict:
        """Evaluate with chunking"""
        print("Starting evaluation with chunked loading...")
        
        # Unsloth inference mode
        FastVisionModel.for_inference(self.model)
        
        test_chunked = ChunkedDataset(test_jsonl_path, images_dir, chunk_size=20)
        
        all_predictions = []
        all_cer_scores = []
        
        instruction = """Extract the jewelry catalog information from this German museum document as a JSON object.

Return a JSON object with ALL fields: Gegenstand, Inv.Nr, Herkunft, Foto Notes, Standort, Material, Datierung, Maße, Gewicht, erworben von, am, Preis, Vers.-Wert, Beschreibung, Literatur, Ausstellungen

Use empty strings "" for missing fields. Preserve exact German text with umlauts (ä, ö, ü, ß). Return ONLY JSON."""
        
        for chunk_idx in range(test_chunked.num_chunks):
            print(f"\nProcessing chunk {chunk_idx + 1}/{test_chunked.num_chunks}")
            
            chunk_data = test_chunked.get_chunk(chunk_idx)
            
            for i, test_item in enumerate(chunk_data):
                print(f"Processing {i+1}/{len(chunk_data)} in chunk {chunk_idx + 1}", end='\r')
                
                try:
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": instruction},
                            {"type": "image", "image": test_item["image_path"]}
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
                        "file_name": test_item['file_name'],
                        "predicted_json": predicted_json,
                        "prediction_string": pred_str,
                        "target_json": self.clean_dict_for_ground_truth(test_item),
                        "target_text": gt_str,
                        "raw_response": generated_text,
                        "cer_score": cer_score,
                        "matched_image_path": test_item["image_path"],
                        "parsing_success": bool(predicted_json and pred_str and pred_str != generated_text)
                    }
                    all_predictions.append(prediction_entry)
                    all_cer_scores.append(cer_score)
                    
                    del inputs, outputs, generated_ids_trimmed, generated_texts
                    
                except Exception as e:
                    print(f"\nError: {test_item['file_name']}: {e}")
                    error_msg = f"Error: {str(e)}"
                    all_predictions.append({
                        "file_name": test_item['file_name'],
                        "predicted_json": {},
                        "prediction_string": error_msg,
                        "target_json": self.clean_dict_for_ground_truth(test_item),
                        "target_text": self.json_to_string_no_sort(test_item),
                        "raw_response": error_msg,
                        "cer_score": 1.0,
                        "matched_image_path": test_item.get("image_path", ""),
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
            f.write("NANONETS-OCR-S FINETUNING - EVALUATION RESULTS\n")
            f.write("SCHMUCK JEWELRY CATALOG DATASET\n")
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
            
            f.write("\n" + "="*60 + "\n")
            f.write("MODEL DETAILS\n")
            f.write("="*60 + "\n")
            f.write("Model: nanonets/Nanonets-OCR-s (original, local)\n")
            f.write("Framework: Unsloth\n")
            f.write("Quantization: 4-bit\n")
            f.write("LoRA: Rank 16, RSLoRA\n")
        
        print(f"\nResults saved to: {cer_file}")
    
    def save_model(self, trainer, output_dir: str):
        """Save finetuned model"""
        print(f"\nSaving model to {output_dir}...")
        trainer.model.save_pretrained(output_dir)
        trainer.tokenizer.save_pretrained(output_dir)
        print("✅ Model saved!")


def main():
    """Main training pipeline"""
    base_dir = "/home/vault/iwi5/iwi5298h/models_image_text/nanonets/schmuck"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}_unsloth")
    os.makedirs(run_dir, exist_ok=True)
    
    config = {
        "model_path": "/home/vault/iwi5/iwi5298h/models/Nanonets-OCR-s",  # ORIGINAL model
        "train_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl",
        "val_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/val.jsonl",
        "test_jsonl_path": "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl",
        "images_dir": "/home/woody/iwi5/iwi5298h/schmuck_images",
        "output_dir": run_dir,
        "num_epochs": 15,
        "batch_size": 2,
        "learning_rate": 5e-5,
    }
    
    config_file = os.path.join(run_dir, "training_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("NANONETS-OCR-S FINETUNING WITH UNSLOTH")
    print("="*60)
    print(f"Model: {config['model_path']}")
    print(f"Output: {run_dir}")
    print("="*60 + "\n")
    
    # Initialize
    finetuner = NanonetsSchmuckFinetune(model_path=config["model_path"])
    
    # Train
    print("\n" + "="*60)
    print("PHASE 1: TRAINING")
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
    
    # Save
    finetuner.save_model(trainer, run_dir)
    
    # Evaluate
    print("\n" + "="*60)
    print("PHASE 2: EVALUATION")
    print("="*60 + "\n")
    
    results = finetuner.evaluate_on_test_set(
        test_jsonl_path=config["test_jsonl_path"],
        images_dir=config["images_dir"],
        output_dir=run_dir
    )
    
    # Print summary
    cer_stats = results['cer_stats']
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Model: Nanonets-OCR-s (finetuned)")
    print(f"Dataset: Schmuck Jewelry Catalog\n")
    print(f"Metrics:")
    print(f"  Avg CER: {cer_stats['average_cer']:.4f} ({cer_stats['average_cer']*100:.2f}%)")
    print(f"  Weighted CER: {cer_stats['weighted_cer']:.4f} ({cer_stats['weighted_cer']*100:.2f}%)")
    print(f"  Median: {cer_stats['median_cer']:.4f} | Std: {cer_stats['std_cer']:.4f}")
    print(f"\nAccuracy:")
    print(f"  Perfect: {cer_stats['perfect_matches']}/{cer_stats['total_images']} ({cer_stats['perfect_matches']/cer_stats['total_images']*100:.2f}%)")
    print(f"  Parsing: {cer_stats['parsing_successes']}/{cer_stats['total_images']} ({cer_stats['parsing_success_rate']*100:.2f}%)")
    print(f"\nSaved to: {run_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
