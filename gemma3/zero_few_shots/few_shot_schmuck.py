#!/usr/bin/env python3
# gemma3_schmuck_fewshot_v100.py
# Few-shot OCR on Schmuck jewelry catalog using Gemma 3 4B (Unsloth)
# ✅ Fixed to match working inventory code pattern

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
import json
import os
from typing import List, Dict
from datetime import datetime
import re
import jiwer


def json_to_string(json_obj):
    """Convert JSON object to a consistent string representation."""
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(',', ':'))


class SchmuckOCRProcessorGemma:
    def __init__(self, model_path: str = "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"):
        """Initialize the OCR processor with Gemma 3 model from Unsloth."""
        print("="*60)
        print("Loading Gemma 3 4B model from Unsloth")
        print("="*60)
        
        # ✅ Use bfloat16 like the working inventory code
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"✅ Gemma 3 model loaded successfully!")
        print("="*60 + "\n")
    
    def create_few_shot_messages(self, few_shot_examples: List[Dict], images_dir: str) -> List[Dict]:
        """Create few-shot messages with both images and expected outputs."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert OCR assistant specialized in extracting structured information from German museum jewelry catalog documents. Extract all information accurately and return it as a JSON object."}]
            }
        ]
        
        # Add few-shot examples
        for example in few_shot_examples:
            # Load image for this example
            image_path = os.path.join(images_dir, example["file_name"])
            
            if not os.path.exists(image_path):
                print(f"Warning: Few-shot image not found: {image_path}")
                continue
                
            image = Image.open(image_path)
            
            # User message with image
            messages.append({
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract the jewelry catalog information from this German museum document as a JSON object with the following fields: Gegenstand, Inv.Nr, Herkunft, Foto Notes, Standort, Material, Datierung, Maße, Gewicht, erworben von, am, Preis, Vers.-Wert, Beschreibung, Literatur, Ausstellungen. Return only the JSON object with empty strings for missing fields."}
                ]
            })
            
            # Assistant response with expected JSON (exclude file_name)
            expected_json = {k: v for k, v in example.items() if k != "file_name"}
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(expected_json, ensure_ascii=False, separators=(',', ':'))}]
            })
        
        return messages
    
    def process_image_with_few_shot(self, image_path: str, few_shot_messages: List[Dict]) -> str:
        """Process a single image with few-shot examples."""
        # Load the test image
        image = Image.open(image_path)
        
        # Create the full conversation including few-shot examples and the test image
        messages = few_shot_messages.copy()
        
        # Add the test image query
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract the jewelry catalog information from this German museum document as a JSON object with the following fields: Gegenstand, Inv.Nr, Herkunft, Foto Notes, Standort, Material, Datierung, Maße, Gewicht, erworben von, am, Preis, Vers.-Wert, Beschreibung, Literatur, Ausstellungen. Return only the JSON object with empty strings for missing fields."}
            ]
        })
        
        # ✅ FIXED: Apply chat template and move to device WITHOUT dtype conversion
        # This matches the working inventory code pattern
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate OCR output
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1
            )
            generation = generation[0][input_len:]
        
        output_text = self.processor.decode(generation, skip_special_tokens=True)
        return output_text


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def extract_json_from_response(response: str) -> Dict:
    """Extract JSON object from model response."""
    # Clean the response
    response = response.strip()
    
    # Try to find JSON in the response
    json_pattern = r'\{.*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                # Try to parse the JSON-like match
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON found, try to parse the entire response
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # If still no valid JSON, return empty structure for Schmuck
    return {
        "Gegenstand": "",
        "Inv.Nr": "",
        "Herkunft": "",
        "Foto Notes": "",
        "Standort": "",
        "Material": "",
        "Datierung": "",
        "Maße": "",
        "Gewicht": "",
        "erworben von": "",
        "am": "",
        "Preis": "",
        "Vers.-Wert": "",
        "Beschreibung": "",
        "Literatur": "",
        "Ausstellungen": ""
    }


def save_predictions_with_timestamp(predictions, test_data, all_cer_scores, base_dir="/home/vault/iwi5/iwi5298h/models_image_text/gemma/schmuck/few_zero_schmuck"):
    """Save predictions and CER results with timestamp directory."""
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_fewshot_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Saving results to: {output_dir}")
    print(f"{'='*60}")
    
    # Save predictions to JSONL
    predictions_file = os.path.join(output_dir, "predictions_fewshot.jsonl")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✅ Predictions saved to: {predictions_file}")
    
    # Calculate CER statistics
    if all_cer_scores:
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        
        # Calculate weighted CER
        total_chars = 0
        total_errors = 0
        
        for i, test_item in enumerate(test_data[:len(predictions)]):
            # Exclude file_name from ground truth for CER calculation
            gt_json = {k: v for k, v in test_item.items() if k != "file_name"}
            gt_json_string = json_to_string(gt_json)
            char_count = len(gt_json_string)
            cer_score = all_cer_scores[i]
            
            total_chars += char_count
            total_errors += int(cer_score * char_count)
        
        weighted_cer = total_errors / total_chars if total_chars > 0 else 0
        
        # Count perfect matches
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)
        
        # Print results
        print("\n" + "="*60)
        print("GEMMA 3 FEW-SHOT CER EVALUATION RESULTS")
        print("SCHMUCK JEWELRY CATALOG DATASET")
        print("="*60)
        print(f"\nCER Statistics across {len(all_cer_scores)} images:")
        print("-" * 50)
        print(f"Average CER:     {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Minimum CER:     {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER:     {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"\nWeighted CER:    {weighted_cer:.4f} ({weighted_cer*100:.2f}%)")
        print(f"Total characters: {total_chars}")
        print(f"Total errors:     {total_errors}")
        print(f"\nPerfect matches:  {perfect_matches}/{len(all_cer_scores)} ({perfect_matches/len(all_cer_scores)*100:.1f}%)")
        
        # Save summary
        summary_file = os.path.join(output_dir, "evaluation_summary_fewshot.json")
        
        summary_data = {
            "timestamp": timestamp,
            "model": "Gemma 3 4B (Unsloth)",
            "dataset": "Schmuck Jewelry Catalog",
            "method": "few-shot",
            "total_images": len(predictions),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "few_shot_examples": ["SCH_3051.jpg", "SCH_3149.jpg"]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Summary saved to: {summary_file}")
        
        return output_dir, avg_cer
    
    return output_dir, None


def main():
    """Main function to process Schmuck jewelry images with Gemma 3 few-shot learning."""
    
    # Paths
    test_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
    train_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl"
    images_dir = "/home/woody/iwi5/iwi5298h/schmuck_images"
    
    # Load test data
    print("="*60)
    print("GEMMA 3 FEW-SHOT SCHMUCK JEWELRY CATALOG OCR")
    print("="*60)
    print("\nLoading test data...")
    test_data = load_jsonl(test_jsonl_path)
    print(f"✅ Loaded {len(test_data)} test samples")
    
    # Load few-shot examples from train data
    print("\nLoading few-shot examples...")
    train_data = load_jsonl(train_jsonl_path)
    
    # Use the 2 specified examples for few-shot learning
    few_shot_examples = []
    target_images = ["SCH_3051.jpg", "SCH_3149.jpg"]
    
    for target_image in target_images:
        for example in train_data:
            if example["file_name"] == target_image:
                few_shot_examples.append(example)
                print(f"  ✅ Found few-shot example: {target_image}")
                break
    
    if len(few_shot_examples) < 2:
        print(f"⚠️  Warning: Only found {len(few_shot_examples)} of the 2 specified few-shot examples")
        # Fallback to first 2 examples if specific ones not found
        few_shot_examples = train_data[:2]
        print("  Using first 2 training samples as fallback")
    
    print(f"\n✅ Using {len(few_shot_examples)} few-shot examples")
    
    # Initialize OCR processor
    ocr_processor = SchmuckOCRProcessorGemma()
    
    # Create few-shot messages with images
    print("\nCreating few-shot messages with images...")
    few_shot_messages = ocr_processor.create_few_shot_messages(few_shot_examples, images_dir)
    print(f"✅ Created few-shot conversation with {len(few_shot_messages)} messages")
    
    # Process test images
    predictions = []
    all_cer_scores = []
    
    print(f"\n{'='*60}")
    print(f"Processing {len(test_data)} test images with Gemma 3 (Few-Shot)...")
    print(f"{'='*60}\n")
    
    for i, test_item in enumerate(test_data):
        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(test_data)}] {test_item['file_name']}")
        print(f"{'─'*60}")
        
        # Construct image path
        image_path = os.path.join(images_dir, test_item['file_name'])
        
        if not os.path.exists(image_path):
            print(f"  ⚠️  Image not found: {image_path}")
            continue
        
        try:
            # Process image with few-shot examples
            print(f"  🔄 Processing with Gemma 3 (Few-Shot)...")
            response = ocr_processor.process_image_with_few_shot(image_path, few_shot_messages)
            predicted_json = extract_json_from_response(response)
            
            # Convert both ground truth and prediction to JSON strings (excluding file_name)
            gt_json = {k: v for k, v in test_item.items() if k != "file_name"}
            gt_json_string = json_to_string(gt_json)
            pred_json_string = json_to_string(predicted_json)
            
            # Calculate CER using jiwer
            cer_score = jiwer.cer(gt_json_string, pred_json_string)
            
            # Store prediction with metadata (file_name is included but not predicted)
            prediction_entry = {
                "file_name": test_item['file_name'],
                "predicted_json": predicted_json,
                "ground_truth": gt_json,
                "raw_response": response,
                "cer_score": cer_score
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(cer_score)
            
            # Enhanced logging
            print(f"  ✅ Processed successfully!")
            print(f"  📊 CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
            print(f"  📏 Lengths: GT={len(gt_json_string)}, Pred={len(pred_json_string)}")
            
            # Show first 100 chars of prediction
            preview = pred_json_string[:100] + "..." if len(pred_json_string) > 100 else pred_json_string
            print(f"  📝 Preview: {preview}")
            
            # Running average
            running_avg = sum(all_cer_scores) / len(all_cer_scores)
            print(f"  📈 Running Avg CER: {running_avg:.4f} ({running_avg*100:.2f}%)")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Add empty prediction for failed cases
            prediction_entry = {
                "file_name": test_item['file_name'],
                "predicted_json": {
                    "Gegenstand": "",
                    "Inv.Nr": "",
                    "Herkunft": "",
                    "Foto Notes": "",
                    "Standort": "",
                    "Material": "",
                    "Datierung": "",
                    "Maße": "",
                    "Gewicht": "",
                    "erworben von": "",
                    "am": "",
                    "Preis": "",
                    "Vers.-Wert": "",
                    "Beschreibung": "",
                    "Literatur": "",
                    "Ausstellungen": ""
                },
                "ground_truth": {k: v for k, v in test_item.items() if k != "file_name"},
                "raw_response": f"Error: {str(e)}",
                "cer_score": 1.0
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}\n")
    
    # Save predictions with timestamp
    output_dir, avg_cer = save_predictions_with_timestamp(predictions, test_data, all_cer_scores)
    
    print(f"\n{'='*60}")
    print(f"GEMMA 3 FEW-SHOT PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model:   Gemma 3 4B (Unsloth)")
    print(f"Dataset: Schmuck Jewelry Catalog")
    print(f"Method:  Few-Shot Learning (2 examples)")
    print(f"\nResults: {output_dir}")
    print(f"Images:  {len(predictions)} processed")
    if avg_cer is not None:
        print(f"Avg CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
