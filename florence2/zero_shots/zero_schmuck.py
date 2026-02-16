#!/usr/bin/env python3
# florence2_schmuck_zeroshot_FIXED.py
# ✅ FIXED: Removed duplicate self.processor line
# ✅ FIXED: use_cache=False to avoid KV cache errors
# ✅ FIXED: If JSON parse fails, compare raw output with GT JSON string

import sys
import os

# ✅ Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import json
from typing import List, Dict
import jiwer
from datetime import datetime



def extract_text_values_from_json(json_data: Dict) -> str:
    """Extract all text values from Schmuck JSON data for CER calculation."""
    values = []
    
    for key, value in json_data.items():
        if key == "file_name":
            continue
        if value:
            values.append(str(value))
    
    return " ".join(values).strip()



def json_to_string(json_obj: Dict) -> str:
    """Convert JSON object to string (without file_name)."""
    clean_obj = {k: v for k, v in json_obj.items() if k != "file_name"}
    return json.dumps(clean_obj, ensure_ascii=False, separators=(',', ':'))



class Florence2SchmuckOCRProcessor:
    def __init__(self, model_name: str = "/home/vault/iwi5/iwi5298h/models/florence2_large"):
        """Initialize the OCR processor with Florence-2-large model."""
        print("="*60)
        print("Loading Florence-2-large model from local cache...")
        print("="*60)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"Device: {self.device}")
        print(f"Dtype: {self.torch_dtype}")
        
        print("⏳ Loading model (this may take 1-2 minutes)...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=self.torch_dtype, 
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(self.device)
        
        print("⏳ Loading processor...")
        
        self.processor = AutoProcessor.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        model_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"✅ Model loaded successfully!")
        print(f"Model parameters: {model_params:.2f}B")
        print("="*60 + "\n")
    
    def extract_ocr_text(self, florence_output) -> str:
        """Extract clean OCR text from Florence-2's structured output."""
        if isinstance(florence_output, dict):
            if '<OCR>' in florence_output:
                return florence_output['<OCR>'].strip()
            elif 'text' in florence_output:
                return florence_output['text'].strip()
            else:
                return str(florence_output).strip()
        elif isinstance(florence_output, str):
            return florence_output.strip()
        else:
            return str(florence_output).strip()
    
    def process_image_zero_shot(self, image_path: str) -> str:
        """Process a single image with zero-shot OCR using Florence-2."""
        try:
            # Load the image
            image = Image.open(image_path).convert('RGB')
            
            # Florence-2 uses specific task prompts for OCR
            task_prompt = "<OCR>"
            
            # Proper processor usage
            inputs = self.processor(
                text=task_prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # Move inputs to device properly
            input_ids = inputs["input_ids"].to(self.device)
            pixel_values = inputs["pixel_values"].to(self.device, self.torch_dtype)
            
            # Check if pixel_values is valid
            if pixel_values is None or pixel_values.numel() == 0:
                return "Error: Failed to process image - empty pixel values"
            
            # ✅ CRITICAL FIX: use_cache=False to avoid KV cache errors
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=1024,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False,  # ✅ Critical fix for past_key_values error
                    early_stopping=False
                )
            
            # Decode the generated text
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Post-process the generation
            parsed_answer = self.processor.post_process_generation(
                generated_text, 
                task=task_prompt, 
                image_size=(image.width, image.height)
            )
            
            # Extract clean text
            clean_text = self.extract_ocr_text(parsed_answer)
            
            return clean_text
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}"
            print(f"  ❌ Error in Florence-2 processing: {e}")
            traceback.print_exc()
            return error_msg



def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data



def save_jsonl(data: List[Dict], file_path: str):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')



def main():
    """Main function to process Schmuck jewelry images with Florence-2 zero-shot OCR."""
    
    print("="*60)
    print("FLORENCE-2 LARGE ZERO-SHOT OCR EVALUATION")
    print("SCHMUCK JEWELRY CATALOG DATASET")
    print("="*60 + "\n")
    
    # Paths
    test_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
    images_dir = "/home/woody/iwi5/iwi5298h/schmuck_images"
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/vault/iwi5/iwi5298h/models_image_text/florence2/shots_schmuck/run_zeroshot_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}\n")
    
    predictions_output_path = os.path.join(output_dir, "test_predictions_zeroshot.jsonl")
    
    # Load test data
    print("Loading test data...")
    try:
        test_data = load_jsonl(test_jsonl_path)
        print(f"✅ Loaded {len(test_data)} test samples\n")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return
    
    # Initialize OCR processor
    ocr_processor = Florence2SchmuckOCRProcessor()
    
    # Process test images
    predictions = []
    all_cer_scores = []
    json_parse_failures = 0
    
    print(f"{'='*60}")
    print(f"Processing {len(test_data)} test images with Florence-2 zero-shot OCR...")
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
            # Process image with zero-shot OCR
            print(f"  🔄 Processing with Florence-2...")
            predicted_text = ocr_processor.process_image_zero_shot(image_path)
            
            # Try to parse as JSON
            predicted_json = None
            json_parse_success = False
            
            try:
                predicted_json = json.loads(predicted_text.strip())
                json_parse_success = True
                print(f"  ✅ JSON parsed successfully")
            except json.JSONDecodeError:
                print(f"  ⚠️  JSON parse failed - using raw output for CER")
                json_parse_failures += 1
            
            # Get ground truth as JSON string (without file_name)
            gt_json_string = json_to_string(test_item)
            
            # Calculate CER based on what we have
            if json_parse_success and predicted_json:
                # If JSON parsed successfully, convert to string for comparison
                pred_json_string = json_to_string({"dummy": "x", **predicted_json})
                cer_score = jiwer.cer(gt_json_string, pred_json_string)
            else:
                # If JSON parse failed, compare raw output with GT JSON string
                cer_score = jiwer.cer(gt_json_string, predicted_text.strip())
            
            # Store prediction
            prediction_entry = {
                "file_name": test_item['file_name'],
                "predicted_text": predicted_text.strip(),
                "predicted_json": predicted_json if json_parse_success else None,
                "json_parse_success": json_parse_success,
                "ground_truth_json_string": gt_json_string,
                "ground_truth_fields": {k: v for k, v in test_item.items() if k != "file_name"},
                "cer_score": cer_score
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(cer_score)
            
            print(f"  ✅ Processed successfully!")
            print(f"  📊 CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
            print(f"  📏 Lengths: GT={len(gt_json_string)}, Pred={len(predicted_text.strip())}")
            
            # Show preview
            preview = predicted_text.strip()[:100] + "..." if len(predicted_text.strip()) > 100 else predicted_text.strip()
            print(f"  📝 Preview: {preview}")
            
            # Running average
            running_avg = sum(all_cer_scores) / len(all_cer_scores)
            print(f"  📈 Running Avg CER: {running_avg:.4f} ({running_avg*100:.2f}%)")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            gt_json_string = json_to_string(test_item)
            prediction_entry = {
                "file_name": test_item['file_name'],
                "predicted_text": f"Error: {str(e)}",
                "predicted_json": None,
                "json_parse_success": False,
                "ground_truth_json_string": gt_json_string,
                "ground_truth_fields": {k: v for k, v in test_item.items() if k != "file_name"},
                "cer_score": 1.0
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}\n")
    
    # Save predictions
    print(f"💾 Saving predictions to {predictions_output_path}...")
    try:
        save_jsonl(predictions, predictions_output_path)
        print(f"✅ Predictions saved successfully!")
    except Exception as e:
        print(f"❌ Error saving predictions: {e}")
        return
    
    # Calculate and print CER statistics
    if all_cer_scores:
        print("\n" + "="*60)
        print("FLORENCE-2 ZERO-SHOT EVALUATION RESULTS")
        print("SCHMUCK JEWELRY CATALOG DATASET")
        print("="*60)
        
        avg_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        
        print(f"\nCER Statistics across {len(all_cer_scores)} images:")
        print("-" * 50)
        print(f"Average CER:  {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Minimum CER:  {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER:  {max_cer:.4f} ({max_cer*100:.2f}%)")
        
        # Calculate weighted CER
        total_chars = 0
        total_errors = 0
        
        for pred in predictions:
            gt_string = pred["ground_truth_json_string"]
            char_count = len(gt_string)
            cer_score = pred["cer_score"]
            
            total_chars += char_count
            total_errors += int(cer_score * char_count)
        
        weighted_cer = total_errors / total_chars if total_chars > 0 else 0
        print(f"\nWeighted CER:  {weighted_cer:.4f} ({weighted_cer*100:.2f}%)")
        print(f"Total characters: {total_chars}")
        print(f"Total errors:     {total_errors}")
        
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)
        json_success_rate = ((len(predictions) - json_parse_failures) / len(predictions) * 100) if len(predictions) > 0 else 0
        
        print(f"\nPerfect matches: {perfect_matches}/{len(all_cer_scores)} ({perfect_matches/len(all_cer_scores)*100:.1f}%)")
        print(f"JSON parse success: {len(predictions) - json_parse_failures}/{len(predictions)} ({json_success_rate:.1f}%)")
        print(f"JSON parse failures: {json_parse_failures}")
        
        # Save summary
        summary_file = os.path.join(output_dir, "evaluation_summary_zeroshot.json")
        summary_data = {
            "timestamp": timestamp,
            "model_name": "Florence-2-large",
            "model_path": "/home/vault/iwi5/iwi5298h/models/florence2_large",
            "dataset": "Schmuck Jewelry Catalog",
            "approach": "zero_shot_ocr",
            "total_images": len(predictions),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "json_parse_success_count": len(predictions) - json_parse_failures,
            "json_parse_failures": json_parse_failures,
            "json_success_rate": float(json_success_rate),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "note": "When JSON parse fails, CER is calculated between raw output and ground truth JSON string"
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Summary saved to: {summary_file}")
        print(f"✅ Predictions saved to: {predictions_output_path}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model:   Florence-2-large")
        print(f"Dataset: Schmuck Jewelry Catalog")
        print(f"Method:  Zero-Shot OCR")
        print(f"Images:  {len(predictions)} processed")
        print(f"Avg CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"JSON success rate: {json_success_rate:.1f}%")
        print(f"\nAll outputs saved to: {output_dir}")
        print("="*60 + "\n")
        
    else:
        print("\n❌ No images were successfully processed.")



if __name__ == "__main__":
    main()
