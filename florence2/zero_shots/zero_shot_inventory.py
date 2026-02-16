import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import json
import os
from typing import List, Dict
from datetime import datetime
import jiwer
import warnings
import unicodedata

warnings.filterwarnings("ignore")


def normalize_unicode(text: str) -> str:
    """Normalize unicode for consistent comparison."""
    return unicodedata.normalize("NFC", str(text))


def extract_text_from_json(json_obj) -> str:
    """
    Recursively extract all text values from a JSON object and concatenate them.
    Preserves field order for consistent comparison.
    """
    texts = []
    
    def extract_recursive(obj):
        if isinstance(obj, dict):
            # Use consistent key order (not sorted, preserve original order)
            for key, value in obj.items():
                extract_recursive(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_recursive(item)
        elif isinstance(obj, str):
            if obj.strip():
                texts.append(obj.strip())
        elif obj is not None:
            texts.append(str(obj))
    
    extract_recursive(json_obj)
    return " ".join(texts)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def run_florence_ocr(image_path: str, model, processor, device, torch_dtype, task="<OCR>"):
    """
    Run Florence-2 OCR on a single image.
    Supports multiple task prompts: <OCR>, <OCR_WITH_REGION>, <MORE_DETAILED_CAPTION>
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Task prompt
    task_prompt = task
    
    # Prepare inputs
    inputs = processor(text=task_prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    
    # Generate with improved parameters
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,  # Increased for longer texts
        num_beams=3,
        do_sample=False,
        early_stopping=True,
    )
    
    # Decode
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Post-process
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )
    
    # Extract text based on task type
    if isinstance(parsed_answer, dict):
        if task_prompt in parsed_answer:
            result = parsed_answer[task_prompt]
            # For OCR_WITH_REGION, extract just the text labels
            if task_prompt == "<OCR_WITH_REGION>" and isinstance(result, dict):
                if "labels" in result:
                    return " ".join(result["labels"])
                elif "quad_boxes" in result and "labels" in result:
                    return " ".join(result["labels"])
            return str(result) if not isinstance(result, str) else result
        else:
            # Return all text values joined
            return " ".join(str(v) for v in parsed_answer.values() if v)
    else:
        return str(parsed_answer)


def main():
    """Main function for Florence-2 zero-shot evaluation."""
    
    # Paths
    test_jsonl_path = "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl"
    images_dir = "/home/woody/iwi5/iwi5298h/inventory_images"
    output_base_dir = "/home/vault/iwi5/iwi5298h/models_image_text/florence2/few_zero_inven"
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"run_zeroshot_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("FLORENCE-2-LARGE ZERO-SHOT OCR EVALUATION")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading Florence-2-large model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large", 
        torch_dtype=torch_dtype, 
        trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large", 
        trust_remote_code=True
    )
    print("Model loaded successfully!")
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_jsonl(test_jsonl_path)
    print(f"Loaded {len(test_data)} test samples")
    
    print(f"\n📝 Evaluation Method:")
    print("Ground truth JSON → Extract all text → Concatenate")
    print("Florence-2 <OCR> task → Extract text")
    print("Compare both texts → Calculate CER\n")
    
    # Process images
    predictions = []
    all_cer_scores = []
    
    print(f"Processing {len(test_data)} images...")
    print("-" * 60)
    
    for i, test_item in enumerate(test_data):
        image_name = test_item['image_name']
        print(f"\n[{i+1}/{len(test_data)}] Processing: {image_name}")
        
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"  ⚠ Image not found!")
            # Add failed entry with CER 1.0
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_text = extract_text_from_json(gt_json)
            predictions.append({
                "image_name": image_name,
                "predicted_text": "",
                "ground_truth_text": gt_text,
                "ground_truth_json": gt_json,
                "cer_score": 1.0,
                "gt_length": len(gt_text),
                "pred_length": 0,
                "error": "Image not found"
            })
            all_cer_scores.append(1.0)
            continue
        
        try:
            # Run Florence-2 OCR with <OCR> task
            predicted_text = run_florence_ocr(image_path, model, processor, device, torch_dtype, task="<OCR>")
            predicted_text = normalize_unicode(predicted_text)
            
            # Get ground truth text
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_text = extract_text_from_json(gt_json)
            
            # Calculate CER
            cer_score = jiwer.cer(gt_text, predicted_text)
            
            # Store results
            prediction_entry = {
                "image_name": image_name,
                "predicted_text": predicted_text,
                "ground_truth_text": gt_text,
                "ground_truth_json": gt_json,
                "cer_score": cer_score,
                "gt_length": len(gt_text),
                "pred_length": len(predicted_text)
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(cer_score)
            
            print(f"  GT length: {len(gt_text)} chars")
            print(f"  Pred length: {len(predicted_text)} chars")
            print(f"  ✓ CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            
            # Add failed entry
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_text = extract_text_from_json(gt_json)
            
            predictions.append({
                "image_name": image_name,
                "predicted_text": f"ERROR: {str(e)}",
                "ground_truth_text": gt_text,
                "ground_truth_json": gt_json,
                "cer_score": 1.0,
                "gt_length": len(gt_text),
                "pred_length": 0
            })
            all_cer_scores.append(1.0)
        
        # Clear GPU cache periodically
        if (i + 1) % 5 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print('='*60)
    print(f"Output directory: {output_dir}")
    
    # Save predictions
    predictions_file = os.path.join(output_dir, "predictions_zeroshot.jsonl")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"✓ Predictions saved: {predictions_file}")
    
    # Calculate statistics
    if all_cer_scores:
        # Macro CER: average of per-sample CER scores
        macro_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        
        # Micro CER: total errors / total characters (character-weighted)
        total_chars = sum(p['gt_length'] for p in predictions)
        total_errors = sum(int(s * predictions[i]['gt_length']) 
                          for i, s in enumerate(all_cer_scores))
        micro_cer = total_errors / total_chars if total_chars > 0 else 0
        
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)
        
        # Print statistics
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS - ZERO-SHOT")
        print('='*60)
        print(f"Total images: {len(all_cer_scores)}")
        print(f"Macro CER (avg per-sample): {macro_cer:.4f} ({macro_cer*100:.2f}%)")
        print(f"Micro CER (char-weighted):  {micro_cer:.4f} ({micro_cer*100:.2f}%)")
        print(f"Minimum CER: {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER: {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"Perfect matches: {perfect_matches}/{len(all_cer_scores)}")
        print(f"Total characters: {total_chars}")
        print(f"Total errors: {total_errors}")
        
        # Save summary
        summary = {
            "timestamp": timestamp,
            "model": "Florence-2-large",
            "method": "zero-shot",
            "task": "<OCR>",
            "total_images": len(all_cer_scores),
            "macro_cer": float(macro_cer),
            "micro_cer": float(micro_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors)
        }
        
        summary_file = os.path.join(output_dir, "evaluation_summary_zeroshot.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Summary saved: {summary_file}")
    
    print(f"\n{'='*60}")
    print("COMPLETE!")
    print('='*60)
    print(f"Results directory: {output_dir}\n")


if __name__ == "__main__":
    main()