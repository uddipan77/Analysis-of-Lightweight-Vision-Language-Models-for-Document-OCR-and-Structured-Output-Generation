import os
# CRITICAL: Set this BEFORE importing PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from huggingface_hub import snapshot_download
import torch
import json
from typing import List, Dict
from datetime import datetime
import re
import jiwer
from PIL import Image
import gc



def json_to_string(json_obj):
    """Convert JSON object to a consistent string representation."""
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(',', ':'))



def resize_image_if_needed(image_path: str, max_size: int = 1024) -> Image.Image:
    """Resize image if it's too large to prevent OOM errors."""
    image = Image.open(image_path)
    
    # Get current size
    width, height = image.size
    
    # Check if resizing is needed
    if width > max_size or height > max_size:
        # Calculate new size maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int((max_size / width) * height)
        else:
            new_height = max_size
            new_width = int((max_size / height) * width)
        
        print(f"  Resizing from {width}x{height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image



class SchmuckOCRProcessor:
    def __init__(self, model_path: str = "/home/vault/iwi5/iwi5298h/models/Nanonets-OCR-s"):
        """Initialize the OCR processor with Nanonets-OCR-s model."""
        print("Loading Nanonets-OCR-s model...")
        
        # Check if model exists locally, if not download it
        if not os.path.exists(model_path) or not os.path.exists(os.path.join(model_path, "config.json")):
            print(f"Model not found at {model_path}. Downloading...")
            
            # Create directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            
            try:
                snapshot_download(
                    repo_id="nanonets/Nanonets-OCR-s",
                    local_dir=model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"Model downloaded successfully to {model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {str(e)}")
        else:
            print(f"Model found at {model_path}")
        
        # Load model with memory optimization
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # Changed from bfloat16 to float16 for better compatibility
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Added
        )
        self.model.eval()
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True
        )
        print("Model loaded successfully!")
        
        # Clear any cached memory
        torch.cuda.empty_cache()
        gc.collect()
    
    def process_image_few_shot(self, test_image_path: str, few_shot_examples: List[Dict]) -> str:
        """Process a single image with few-shot prompting."""
        
        # Clear memory before processing
        torch.cuda.empty_cache()
        gc.collect()
        
        # Resize test image
        test_image = resize_image_if_needed(test_image_path, max_size=1024)
        
        # Create base prompt for the task
        base_prompt = """Extract the jewelry catalog information from this German museum document as a JSON object with the following fields:

- Gegenstand: Object type/name
- Inv.Nr: Inventory number
- Herkunft: Origin/provenance
- Foto Notes: Photo notes
- Standort: Location/storage
- Material: Material description
- Datierung: Dating/period
- Maße: Measurements/dimensions
- Gewicht: Weight
- erworben von: Acquired from (person/source)
- am: Date of acquisition
- Preis: Price
- Vers.-Wert: Insurance value
- Beschreibung: Description
- Literatur: Literature references
- Ausstellungen: Exhibitions

Return ONLY a valid JSON object. Use empty strings "" for missing fields."""

        # Build messages with few-shot examples
        messages = [{"role": "system", "content": "You are a helpful assistant specialized in extracting structured data from museum catalog documentation."}]
        
        # Resize and load all example images
        example_images = []
        for example in few_shot_examples:
            example_img = resize_image_if_needed(example['image_path'], max_size=1024)
            example_images.append(example_img)
        
        # Add few-shot examples as user-assistant pairs
        for i, example in enumerate(few_shot_examples):
            # User message with example image and prompt
            messages.append({
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": base_prompt},
                ]
            })
            
            # Assistant message with expected JSON output
            messages.append({
                "role": "assistant",
                "content": json_to_string(example['ground_truth'])
            })
        
        # Add the test image query
        messages.append({
            "role": "user", 
            "content": [
                {"type": "image"},
                {"type": "text", "text": base_prompt},
            ]
        })
        
        try:
            # Process the inputs with all images (examples + test)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Combine all images: example images + test image
            all_images = example_images + [test_image]
            
            inputs = self.processor(text=[text], images=all_images, padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            # Generate OCR output with torch.inference_mode
            with torch.inference_mode():  # More efficient than torch.no_grad()
                output_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=2048,
                    do_sample=False,
                    use_cache=True  # Enable KV cache for efficiency
                )
            
            # Extract only the generated tokens (remove input tokens)
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            
            # Decode the results
            output_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            result = output_text[0]
            
        finally:
            # CRITICAL: Clean up memory immediately in finally block
            del inputs, output_ids, generated_ids, test_image, example_images, all_images, text, messages
            if 'output_text' in locals():
                del output_text
            torch.cuda.empty_cache()
            gc.collect()
        
        return result



def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data



def get_empty_schmuck_structure() -> Dict:
    """Return empty jewelry catalog structure matching the expected schema."""
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
    
    # If still no valid JSON, return empty structure
    return get_empty_schmuck_structure()



def save_predictions_with_timestamp(predictions, test_data, all_cer_scores, few_shot_examples,
                                    base_dir="/home/vault/iwi5/iwi5298h/models_image_text/nanonets/shots_schmuck"):
    """Save predictions and CER results with timestamp directory."""
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"few_shots_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    # Save predictions to JSONL
    predictions_file = os.path.join(output_dir, "predictions_fewshot.jsonl")
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Predictions saved to: {predictions_file}")
    
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
        print("NANONETS-OCR-S FEW-SHOT CER EVALUATION RESULTS")
        print("JEWELRY CATALOG DATASET")
        print("="*60)
        print(f"\nFew-Shot Examples Used: {len(few_shot_examples)}")
        for i, ex in enumerate(few_shot_examples):
            print(f"  Example {i+1}: {ex['file_name']}")
        print(f"\nCER Statistics across {len(all_cer_scores)} images:")
        print("-" * 50)
        print(f"Average CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"Minimum CER: {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER: {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"\nWeighted CER: {weighted_cer:.4f} ({weighted_cer*100:.2f}%)")
        print(f"Total characters: {total_chars}")
        print(f"Total errors: {total_errors}")
        print(f"Perfect matches: {perfect_matches}/{len(all_cer_scores)} ({perfect_matches/len(all_cer_scores)*100:.1f}%)")
        
        # Save summary
        summary_file = os.path.join(output_dir, "evaluation_summary_fewshot.json")
        summary_data = {
            "timestamp": timestamp,
            "model": "Nanonets-OCR-s",
            "dataset": "jewelry_catalog",
            "method": "few-shot",
            "num_few_shot_examples": len(few_shot_examples),
            "few_shot_examples": [ex['file_name'] for ex in few_shot_examples],
            "total_images": len(predictions),
            "average_cer": float(avg_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "weighted_cer": float(weighted_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors)
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_file}")
        
        return output_dir, avg_cer
    
    return output_dir, None



def main():
    """Main function to process jewelry catalog images with few-shot learning."""
    
    # Paths
    train_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl"
    test_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
    images_dir = "/home/woody/iwi5/iwi5298h/schmuck_images"
    
    # Load train and test data
    print("="*60)
    print("NANONETS-OCR-S FEW-SHOT JEWELRY CATALOG OCR PROCESSING")
    print("="*60)
    print("\nLoading training data for few-shot examples...")
    train_data = load_jsonl(train_jsonl_path)
    print(f"Loaded {len(train_data)} training samples")
    
    print("\nLoading test data...")
    test_data = load_jsonl(test_jsonl_path)
    print(f"Loaded {len(test_data)} test samples")
    
    # Define few-shot examples (using the example you provided)
    few_shot_file_names = [
        "SCH_3149.jpg",  # Example from train.jsonl
        # Add more examples here if needed, e.g.:
        # "SCH_XXXX.jpg",  # Example 2
    ]
    
    # Find the examples in train data
    few_shot_examples = []
    for file_name in few_shot_file_names:
        for train_item in train_data:
            if train_item['file_name'] == file_name:
                # Prepare example with image path and ground truth (excluding file_name)
                example = {
                    'file_name': train_item['file_name'],
                    'image_path': os.path.join(images_dir, train_item['file_name']),
                    'ground_truth': {k: v for k, v in train_item.items() if k != "file_name"}
                }
                few_shot_examples.append(example)
                break
    
    print(f"\nUsing {len(few_shot_examples)} few-shot example(s):")
    for i, ex in enumerate(few_shot_examples):
        print(f"  {i+1}. {ex['file_name']}")
    
    # Verify all example images exist
    for ex in few_shot_examples:
        if not os.path.exists(ex['image_path']):
            raise FileNotFoundError(f"Few-shot example image not found: {ex['image_path']}")
    
    # Initialize OCR processor
    ocr_processor = SchmuckOCRProcessor()
    
    # Process test images with few-shot examples
    predictions = []
    all_cer_scores = []
    
    print(f"\nProcessing {len(test_data)} test images with {len(few_shot_examples)}-shot prompting...")
    print("-" * 60)
    
    for i, test_item in enumerate(test_data):
        print(f"Processing image {i+1}/{len(test_data)}: {test_item['file_name']}")
        
        # Construct image path using file_name
        image_name = test_item['file_name']
        image_path = os.path.join(images_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Process image with few-shot examples
            response = ocr_processor.process_image_few_shot(image_path, few_shot_examples)
            predicted_json = extract_json_from_response(response)
            
            # Convert both ground truth and prediction to JSON strings (excluding file_name)
            gt_json = {k: v for k, v in test_item.items() if k != "file_name"}
            gt_json_string = json_to_string(gt_json)
            pred_json_string = json_to_string(predicted_json)
            
            # Calculate CER using jiwer
            cer_score = jiwer.cer(gt_json_string, pred_json_string)
            
            # Store prediction with metadata
            prediction_entry = {
                "file_name": test_item['file_name'],
                "predicted_json": predicted_json,
                "ground_truth": gt_json,
                "raw_response": response,
                "cer_score": cer_score
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(cer_score)
            
            print(f"  Processed successfully. CER: {cer_score:.4f}")
            
            # Explicit cleanup after each image
            del response, predicted_json, gt_json, prediction_entry
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {test_item['file_name']}: {str(e)}")
            # Add empty prediction for failed cases
            prediction_entry = {
                "file_name": test_item['file_name'],
                "predicted_json": get_empty_schmuck_structure(),
                "ground_truth": {k: v for k, v in test_item.items() if k != "file_name"},
                "raw_response": f"Error: {str(e)}",
                "cer_score": 1.0
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)
            
            # Clear memory even on error
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    # Save predictions with timestamp
    output_dir, avg_cer = save_predictions_with_timestamp(
        predictions, test_data, all_cer_scores, few_shot_examples
    )
    
    print(f"\n{'='*60}")
    print(f"FEW-SHOT PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model: Nanonets-OCR-s")
    print(f"Dataset: Jewelry Catalog")
    print(f"Method: Few-Shot Learning ({len(few_shot_examples)} example(s))")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER on test set: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()
