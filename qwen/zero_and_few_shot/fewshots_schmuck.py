import os
# CRITICAL: Set this BEFORE importing PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from typing import List, Dict, Any
from datetime import datetime
import re
import jiwer
from PIL import Image
import gc



def json_to_string(json_obj):
    """Convert JSON object to a consistent string representation."""
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(',', ':'))



def resize_image_if_needed(image_path: str, max_size: int = 1024) -> str:
    """
    Resize image if it's too large to prevent OOM errors.
    Returns path to resized image (or original if no resize needed).
    """
    image = Image.open(image_path)
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
        
        # Save to temporary file
        temp_path = image_path.replace('.jpg', '_resized.jpg')
        image.save(temp_path, 'JPEG', quality=95)
        image.close()
        return temp_path
    
    image.close()
    return image_path



class SchmuckOCRProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize the OCR processor with Qwen2.5-VL model."""
        print("Loading Qwen2.5-VL model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded successfully!")
        
        # Clear cache after loading
        torch.cuda.empty_cache()
        gc.collect()
    
    def create_few_shot_messages(self, few_shot_examples: List[Dict], images_dir: str) -> List[Dict]:
        """Create few-shot messages with both images and expected outputs."""
        messages = []
        
        # Add few-shot examples
        for example in few_shot_examples:
            # Resize few-shot example image
            original_path = os.path.join(images_dir, example["file_name"])
            resized_path = resize_image_if_needed(original_path, max_size=1024)
            
            # User message with image
            messages.append({
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "image": resized_path
                    },
                    {
                        "type": "text", 
                        "text": """Extract the jewelry catalog information from this German museum document as a JSON object with the following fields:
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

Return only a valid JSON object with these exact field names. Use empty strings "" for missing fields."""
                    }
                ]
            })
            
            # Assistant response with expected JSON
            expected_json = {k: v for k, v in example.items() if k != "file_name"}
            messages.append({
                "role": "assistant",
                "content": json.dumps(expected_json, ensure_ascii=False, separators=(',', ':'))
            })
        
        return messages
    
    def process_image_with_few_shot(self, image_path: str, few_shot_messages: List[Dict]) -> str:
        """Process a single image with few-shot examples."""
        
        # Clear memory before processing
        torch.cuda.empty_cache()
        gc.collect()
        
        # Resize test image
        processed_image_path = resize_image_if_needed(image_path, max_size=1024)
        
        try:
            # Create the full conversation including few-shot examples and the test image
            messages = few_shot_messages.copy()
            
            # Add the test image query
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": processed_image_path
                    },
                    {
                        "type": "text",
                        "text": """Extract the jewelry catalog information from this German museum document as a JSON object with the following fields:
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

Return only a valid JSON object with these exact field names. Use empty strings "" for missing fields."""
                    }
                ],
            })
            
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move to GPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            inputs = inputs.to(device)
            
            # Generate with inference_mode for memory efficiency
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)
            
            # Extract generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode results
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            result = output_text[0]
            
        finally:
            # CRITICAL: Clean up ALL intermediate variables
            if 'inputs' in locals():
                del inputs
            if 'generated_ids' in locals():
                del generated_ids
            if 'generated_ids_trimmed' in locals():
                del generated_ids_trimmed
            if 'output_text' in locals():
                del output_text
            if 'messages' in locals():
                del messages
            if 'text' in locals():
                del text
            if 'image_inputs' in locals():
                del image_inputs
            if 'video_inputs' in locals():
                del video_inputs
            
            # Remove resized temp file if created
            if processed_image_path != image_path:
                try:
                    os.remove(processed_image_path)
                except:
                    pass
            
            # Force memory cleanup
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



def save_predictions_with_timestamp(predictions, test_data, all_cer_scores, few_shot_examples, base_dir="/home/vault/iwi5/iwi5298h/models_image_text/qwen/shots_schmuck"):
    """Save predictions and CER results with timestamp directory."""
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"few_shot_{timestamp}")
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
        print("QWEN2.5-VL FEW-SHOT CER EVALUATION RESULTS")
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
            "model": "Qwen2.5-VL-7B-Instruct",
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
    test_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/test.jsonl"
    train_jsonl_path = "/home/woody/iwi5/iwi5298h/json_schmuck/train.jsonl"
    images_dir = "/home/woody/iwi5/iwi5298h/schmuck_images"
    
    # Load test data
    print("="*60)
    print("QWEN2.5-VL FEW-SHOT JEWELRY CATALOG OCR PROCESSING")
    print("="*60)
    print("\nLoading test data...")
    test_data = load_jsonl(test_jsonl_path)
    print(f"Loaded {len(test_data)} test samples")
    
    # Load few-shot examples from train data
    print("\nLoading few-shot examples...")
    train_data = load_jsonl(train_jsonl_path)
    
    # Use the same example as Nanonets few-shot: SCH_3149.jpg
    few_shot_examples = []
    target_files = ["SCH_3149.jpg"]  # Same as Nanonets few-shot example
    
    for target_file in target_files:
        for example in train_data:
            if example["file_name"] == target_file:
                few_shot_examples.append(example)
                break
    
    if len(few_shot_examples) < len(target_files):
        print(f"Warning: Only found {len(few_shot_examples)} of the {len(target_files)} specified few-shot examples")
        # Fallback to first example if specific one not found
        if len(few_shot_examples) == 0:
            few_shot_examples = train_data[:1]
    
    print(f"\nUsing {len(few_shot_examples)} few-shot example(s):")
    for i, ex in enumerate(few_shot_examples):
        print(f"  {i+1}. {ex['file_name']}")
    
    # Verify few-shot example images exist
    for ex in few_shot_examples:
        image_path = os.path.join(images_dir, ex['file_name'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Few-shot example image not found: {image_path}")
    
    # Initialize OCR processor
    ocr_processor = SchmuckOCRProcessor()
    
    # Create few-shot messages with images (resized in the function)
    print("\nCreating few-shot messages with images...")
    few_shot_messages = ocr_processor.create_few_shot_messages(few_shot_examples, images_dir)
    print(f"Created few-shot conversation with {len(few_shot_messages)} messages")
    
    # Process test images
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
            
            print(f"  Processed successfully. CER: {cer_score:.4f}")
            
            # Clean up after each successful processing
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
            
            # Clean memory even on error
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
    print(f"Model: Qwen2.5-VL-7B-Instruct")
    print(f"Dataset: Jewelry Catalog")
    print(f"Method: Few-Shot Learning ({len(few_shot_examples)} example(s))")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if avg_cer is not None:
        print(f"Average CER on test set: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"{'='*60}\n")



if __name__ == "__main__":
    main()
