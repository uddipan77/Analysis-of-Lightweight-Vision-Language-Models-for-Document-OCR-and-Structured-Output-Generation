from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import re
import jiwer


def json_to_string(json_obj):
    """Convert JSON object to a consistent string representation."""
    return json.dumps(json_obj, ensure_ascii=False, sort_keys=False, separators=(',', ':'))


class InventoryOCRProcessor:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """Initialize the OCR processor with Qwen2.5-VL model."""
        print("Loading Qwen2.5-VL model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("Model loaded successfully!")
    
    def create_few_shot_messages(self, few_shot_examples: List[Dict], images_dir: str) -> List[Dict]:
        """Create few-shot messages with both images and expected outputs."""
        messages = []
        
        instruction = """Extract the inventory information from this German historical document image as a JSON object with exactly these fields:
- Überschrift (heading/title)
- Inventarnummer (inventory number)
- Maße (measurements object with keys L, B, D for Länge/length, Breite/breadth, Dicke/depth)
- Objektbezeichnung (object designation/material)
- Fundort (find location)
- Fundzeit (find date/period)
- Beschreibungstext (description text)

Use empty string "" for any missing fields. The Maße field must always be an object with L, B, D keys even if empty.
Return only the JSON object."""
        
        # Add few-shot examples
        for example in few_shot_examples:
            # User message with image
            messages.append({
                "role": "user", 
                "content": [
                    {
                        "type": "image",
                        "image": os.path.join(images_dir, example["image_name"])
                    },
                    {
                        "type": "text", 
                        "text": instruction
                    }
                ]
            })
            
            # Assistant response with expected JSON
            expected_json = {k: v for k, v in example.items() if k != "image_name"}
            messages.append({
                "role": "assistant",
                "content": json.dumps(expected_json, ensure_ascii=False, separators=(',', ':'))
            })
        
        return messages
    
    def process_image_with_few_shot(self, image_path: str, few_shot_messages: List[Dict]) -> str:
        """Process a single image with few-shot examples."""
        # Create the full conversation including few-shot examples and the test image
        messages = few_shot_messages.copy()
        
        # Add the test image query
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {
                    "type": "text",
                    "text": """Extract the inventory information from this German historical document image as a JSON object with exactly these fields:
- Überschrift (heading/title)
- Inventarnummer (inventory number)
- Maße (measurements object with keys L, B, D for Länge/length, Breite/breadth, Dicke/depth)
- Objektbezeichnung (object designation/material)
- Fundort (find location)
- Fundzeit (find date/period)
- Beschreibungstext (description text)

Use empty string "" for any missing fields. The Maße field must always be an object with L, B, D keys even if empty.
Return only the JSON object."""
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
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = inputs.to(device)
        
        # Generate OCR output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]


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
    
    # If still no valid JSON, return empty structure
    return {
        "Überschrift": "",
        "Inventarnummer": "",
        "Maße": {"L": "", "B": "", "D": ""},
        "Objektbezeichnung": "",
        "Fundort": "",
        "Fundzeit": "",
        "Beschreibungstext": ""
    }


def save_predictions_with_timestamp(predictions, test_data, all_cer_scores, base_dir="/home/vault/iwi5/iwi5298h/models_image_text/qwen/few_zero_inven"):
    """Save predictions and CER results with timestamp directory."""
    
    # Create timestamp directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"run_fewshot_{timestamp}")
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
        # Macro CER: average of per-sample CER scores
        macro_cer = sum(all_cer_scores) / len(all_cer_scores)
        min_cer = min(all_cer_scores)
        max_cer = max(all_cer_scores)
        
        # Micro CER: total errors / total characters (weighted by sample length)
        total_chars = 0
        total_errors = 0
        
        for i, test_item in enumerate(test_data[:len(predictions)]):
            # Exclude image_name from ground truth for CER calculation
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)
            char_count = len(gt_json_string)
            cer_score = all_cer_scores[i]
            
            total_chars += char_count
            total_errors += int(cer_score * char_count)
        
        micro_cer = total_errors / total_chars if total_chars > 0 else 0
        
        # Count perfect matches
        perfect_matches = sum(1 for cer in all_cer_scores if cer == 0.0)
        
        # Print results
        print("\n" + "="*60)
        print("FEW-SHOT CER EVALUATION RESULTS")
        print("="*60)
        print(f"\nCER Statistics across {len(all_cer_scores)} images:")
        print("-" * 50)
        print(f"Macro CER (avg per-sample): {macro_cer:.4f} ({macro_cer*100:.2f}%)")
        print(f"Micro CER (char-weighted):  {micro_cer:.4f} ({micro_cer*100:.2f}%)")
        print(f"Minimum CER: {min_cer:.4f} ({min_cer*100:.2f}%)")
        print(f"Maximum CER: {max_cer:.4f} ({max_cer*100:.2f}%)")
        print(f"\nTotal characters: {total_chars}")
        print(f"Total errors: {total_errors}")
        print(f"Perfect matches: {perfect_matches}/{len(all_cer_scores)} ({perfect_matches/len(all_cer_scores)*100:.1f}%)")
        
        # Save summary
        summary_file = os.path.join(output_dir, "evaluation_summary_fewshot.json")
        summary_data = {
            "timestamp": timestamp,
            "method": "few-shot",
            "total_images": len(predictions),
            "macro_cer": float(macro_cer),
            "micro_cer": float(micro_cer),
            "minimum_cer": float(min_cer),
            "maximum_cer": float(max_cer),
            "perfect_matches": int(perfect_matches),
            "total_characters": int(total_chars),
            "total_errors": int(total_errors),
            "few_shot_examples": ["inventarbuch-265.jpg", "inventarbuch-189.jpg", "inventarbuch-026.jpg"]
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {summary_file}")
        
        return output_dir, macro_cer, micro_cer
    
    return output_dir, None, None


def main():
    """Main function to process inventory images and calculate CER."""
    
    # Paths
    test_jsonl_path = "/home/woody/iwi5/iwi5298h/updated_json_inven/test.jsonl"
    train_jsonl_path = "/home/woody/iwi5/iwi5298h/updated_json_inven/train.jsonl"
    images_dir = "/home/woody/iwi5/iwi5298h/inventory_images"
    
    # Load test data
    print("Loading test data...")
    test_data = load_jsonl(test_jsonl_path)
    print(f"Loaded {len(test_data)} test samples")
    
    # Load few-shot examples from train data
    print("Loading few-shot examples...")
    train_data = load_jsonl(train_jsonl_path)
    
    # Use the specified 3 examples for few-shot learning
    few_shot_examples = []
    target_images = ["inventarbuch-265.jpg", "inventarbuch-189.jpg", "inventarbuch-026.jpg"]
    
    for target_image in target_images:
        for example in train_data:
            if example["image_name"] == target_image:
                few_shot_examples.append(example)
                break
    
    if len(few_shot_examples) < 3:
        print(f"Warning: Only found {len(few_shot_examples)} of the 3 specified few-shot examples")
        # Fallback to first 3 examples if specific ones not found
        few_shot_examples = train_data[:3]
    
    print(f"Using {len(few_shot_examples)} few-shot examples")
    
    # Initialize OCR processor
    ocr_processor = InventoryOCRProcessor()
    
    # Create few-shot messages with images
    print("Creating few-shot messages with images...")
    few_shot_messages = ocr_processor.create_few_shot_messages(few_shot_examples, images_dir)
    print(f"Created few-shot conversation with {len(few_shot_messages)} messages")
    
    # Process test images
    predictions = []
    all_cer_scores = []
    
    print(f"\nProcessing {len(test_data)} test images...")
    
    for i, test_item in enumerate(test_data):
        print(f"Processing image {i+1}/{len(test_data)}: {test_item['image_name']}")
        
        # Construct image path
        image_path = os.path.join(images_dir, test_item['image_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        try:
            # Process image with few-shot examples
            response = ocr_processor.process_image_with_few_shot(image_path, few_shot_messages)
            predicted_json = extract_json_from_response(response)
            
            # Convert both ground truth and prediction to JSON strings (excluding image_name)
            gt_json = {k: v for k, v in test_item.items() if k != "image_name"}
            gt_json_string = json_to_string(gt_json)
            pred_json_string = json_to_string(predicted_json)
            
            # Calculate CER using jiwer
            cer_score = jiwer.cer(gt_json_string, pred_json_string)
            
            # Store prediction with metadata (image_name is included but not predicted)
            prediction_entry = {
                "image_name": test_item['image_name'],
                "predicted_json": predicted_json,
                "ground_truth": gt_json,
                "raw_response": response,
                "cer_score": cer_score
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(cer_score)
            
            print(f"  Processed successfully. CER: {cer_score:.4f}")
            
        except Exception as e:
            print(f"Error processing {test_item['image_name']}: {str(e)}")
            # Add empty prediction for failed cases
            prediction_entry = {
                "image_name": test_item['image_name'],
                "predicted_json": {
                    "Überschrift": "",
                    "Inventarnummer": "",
                    "Maße": {"L": "", "B": "", "D": ""},
                    "Objektbezeichnung": "",
                    "Fundort": "",
                    "Fundzeit": "",
                    "Beschreibungstext": ""
                },
                "ground_truth": {k: v for k, v in test_item.items() if k != "image_name"},
                "raw_response": f"Error: {str(e)}",
                "cer_score": 1.0
            }
            predictions.append(prediction_entry)
            all_cer_scores.append(1.0)
            continue
    
    # Save predictions with timestamp
    output_dir, macro_cer, micro_cer = save_predictions_with_timestamp(predictions, test_data, all_cer_scores)
    
    print(f"\n{'='*60}")
    print(f"FEW-SHOT PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"Method: Few-Shot Learning (3 training examples)")
    print(f"Results saved to: {output_dir}")
    print(f"Total images processed: {len(predictions)}")
    if macro_cer is not None:
        print(f"Macro CER (avg per-sample): {macro_cer:.4f} ({macro_cer*100:.2f}%)")
        print(f"Micro CER (char-weighted):  {micro_cer:.4f} ({micro_cer*100:.2f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
