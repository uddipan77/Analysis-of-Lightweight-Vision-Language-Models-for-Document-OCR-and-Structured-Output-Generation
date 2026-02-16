#this is an update for schmuck dataset - uses file_name instead of image_name
import json
import os
import torch
import argparse
import editdistance
import unicodedata
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    GenerationConfig,
    EarlyStoppingCallback
)
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import glob

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text

def canonical_json_string(obj):
    # Remove sort_keys to maintain original order
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))

def parse_json_string(text):
    """Parse the predicted text back to JSON structure"""
    try:
        # Try to parse as direct JSON first
        return json.loads(text)
    except:
        # If that fails, return the text as-is for comparison
        return text

def calculate_cer(predictions, targets, json_mode=False):
    total_chars = 0
    total_errors = 0
    for pred, target in zip(predictions, targets):
        if json_mode:
            try:
                pred_json = parse_json_string(pred)
                target_json = parse_json_string(target)
                pred_str = canonical_json_string(pred_json) if isinstance(pred_json, dict) else str(pred_json)
                target_str = canonical_json_string(target_json) if isinstance(target_json, dict) else str(target_json)
            except:
                pred_str, target_str = pred, target
        else:
            pred_str, target_str = pred, target
        total_chars += len(target_str)
        total_errors += editdistance.eval(pred_str, target_str)
    return (total_errors / total_chars) * 100 if total_chars > 0 else 0

def compute_metrics_for_training(eval_pred):
    """
    Compute CER metrics during training for model selection
    """
    predictions, labels = eval_pred
    
    # Decode predictions
    predictions = predictions[0] if isinstance(predictions, tuple) else predictions
    if len(predictions.shape) == 3:
        predictions = torch.argmax(torch.tensor(predictions), dim=-1)
    
    # Decode both predictions and labels
    decoded_preds = []
    decoded_labels = []
    
    for pred_ids, label_ids in zip(predictions, labels):
        # Remove padding and special tokens
        pred_ids = pred_ids[pred_ids != processor.tokenizer.pad_token_id]
        label_ids = label_ids[label_ids != -100]
        label_ids = label_ids[label_ids != processor.tokenizer.pad_token_id]
        
        # Skip the start token for predictions (first token is usually <s_cord-v2>)
        if len(pred_ids) > 0:
            pred_text = processor.tokenizer.decode(pred_ids, skip_special_tokens=True)
            # Remove the start token text if it exists
            pred_text = pred_text.replace("<s_cord-v2>", "").strip()
        else:
            pred_text = ""
            
        if len(label_ids) > 0:
            label_text = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
            label_text = label_text.replace("<s_cord-v2>", "").strip()
        else:
            label_text = ""
            
        decoded_preds.append(pred_text)
        decoded_labels.append(label_text)
    
    # Calculate CER
    cer_string = calculate_cer(decoded_preds, decoded_labels, json_mode=False)
    cer_structured = calculate_cer(decoded_preds, decoded_labels, json_mode=True)
    
    return {
        "cer_string": cer_string,
        "cer_structured": cer_structured,
    }

def download_and_cache_model(model_name, local_cache_dir):
    """
    Download the model and processor to local directory if not already present
    """
    print(f"Checking for model at: {local_cache_dir}")
    
    # Create the directory if it doesn't exist
    os.makedirs(local_cache_dir, exist_ok=True)
    
    # Check if model files already exist
    config_path = os.path.join(local_cache_dir, "config.json")
    
    if os.path.exists(config_path):
        print(f"Model already exists at {local_cache_dir}, loading from local cache...")
        processor = DonutProcessor.from_pretrained(local_cache_dir)
        model = VisionEncoderDecoderModel.from_pretrained(local_cache_dir)
    else:
        print(f"Downloading model {model_name} to {local_cache_dir}...")
        # Download from HuggingFace Hub and save locally
        processor = DonutProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # Save to local directory
        print(f"Saving model to {local_cache_dir}...")
        processor.save_pretrained(local_cache_dir)
        model.save_pretrained(local_cache_dir)
        print("Model downloaded and cached successfully!")
    
    return processor, model

def find_image_path(file_name: str, images_dir: str) -> str:
    """Find image path using flexible matching for schmuck dataset."""
    # First try exact match
    exact_path = os.path.join(images_dir, file_name)
    if os.path.exists(exact_path):
        return exact_path
    
    # Try without extension and search with different extensions
    name_without_ext = os.path.splitext(file_name)[0]
    for ext in ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']:
        search_path = os.path.join(images_dir, f"{name_without_ext}{ext}")
        if os.path.exists(search_path):
            return search_path
    
    # Use glob for case-insensitive search
    search_pattern = os.path.join(images_dir, f"*{name_without_ext}*")
    matching_files = glob.glob(search_pattern)
    if matching_files:
        return matching_files[0]
    
    # Return original path even if it doesn't exist (for error handling)
    return exact_path

class SchmuckOCRDataset(Dataset):
    def __init__(self, json_path, image_dir, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        self.image_dir = Path(image_dir)
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f] #converts the JSON string into a Python dictionary, so we have list of dict
        print(f"Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # UPDATED: Use file_name instead of image_name and flexible path matching
        image_path_str = find_image_path(item['file_name'], str(self.image_dir))
        image_path = Path(image_path_str)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        target_text = self.format_target_text(item) #we get back json string without file_name key
        
        #Image Encoding,Convert the PIL image to tensor format that the vision encoder can process. 
        #The squeeze(0) removes the batch dimension since we're processing one image at a time.
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        
        #Decoder Start Token, : Create the special start token that tells the decoder this is a CORD-v2 task. This is crucial for the model to understand the expected output format.
        decoder_input_ids = self.processor.tokenizer(
            "<s_cord-v2>", add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Target Text Tokenization
        #add_special_tokens=False: No extra tokens (you're handling them manually)
        #padding=False: No padding yet (done later in data collator)
        #truncation=True: Cut off if too long
        #max_length - len(decoder_input_ids) - 1: Reserve space for start token and EOS token
        target_tokenized = self.processor.tokenizer(
            target_text, add_special_tokens=False, padding=False, truncation=True,
            max_length=self.max_length - len(decoder_input_ids) - 1, return_tensors="pt"
        )
        
        #Add EOS Token,  Append the end-of-sequence token to signal when the model should stop generating text.
        eos_token_id = self.processor.tokenizer.eos_token_id
        target_ids = target_tokenized.input_ids.squeeze(0)
        target_ids = torch.cat([target_ids, torch.tensor([eos_token_id], dtype=torch.long)])
        
        # Combine Start Token + Target, Create the complete sequence: <s_cord-v2> + JSON_content + <EOS>
        full_target_ids = torch.cat([decoder_input_ids, target_ids])
        
        # Length Validation and Truncation, Ensure the sequence doesn't exceed maximum length. 
        #If truncated, replace the last token with EOS to maintain proper sequence ending.
        if len(full_target_ids) > self.max_length:
            full_target_ids = full_target_ids[:self.max_length]
            full_target_ids[-1] = eos_token_id
        
        #Vocabulary Bounds Checking, Safety check to prevent out-of-vocabulary token IDs that could cause runtime 
        #errors during training.
        vocab_size = len(self.processor.tokenizer)
        if torch.any(full_target_ids >= vocab_size):
            full_target_ids = torch.clamp(full_target_ids, 0, vocab_size - 1).long()
        full_target_ids = full_target_ids.long()
        
        #Create Training Labels, The -100 values for the start token positions tell PyTorch to ignore these positions 
        #during loss computation - the model shouldn't be penalized for the start token since it's given as input.
        labels = full_target_ids.clone()
        labels[:len(decoder_input_ids)] = -100
        
        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": full_target_ids,
            "labels": labels,
        }

    def format_target_text(self, item):
        """Convert the original JSON item to a JSON string, excluding file_name"""
        # UPDATED: Create a copy and remove file_name (instead of image_name)
        target_dict = {k: v for k, v in item.items() if k != 'file_name'}
        
        # Convert to JSON string without sorting keys
        return json.dumps(target_dict, ensure_ascii=False, separators=(',', ':'))

class ImprovedDataCollator:
    def __init__(self, processor, max_length=512):
        self.processor = processor
        self.max_length = max_length
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.vocab_size = len(processor.tokenizer)

    def __call__(self, batch):
        #batch is a list containing multiple samples from your dataset.
        """
        batch = [
        {"pixel_values": tensor1, "decoder_input_ids": tensor1, "labels": tensor1},
        {"pixel_values": tensor2, "decoder_input_ids": tensor2, "labels": tensor2},
        {"pixel_values": tensor3, "decoder_input_ids": tensor3, "labels": tensor3},
        # ... more samples
        ]
        #"""
        #The key problem is that sequences in your batch have different lengths:
        #You cannot directly stack tensors of different lengths - PyTorch requires all tensors in a batch to have the same shape.
        #Images are already processed to the same size in your dataset, so they can be directly stacked.
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        #Find the longest sequence in the batch, but cap it at your maximum allowed length.
        max_len = min(max(len(item["decoder_input_ids"]) for item in batch), self.max_length)
        decoder_input_ids = []
        labels = []
        
        # Process Each Sample Individually,  If a sequence is too long, truncate it and ensure it ends with EOS token.
        for item in batch:
            input_ids = item["decoder_input_ids"]
            item_labels = item["labels"]
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                item_labels = item_labels[:max_len]
                if input_ids[-1] != self.processor.tokenizer.eos_token_id:
                    input_ids[-1] = self.processor.tokenizer.eos_token_id
            if torch.any(input_ids >= self.vocab_size) or torch.any(input_ids < 0):
                input_ids = torch.clamp(input_ids, 0, self.vocab_size - 1).long()
            input_ids = input_ids.long()
            item_labels = item_labels.long()
            seq_len = len(input_ids)
            pad_len = max_len - seq_len
            if pad_len > 0:
                padded_input = torch.cat([
                    input_ids, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                ])
                padded_labels = torch.cat([
                    item_labels, torch.full((pad_len,), -100, dtype=torch.long)
                ])
            else:
                padded_input = input_ids
                padded_labels = item_labels
            decoder_input_ids.append(padded_input)
            labels.append(padded_labels)
        decoder_input_ids_tensor = torch.stack(decoder_input_ids).long()
        labels_tensor = torch.stack(labels).long()
        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids_tensor,
            "labels": labels_tensor,
        }

def generate_predictions(model, dataset, processor, device, max_length=512, batch_size=1):
    """
    For each image, create a record containing:
    Image name (e.g., "SCH_3187.jpg")
    Model's prediction (the JSON it generated)
    Ground truth (the correct answer from your dataset)
    Sample index (which image this was in the dataset)
    Original data (the raw data from your dataset)
    """
    model.eval()
    predictions = []
    data_collator = ImprovedDataCollator(processor, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=data_collator)
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Generating predictions")):
            pixel_values = batch_data["pixel_values"].to(device)
            prompt_ids = processor.tokenizer(
                "<s_cord-v2>", add_special_tokens=False, return_tensors="pt"
            ).input_ids.to(device).long()
            generated_ids = model.generate(
                pixel_values=pixel_values,
                decoder_input_ids=prompt_ids.repeat(pixel_values.size(0), 1),
                max_length=max_length,
                num_beams=3,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            for i in range(len(generated_ids)):
                actual_idx = batch_idx * batch_size + i
                predicted_text = processor.tokenizer.decode(
                    generated_ids[i][len(prompt_ids[0]):],
                    skip_special_tokens=True
                ).strip()
                ground_truth = dataset.format_target_text(dataset.data[actual_idx])
                
                # UPDATED: Use file_name instead of image_name
                prediction_record = {
                    "file_name": dataset.data[actual_idx]["file_name"],
                    "prediction": predicted_text,
                    "ground_truth": ground_truth,
                    "sample_index": actual_idx
                }
                #original data is retrieved from the dataset dataset.data contains: The original dictionaries from your JSONL file:, not what getitem returns
                original_data = dataset.data[actual_idx].copy()
                prediction_record["original_data"] = original_data
                predictions.append(prediction_record)
    return predictions

def save_predictions_jsonl(predictions, output_path):
    print(f"Saving {len(predictions)} predictions to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"Predictions saved successfully!")

def cer_on_predictions(pred_jsonl):
    with open(pred_jsonl, 'r', encoding='utf-8') as f:
        preds = [json.loads(line) for line in f]
    preds_text = [normalize_text(d["prediction"]) for d in preds]
    gt_text = [normalize_text(d["ground_truth"]) for d in preds]
    cer_str = calculate_cer(preds_text, gt_text, json_mode=False)
    cer_struct = calculate_cer(preds_text, gt_text, json_mode=True)
    return cer_str, cer_struct

def main():
    parser = argparse.ArgumentParser(description="DONUT OCR training for Schmuck dataset")
    # UPDATED: Set default paths for schmuck dataset
    parser.add_argument("--data_dir", default="/home/woody/iwi5/iwi5298h/json_schmuck", help="Directory with train/val/test JSONL files")
    parser.add_argument("--image_dir", default="/home/woody/iwi5/iwi5298h/schmuck_images", help="Directory containing images")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--predict_on", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--prediction_batch_size", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Patience epochs for early stopping")
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0, help="Delta threshold for early stopping")
    args = parser.parse_args()

    # UPDATED: Timestamped output dir for schmuck dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTPUT_DIR = f"/home/vault/iwi5/iwi5298h/models_image_text/donut/schmuck/schmuck_data_{timestamp}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tb_log_dir = os.path.join(OUTPUT_DIR, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"All output will be saved in: {OUTPUT_DIR}")

    # Model/processor loading - download to local directory if needed
    model_name = "naver-clova-ix/donut-base"
    local_model_dir = "/home/vault/iwi5/iwi5298h/models/donut_base"
    
    # Download and cache the model locally
    global processor  # Make processor global for compute_metrics function
    processor, model = download_and_cache_model(model_name, local_model_dir)
    
    # Set config
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_cord-v2>")
    model.config.max_length = args.max_length
    
    generation_config = GenerationConfig(
        max_length=args.max_length,
        num_beams=3,
        early_stopping=True,
        no_repeat_ngram_size=3,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<s_cord-v2>")
    )
    model.generation_config = generation_config
    model = model.to(device)

    # UPDATED: Datasets using SchmuckOCRDataset
    dataset_max_length = min(args.max_length, 512) #If the tokenized sequence exceeds 512 tokens, it gets truncated:
    train_dataset = SchmuckOCRDataset(
        os.path.join(args.data_dir, 'train.jsonl'),
        args.image_dir, processor, dataset_max_length)
    val_dataset = SchmuckOCRDataset(
        os.path.join(args.data_dir, 'val.jsonl'),
        args.image_dir, processor, dataset_max_length)
    test_dataset = None
    test_path = os.path.join(args.data_dir, 'test.jsonl')
    if os.path.exists(test_path) and args.predict_on in ["test", "all"]:
        test_dataset = SchmuckOCRDataset(
            test_path, args.image_dir, processor, dataset_max_length)

    # TensorBoard logger
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Training args - UPDATED to use CER for model selection
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=dataset_max_length,
        generation_num_beams=3,
        logging_steps=50,
        remove_unused_columns=False,
        fp16=False,
        dataloader_num_workers=0,
        report_to="tensorboard",
        logging_dir=tb_log_dir,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_cer_string",  # Use CER instead of eval_loss
        greater_is_better=False,  # Lower CER is better
    )

    data_collator = ImprovedDataCollator(processor, dataset_max_length)

    # Trainer - includes compute_metrics for CER-based model selection
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_for_training,  # Enable CER calculation
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            ),
        ],
    )

    print("Starting training...")
    print("Model selection based on: eval_cer_string (lower is better)")
    trainer.train()

    print("Saving final model and processor...")
    final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
    model.save_pretrained(final_model_dir) #Save the model to disk (final_model_dir)
    processor.save_pretrained(final_model_dir)
    generation_config.save_pretrained(final_model_dir)
    print("Model and processor saved.")

    # Generate and save predictions + CER summary
    CER_summary = []
    if args.predict_on in ["test", "all"]:
        print("Generating predictions on test set...")
        test_predictions = generate_predictions(
            model, test_dataset, processor, device,
            dataset_max_length, args.prediction_batch_size)
        test_predictions_path = os.path.join(OUTPUT_DIR, "test_predictions.jsonl")
        save_predictions_jsonl(test_predictions, test_predictions_path)
        test_cer_str, test_cer_struct = cer_on_predictions(test_predictions_path)
        CER_summary.append(f"Test CER (string): {test_cer_str:.4f}")
        CER_summary.append(f"Test CER (structured): {test_cer_struct:.4f}")

    if args.predict_on in ["val", "all"]:
        print("Generating predictions on validation set...")
        val_predictions = generate_predictions(
            model, val_dataset, processor, device,
            dataset_max_length, args.prediction_batch_size)
        val_predictions_path = os.path.join(OUTPUT_DIR, "val_predictions.jsonl")
        save_predictions_jsonl(val_predictions, val_predictions_path)
        val_cer_str, val_cer_struct = cer_on_predictions(val_predictions_path)
        CER_summary.append(f"Validation CER (string): {val_cer_str:.4f}")
        CER_summary.append(f"Validation CER (structured): {val_cer_struct:.4f}")

    if args.predict_on in ["train", "all"]:
        print("Generating predictions on training set...")
        train_predictions = generate_predictions(
            model, train_dataset, processor, device,
            dataset_max_length, args.prediction_batch_size)
        train_predictions_path = os.path.join(OUTPUT_DIR, "train_predictions.jsonl")
        save_predictions_jsonl(train_predictions, train_predictions_path)
        train_cer_str, train_cer_struct = cer_on_predictions(train_predictions_path)
        CER_summary.append(f"Train CER (string): {train_cer_str:.4f}")
        CER_summary.append(f"Train CER (structured): {train_cer_struct:.4f}")

    # Save CER summary to file
    scores_path = os.path.join(OUTPUT_DIR, "final_CER_scores.txt")
    with open(scores_path, "w") as f:
        f.write("DONUT MODEL TRAINING RESULTS - SCHMUCK DATASET\n")
        f.write("="*60 + "\n")
        f.write("Model Selection: Based on eval_cer_string (lowest CER)\n")
        f.write("Dataset: Schmuck (jewelry) historical documents\n")
        f.write("Image field: file_name (instead of image_name)\n")
        f.write("Key ordering: Original order preserved (no sorting)\n")
        f.write("="*60 + "\n")
        f.write("\n".join(CER_summary) + "\n")
    print(f"CER scores written to {scores_path}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"\nTo view TensorBoard loss curves, run:\n  tensorboard --logdir {tb_log_dir}")

if __name__ == '__main__':
    main()
