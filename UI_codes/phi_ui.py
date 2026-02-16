#!/usr/bin/env python3
"""
Staircase OCR service with:

- FastAPI backend (JSON + Markdown inference endpoints)
- Gradio UI mounted on FastAPI
- Dropdown to choose the OCR model (currently: finetuned Phi-3.5-Vision)
- Latency measurement and simple in-memory cache

Run:
    python phi_ui.py

Or:
    uvicorn phi_ui:app --host 0.0.0.0 --port 8000

UI URL:
    http://<server>:8000/ui
"""

import io
import json
import time
import hashlib
import unicodedata
from typing import List, Dict, Any

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel

import gradio as gr


# ========= PATH CONFIG – EDIT THESE ON NEW MACHINES =========
BASE_MODEL_PATH = "/home/vault/iwi5/iwi5298h/models/phi_3_5_vision"
FINETUNED_MODEL_PATH = "/home/vault/iwi5/iwi5298h/models_image_text/phi/multistage/stair/best_model"
# ============================================================


# ======================================================================
# CONFIG
# ======================================================================

INSTRUCTION = """You are an OCR model for historical staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this dataset (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
- Never drop a key that appears in the form’s JSON structure. If a field is empty on the form, still include it with an empty string "" (or false for an unchecked box).
- Use booleans for checkbox options: true if the box is checked, false if it is empty.
- Use strings for numbers and free-text fields (measurements, dates, names, notes).
- Do NOT invent new fields.
"""

# NOTE: finetuned_model_path points to your QLoRA adapter dir (best_model/)
AVAILABLE_MODELS: Dict[str, Dict[str, Any]] = {
    "phi": {
        "label": "Phi-3.5-Vision (multistage, finetuned with QLoRA)",
        "base_model_path": BASE_MODEL_PATH,
        "finetuned_model_path": FINETUNED_MODEL_PATH,
        "num_crops": 16,
        "type": "phi",
    },
    # In future you can add more models here, e.g. another key "qwen2_5_vl"
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# ======================================================================
# Helpers (JSON extraction + Markdown rendering)
# ======================================================================

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def extract_json_from_response(response: str) -> str:
    response = response.strip()

    if "{" in response and "}" in response:
        start = response.find("{")
        end = response.rfind("}") + 1
        json_str = response[start:end]

        try:
            brace_count = 0
            first_end = -1
            for i, char in enumerate(json_str):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        first_end = i + 1
                        break

            if first_end > 0:
                first_json = json_str[:first_end]
                parsed = json.loads(first_json)
                return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            pass

        try:
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
        except Exception:
            return json_str

    return response


def escape_md(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    return s.replace("|", "\\|" )


def format_scalar(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "✓" if value else ""
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def format_list(lst):
    if not lst:
        return ""
    if all(not isinstance(x, (dict, list)) for x in lst):
        return ", ".join(format_scalar(x) for x in lst)
    return json.dumps(lst, ensure_ascii=False)


def dict_of_dicts_columns(d):
    if not isinstance(d, dict) or not d:
        return None

    first_inner = None
    for v in d.values():
        if not isinstance(v, dict):
            return None
        if first_inner is None:
            first_inner = v

    cols = list(first_inner.keys())
    for v in d.values():
        if set(v.keys()) != set(cols):
            return None
    return cols


def render_dict_as_table(name, d, depth=3):
    cols = dict_of_dicts_columns(d)
    if not cols:
        return None

    lines = []
    heading = "#" * depth + f" {name}\n\n"
    lines.append(heading)

    header_row = "|  | " + " | ".join(escape_md(c) for c in cols) + " |\n"
    sep_row = "| --- | " + " | ".join("---" for _ in cols) + " |\n"
    lines.append(header_row)
    lines.append(sep_row)

    for row_key, inner in d.items():
        row_vals = []
        for c in cols:
            v = inner.get(c)
            if isinstance(v, list):
                cell = format_list(v)
            else:
                cell = format_scalar(v)
            row_vals.append(escape_md(cell))
        lines.append(
            f"| {escape_md(row_key)} | " + " | ".join(row_vals) + " |\n"
        )

    lines.append("\n")
    return "".join(lines)


def render_dict_generic(d, depth=3, section_name=None):
    lines = []

    if section_name is not None:
        lines.append("#" * depth + f" {section_name}\n\n")

    simple_items = []
    complex_items = []

    for k, v in d.items():
        if isinstance(v, (dict, list)):
            complex_items.append((k, v))
        else:
            simple_items.append((k, v))

    if simple_items:
        lines.append("| Field | Value |\n")
        lines.append("| --- | --- |\n")
        for k, v in simple_items:
            lines.append(
                f"| {escape_md(k)} | {escape_md(format_scalar(v))} |\n"
            )
        lines.append("\n")

    for k, v in complex_items:
        if isinstance(v, dict):
            table_md = render_dict_as_table(k, v, depth + 1)
            if table_md is not None:
                lines.append(table_md)
            else:
                lines.append(render_dict_generic(v, depth + 1, k))

        elif isinstance(v, list):
            if all(not isinstance(x, (dict, list)) for x in v):
                lines.append("#" * (depth + 1) + f" {k}\n\n")
                lines.append("| Value |\n")
                lines.append("| --- |\n")
                for x in v:
                    lines.append(f"| {escape_md(format_scalar(x))} |\n")
                lines.append("\n")
            elif v and all(isinstance(x, dict) for x in v):
                cols = list(v[0].keys())
                if all(set(x.keys()) == set(cols) for x in v):
                    lines.append("#" * (depth + 1) + f" {k}\n\n")
                    header_row = "|  | " + " | ".join(
                        escape_md(c) for c in cols
                    ) + " |\n"
                    sep_row = "| --- | " + " | ".join(
                        "---" for _ in cols
                    ) + " |\n"
                    lines.append(header_row)
                    lines.append(sep_row)
                    for idx, item in enumerate(v, start=1):
                        row_vals = []
                        for c in cols:
                            val = item.get(c)
                            if isinstance(val, list):
                                cell = format_list(val)
                            else:
                                cell = format_scalar(val)
                            row_vals.append(escape_md(cell))
                        lines.append(
                            f"| #{idx} | " + " | ".join(row_vals) + " |\n"
                        )
                    lines.append("\n")
                else:
                    for idx, item in enumerate(v, start=1):
                        lines.append(
                            render_dict_generic(
                                item, depth + 1, f"{k} #{idx}"
                            )
                        )
            else:
                lines.append("#" * (depth + 1) + f" {k}\n\n")
                lines.append("```json\n")
                lines.append(json.dumps(v, ensure_ascii=False, indent=2))
                lines.append("\n```\n\n")

    return "".join(lines)


def prediction_json_to_markdown(predicted_json_str: str) -> str:
    try:
        obj = json.loads(predicted_json_str)
    except Exception:
        return "```json\n" + predicted_json_str + "\n```"

    if not isinstance(obj, dict):
        return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"

    md = "## Predicted Staircase Form\n\n"
    md += render_dict_generic(obj, depth=3)
    return md


def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


# ======================================================================
# Model loading (lazy)
# ======================================================================

MODEL_BUNDLES: Dict[str, Dict[str, Any]] = {}


def get_model_bundle(model_id: str) -> Dict[str, Any]:
    if model_id not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model_id '{model_id}'. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    if model_id in MODEL_BUNDLES:
        return MODEL_BUNDLES[model_id]

    cfg = AVAILABLE_MODELS[model_id]
    model_type = cfg["type"]
    base_model_path = cfg["base_model_path"]
    finetuned_path = cfg["finetuned_model_path"]

    print(f"⏳ Loading model '{model_id}' ({cfg['label']})")
    print(f"   base_model_path     = {base_model_path}")
    print(f"   finetuned_model_path= {finetuned_path}")

    if model_type == "phi":
        # Processor (tokenizer + image processor) from base
        processor_kwargs = {"trust_remote_code": True}
        if cfg.get("num_crops") is not None:
            processor_kwargs["num_crops"] = cfg["num_crops"]

        processor = AutoProcessor.from_pretrained(
            base_model_path,
            **processor_kwargs,
        )

        # Base Phi model in 4-bit
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
        )

        # Attach QLoRA adapter (your best_model folder)
        model = PeftModel.from_pretrained(
            base_model,
            finetuned_path,
        )

        model.eval()
        device = next(model.parameters()).device

        bundle = {
            "processor": processor,
            "model": model,
            "device": device,
            "type": "phi",
        }

    else:
        raise ValueError(f"Unsupported model type '{model_type}'")

    print(f"✅ Model '{model_id}' loaded on device: {bundle['device']}")
    MODEL_BUNDLES[model_id] = bundle
    return bundle


# ======================================================================
# Simple in-memory cache
# ======================================================================

CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_MAX_SIZE = 512


def hash_image_bytes(model_id: str, image_bytes: bytes) -> str:
    return f"{model_id}:" + hashlib.sha256(image_bytes).hexdigest()


def cache_get(key: str):
    return CACHE.get(key)


def cache_set(key: str, value: Dict[str, Any]):
    if len(CACHE) >= CACHE_MAX_SIZE:
        # pop an arbitrary item (FIFO-ish)
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = value


# ======================================================================
# Core model inference (Phi)
# ======================================================================

def run_phi_on_images(images: List[Image.Image], bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Phi inference (one image at a time).

    IMPORTANT: processor receives a single prompt string for each call,
    so processing_phi3_v.py sees `texts` as str, not list[str],
    avoiding the `TypeError: expected string or bytes-like object`.
    """
    processor = bundle["processor"]
    model = bundle["model"]
    device = bundle["device"]

    results: List[Dict[str, str]] = []

    for img in images:
        messages = [
            {
                "role": "user",
                "content": f"<|image_1|>\n{INSTRUCTION}",
            }
        ]

        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = processor(
            text=prompt,
            images=[img],
            return_tensors="pt",
            padding=True,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Strip prompt part
        input_ids = inputs["input_ids"][0]
        out_ids = generate_ids[0]
        trimmed = out_ids[len(input_ids):].unsqueeze(0)

        raw_output = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        raw_output = normalize_unicode(raw_output)
        predicted_json = extract_json_from_response(raw_output)
        markdown = prediction_json_to_markdown(predicted_json)

        results.append(
            {
                "markdown": markdown,
                "json": predicted_json,
            }
        )

    return results


def run_model_on_images(images: List[Image.Image], model_id: str) -> List[Dict[str, str]]:
    bundle = get_model_bundle(model_id)
    model_type = bundle["type"]

    if model_type == "phi":
        return run_phi_on_images(images, bundle)
    else:
        raise ValueError(f"Unsupported model type '{model_type}'")


def run_single_ocr_pil(image: Image.Image, model_id: str) -> Dict[str, Any]:
    start = time.time()
    image_bytes = pil_to_bytes(image)
    key = hash_image_bytes(model_id, image_bytes)

    cached = cache_get(key)
    if cached is not None:
        latency = time.time() - start
        return {
            "markdown": cached["markdown"],
            "json": cached["json"],
            "latency_seconds": latency,
            "cache_hit": True,
        }

    results = run_model_on_images([image], model_id=model_id)
    result = results[0]

    cache_set(key, result)
    latency = time.time() - start

    return {
        "markdown": result["markdown"],
        "json": result["json"],
        "latency_seconds": latency,
        "cache_hit": False,
    }


# ======================================================================
# FastAPI app
# ======================================================================

app = FastAPI(
    title="Staircase Form Reader (Phi-3.5-Vision, finetuned)",
    description=(
        "Upload staircase survey form images.\n"
        "Currently uses fine-tuned Phi-3.5-Vision (base + QLoRA adapter).\n"
        "Service returns JSON + Markdown + latency."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    markdown: str
    json: str
    latency_seconds: float
    cache_hit: bool
    model_id: str


class BatchPredictionItem(BaseModel):
    markdown: str
    json: str
    latency_seconds: float
    cache_hit: bool
    model_id: str


class BatchPredictionResponse(BaseModel):
    results: List[BatchPredictionItem]


@app.get("/health")
async def health():
    return {"status": "ok", "available_models": list(AVAILABLE_MODELS.keys())}


@app.post("/predict/{model_id}", response_model=PredictionResponse)
async def predict_single(model_id: str, file: UploadFile = File(...)):
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id '{model_id}'. Use one of {list(AVAILABLE_MODELS.keys())}",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = run_single_ocr_pil(image, model_id=model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {e}")

    return PredictionResponse(
        markdown=result["markdown"],
        json=result["json"],
        latency_seconds=result["latency_seconds"],
        cache_hit=result["cache_hit"],
        model_id=model_id,
    )


@app.post("/batch_predict/{model_id}", response_model=BatchPredictionResponse)
async def predict_batch(model_id: str, files: List[UploadFile] = File(...)):
    if model_id not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model_id '{model_id}'. Use one of {list(AVAILABLE_MODELS.keys())}",
        )

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: List[BatchPredictionItem] = []

    for f in files:
        file_bytes = await f.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail=f"Empty file: {f.filename}")

        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            raise HTTPException(
                status_code=400, detail=f"Invalid image file: {f.filename}"
            )

        try:
            r = run_single_ocr_pil(img, model_id=model_id)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error during inference on {f.filename}: {e}"
            )

        results.append(
            BatchPredictionItem(
                markdown=r["markdown"],
                json=r["json"],
                latency_seconds=r["latency_seconds"],
                cache_hit=r["cache_hit"],
                model_id=model_id,
            )
        )

    return BatchPredictionResponse(results=results)


# ======================================================================
# Gradio UI
# ======================================================================

def build_gradio_app() -> gr.Blocks:
    title = "Staircase Form Reader (Phi-3.5-Vision, finetuned)"
    description = (
        "1. Choose the OCR model (currently: Phi-3.5-Vision finetuned with QLoRA)\n"
        "2. Upload a scanned staircase survey form image\n"
        "3. The model outputs JSON, rendered as Markdown + raw JSON\n"
        "4. Latency (seconds) is displayed under the Run button"
    )

    def gradio_predict(model_id: str, image: Image.Image):
        if image is None:
            return "Please upload an image.", "", "Latency: – s"
        if model_id not in AVAILABLE_MODELS:
            return f"Invalid model_id '{model_id}'", "", "Latency: – s"

        try:
            result = run_single_ocr_pil(image, model_id=model_id)
        except Exception as e:
            return f"Error during inference: {e}", "", "Latency: – s"

        latency_str = (
            f"Latency: {result['latency_seconds']:.2f} s "
            f"(cache_hit={result['cache_hit']}, model='{model_id}')"
        )
        return result["markdown"], result["json"], latency_str

    with gr.Blocks() as demo:
        gr.Markdown(f"## {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="phi",
                    label="Choose OCR model",
                )
                image_input = gr.Image(
                    type="pil",
                    label="Upload staircase form image",
                )
                run_button = gr.Button("Run OCR")
                latency_output = gr.Markdown("Latency: – s")

            with gr.Column(scale=1):
                markdown_output = gr.Markdown(
                    label="Markdown view of predicted form"
                )
                json_output = gr.Textbox(
                    label="Raw JSON prediction",
                    lines=18,
                )

        run_button.click(
            fn=gradio_predict,
            inputs=[model_dropdown, image_input],
            outputs=[markdown_output, json_output, latency_output],
        )

    return demo


gradio_app = build_gradio_app()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
