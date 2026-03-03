#!/usr/bin/env python3
"""
Multi-Dataset OCR UI  —  Gemma-3 4B (finetuned, single adapter for all datasets)
=================================================================================

FastAPI backend + Gradio UI in one file (same pattern as demo_ui.py).
Supports staircase, inventory, and schmuck images with a single LoRA adapter.

Run:
    python gemma_multidataset_ui.py

UI URL:
    http://<server>:8000/ui
"""

import io
import json
import time
import hashlib
import gc
import unicodedata
from typing import List, Dict, Any

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import gradio as gr


# ========= PATH CONFIG =========
BASE_MODEL_PATH = (
    "/home/vault/iwi5/iwi5298h/models/hf_cache/hub/"
    "models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/"
    "snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe"
)
FINETUNED_ADAPTER_PATH = (
    "/home/vault/iwi5/iwi5298h/models_image_text/gemma/general/"
    "run_20260215_100553_multidataset_bestCER/best_model"
)
# ================================


# ======================================================================
# INSTRUCTIONS  (per document type)
# ======================================================================

INSTRUCTION_STAIRCASE = """You are an OCR model for historical staircase survey forms.

Task:
Given ONE image of a filled-in staircase form, read all printed text, handwritten notes and all checked/unchecked boxes and output a single JSON object that represents the complete form.

Rules:
- Return ONLY one valid JSON object, with no extra text before or after it.
- Use exactly the same field names, nesting, accents, and capitalization as in the training JSON for this form type (e.g. keys like "stair_type", "Name des Hauses", "Adresse", "LÄUFE", "GELÄNDER", etc.).
- Never drop a key that appears in the form's JSON structure. If a field is empty on the form, still include it with an empty string "" (or false for an unchecked box).
- Use booleans for checkbox options: true if the box is checked, false if it is empty.
- Use strings for numbers and free-text fields (measurements, dates, names, notes).
- Do NOT invent new fields."""

INSTRUCTION_INVENTORY = """Du bist ein OCR- und Information-Extraction-Modell für deutsche historische Inventardokumente.

Aufgabe:
Lies ALLE Informationen aus dem Bild dieses Inventarblatts und gib GENAU EIN JSON-Objekt zurück.

Das JSON MUSS folgende Felder enthalten:
- Überschrift: Dokumenttitel
- Inventarnummer: Inventar- oder Katalognummer
- Maße: Objekt mit L, B, D (Länge, Breite, Tiefe)
- Objektbezeichnung: Beschreibung/Name des Objekts
- Fundort: Fundort des Objekts
- Fundzeit: Zeit der Auffindung
- Beschreibungstext: Ausführlicher Beschreibungstext

Regeln:
- Gib NUR ein gültiges JSON-Objekt zurück (kein extra Text davor oder danach).
- Verwende GENAU diese Feldnamen und Groß-/Kleinschreibung.
- Wenn ein Feld leer ist oder nicht sichtbar, gib einen leeren String "" zurück.
- Das Feld "Maße" MUSS immer ein Objekt mit den Schlüsseln "L", "B", "D" sein, auch wenn leer.
- Erfinde keine zusätzlichen Felder."""

INSTRUCTION_SCHMUCK = """Extract all information from this German jewelry catalog document image as a structured JSON object.

The JSON should contain these fields:
- Gegenstand: Object/item name
- Inv.Nr: Inventory number
- Herkunft: Origin/provenance
- Foto Notes: Photo notes
- Standort: Location
- Material: Material description
- Datierung: Dating/time period
- Maße: Measurements
- Gewicht: Weight
- erworben von: Acquired from
- am: Acquired on (date)
- Preis: Price
- Vers.-Wert: Insurance value
- Beschreibung: Description
- Literatur: Literature references
- Ausstellungen: Exhibitions

Return ONLY the JSON object, properly formatted."""

DOC_TYPE_INSTRUCTIONS: Dict[str, str] = {
    "staircase": INSTRUCTION_STAIRCASE,
    "inventory": INSTRUCTION_INVENTORY,
    "schmuck": INSTRUCTION_SCHMUCK,
}

DOC_TYPE_LABELS: Dict[str, str] = {
    "staircase": "Staircase Survey Form",
    "inventory": "Inventory Entry",
    "schmuck": "Schmuck / Jewelry Catalog",
}


# ======================================================================
# Helpers (JSON extraction + Markdown rendering)
# ======================================================================

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def extract_json_from_response(response: str) -> str:
    if isinstance(response, list):
        response = response[0] if response else ""
    if response is None:
        return "{}"

    text = str(response).strip()
    if not text:
        return "{}"

    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 3:
            body = parts[1]
            if body.lstrip().startswith("json"):
                body = body[4:].strip()
            text = body.strip()

    if "{" not in text or "}" not in text:
        return text

    start = text.find("{")
    end = text.rfind("}") + 1
    json_str = text[start:end]

    # First complete JSON object (brace-balanced)
    try:
        depth = 0
        for i, ch in enumerate(json_str):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = json_str[: i + 1]
                    parsed = json.loads(candidate)
                    return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
    except Exception:
        pass

    try:
        parsed = json.loads(json_str)
        return json.dumps(parsed, ensure_ascii=False, sort_keys=False)
    except Exception:
        return json_str


def escape_md(text) -> str:
    if text is None:
        return ""
    return str(text).replace("|", "\\|")


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
    lines = ["#" * depth + f" {name}\n\n"]
    lines.append("|  | " + " | ".join(escape_md(c) for c in cols) + " |\n")
    lines.append("| --- | " + " | ".join("---" for _ in cols) + " |\n")
    for row_key, inner in d.items():
        row_vals = []
        for c in cols:
            v = inner.get(c)
            cell = format_list(v) if isinstance(v, list) else format_scalar(v)
            row_vals.append(escape_md(cell))
        lines.append(f"| {escape_md(row_key)} | " + " | ".join(row_vals) + " |\n")
    lines.append("\n")
    return "".join(lines)


def render_dict_generic(d, depth=3, section_name=None):
    lines = []
    if section_name is not None:
        lines.append("#" * depth + f" {section_name}\n\n")

    simple_items, complex_items = [], []
    for k, v in d.items():
        (complex_items if isinstance(v, (dict, list)) else simple_items).append((k, v))

    if simple_items:
        lines.append("| Field | Value |\n| --- | --- |\n")
        for k, v in simple_items:
            lines.append(f"| {escape_md(k)} | {escape_md(format_scalar(v))} |\n")
        lines.append("\n")

    for k, v in complex_items:
        if isinstance(v, dict):
            table_md = render_dict_as_table(k, v, depth + 1)
            lines.append(table_md if table_md else render_dict_generic(v, depth + 1, k))
        elif isinstance(v, list):
            if all(not isinstance(x, (dict, list)) for x in v):
                lines.append("#" * (depth + 1) + f" {k}\n\n| Value |\n| --- |\n")
                for x in v:
                    lines.append(f"| {escape_md(format_scalar(x))} |\n")
                lines.append("\n")
            elif v and all(isinstance(x, dict) for x in v):
                cols = list(v[0].keys())
                if all(set(x.keys()) == set(cols) for x in v):
                    lines.append("#" * (depth + 1) + f" {k}\n\n")
                    lines.append("|  | " + " | ".join(escape_md(c) for c in cols) + " |\n")
                    lines.append("| --- | " + " | ".join("---" for _ in cols) + " |\n")
                    for idx, item in enumerate(v, 1):
                        rv = []
                        for c in cols:
                            val = item.get(c)
                            rv.append(escape_md(format_list(val) if isinstance(val, list) else format_scalar(val)))
                        lines.append(f"| #{idx} | " + " | ".join(rv) + " |\n")
                    lines.append("\n")
                else:
                    for idx, item in enumerate(v, 1):
                        lines.append(render_dict_generic(item, depth + 1, f"{k} #{idx}"))
            else:
                lines.append("#" * (depth + 1) + f" {k}\n\n```json\n")
                lines.append(json.dumps(v, ensure_ascii=False, indent=2))
                lines.append("\n```\n\n")
    return "".join(lines)


def prediction_json_to_markdown(predicted_json_str: str, doc_type: str = "document") -> str:
    try:
        obj = json.loads(predicted_json_str)
    except Exception:
        return "```json\n" + predicted_json_str + "\n```"
    if not isinstance(obj, dict):
        return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"
    title = DOC_TYPE_LABELS.get(doc_type, "Predicted Form")
    return f"## {title}\n\n" + render_dict_generic(obj, depth=3)


def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


# ======================================================================
# Model loading (lazy)
# ======================================================================

_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    from unsloth import FastVisionModel
    from peft import PeftModel

    print(f"⏳ Loading Gemma-3 4B base model (4-bit via Unsloth) …")
    print(f"   base_model_path     = {BASE_MODEL_PATH}")
    print(f"   finetuned_adapter   = {FINETUNED_ADAPTER_PATH}")

    base_model, _tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("✅ Base model loaded")

    _model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)
    print("✅ Multi-dataset adapter attached")

    FastVisionModel.for_inference(_model)
    _model.eval()
    print("✅ Gemma-3 multi-dataset model ready!")

    return _model, _tokenizer


# ======================================================================
# Simple in-memory cache
# ======================================================================

CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_MAX_SIZE = 512


def hash_key(doc_type: str, image_bytes: bytes) -> str:
    return f"{doc_type}:" + hashlib.sha256(image_bytes).hexdigest()


def cache_get(key: str):
    return CACHE.get(key)


def cache_set(key: str, value: Dict[str, Any]):
    if len(CACHE) >= CACHE_MAX_SIZE:
        CACHE.pop(next(iter(CACHE)))
    CACHE[key] = value


# ======================================================================
# Core inference
# ======================================================================

def run_gemma_inference(image: Image.Image, doc_type: str) -> Dict[str, str]:
    """Run a single image through the Gemma-3 multi-dataset model."""
    model, tokenizer = get_model()
    device = next(model.parameters()).device

    instruction = DOC_TYPE_INSTRUCTIONS.get(doc_type, INSTRUCTION_STAIRCASE)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True,
            repetition_penalty=1.0,
        )

    input_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][input_len:]
    raw_output = tokenizer.decode(
        gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    del inputs, outputs, gen_ids
    torch.cuda.empty_cache()
    gc.collect()

    raw_output = normalize_unicode(raw_output.strip())
    predicted_json = extract_json_from_response(raw_output)
    markdown = prediction_json_to_markdown(predicted_json, doc_type)

    return {"markdown": markdown, "json": predicted_json}


def run_single_ocr(image: Image.Image, doc_type: str) -> Dict[str, Any]:
    start = time.time()
    image_bytes = pil_to_bytes(image)
    key = hash_key(doc_type, image_bytes)

    cached = cache_get(key)
    if cached is not None:
        return {
            "markdown": cached["markdown"],
            "json": cached["json"],
            "latency_seconds": time.time() - start,
            "cache_hit": True,
        }

    result = run_gemma_inference(image, doc_type)
    cache_set(key, result)

    return {
        "markdown": result["markdown"],
        "json": result["json"],
        "latency_seconds": time.time() - start,
        "cache_hit": False,
    }


# ======================================================================
# FastAPI app
# ======================================================================

app = FastAPI(
    title="Multi-Dataset OCR (Gemma-3 4B, finetuned)",
    description=(
        "Upload staircase / inventory / schmuck form images.\n"
        "Single Gemma-3 4B model with multi-dataset LoRA adapter.\n"
        "Returns JSON + Markdown + latency."
    ),
    version="1.0.0",
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
    doc_type: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Gemma-3-4B-IT (multi-dataset LoRA)",
        "doc_types": list(DOC_TYPE_LABELS.keys()),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    doc_type: str = Query("staircase", enum=["staircase", "inventory", "schmuck"]),
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = run_single_ocr(image, doc_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictionResponse(
        markdown=result["markdown"],
        json=result["json"],
        latency_seconds=result["latency_seconds"],
        cache_hit=result["cache_hit"],
        doc_type=doc_type,
    )


# ======================================================================
# Gradio UI
# ======================================================================

def build_gradio_app() -> gr.Blocks:
    title = "Multi-Dataset OCR using Gemma-3 4B (finetuned)"
    description = (
        "1. Select the **document type** (staircase / inventory / schmuck)\n"
        "2. Upload a scanned form image\n"
        "3. The model extracts structured data as JSON, rendered as Markdown + raw JSON\n"
        "4. Latency is shown below the Run button\n\n"
        "**Model:** Gemma-3-4B-IT with multi-dataset LoRA adapter "
        "(trained on all three datasets)"
    )

    def gradio_predict(doc_type: str, image: Image.Image):
        if image is None:
            return "Please upload an image.", "", "Latency: – s"
        if doc_type not in DOC_TYPE_INSTRUCTIONS:
            return f"Invalid document type '{doc_type}'", "", "Latency: – s"

        try:
            result = run_single_ocr(image, doc_type)
        except Exception as e:
            return f"Error during inference: {e}", "", "Latency: – s"

        latency_str = (
            f"Latency: {result['latency_seconds']:.2f} s  "
            f"(cache_hit={result['cache_hit']}, doc_type='{doc_type}')"
        )
        return result["markdown"], result["json"], latency_str

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                doc_type_dropdown = gr.Dropdown(
                    choices=list(DOC_TYPE_LABELS.keys()),
                    value="staircase",
                    label="Document Type",
                    info="Choose the type of document you are uploading",
                )
                image_input = gr.Image(
                    type="pil",
                    label="Upload document image",
                )
                run_button = gr.Button("🔍 Run OCR", variant="primary")
                latency_output = gr.Markdown("Latency: – s")

            with gr.Column(scale=1):
                markdown_output = gr.Markdown(
                    label="Markdown view of predicted form",
                )
                json_output = gr.Textbox(
                    label="Raw JSON prediction",
                    lines=20,
                )

        run_button.click(
            fn=gradio_predict,
            inputs=[doc_type_dropdown, image_input],
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

    print("─" * 60)
    print("  Multi-Dataset OCR  –  Gemma-3 4B (finetuned)")
    print("  UI:  http://localhost:8000/ui")
    print("  API: http://localhost:8000/docs")
    print("─" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
