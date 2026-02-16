"""
Batch OCR + structured JSON extraction with Gemini 3 Flash.

- Reads images inventarbuch-001.jpg ... inventarbuch-308.jpg
- Uses prompt-only JSON (NO schema, NO SDK validation)
- One JSON file per image
- Missing fields => "" (empty string)
- Quota-safe for free tier

Requirements:
  pip install google-genai
"""

import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from google import genai
from google.genai import types


# -----------------------
# CONFIG
# -----------------------
# Gemini 3 Flash model
MODEL = "models/gemini-3-flash-preview"

INPUT_DIR = Path(r"C:\FAU\sem6\Thesis\inventarkarten-ocr\inventarkarten\erlangen")
OUTPUT_DIR = INPUT_DIR / "erlangen_json"

START_IDX = 87
END_IDX = 100

API_KEY_ENV = "GEMINI_API_KEY"

SLEEP_BETWEEN_IMAGES = 3  # seconds
MAX_RETRIES = 2


# -----------------------
# JSON TEMPLATE
# -----------------------
TEMPLATE: Dict[str, Any] = {
    "image_name": "",
    "Überschrift": "",
    "Inventarnummer": "",
    "Maße": {"L": "", "B": "", "D": ""},
    "Objektbezeichnung": "",
    "Fundort": "",
    "Fundzeit": "",
    "Beschreibungstext": "",
}

# -----------------------
# JSON SCHEMA FOR STRUCTURED OUTPUT
# -----------------------
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "image_name": {"type": "string"},
        "Überschrift": {"type": "string"},
        "Inventarnummer": {"type": "string"},
        "Maße": {
            "type": "object",
            "properties": {
                "L": {"type": "string"},
                "B": {"type": "string"},
                "D": {"type": "string"}
            },
            "required": ["L", "B", "D"]
        },
        "Objektbezeichnung": {"type": "string"},
        "Fundort": {"type": "string"},
        "Fundzeit": {"type": "string"},
        "Beschreibungstext": {"type": "string"}
    },
    "required": ["image_name", "Überschrift", "Inventarnummer", "Maße", 
                 "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]
}


# -----------------------
# PROMPT (OCR-friendly)
# -----------------------
PROMPT = """Extract all visible text from this inventory card image.

Field descriptions:
- image_name: The filename provided
- Überschrift: The header/title at the top of the card
- Inventarnummer: The inventory number (e.g., "I.40")
- Maße: Measurements with L (length), B (width), D (depth/thickness)
- Objektbezeichnung: Object designation/name
- Fundort: Location where the object was found
- Fundzeit: Time/date when the object was found  
- Beschreibungstext: The main description text on the card

Rules:
- If a field is not visible in the image, use an empty string ""
- Do not invent or guess content
- Preserve original German spelling and punctuation
- Preserve decimal commas (e.g., 10,93)
"""


# -----------------------
# HELPERS
# -----------------------
def guess_mime_type(path: Path) -> str:
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"


def deep_copy_template() -> Dict[str, Any]:
    return json.loads(json.dumps(TEMPLATE, ensure_ascii=False))


def merge_into_template(template: Dict[str, Any], data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = deep_copy_template()

    if isinstance(data, dict):
        for key in out:
            if key == "Maße" and isinstance(data.get("Maße"), dict):
                for mk in out["Maße"]:
                    val = data["Maße"].get(mk)
                    out["Maße"][mk] = "" if val is None else str(val)
            else:
                val = data.get(key)
                out[key] = "" if val is None else str(val)

    return out


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    text = text.strip()
    
    # Handle markdown code blocks: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # Remove opening ```json or ```
        lines = text.split("\n", 1)
        if len(lines) > 1:
            text = lines[1]
        else:
            text = text[3:]  # Just remove ```
        
        # Remove closing ```
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()

    try:
        return json.loads(text)
    except Exception as e:
        print(f"[WARN] JSON parse error: {e}")
        print(f"[DEBUG] Raw text: {text[:500]}...")  # Print first 500 chars for debugging
        return None


# -----------------------
# GEMINI CALL (IMAGE FIRST + STRUCTURED OUTPUT)
# -----------------------
def call_gemini(
    client: genai.Client,
    image_bytes: bytes,
    mime_type: str,
    file_name: str,
) -> Optional[Dict[str, Any]]:

    contents = [
        types.Content(
            role="user",
            parts=[
                # ✅ IMAGE FIRST (CRITICAL)
                types.Part(
                    inline_data=types.Blob(
                        mime_type=mime_type,
                        data=image_bytes,
                    )
                ),
                # ✅ TEXT AFTER IMAGE
                types.Part(
                    text=f"image_name: {file_name}\n\n{PROMPT}"
                ),
            ],
        )
    ]

    # ✅ STRUCTURED OUTPUT CONFIG - Forces JSON response
    config = types.GenerateContentConfig(
        temperature=0.2,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            
            # Debug: print raw response
            if resp.text:
                print(f"[DEBUG] Response length: {len(resp.text)} chars")
            else:
                print(f"[WARN] Empty response from API")
                
            return parse_json(resp.text)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"[ERROR] {file_name}: {e}")
                return None
            print(f"[WARN] {file_name}: retry {attempt}/{MAX_RETRIES}")
            time.sleep(5)


# -----------------------
# MAIN
# -----------------------
def main() -> None:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Environment variable {API_KEY_ENV} not set")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=api_key)

    for i in range(START_IDX, END_IDX + 1):
        num = f"{i:03d}"
        file_name = f"inventarbuch-{num}.jpg"
        img_path = INPUT_DIR / file_name
        out_path = OUTPUT_DIR / f"inventarbuch-{num}.json"

        if not img_path.exists():
            print(f"[SKIP] missing: {file_name}")
            continue

        if out_path.exists():
            print(f"[SKIP] exists: {file_name}")
            continue

        print(f"[INFO] processing: {file_name}")

        parsed = call_gemini(
            client,
            image_bytes=img_path.read_bytes(),
            mime_type=guess_mime_type(img_path),
            file_name=file_name,
        )

        final_obj = merge_into_template(TEMPLATE, parsed)
        final_obj["image_name"] = file_name

        out_path.write_text(
            json.dumps(final_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[OK] wrote: {out_path}")

        time.sleep(SLEEP_BETWEEN_IMAGES)


if __name__ == "__main__":
    main()
