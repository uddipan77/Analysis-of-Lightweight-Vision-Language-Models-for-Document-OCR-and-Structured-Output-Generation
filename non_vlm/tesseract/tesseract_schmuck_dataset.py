import os
import json
import re
import unicodedata
from datetime import datetime
from typing import Optional
from PIL import Image
import pytesseract

# ---------------- Tesseract setup ----------------
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

LANG = "deu"
CONFIG = r"--oem 1 --psm 6"

# ---------------- Paths ----------------
IMAGES_DIR = r"C:\FAU\sem6\Thesis\schmuck-benchmark\dataset\images"
JSONL_PATH = r"C:\FAU\sem6\Thesis\schmuck-benchmark\dataset\test.jsonl"

# output folder (timestamped)
BASE_OUTPUT_DIR = os.path.join(os.path.dirname(IMAGES_DIR), "output_tesseract_schmuck_json_vs_json")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, "run_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
os.makedirs(RUN_DIR, exist_ok=True)

# ---------------- JSON helpers ----------------
def canonical_json_string(obj: dict) -> str:
    """Stable JSON string for CER."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

# ---------------- Levenshtein (char) ----------------
def levenshtein_char(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]

def cer_raw(hyp: str, ref: str):
    """Raw CER on raw strings (schema-sensitive JSON vs JSON)."""
    edits = levenshtein_char(hyp, ref)
    denom = max(1, len(ref))
    return edits / denom, edits, len(ref)

# ---------------- OCR ----------------
def ocr_tesseract(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(img, lang=LANG, config=CONFIG)
    return text.strip()

# ---------------- Image resolver ----------------
def resolve_image_path(images_dir: str, file_name: str) -> Optional[str]:
    if not file_name:
        return None

    base = os.path.splitext(file_name)[0]
    for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF"]:
        candidate = os.path.join(images_dir, base + ext)
        if os.path.exists(candidate):
            return candidate

    candidate = os.path.join(images_dir, file_name)
    if os.path.exists(candidate):
        return candidate

    return None

# ---------------- Rule-based text -> JSON (Schmuck) ----------------
KEYS = [
    "Gegenstand",
    "Inv.Nr",
    "Herkunft",
    "Foto Notes",
    "Standort",
    "Material",
    "Datierung",
    "Maße",
    "Gewicht",
    "erworben von",
    "am",
    "Preis",
    "Vers.-Wert",
    "Beschreibung",
    "Literatur",
    "Ausstellungen",
]

KEY_ALIASES = {
    "inv nr": "Inv.Nr",
    "invnr": "Inv.Nr",
    "inv nr ": "Inv.Nr",
    "inv. nr": "Inv.Nr",
    "inv.nr": "Inv.Nr",

    "foto": "Foto Notes",
    "foto nr": "Foto Notes",
    "foto nr.": "Foto Notes",
    "foto notes": "Foto Notes",

    "vers wert": "Vers.-Wert",
    "vers.-wert": "Vers.-Wert",
    "vers. wert": "Vers.-Wert",
    "vers.—wert": "Vers.-Wert",

    "maße": "Maße",
    "masse": "Maße",
}

def _norm_key_fragment(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(":", "").strip()
    return s

def _canonical_key(raw_key: str) -> Optional[str]:
    nk = _norm_key_fragment(raw_key)
    if nk in KEY_ALIASES:
        return KEY_ALIASES[nk]
    for k in KEYS:
        if _norm_key_fragment(k) == nk:
            return k
    return None

def parse_schmuck_text_to_json(ocr_text: str) -> dict:
    out = {k: "" for k in KEYS}
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    if not lines:
        return out

    current_key = None

    def append_value(key: str, val: str):
        val = val.strip()
        if not val:
            return
        if out[key]:
            out[key] = (out[key] + " " + val).strip()
        else:
            out[key] = val

    for ln in lines:
        matches = list(re.finditer(r"([A-Za-zÄÖÜäöüß\.\-\s]{2,}?)\s*:\s*([^:]+)", ln))
        if matches:
            current_key = None
            for m in matches:
                raw_k = m.group(1).strip()
                raw_v = m.group(2).strip()
                ck = _canonical_key(raw_k)
                if ck is None:
                    continue
                append_value(ck, raw_v)
                current_key = ck
        else:
            if current_key:
                append_value(current_key, ln)
            else:
                # Maße lines like "L 14,8 cm" "B 5,8 cm"
                if re.search(r"\b[LBHD]\s*\d", ln, flags=re.IGNORECASE):
                    append_value("Maße", ln)

    for k in out:
        out[k] = re.sub(r"\s+", " ", out[k]).strip()

    return out

# ---------------- Main ----------------
def main():
    records = []
    found = 0
    missing = 0

    total_edits = 0
    total_ref_chars = 0

    preds_jsonl_path = os.path.join(RUN_DIR, "predictions.jsonl")

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            file_name = (obj.get("file_name") or "").strip()
            if not file_name:
                missing += 1
                continue

            img_path = resolve_image_path(IMAGES_DIR, file_name)
            if not img_path:
                missing += 1
                rec = {
                    "file_name": file_name,
                    "matched_image_path": None,
                    "ocr_text": "",
                    "predicted_json": {},
                    "ground_truth_json": {k: v for k, v in obj.items() if k != "file_name"},
                    "pred_json_str": "",
                    "gt_json_str": "",
                    "cer_json": 1.0,
                    "error": "image_not_found",
                }
                records.append(rec)
                continue

            found += 1

            gt_json = {k: v for k, v in obj.items() if k != "file_name"}
            gt_json_str = canonical_json_string(gt_json)

            try:
                ocr_text = ocr_tesseract(img_path)
            except Exception as e:
                missing += 1
                rec = {
                    "file_name": file_name,
                    "matched_image_path": img_path,
                    "ocr_text": "",
                    "predicted_json": {},
                    "ground_truth_json": gt_json,
                    "pred_json_str": "",
                    "gt_json_str": gt_json_str,
                    "cer_json": 1.0,
                    "error": "ocr_error: {0}".format(str(e)),
                }
                records.append(rec)
                continue

            base = os.path.splitext(os.path.basename(file_name))[0]

            ocr_out_path = os.path.join(RUN_DIR, "{0}_tesseract.txt".format(base))
            with open(ocr_out_path, "w", encoding="utf-8") as fo:
                fo.write(ocr_text)

            pred_json = parse_schmuck_text_to_json(ocr_text)
            pred_json_str = canonical_json_string(pred_json)

            cer_val, edits, ref_len = cer_raw(pred_json_str, gt_json_str)

            total_edits += edits
            total_ref_chars += ref_len

            pred_json_path = os.path.join(RUN_DIR, "{0}_pred.json".format(base))
            with open(pred_json_path, "w", encoding="utf-8") as fj:
                json.dump(pred_json, fj, ensure_ascii=False, indent=2)

            rec = {
                "file_name": file_name,
                "matched_image_path": img_path,
                "ocr_text_file": ocr_out_path,
                "predicted_json_file": pred_json_path,
                "predicted_json": pred_json,
                "ground_truth_json": gt_json,
                "pred_json_str": pred_json_str,
                "gt_json_str": gt_json_str,
                "cer_json": cer_val,
                "edits": edits,
                "ref_len": ref_len,
            }
            records.append(rec)

            print("{0}: CER_json={1:.4f} (edits={2}/{3})".format(base, cer_val, edits, ref_len))

    with open(preds_jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = os.path.join(RUN_DIR, "metrics_tesseract_json_vs_json.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Tesseract Schmuck — JSON vs JSON CER\n")
        f.write("=" * 60 + "\n")
        f.write("JSONL: {0}\n".format(JSONL_PATH))
        f.write("Images: {0}\n".format(IMAGES_DIR))
        f.write("Run dir: {0}\n\n".format(RUN_DIR))

        cer_list = [r.get("cer_json") for r in records if isinstance(r.get("cer_json"), (int, float))]
        valid = [c for c in cer_list if c is not None]
        if valid:
            macro = sum(valid) / len(valid)
            micro = total_edits / max(1, total_ref_chars)
            f.write("Macro CER_json: {0:.6f}\n".format(macro))
            f.write("Micro CER_json: {0:.6f}\n".format(micro))
            f.write("Total edits: {0}\n".format(total_edits))
            f.write("Total ref chars: {0}\n".format(total_ref_chars))

        f.write("\nImages found: {0}\n".format(found))
        f.write("Missing/skipped: {0}\n".format(missing))
        f.write("Predictions file: {0}\n".format(preds_jsonl_path))

    print("\nDone.")
    print("Run dir: {0}".format(RUN_DIR))
    print("Predictions: {0}".format(preds_jsonl_path))
    print("Summary: {0}".format(summary_path))
    print("Images found: {0}".format(found))
    print("Images missing/skipped: {0}".format(missing))

if __name__ == "__main__":
    main()
