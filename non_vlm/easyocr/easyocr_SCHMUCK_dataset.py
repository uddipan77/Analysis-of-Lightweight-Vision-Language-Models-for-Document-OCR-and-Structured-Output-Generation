import os
import sys
import re
import unicodedata
import json

# --- dependency checks ---
try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except Exception:
    print("NumPy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import easyocr
    print("EasyOCR imported successfully")
except Exception:
    print("EasyOCR is required. Install with: pip install easyocr")
    sys.exit(1)

# ---------- config ----------
IMAGES_DIR = r"C:\FAU\sem6\Thesis\schmuck-benchmark\dataset\images"
JSONL_PATH = r"C:\FAU\sem6\Thesis\schmuck-benchmark\dataset\test.jsonl"

LANGS = ['de']
USE_GPU = True
MIN_CONF = 0.0
LINE_TOL = 0.7

# ---------- output ----------
OUTPUT_DIR = os.path.join(os.path.dirname(IMAGES_DIR), "output_easyocr_schmuck_json_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRED_JSONL = os.path.join(OUTPUT_DIR, "predictions_easyocr.jsonl")
METRICS_TXT = os.path.join(OUTPUT_DIR, "metrics_easyocr_json.txt")

# =========================
# Normalization + CER core
# =========================
def normalize_for_cer_schema(s: str) -> str:
    """Schema CER: punctuation-sensitive (keeps ':' etc.)."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s

def normalize_for_cer_content(s: str) -> str:
    """Content CER: punctuation-insensitive (drop punctuation; keep 0-9, a-z, äöüß, hyphen)."""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9a-zäöüß\-]+", "", s, flags=re.UNICODE)
    return s

def levenshtein_char(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    return prev[m]

def cer(hyp: str, ref: str, ignore_punct: bool = False):
    norm = normalize_for_cer_content if ignore_punct else normalize_for_cer_schema
    h = norm(hyp or "")
    r = norm(ref or "")
    dist = levenshtein_char(h, r)
    return dist / max(1, len(r)), dist, len(r)

# =========================
# OCR utils
# =========================
def result_to_text(detections, min_conf=0.0, line_tol=LINE_TOL) -> str:
    """Convert EasyOCR detections [(bbox, text, conf), ...] into a reading-order string."""
    items = []
    for bbox, text, conf in detections:
        if conf is not None and conf < min_conf:
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x_left, y_top = min(xs), min(ys)
        height = max(ys) - min(ys) + 1e-6
        items.append((y_top, x_left, height, text))
    if not items:
        return ""
    items.sort(key=lambda t: (t[0], t[1]))  # y then x

    lines, current = [], [items[0]]
    for it in items[1:]:
        prev = current[-1]
        same_line = abs(it[0] - prev[0]) <= max(it[2], prev[2]) * line_tol
        if same_line:
            current.append(it)
        else:
            lines.append(current); current = [it]
    lines.append(current)

    out_lines = []
    for line in lines:
        line_sorted = sorted(line, key=lambda t: t[1])
        out_lines.append(" ".join(t[3] for t in line_sorted))
    return "\n".join(out_lines)

# =========================
# Schmuck: OCR-text -> JSON
# =========================
SCHMUCK_KEYS_ORDER = [
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

# Variants that appear in OCR (left side) -> canonical key
KEY_VARIANTS = {
    "gegenstand": "Gegenstand",
    "inv.nr": "Inv.Nr",
    "inv. nr": "Inv.Nr",
    "inv nr": "Inv.Nr",
    "inv-nr": "Inv.Nr",
    "inv.": "Inv.Nr",  # sometimes OCR truncates
    "herkunft": "Herkunft",
    "foto notes": "Foto Notes",
    "foto nr": "Foto Notes",
    "foto nr.": "Foto Notes",
    "foto": "Foto Notes",  # fallback (use carefully; works for these cards)
    "standort": "Standort",
    "material": "Material",
    "datierung": "Datierung",
    "maße": "Maße",
    "masse": "Maße",  # OCR sometimes
    "gewicht": "Gewicht",
    "erworben von": "erworben von",
    "erworben von.": "erworben von",
    "erworben von :": "erworben von",
    "am": "am",
    "preis": "Preis",
    "vers.-wert": "Vers.-Wert",
    "vers.—wert": "Vers.-Wert",
    "vers.- wert": "Vers.-Wert",
    "vers.wert": "Vers.-Wert",
    "vers-wert": "Vers.-Wert",
    "beschreibung": "Beschreibung",
    "literatur": "Literatur",
    "ausstellungen": "Ausstellungen",
}

def _clean_ocr_for_parsing(text: str) -> str:
    t = text or ""
    # normalize unicode, unify line endings
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # unify common dash variants
    t = t.replace("—", "-").replace("–", "-")

    # normalize some known label variants (helps matching)
    # e.g., "Inv. Nr.:" -> "Inv.Nr:"
    t = re.sub(r"\bInv\.\s*Nr\.?\b", "Inv.Nr", t, flags=re.IGNORECASE)
    t = re.sub(r"\bFoto\s*Nr\.?\b", "Foto Nr", t, flags=re.IGNORECASE)
    t = re.sub(r"\bVers\.\s*-\s*Wert\b", "Vers.-Wert", t, flags=re.IGNORECASE)
    t = re.sub(r"\bVers\.\s*Wert\b", "Vers.-Wert", t, flags=re.IGNORECASE)

    # collapse repeated spaces
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def parse_schmuck_text_to_json(hyp_text: str) -> dict:
    """
    Rule-based parser for Schmuck cards.
    Strategy:
      - Detect occurrences of known keys (with optional ':' after them)
      - Segment the text between key occurrences as that key's value
      - Fill a fixed schema (missing -> "")
    """
    text = _clean_ocr_for_parsing(hyp_text)

    # Build a regex that finds any key variant as a "label" token.
    # Allow optional punctuation/colon after label.
    # We match in a case-insensitive way.
    # IMPORTANT: sort by length so "erworben von" matches before "am", etc.
    variants = sorted(KEY_VARIANTS.keys(), key=len, reverse=True)
    # escape for regex alternation
    alt = "|".join(re.escape(v) for v in variants)
    # label pattern: word boundary + (variant) + optional spaces + optional ":".
    label_re = re.compile(rf"(?i)\b({alt})\b\s*:?", flags=re.UNICODE)

    matches = list(label_re.finditer(text))
    pred = {k: "" for k in SCHMUCK_KEYS_ORDER}

    if not matches:
        # no labels detected -> nothing we can do reliably
        return pred

    # Build segments
    for idx, m in enumerate(matches):
        raw_label = m.group(1)
        label_norm = raw_label.strip().casefold()
        canonical = KEY_VARIANTS.get(label_norm)

        if not canonical:
            continue

        start_val = m.end()
        end_val = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        val = text[start_val:end_val].strip()

        # light cleanup: remove stray leading separators
        val = re.sub(r"^[\s\-:]+", "", val).strip()

        # join internal newlines as spaces for values
        val = re.sub(r"\s*\n\s*", " ", val).strip()

        # If we already have content for this canonical key, append (rare but happens)
        if pred.get(canonical):
            pred[canonical] = (pred[canonical].rstrip() + " " + val).strip()
        else:
            pred[canonical] = val

    # Heuristic: Maße sometimes continues on next line without key,
    # but since we segment until next key, it's already captured.
    # Optional: normalize Material commas/spacing
    if pred.get("Material"):
        pred["Material"] = re.sub(r"\s*,\s*", ", ", pred["Material"]).strip()
        pred["Material"] = re.sub(r"\s+", " ", pred["Material"]).strip()

    # Optional: move obvious currency to Vers.-Wert if empty but appears near "Preis"/end
    # Keep conservative to avoid harm.
    if not pred.get("Vers.-Wert"):
        mcur = re.search(r"\b(\d+[.,]?\d*)\s*(DM|€|EUR)\b", text, flags=re.IGNORECASE)
        if mcur and ("vers" in text.lower() or "wert" in text.lower()):
            pred["Vers.-Wert"] = f"{mcur.group(1)} {mcur.group(2)}".strip()

    return pred

# =========================
# JSON-vs-JSON CER (field-wise)
# =========================
def gt_json_from_obj(obj: dict) -> dict:
    """Ground truth dict without file_name."""
    return {k: v for k, v in obj.items() if k != "file_name"}

def compute_json_field_cer(pred_json: dict, gt_json: dict, ignore_punct: bool):
    """
    Returns:
      - macro CER over fields
      - micro CER over fields (sum edits / sum ref_len)
      - per_field: {key: {cer, edits, ref_len}}
    """
    per_field = {}
    cers = []
    tot_edits = 0
    tot_ref = 0

    # Evaluate only keys in GT schema (except file_name)
    keys = [k for k in gt_json.keys() if k != "file_name"]

    for k in keys:
        ref = str(gt_json.get(k, "") or "")
        hyp = str(pred_json.get(k, "") or "")

        cer_v, edits, ref_len = cer(hyp, ref, ignore_punct=ignore_punct)

        per_field[k] = {"cer": cer_v, "edits": edits, "ref_len": ref_len}
        cers.append(cer_v)
        tot_edits += edits
        tot_ref += ref_len

    macro = float(np.mean(cers)) if cers else 0.0
    micro = (tot_edits / max(1, tot_ref)) if tot_ref > 0 else 0.0
    return macro, micro, per_field

# =========================
# Main
# =========================
def main():
    reader = easyocr.Reader(LANGS, gpu=USE_GPU)

    found, missing = 0, 0
    per_image_rows = []

    # Totals for global micro
    tot_schema_edits = tot_schema_ref = 0
    tot_content_edits = tot_content_ref = 0

    # For global macro, store per-image macro
    macro_schema_list = []
    macro_content_list = []

    # write predictions jsonl
    pred_f = open(PRED_JSONL, "w", encoding="utf-8")

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            file_name = (obj.get("file_name") or "").strip()
            if not file_name:
                missing += 1
                continue

            # resolve image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(
                    IMAGES_DIR,
                    file_name.replace(".jpeg", ext).replace(".jpg", ext).replace(".png", ext)
                )
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                candidate = os.path.join(IMAGES_DIR, file_name)
                if os.path.exists(candidate):
                    img_path = candidate

            if not img_path:
                missing += 1
                continue

            found += 1
            base = os.path.splitext(file_name)[0]

            gt = gt_json_from_obj(obj)

            # OCR
            try:
                detections = reader.readtext(img_path, detail=1, paragraph=False)
            except Exception as e:
                print(f"[ERROR] OCR failed for {img_path}: {e}")
                missing += 1
                continue

            hyp_text = result_to_text(detections, min_conf=MIN_CONF, line_tol=LINE_TOL)

            # save raw OCR text
            ocr_out = os.path.join(OUTPUT_DIR, f"{base}_easyocr.txt")
            with open(ocr_out, "w", encoding="utf-8") as focr:
                focr.write(hyp_text)

            # Parse OCR text -> predicted JSON
            pred_json = parse_schmuck_text_to_json(hyp_text)

            # JSON-vs-JSON CER (field-wise)
            macro_schema, micro_schema, per_field_schema = compute_json_field_cer(
                pred_json, gt, ignore_punct=False
            )
            macro_content, micro_content, per_field_content = compute_json_field_cer(
                pred_json, gt, ignore_punct=True
            )

            # Update global totals for micro
            # micro computed by summing edits/ref across ALL fields/images
            for k, d in per_field_schema.items():
                tot_schema_edits += d["edits"]
                tot_schema_ref += d["ref_len"]
            for k, d in per_field_content.items():
                tot_content_edits += d["edits"]
                tot_content_ref += d["ref_len"]

            macro_schema_list.append(macro_schema)
            macro_content_list.append(macro_content)

            # Store row
            per_image_rows.append({
                "file_name": file_name,
                "image_path": img_path,
                "ocr_text_file": ocr_out,
                "macro_cer_schema": macro_schema,
                "macro_cer_content": macro_content,
                "micro_cer_schema": micro_schema,
                "micro_cer_content": micro_content,
            })

            # Save prediction record
            rec = {
                "file_name": file_name,
                "image_path": img_path,
                "predicted_json": pred_json,
                "ground_truth": gt,
                "macro_cer_schema": macro_schema,
                "macro_cer_content": macro_content,
                "micro_cer_schema": micro_schema,
                "micro_cer_content": micro_content,
            }
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(
                f"{base}: "
                f"JSON_CER_schema(macro)={macro_schema:.4f} | JSON_CER_content(macro)={macro_content:.4f}"
            )

    pred_f.close()

    # Global averages
    macro_schema_global = float(np.mean(macro_schema_list)) if macro_schema_list else 0.0
    macro_content_global = float(np.mean(macro_content_list)) if macro_content_list else 0.0
    micro_schema_global = tot_schema_edits / max(1, tot_schema_ref)
    micro_content_global = tot_content_edits / max(1, tot_content_ref)

    # Write metrics file
    with open(METRICS_TXT, "w", encoding="utf-8") as mf:
        mf.write("EASYOCR -> RULE PARSE -> JSON  (Schmuck)\n")
        mf.write("=" * 60 + "\n\n")

        for r in per_image_rows:
            mf.write(
                f"{os.path.splitext(r['file_name'])[0]}: "
                f"macro_schema={r['macro_cer_schema']:.4f} | "
                f"macro_content={r['macro_cer_content']:.4f} | "
                f"micro_schema={r['micro_cer_schema']:.4f} | "
                f"micro_content={r['micro_cer_content']:.4f} | "
                f"ocr_file={r['ocr_text_file']}\n"
            )

        mf.write("\nAverages (over images):\n")
        mf.write(f"Macro CER (schema):  {macro_schema_global:.4f}\n")
        mf.write(f"Macro CER (content): {macro_content_global:.4f}\n")
        mf.write("\nMicro CER (over all fields/chars):\n")
        mf.write(f"Micro CER (schema):  {micro_schema_global:.4f}\n")
        mf.write(f"Micro CER (content): {micro_content_global:.4f}\n")

        mf.write(f"\nImages found: {found}\nImages missing/skipped: {missing}\n")
        mf.write(f"\nPredictions JSONL: {PRED_JSONL}\n")

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")
    print(f"- Predictions: {PRED_JSONL}")
    print(f"- Metrics:     {METRICS_TXT}")
    print(f"Images found: {found}")
    print(f"Images missing/skipped: {missing}")

if __name__ == "__main__":
    main()
