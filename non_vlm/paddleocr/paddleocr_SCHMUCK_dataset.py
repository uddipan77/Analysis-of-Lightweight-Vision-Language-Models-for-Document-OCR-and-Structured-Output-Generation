import os
import sys
import re
import unicodedata
import json
from collections import OrderedDict

# --- dependency checks ---
try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except Exception:
    print("NumPy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import paddle
    print(f"PaddlePaddle: {paddle.__version__}  (CUDA: {paddle.is_compiled_with_cuda()})")
except Exception:
    print("PaddlePaddle is required. Install with: pip install paddlepaddle")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
    print("PaddleOCR imported successfully")
except Exception:
    print("PaddleOCR is required. Install with: pip install paddleocr")
    sys.exit(1)

# ---------- config ----------
IMAGES_DIR = r"C:\FAU\sem6\Thesis\schmuck-benchmark\dataset\images"
JSONL_PATH = r"C:\FAU\sem6\Thesis\schmuck-benchmark\dataset\test.jsonl"

LANG = "german"
MIN_CONF = 0.0
LINE_TOL = 0.7

# ---------- output ----------
OUTPUT_DIR = os.path.join(os.path.dirname(IMAGES_DIR), "output_paddleocr_test_schmuck_json")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MISSING_LOG = os.path.join(OUTPUT_DIR, "missing_images.txt")
ERROR_LOG   = os.path.join(OUTPUT_DIR, "ocr_errors.txt")
PRED_JSONL  = os.path.join(OUTPUT_DIR, "predictions_paddleocr.jsonl")

# ---------- normalization ----------
def normalize_for_cer_schema(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"-\s*\n\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s

def normalize_for_cer_content(s: str) -> str:
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
    h = norm(hyp)
    r = norm(ref)
    dist = levenshtein_char(h, r)
    return dist / max(1, len(r)), dist, len(r)

def json_to_string_stable(d: dict, key_order: list) -> str:
    """Stable JSON string with fixed key order (important for CER stability)."""
    od = OrderedDict()
    for k in key_order:
        od[k] = d.get(k, "")
    return json.dumps(od, ensure_ascii=False, separators=(",", ":"))

# ---------- OCR utils ----------
def result_to_text(detections, min_conf=0.0, line_tol=LINE_TOL) -> str:
    items = []
    for bbox, text, conf in detections:
        if conf is not None and conf < min_conf:
            continue
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and isinstance(bbox[0], (list, tuple))):
            continue
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        x_left, y_top = float(min(xs)), float(min(ys))
        height = float(max(ys) - min(ys) + 1e-6)
        items.append((y_top, x_left, height, text))
    if not items:
        return ""
    items.sort(key=lambda t: (t[0], t[1]))

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

def _to_list(x):
    try:
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x

def paddle_predict_to_detections(predict_output):
    if not isinstance(predict_output, (list, tuple)):
        try:
            predict_output = list(predict_output)
        except Exception:
            predict_output = [predict_output]
    if not predict_output:
        return []

    res0 = predict_output[0]
    res_dict = None

    jd = getattr(res0, "json", None)
    if callable(jd):
        try: jd = jd()
        except Exception: jd = None
    if isinstance(jd, dict):
        res_dict = jd

    if res_dict is None:
        rd = getattr(res0, "res", None)
        if isinstance(rd, dict):
            res_dict = {"res": rd}
    if res_dict is None and isinstance(res0, dict):
        res_dict = res0

    if not isinstance(res_dict, dict) or ("res" not in res_dict and "prunedResult" not in res_dict):
        core = res_dict if isinstance(res_dict, dict) else {}
    else:
        core = res_dict.get("res") or res_dict.get("prunedResult") or {}

    texts  = _to_list(core.get("rec_texts")) or []
    scores = _to_list(core.get("rec_scores")) or []
    polys  = _to_list(core.get("rec_polys")) or _to_list(core.get("dt_polys")) or []

    detections = []
    n = max(len(texts), len(scores), len(polys) if isinstance(polys, list) else 0)
    for i in range(n):
        text = str(texts[i]) if i < len(texts) else ""
        conf = float(scores[i]) if i < len(scores) else None
        poly = polys[i] if (isinstance(polys, list) and i < len(polys)) else None

        bbox = None
        if isinstance(poly, (list, tuple)) and len(poly) >= 4 and isinstance(poly[0], (list, tuple)):
            bbox = [[float(p[0]), float(p[1])] for p in poly[:4]]
        elif isinstance(poly, (list, tuple)) and len(poly) == 8:
            pts = list(map(float, poly))
            bbox = [[pts[0], pts[1]], [pts[2], pts[3]], [pts[4], pts[5]], [pts[6], pts[7]]]
        if bbox is None:
            continue
        detections.append((bbox, text, conf))
    return detections

# ---------- Image resolver (robust) ----------
def build_image_index(root_dir):
    idx_full = {}
    idx_base = {}
    for r, _dirs, files in os.walk(root_dir):
        for fn in files:
            name_lower = fn.lower()
            path = os.path.join(r, fn)
            idx_full[name_lower] = path
            base = os.path.splitext(name_lower)[0]
            idx_base.setdefault(base, path)
    return idx_full, idx_base

def resolve_image_path(file_name, idx_full, idx_base):
    if not file_name:
        return None, "empty file_name"
    name_lower = file_name.strip().lower()
    if name_lower in idx_full:
        return idx_full[name_lower], None
    base = os.path.splitext(name_lower)[0]
    if base in idx_base:
        return idx_base[base], None
    return None, "not found (case-insensitive, any extension)"

# ---------- OCR factory ----------
def build_ocr(lang: str):
    try:
        return PaddleOCR(device="cpu", lang=lang, use_textline_orientation=True)
    except TypeError:
        return PaddleOCR(device="cpu", lang=lang)

# ---------- Plain text -> JSON (rule-based for Schmuck) ----------
SCHMUCK_KEYS = [
    "Gegenstand", "Inv.Nr", "Herkunft", "Foto Notes", "Standort", "Material",
    "Datierung", "Maße", "Gewicht", "erworben von", "am", "Preis", "Vers.-Wert",
    "Beschreibung", "Literatur", "Ausstellungen"
]

KEY_PATTERNS = [
    (r"^\s*gegenstand\s*[:\-]?\s*(.*)$", "Gegenstand"),
    (r"^\s*inv\.?\s*nr\.?\s*[:\-]?\s*(.*)$", "Inv.Nr"),
    (r"^\s*inv\.?\s*nr\s*[:\-]?\s*(.*)$", "Inv.Nr"),
    (r"^\s*herkunft\s*[:\-]?\s*(.*)$", "Herkunft"),
    (r"^\s*foto\s*(notes|nr\.?)\s*[:\-]?\s*(.*)$", "Foto Notes"),
    (r"^\s*foto\s*nr\.?\s*[:\-]?\s*(.*)$", "Foto Notes"),
    (r"^\s*standort\s*[:\-]?\s*(.*)$", "Standort"),
    (r"^\s*material\s*[:\-]?\s*(.*)$", "Material"),
    (r"^\s*datierung\s*[:\-]?\s*(.*)$", "Datierung"),
    (r"^\s*maße\s*[:\-]?\s*(.*)$", "Maße"),
    (r"^\s*gewicht\s*[:\-]?\s*(.*)$", "Gewicht"),
    (r"^\s*erworben\s+von\s*[:\-]?\s*(.*)$", "erworben von"),
    (r"^\s*am\s*[:\-]?\s*(.*)$", "am"),
    (r"^\s*preis\s*[:\-]?\s*(.*)$", "Preis"),
    (r"^\s*vers\.\s*[-–—]?\s*wert\s*[:\-]?\s*(.*)$", "Vers.-Wert"),
    (r"^\s*vers\.?-?wert\s*[:\-]?\s*(.*)$", "Vers.-Wert"),
    (r"^\s*beschreibung\s*[:\-]?\s*(.*)$", "Beschreibung"),
    (r"^\s*literatur\s*[:\-]?\s*(.*)$", "Literatur"),
    (r"^\s*ausstellungen\s*[:\-]?\s*(.*)$", "Ausstellungen"),
]

def _clean_val(v: str) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v)
    return v

def parse_schmuck_text_to_json(ocr_text: str) -> dict:
    """
    Simple robust parser:
    - Detects field headers line-by-line
    - Captures value on same line (after ':') or in following lines until next header
    """
    out = OrderedDict((k, "") for k in SCHMUCK_KEYS)
    if not ocr_text:
        return out

    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    current_key = None

    def set_or_append(key, val):
        val = _clean_val(val)
        if not val:
            return
        if out[key]:
            # avoid duplicating identical fragments
            if val.lower() not in out[key].lower():
                out[key] = _clean_val(out[key] + " " + val)
        else:
            out[key] = val

    for ln in lines:
        low = ln.casefold()

        matched = False
        for pat, k in KEY_PATTERNS:
            m = re.match(pat, low, flags=re.IGNORECASE)
            if m:
                matched = True
                current_key = k

                # pattern variants: some have 2 capture groups
                if k == "Foto Notes" and m.lastindex == 2:
                    val = m.group(2)
                else:
                    val = m.group(m.lastindex) if m.lastindex else ""

                # if original line contains something after ":" use original slice when possible
                # (keeps commas/case a bit better than low)
                if ":" in ln:
                    val2 = ln.split(":", 1)[1]
                    val = val2 if val2.strip() else val

                set_or_append(current_key, val)
                break

        if matched:
            continue

        # continuation line for the current field
        if current_key:
            # stop appending if a new header appears inline (rare but happens)
            # e.g. "Inv. Nr.: ... Herkunft: ..."
            inline_hit = False
            for pat, k2 in KEY_PATTERNS:
                if re.search(pat.replace("^", "").replace("$", ""), low, flags=re.IGNORECASE):
                    inline_hit = True
                    break
            if not inline_hit:
                set_or_append(current_key, ln)

    # Post-fix common join issues:
    # If Material accidentally got split into later standalone tokens like "Email", "Zitrine" lines, we already append.
    return out

# ---------- main ----------
def main():
    # Force CPU
    try:
        paddle.device.set_device("cpu")
    except Exception:
        try:
            paddle.set_device("cpu")
        except Exception:
            pass
    print("Device set to CPU.")

    missing_log_lines = []
    error_log_lines = []

    ocr = build_ocr(LANG)

    idx_full, idx_base = build_image_index(IMAGES_DIR)
    print(f"Indexed {len(idx_full)} files under {IMAGES_DIR}")

    found, missing = 0, 0
    per_image = []

    tot_schema_edits = tot_schema_ref = 0
    tot_content_edits = tot_content_ref = 0
    total_lines = 0

    # write predictions jsonl progressively
    pred_f = open(PRED_JSONL, "w", encoding="utf-8")

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for rawline in f:
            total_lines += 1
            line = rawline.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                missing += 1
                msg = f"[JSON ERROR] line {total_lines}: {e}"
                print(msg)
                missing_log_lines.append(msg)
                continue

            file_name = (obj.get("file_name") or "").strip()
            img_path, why = resolve_image_path(file_name, idx_full, idx_base)
            if not img_path:
                missing += 1
                note = f"{file_name or '<EMPTY>'}: {why}"
                print(f"[MISSING] {note}")
                missing_log_lines.append(note)
                continue

            found += 1

            # ---- Ground truth JSON (exclude file_name) ----
            gt_json = {k: v for k, v in obj.items() if k != "file_name"}
            gt_json_str = json_to_string_stable(gt_json, SCHMUCK_KEYS)

            # ---- OCR ----
            try:
                raw = ocr.predict(img_path)
                detections = paddle_predict_to_detections(raw)
            except Exception as e:
                missing += 1
                note = f"[OCR ERROR] {file_name}: {e}"
                print(note)
                error_log_lines.append(note)
                continue

            hyp_text = result_to_text(detections, min_conf=MIN_CONF, line_tol=LINE_TOL)

            base = os.path.splitext(os.path.basename(img_path))[0]

            # save OCR plain text
            ocr_out = os.path.join(OUTPUT_DIR, f"{base}_paddleocr.txt")
            with open(ocr_out, "w", encoding="utf-8") as focr:
                focr.write(hyp_text)

            # ---- Parse OCR text -> predicted JSON ----
            pred_json = parse_schmuck_text_to_json(hyp_text)
            pred_json_str = json_to_string_stable(pred_json, SCHMUCK_KEYS)

            # ---- CER on JSON vs JSON ----
            cer_schema, edits_schema, ref_schema = cer(pred_json_str, gt_json_str, ignore_punct=False)
            cer_content, edits_content, ref_content = cer(pred_json_str, gt_json_str, ignore_punct=True)

            per_image.append((
                base,
                cer_schema, edits_schema, ref_schema,
                cer_content, edits_content, ref_content,
                ocr_out
            ))
            tot_schema_edits += edits_schema; tot_schema_ref += ref_schema
            tot_content_edits += edits_content; tot_content_ref += ref_content

            # save prediction record
            rec = {
                "file_name": file_name,
                "image_base": base,
                "matched_image_path": img_path,
                "ocr_text": hyp_text,
                "predicted_json": pred_json,
                "ground_truth_json": gt_json,
                "predicted_json_string": pred_json_str,
                "ground_truth_json_string": gt_json_str,
                "cer_schema_json": cer_schema,
                "cer_content_json": cer_content,
            }
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(
                f"{base}: "
                f"CER_schema_JSON={cer_schema:.4f} (edits={edits_schema}/{ref_schema})  "
                f"CER_content_JSON={cer_content:.4f} (edits={edits_content}/{ref_content})"
            )

    pred_f.close()

    # save metrics
    summary = os.path.join(OUTPUT_DIR, "metrics_paddleocr.txt")
    with open(summary, "w", encoding="utf-8") as f:
        for (base,
             cer_schema, edits_schema, ref_schema,
             cer_content, edits_content, ref_content,
             ocr_out) in per_image:
            f.write(
                f"{base}: "
                f"CER_schema_JSON={cer_schema:.4f} (char_edits={edits_schema}, char_ref={ref_schema}) | "
                f"CER_content_JSON={cer_content:.4f} (char_edits={edits_content}, char_ref={ref_content}) | "
                f"ocr_file={ocr_out}\n"
            )
        if per_image:
            macro_schema = float(np.mean([r[1] for r in per_image]))
            macro_content = float(np.mean([r[4] for r in per_image]))
            micro_schema = tot_schema_edits / max(1, tot_schema_ref)
            micro_content = tot_content_edits / max(1, tot_content_ref)

            f.write("\nAverages:\n")
            f.write(f"Macro CER (schema JSON): {macro_schema:.4f}\n")
            f.write(f"Micro CER (schema JSON): {micro_schema:.4f}\n")
            f.write(f"Macro CER (content JSON): {macro_content:.4f}\n")
            f.write(f"Micro CER (content JSON): {micro_content:.4f}\n")

        f.write(f"\nJSONL lines read: {total_lines}\n")
        f.write(f"Images found: {found}\nImages missing/skipped: {missing}\n")
        f.write(f"Predictions JSONL: {PRED_JSONL}\n")

    if missing_log_lines:
        with open(MISSING_LOG, "w", encoding="utf-8") as mf:
            mf.write("\n".join(missing_log_lines))
    if error_log_lines:
        with open(ERROR_LOG, "w", encoding="utf-8") as ef:
            ef.write("\n".join(error_log_lines))

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")
    print(f"- Summary: {summary}")
    print(f"- Predictions: {PRED_JSONL}")
    if missing_log_lines:
        print(f"- Missing list: {MISSING_LOG}  ({len(missing_log_lines)} items)")
    if error_log_lines:
        print(f"- OCR errors:   {ERROR_LOG}  ({len(error_log_lines)} items)")
    print(f"JSONL lines read: {total_lines}")
    print(f"Images found: {found}")
    print(f"Images missing/skipped: {missing}")

if __name__ == "__main__":
    main()
