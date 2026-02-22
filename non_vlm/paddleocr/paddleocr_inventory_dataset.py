import os
import sys
import re
import unicodedata
import json
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

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

# ---------- config (Inventory dataset) ----------
IMAGES_DIR = r"C:\FAU\sem6\Thesis\inventarkarten-ocr\inventarkarten\erlangen"
JSONL_PATH = r"C:\FAU\sem6\Thesis\inventarkarten-ocr\gemini3\test.jsonl"

LANG = "german"
MIN_CONF = 0.0
LINE_TOL = 0.7

# ---------- output ----------
# output can be saved in a new folder named new_inventory_paddleocr (in CWD)
OUTPUT_DIR = os.path.join(os.getcwd(), "new_inventory_paddleocr")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MISSING_LOG = os.path.join(OUTPUT_DIR, "missing_images.txt")
ERROR_LOG   = os.path.join(OUTPUT_DIR, "ocr_errors.txt")
PRED_JSONL  = os.path.join(OUTPUT_DIR, "predictions_paddleocr_inventory.jsonl")

# ---------- Inventory schema ----------
# New schema keys matching ground truth structure
SCHEMA_KEYS = ["Überschrift", "Inventarnummer", "Maße", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]
MEAS_KEYS = ["L", "B", "D"]

def empty_schema() -> dict:
    return {
        "Überschrift": "",
        "Inventarnummer": "",
        "Maße": {"L": "", "B": "", "D": ""},
        "Objektbezeichnung": "",
        "Fundort": "",
        "Fundzeit": "",
        "Beschreibungstext": "",
    }

def normalize_to_schema(obj: dict) -> dict:
    out = empty_schema()
    if not isinstance(obj, dict):
        return out

    # top-level
    for k in ["Überschrift", "Inventarnummer", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]:
        v = obj.get(k, "")
        out[k] = "" if v is None else str(v).strip()

    # Maße
    meas = obj.get("Maße", {})
    if isinstance(meas, dict):
        for mk in MEAS_KEYS:
            v = meas.get(mk, "")
            out["Maße"][mk] = "" if v is None else str(v).strip()

    # normalize whitespace
    for k in ["Überschrift", "Inventarnummer", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]:
        out[k] = re.sub(r"\s+", " ", out[k]).strip()
    for mk in MEAS_KEYS:
        out["Maße"][mk] = re.sub(r"\s+", " ", out["Maße"][mk]).strip()

    return out

def canonical_json_string(obj: dict) -> str:
    """Stable JSON string for CER (sorted keys; stable separators)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

# ---------- CER (char) ----------
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
        return None, "empty image_name"
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

# ---------- Inventory parsing (same as Tesseract logic; keys NOT required) ----------
def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _decimal_to_comma(s: str) -> str:
    return re.sub(r"(?<=\d)\.(?=\d)", ",", s)

def _clean_lines_inventory(text: str) -> List[str]:
    text = _nfkc(text).replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [ln.strip() for ln in text.split("\n")]
    raw_lines = [ln for ln in raw_lines if ln]

    # de-hyphenate across lines
    out = []
    i = 0
    while i < len(raw_lines):
        ln = re.sub(r"\s+", " ", raw_lines[i]).strip()
        if ln.endswith("-") and i + 1 < len(raw_lines):
            nxt = re.sub(r"\s+", " ", raw_lines[i + 1]).strip()
            if nxt and re.match(r"^[a-zäöüß]", nxt):
                ln = ln[:-1] + nxt
                i += 2
                out.append(_decimal_to_comma(ln.strip()))
                continue
        out.append(_decimal_to_comma(ln))
        i += 1

    out = [re.sub(r"\s+", " ", ln).strip() for ln in out if ln.strip()]
    # drop noise lines
    out2 = []
    for ln in out:
        alnum = sum(ch.isalnum() for ch in ln)
        if alnum == 0 and len(ln.strip()) <= 3:
            continue
        if len(ln.strip()) <= 1:
            continue
        sym_ratio = len(re.sub(r"[A-Za-zÄÖÜäöüß0-9\s]", "", ln)) / max(1, len(ln))
        if sym_ratio > 0.85 and alnum == 0:
            continue
        out2.append(ln)
    return out2

def _is_paragraph_like(line: str) -> bool:
    if len(line) >= 75:
        return True
    if len(line.split()) >= 12 and len(line) >= 55:
        return True
    if sum(line.count(ch) for ch in [".", "!", "?"]) >= 1 and len(line) >= 60:
        return True
    return False

def _looks_like_measurement_line(line: str) -> bool:
    if re.search(r"[£L]\s*\d", line):
        return True
    if re.search(r"\bB\s*\d", line):
        return True
    if re.search(r"\b(D|T)\s*\d", line):
        return True
    if re.search(r"\d+(?:,\d+)?\s*[xX×]\s*\d+(?:,\d+)?\s*[xX×]\s*\d+(?:,\d+)?", line):
        return True
    if "cm" in line.lower() and re.search(r"\d", line):
        return True
    if len(re.findall(r"\d+,\d+", line)) >= 2:
        return True
    return False

def _extract_measurements(lines: List[str]) -> Tuple[Dict[str, str], List[int]]:
    meas = {"L": "", "B": "", "D": ""}
    used: List[int] = []

    def assign_if_empty(key: str, val: str, idx: int):
        val = val.strip()
        if not val:
            return
        if not meas[key]:
            meas[key] = val
            used.append(idx)

    for i, ln in enumerate(lines):
        if not _looks_like_measurement_line(ln):
            continue

        m = re.search(r"[£L]\s*[:=]?\s*(\d+(?:,\d+)?)", ln)
        if m:
            assign_if_empty("L", m.group(1), i)

        m = re.search(r"\bB\s*[:=]?\s*(\d+(?:,\d+)?)(\s*cm)?", ln, flags=re.IGNORECASE)
        if m:
            val = m.group(1) + ("cm" if m.group(2) else "")
            assign_if_empty("B", val, i)

        m = re.search(r"\b(D|T)\s*[:=]?\s*(\d+(?:,\d+)?)", ln, flags=re.IGNORECASE)
        if m:
            assign_if_empty("D", m.group(2), i)

    if not (meas["L"] and meas["B"] and meas["D"]):
        for i, ln in enumerate(lines):
            m = re.search(r"(\d+(?:,\d+)?)\s*[xX×]\s*(\d+(?:,\d+)?)\s*[xX×]\s*(\d+(?:,\d+)?)", ln)
            if m:
                if not meas["L"]:
                    meas["L"] = m.group(1)
                if not meas["B"]:
                    meas["B"] = m.group(2)
                if not meas["D"]:
                    meas["D"] = m.group(3)
                used.append(i)
                break

    for k in meas:
        meas[k] = re.sub(r"\s+", " ", meas[k]).strip()
    return meas, sorted(set(used))

def _inv_token_candidates(text: str) -> List[str]:
    t = text.strip().replace(" ", "")
    patt = re.compile(r"[A-Za-zÄÖÜäöüß]{1,3}=?[A-Za-zÄÖÜäöüß]{0,4}\.?[A-Za-zÄÖÜäöüß]{0,3}\d{1,4}")
    hits = patt.findall(t)
    if not hits:
        patt2 = re.compile(r"[A-Za-zÄÖÜäöüß0-9=/\.\-]{4,25}")
        for tok in patt2.findall(t):
            if re.search(r"[A-Za-zÄÖÜäöüß]", tok) and re.search(r"\d", tok):
                hits.append(tok)
    out = []
    seen = set()
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out

def _score_inv_token(tok: str) -> int:
    if not tok:
        return -999
    if not (re.search(r"[A-Za-zÄÖÜäöüß]", tok) and re.search(r"\d", tok)):
        return -999
    score = 0
    if "=" in tok: score += 15
    if "." in tok: score += 6
    if "/" in tok: score += 4
    if 6 <= len(tok) <= 14: score += 10
    elif 4 <= len(tok) <= 20: score += 5
    else: score -= 3
    weird = re.sub(r"[A-Za-zÄÖÜäöüß0-9=/\.\-]", "", tok)
    score -= 5 * len(weird)
    return score

def _inventory_line_score(ln: str) -> int:
    t = ln.strip()
    if len(t) < 3 or len(t) > 35:
        return -999
    has_digit = bool(re.search(r"\d", t))
    has_letter = bool(re.search(r"[A-Za-zÄÖÜäöüß]", t))
    score = 0
    if has_digit: score += 10
    if has_letter: score += 5
    if "=" in t: score += 20
    if "." in t: score += 6
    if "/" in t: score += 6
    score -= t.count(" ") * 2
    if len(t.split()) >= 10:
        score -= 10
    if "fundort" in t.casefold() or "fundzeit" in t.casefold():
        score -= 10
    sym_ratio = len(re.sub(r"[A-Za-zÄÖÜäöüß0-9\s=/\.\-]", "", t)) / max(1, len(t))
    if sym_ratio > 0.6:
        score -= 10
    return score

def _extract_inventory_number(lines: List[str], search_upto: int = 10) -> Tuple[str, Optional[int]]:
    best_tok, best_idx, best_sc = "", None, -10**9
    upto = min(len(lines), max(1, search_upto))
    for i in range(upto):
        for tok in _inv_token_candidates(lines[i]):
            sc = _score_inv_token(tok) + (upto - i)
            if sc > best_sc:
                best_sc, best_tok, best_idx = sc, tok, i
    if best_sc >= 10:
        return best_tok.strip(), best_idx

    best_idx2, best_sc2 = None, -10**9
    for i in range(max(0, len(lines) - 10), len(lines)):
        sc = _inventory_line_score(lines[i]) + i
        if sc > best_sc2:
            best_sc2, best_idx2 = sc, i
    if best_idx2 is not None and best_sc2 >= 8:
        return lines[best_idx2].strip(), best_idx2

    return "", None

def _find_marker_idx(lines: List[str], marker: str) -> Optional[int]:
    m = marker.casefold()
    for i, ln in enumerate(lines):
        if m in ln.casefold():
            return i
    return None

def _extract_after_marker_until(line: str, marker: str, stop_markers: List[str]) -> str:
    low = line.casefold()
    idx = low.find(marker.casefold())
    if idx == -1:
        return ""
    after = line[idx + len(marker):]
    after_low = after.casefold()

    cut_pos = None
    for sm in stop_markers:
        pos = after_low.find(sm.casefold())
        if pos != -1:
            cut_pos = pos if cut_pos is None else min(cut_pos, pos)
    if cut_pos is not None:
        after = after[:cut_pos]

    after = re.sub(r"^[\s:;\-–—]+", "", after).strip()
    after = re.sub(r"\s+", " ", after).strip()
    return after

def parse_inventory_text_to_json(ocr_text: str) -> dict:
    """
    Inventory parsing:
      - Überschrift = first line
      - Inventarnummer = token from early lines, fallback to bottom candidate
      - Maße = tolerant patterns
      - Fundort/Fundzeit markers used if present
      - Objektbezeichnung = 1-2 lines after fundzeit/fundort/heading
      - Beschreibungstext = remaining non-consumed lines (prevents "everything into Beschreibungstext")
    """
    out = empty_schema()
    lines = _clean_lines_inventory(ocr_text)
    if not lines:
        return out

    consumed = set()

    # Überschrift
    out["Überschrift"] = lines[0].strip()
    consumed.add(0)

    # Inventarnummer
    inv, inv_idx = _extract_inventory_number(lines, search_upto=10)
    if inv:
        out["Inventarnummer"] = inv
        if inv_idx is not None:
            consumed.add(inv_idx)

    # Maße
    meas, meas_used = _extract_measurements(lines)
    out["Maße"] = meas
    for idx in meas_used:
        consumed.add(idx)

    # markers
    fundort_idx = _find_marker_idx(lines, "Fundort")
    fundzeit_idx = _find_marker_idx(lines, "Fundzeit")

    if fundort_idx is not None:
        out["Fundort"] = _extract_after_marker_until(lines[fundort_idx], "Fundort", stop_markers=["Fundzeit"])
        consumed.add(fundort_idx)

    if fundzeit_idx is not None:
        out["Fundzeit"] = _extract_after_marker_until(lines[fundzeit_idx], "Fundzeit", stop_markers=["Fundort"])
        consumed.add(fundzeit_idx)

    # Objektbezeichnung
    start_idx = 1
    if fundzeit_idx is not None:
        start_idx = fundzeit_idx + 1
    elif fundort_idx is not None:
        start_idx = fundort_idx + 1

    objekt_parts = []
    for i in range(start_idx, min(len(lines), start_idx + 6)):
        if i in consumed:
            continue
        ln = lines[i].strip()
        if not ln:
            continue
        if _is_paragraph_like(ln):
            break
        objekt_parts.append(ln)
        consumed.add(i)
        if len(objekt_parts) >= 2:
            break
    out["Objektbezeichnung"] = re.sub(r"\s+", " ", " ".join(objekt_parts)).strip()

    # Beschreibungstext = remaining
    remaining = [lines[i] for i in range(len(lines)) if i not in consumed]
    out["Beschreibungstext"] = re.sub(r"\s+", " ", " ".join(remaining)).strip()

    return normalize_to_schema(out)

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

            # IMPORTANT: discard image_name during CER calculation
            image_name = (obj.get("image_name") or "").strip()
            img_path, why = resolve_image_path(image_name, idx_full, idx_base)
            if not img_path:
                missing += 1
                note = f"{image_name or '<EMPTY>'}: {why}"
                print(f"[MISSING] {note}")
                missing_log_lines.append(note)
                continue

            found += 1

            # ---- Ground truth JSON (exclude image_name) ----
            gt_json = normalize_to_schema({k: v for k, v in obj.items() if k != "image_name"})
            gt_json_str = canonical_json_string(gt_json)

            # ---- OCR ----
            try:
                raw = ocr.predict(img_path)
                detections = paddle_predict_to_detections(raw)
            except Exception as e:
                missing += 1
                note = f"[OCR ERROR] {image_name}: {e}"
                print(note)
                error_log_lines.append(note)
                continue

            hyp_text = result_to_text(detections, min_conf=MIN_CONF, line_tol=LINE_TOL)
            base = os.path.splitext(os.path.basename(img_path))[0]

            # save OCR text
            ocr_out = os.path.join(OUTPUT_DIR, f"{base}_paddleocr.txt")
            with open(ocr_out, "w", encoding="utf-8") as focr:
                focr.write(hyp_text)

            # ---- Parse OCR text -> predicted JSON ----
            pred_json = parse_inventory_text_to_json(hyp_text)
            pred_json_str = canonical_json_string(pred_json)

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

            rec = {
                "image_name": image_name,
                "image_base": base,
                "matched_image_path": img_path,
                "ocr_text_file": ocr_out,
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

    # ---------- save metrics ----------
    summary = os.path.join(OUTPUT_DIR, "metrics_paddleocr_inventory.txt")
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
            f.write(f"Macro CER (schema JSON): {macro_schema:.6f}\n")
            f.write(f"Micro CER (schema JSON): {micro_schema:.6f}\n")
            f.write(f"Macro CER (content JSON): {macro_content:.6f}\n")
            f.write(f"Micro CER (content JSON): {micro_content:.6f}\n")

        f.write(f"\nJSONL lines read: {total_lines}\n")
        f.write(f"Images found: {found}\nImages missing/skipped: {missing}\n")
        f.write(f"Predictions JSONL: {PRED_JSONL}\n")

    # requested extra file: macro/micro CER only (schema JSON)
    cer_only = os.path.join(OUTPUT_DIR, "cer_macro_micro.txt")
    if per_image:
        macro_schema = float(np.mean([r[1] for r in per_image]))
        micro_schema = tot_schema_edits / max(1, tot_schema_ref)
    else:
        macro_schema = 1.0
        micro_schema = 1.0
    with open(cer_only, "w", encoding="utf-8") as f:
        f.write(f"Macro CER_json: {macro_schema:.6f}\n")
        f.write(f"Micro CER_json: {micro_schema:.6f}\n")
        f.write(f"Images found: {found}\n")
        f.write(f"Images missing/skipped: {missing}\n")

    if missing_log_lines:
        with open(MISSING_LOG, "w", encoding="utf-8") as mf:
            mf.write("\n".join(missing_log_lines))
    if error_log_lines:
        with open(ERROR_LOG, "w", encoding="utf-8") as ef:
            ef.write("\n".join(error_log_lines))

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")
    print(f"- Summary: {summary}")
    print(f"- Predictions: {PRED_JSONL}")
    print(f"- CER macro/micro: {cer_only}")
    if missing_log_lines:
        print(f"- Missing list: {MISSING_LOG}  ({len(missing_log_lines)} items)")
    if error_log_lines:
        print(f"- OCR errors:   {ERROR_LOG}  ({len(error_log_lines)} items)")
    print(f"JSONL lines read: {total_lines}")
    print(f"Images found: {found}")
    print(f"Images missing/skipped: {missing}")

if __name__ == "__main__":
    main()
