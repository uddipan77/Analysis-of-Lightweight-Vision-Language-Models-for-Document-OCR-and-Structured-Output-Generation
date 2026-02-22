import os
import json
import re
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Tuple, List
from PIL import Image
import pytesseract

# ---------------- Tesseract setup ----------------
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

LANG = "deu"
CONFIG = r"--oem 1 --psm 6"

# ---------------- Paths (Inventory dataset) ----------------
IMAGES_DIR = r"C:\FAU\sem6\Thesis\inventarkarten-ocr\inventarkarten\erlangen"
JSONL_PATH = r"C:\FAU\sem6\Thesis\inventarkarten-ocr\gemini3\test.jsonl"

# output folder in CWD
BASE_OUTPUT_DIR = os.path.join(os.getcwd(), "new_inventory_tesseract")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, "run_{0}".format(datetime.now().strftime("%Y%m%d_%H%M%S")))
os.makedirs(RUN_DIR, exist_ok=True)

# ---------------- Schema ----------------
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
    """Ensure schema + stable types + whitespace cleanup."""
    out = empty_schema()
    if not isinstance(obj, dict):
        return out

    for k in SCHEMA_KEYS:
        if k == "Maße":
            meas = obj.get("Maße", {})
            if isinstance(meas, dict):
                for mk in MEAS_KEYS:
                    v = meas.get(mk, "")
                    out["Maße"][mk] = "" if v is None else str(v).strip()
        else:
            v = obj.get(k, "")
            out[k] = "" if v is None else str(v).strip()

    # collapse whitespace everywhere
    for k in ["Überschrift", "Inventarnummer", "Objektbezeichnung", "Fundort", "Fundzeit", "Beschreibungstext"]:
        out[k] = re.sub(r"\s+", " ", out[k]).strip()
    for mk in MEAS_KEYS:
        out["Maße"][mk] = re.sub(r"\s+", " ", out["Maße"][mk]).strip()

    return out

# ---------------- JSON helpers ----------------
def canonical_json_string(obj: dict) -> str:
    """Stable JSON string for CER (sorted keys so order doesn't affect CER)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

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

# ---------------- Text cleanup helpers ----------------
def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)

def _decimal_to_comma(s: str) -> str:
    return re.sub(r"(?<=\d)\.(?=\d)", ",", s)

def _clean_lines(ocr_text: str) -> List[str]:
    text = _nfkc(ocr_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [ln.strip() for ln in text.split("\n")]
    raw_lines = [ln for ln in raw_lines if ln]

    # de-hyphenate across line breaks (word- \n continuation)
    lines: List[str] = []
    i = 0
    while i < len(raw_lines):
        ln = re.sub(r"\s+", " ", raw_lines[i]).strip()
        if ln.endswith("-") and i + 1 < len(raw_lines):
            nxt = re.sub(r"\s+", " ", raw_lines[i + 1]).strip()
            if nxt and re.match(r"^[a-zäöüß]", nxt):
                ln = ln[:-1] + nxt
                i += 2
                lines.append(_decimal_to_comma(ln.strip()))
                continue
        lines.append(_decimal_to_comma(ln))
        i += 1

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in lines if ln.strip()]
    return lines

def _is_paragraph_like(line: str) -> bool:
    if len(line) >= 75:
        return True
    if len(line.split()) >= 12 and len(line) >= 55:
        return True
    if sum(line.count(ch) for ch in [".", "!", "?"]) >= 1 and len(line) >= 60:
        return True
    return False

def _is_noise_line(ln: str) -> bool:
    # drop lines that are basically symbols (like "%", "®", "{")
    alnum = sum(ch.isalnum() for ch in ln)
    if alnum == 0 and len(ln.strip()) <= 3:
        return True
    if len(ln.strip()) <= 1:
        return True
    # lines that are mostly symbols
    sym_ratio = len(re.sub(r"[A-Za-zÄÖÜäöüß0-9\s]", "", ln)) / max(1, len(ln))
    if sym_ratio > 0.85 and alnum == 0:
        return True
    return False

# ---------------- Measurements extraction (tolerant) ----------------
def _looks_like_measurement_line(line: str) -> bool:
    # tolerate OCR: '£' often becomes 'L'
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

    # Pass 1: labelled L/£, B, D/T
    for i, ln in enumerate(lines):
        if not _looks_like_measurement_line(ln):
            continue
        t = ln

        # L: L or £
        m = re.search(r"[£L]\s*[:=]?\s*(\d+(?:,\d+)?)", t)
        if m:
            assign_if_empty("L", m.group(1), i)

        # B: B (possibly with cm)
        m = re.search(r"\bB\s*[:=]?\s*(\d+(?:,\d+)?)(\s*cm)?", t, flags=re.IGNORECASE)
        if m:
            val = m.group(1) + ("cm" if m.group(2) else "")
            assign_if_empty("B", val, i)

        # D: D/T
        m = re.search(r"\b(D|T)\s*[:=]?\s*(\d+(?:,\d+)?)", t, flags=re.IGNORECASE)
        if m:
            assign_if_empty("D", m.group(2), i)

    # Pass 2: x-separated triple
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

# ---------------- Inventory-number helpers ----------------
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

    # Penalize mostly symbols
    sym_ratio = len(re.sub(r"[A-Za-zÄÖÜäöüß0-9\s=/\.\-]", "", t)) / max(1, len(t))
    if sym_ratio > 0.6:
        score -= 10

    return score

def _inv_token_candidates(text: str) -> List[str]:
    """
    Extract inventory-like tokens INSIDE a line.
    This handles cases where OCR merges multiple fields into one line.
    """
    t = text.strip()
    t2 = t.replace(" ", "")

    patt = re.compile(r"[A-Za-zÄÖÜäöüß]{1,3}=?[A-Za-zÄÖÜäöüß]{0,4}\.?[A-Za-zÄÖÜäöüß]{0,3}\d{1,4}")
    hits = patt.findall(t2)

    if not hits:
        patt2 = re.compile(r"[A-Za-zÄÖÜäöüß0-9=/\.\-]{4,25}")
        for tok in patt2.findall(t2):
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
    score = 0
    if not tok:
        return -999
    if not (re.search(r"[A-Za-zÄÖÜäöüß]", tok) and re.search(r"\d", tok)):
        return -999
    if "=" in tok:
        score += 15
    if "." in tok:
        score += 6
    if "/" in tok:
        score += 4
    if 6 <= len(tok) <= 14:
        score += 10
    elif 4 <= len(tok) <= 20:
        score += 5
    else:
        score -= 3
    weird = re.sub(r"[A-Za-zÄÖÜäöüß0-9=/\.\-]", "", tok)
    score -= 5 * len(weird)
    return score

def _extract_inventory_number(lines: List[str], search_upto: int = 10) -> Tuple[str, Optional[int]]:
    """
    Prefer token extraction from early lines; fallback to bottom line scoring.
    Returns (inventory_number, line_idx).
    """
    # A) token-based from early lines
    best_tok, best_idx, best_sc = "", None, -10**9
    upto = min(len(lines), max(1, search_upto))
    for i in range(upto):
        ln = lines[i]
        for tok in _inv_token_candidates(ln):
            sc = _score_inv_token(tok) + (upto - i)  # prefer earlier
            if sc > best_sc:
                best_sc, best_tok, best_idx = sc, tok, i
    if best_sc >= 10:
        return best_tok.strip(), best_idx

    # B) fallback: line-based near bottom
    best_idx2, best_sc2 = None, -10**9
    for i in range(max(0, len(lines) - 10), len(lines)):
        sc = _inventory_line_score(lines[i]) + i  # prefer lower
        if sc > best_sc2:
            best_sc2, best_idx2 = sc, i
    if best_idx2 is not None and best_sc2 >= 8:
        return lines[best_idx2].strip(), best_idx2

    return "", None

# ---------------- Marker extraction ----------------
def _find_marker_idx(lines: List[str], marker: str) -> Optional[int]:
    m = marker.casefold()
    for i, ln in enumerate(lines):
        if m in ln.casefold():
            return i
    return None

def _extract_after_marker_until(line: str, marker: str, stop_markers: List[str]) -> str:
    """
    Extract substring after marker. If any stop_marker appears after it, cut before stop_marker.
    """
    low = line.casefold()
    mlow = marker.casefold()
    idx = low.find(mlow)
    if idx == -1:
        return ""
    after = line[idx + len(marker):]

    # Cut at stop markers if present
    after_low = after.casefold()
    cut_pos = None
    for sm in stop_markers:
        pos = after_low.find(sm.casefold())
        if pos != -1:
            if cut_pos is None or pos < cut_pos:
                cut_pos = pos
    if cut_pos is not None:
        after = after[:cut_pos]

    after = re.sub(r"^[\s:;\-–—]+", "", after).strip()
    after = re.sub(r"\s+", " ", after).strip()
    return after

# ---------------- Final parser (fix: no "everything in Beschreibungstext") ----------------
def parse_inventory_text_to_json(ocr_text: str) -> dict:
    """
    Inventory-card parsing where keys are NOT required.
    If markers 'Fundort'/'Fundzeit' appear in OCR, use them as separators.
    Beschreibungstext = all non-consumed lines (never "everything by default").
    """
    out = empty_schema()

    lines = _clean_lines(ocr_text)
    lines = [ln for ln in lines if not _is_noise_line(ln)]
    if not lines:
        return out

    consumed = set()

    # 1) Überschrift: first line as heading (most cards follow this strongly)
    out["Überschrift"] = lines[0].strip()
    consumed.add(0)

    # 2) Inventarnummer: token extraction early, fallback bottom line
    inv, inv_idx = _extract_inventory_number(lines, search_upto=10)
    if inv:
        out["Inventarnummer"] = inv
        if inv_idx is not None:
            consumed.add(inv_idx)

    # 3) Maße: run on all lines, consume those used
    meas, meas_used = _extract_measurements(lines)
    out["Maße"] = meas
    for idx in meas_used:
        consumed.add(idx)

    # 4) Fundort & Fundzeit using markers if present
    fundort_idx = _find_marker_idx(lines, "Fundort")
    fundzeit_idx = _find_marker_idx(lines, "Fundzeit")

    if fundort_idx is not None:
        # stop at Fundzeit if it appears on the same line after Fundort
        out["Fundort"] = _extract_after_marker_until(
            lines[fundort_idx], "Fundort", stop_markers=["Fundzeit"]
        )
        consumed.add(fundort_idx)

    if fundzeit_idx is not None:
        out["Fundzeit"] = _extract_after_marker_until(
            lines[fundzeit_idx], "Fundzeit", stop_markers=["Fundort"]
        )
        consumed.add(fundzeit_idx)

    # 5) Objektbezeichnung heuristic:
    # choose the first non-consumed, non-paragraph-like line after (Fundzeit or Fundort or heading)
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
        # don't take very paragraph-like stuff as Objektbezeichnung
        if _is_paragraph_like(ln):
            break
        objekt_parts.append(ln)
        consumed.add(i)
        # 1-2 lines are enough
        if len(objekt_parts) >= 2:
            break

    out["Objektbezeichnung"] = re.sub(r"\s+", " ", " ".join(objekt_parts)).strip()

    # 6) Beschreibungstext = everything not consumed
    remaining = [lines[i] for i in range(len(lines)) if i not in consumed]
    out["Beschreibungstext"] = re.sub(r"\s+", " ", " ".join(remaining)).strip()

    return normalize_to_schema(out)

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

            file_name = (obj.get("image_name") or "").strip()
            if not file_name:
                missing += 1
                continue

            img_path = resolve_image_path(IMAGES_DIR, file_name)
            if not img_path:
                missing += 1
                gt_json = normalize_to_schema({k: v for k, v in obj.items() if k != "image_name"})
                gt_json_str = canonical_json_string(gt_json)

                rec = {
                    "image_name": file_name,
                    "matched_image_path": None,
                    "ocr_text": "",
                    "predicted_json": empty_schema(),
                    "ground_truth_json": gt_json,
                    "pred_json_str": canonical_json_string(empty_schema()),
                    "gt_json_str": gt_json_str,
                    "cer_json": 1.0,
                    "error": "image_not_found",
                }
                records.append(rec)
                continue

            found += 1

            gt_json = normalize_to_schema({k: v for k, v in obj.items() if k != "image_name"})
            gt_json_str = canonical_json_string(gt_json)

            try:
                ocr_text = ocr_tesseract(img_path)
            except Exception as e:
                missing += 1
                rec = {
                    "image_name": file_name,
                    "matched_image_path": img_path,
                    "ocr_text": "",
                    "predicted_json": empty_schema(),
                    "ground_truth_json": gt_json,
                    "pred_json_str": canonical_json_string(empty_schema()),
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

            pred_json = parse_inventory_text_to_json(ocr_text)
            pred_json_str = canonical_json_string(pred_json)

            cer_val, edits, ref_len = cer_raw(pred_json_str, gt_json_str)
            total_edits += edits
            total_ref_chars += ref_len

            pred_json_path = os.path.join(RUN_DIR, "{0}_pred.json".format(base))
            with open(pred_json_path, "w", encoding="utf-8") as fj:
                json.dump(pred_json, fj, ensure_ascii=False, indent=2)

            rec = {
                "image_name": file_name,
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

    # write predictions.jsonl
    with open(preds_jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # macro/micro CER
    cer_list = [r.get("cer_json") for r in records if isinstance(r.get("cer_json"), (int, float))]
    valid = [c for c in cer_list if c is not None]
    macro = (sum(valid) / len(valid)) if valid else 1.0
    micro = (total_edits / max(1, total_ref_chars))

    summary_path = os.path.join(RUN_DIR, "metrics_tesseract_inventory_json_vs_json.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Tesseract Inventory — JSON vs JSON CER (keys NOT required)\n")
        f.write("=" * 70 + "\n")
        f.write("JSONL: {0}\n".format(JSONL_PATH))
        f.write("Images: {0}\n".format(IMAGES_DIR))
        f.write("Run dir: {0}\n\n".format(RUN_DIR))
        f.write("Macro CER_json: {0:.6f}\n".format(macro))
        f.write("Micro CER_json: {0:.6f}\n".format(micro))
        f.write("Total edits: {0}\n".format(total_edits))
        f.write("Total ref chars: {0}\n".format(total_ref_chars))
        f.write("\nImages found: {0}\n".format(found))
        f.write("Missing/skipped: {0}\n".format(missing))
        f.write("Predictions file: {0}\n".format(preds_jsonl_path))

    # requested extra file: only macro/micro
    cer_only_path = os.path.join(RUN_DIR, "cer_macro_micro.txt")
    with open(cer_only_path, "w", encoding="utf-8") as f:
        f.write("Macro CER_json: {0:.6f}\n".format(macro))
        f.write("Micro CER_json: {0:.6f}\n".format(micro))
        f.write("Total edits: {0}\n".format(total_edits))
        f.write("Total ref chars: {0}\n".format(total_ref_chars))
        f.write("Images found: {0}\n".format(found))
        f.write("Missing/skipped: {0}\n".format(missing))

    print("\nDone.")
    print("Run dir: {0}".format(RUN_DIR))
    print("Predictions: {0}".format(preds_jsonl_path))
    print("Summary: {0}".format(summary_path))
    print("CER only: {0}".format(cer_only_path))
    print("Images found: {0}".format(found))
    print("Images missing/skipped: {0}".format(missing))

if __name__ == "__main__":
    main()
