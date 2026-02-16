#!/usr/bin/env python
"""
Compute mAP-style precision / recall / F1 using per-field CER overlap.

Evaluation policy:
  - GT string is always the source of truth.
  - If both pred JSON and GT JSON are available → per-field CER (dict vs dict).
  - If pred JSON is not available → all non-empty GT fields count as FN
    (the model is penalised for not producing valid JSON).
  - Reports JSON validity rate.
"""
import json
import os
from typing import Dict, Any, Optional, List

from jiwer import cer  # pip install jiwer


# ================== USER CONFIG ==================
INPUT_JSONL_PATH = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/inventory_dataset/run_A100_genCER_20260215_095759/test_predictions.jsonl"
OUTPUT_PATH = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/inventory_dataset/run_A100_genCER_20260215_095759/map_f1_metrics.json"

# Overlap thresholds for mAP-style CER (1 - CER >= threshold)
OVERLAP_THRESHOLDS = [0.8, 0.7, 0.5, 0.3]
# =================================================

# ---- Universal key names (checked in order, first valid match wins) ----
PRED_DICT_KEYS = ["predicted_json", "prediction_json"]
PRED_STRING_KEYS = [
    "prediction_string", "predicted_text", "prediction_text",
    "raw_response", "raw_output", "prediction_stripped",
    "prediction_raw", "prediction",
]
GT_DICT_KEYS = [
    "target_json", "ground_truth", "gt_json",
    "ground_truth_fields", "ground_truth_json", "original_data",
]
GT_STRING_KEYS = [
    "target_text", "ground_truth_text", "ground_truth",
    "ground_truth_json_string", "gt_string", "ground_truth_str",
]


# ===================== helpers =====================

def _try_parse_json_dict(s: str) -> Optional[Dict[str, Any]]:
    """json.loads → dict, or None."""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _resolve_dict(obj: Dict[str, Any], dict_keys: List[str],
                  string_keys: List[str]) -> Optional[Dict[str, Any]]:
    """
    Try to get a non-empty dict from *obj*.
      1. Check dict_keys for a value that is already a non-empty dict.
      2. Check string_keys for a JSON string that parses into a non-empty dict.
    """
    for k in dict_keys:
        v = obj.get(k)
        if isinstance(v, dict) and v:
            return v
    for k in string_keys:
        v = obj.get(k)
        if isinstance(v, str):
            parsed = _try_parse_json_dict(v)
            if parsed is not None:
                return parsed
    return None


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """
    Flatten nested dictionaries using dot notation for keys.
    Non-dict values are converted to strings.
    """
    items: Dict[str, str] = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = "" if v is None else str(v)
    return items


def _safe_f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    if p is None or r is None:
        return None
    return (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0


# ===================== main =====================

def compute_all_metrics(input_path: str, output_path: str) -> None:
    # map_stats[field][thresh] = {"tp": int, "fp": int, "fn": int}
    map_stats: Dict[str, Dict[float, Dict[str, int]]] = {}

    num_lines_total = 0
    num_json_vs_json = 0
    num_pred_json_failed = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            num_lines_total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: could not parse line {line_idx}, skipping.")
                continue

            # ---- resolve GT dict ----
            gt_dict = _resolve_dict(obj, GT_DICT_KEYS, GT_STRING_KEYS)
            if gt_dict is None:
                print(f"Warning: line {line_idx} has no ground truth, skipping.")
                continue

            # ---- resolve Pred dict ----
            pred_dict = _resolve_dict(obj, PRED_DICT_KEYS, PRED_STRING_KEYS)

            gt_flat = flatten_dict(gt_dict)

            if pred_dict is not None:
                # ---- Path A: both dicts available → per-field CER comparison ----
                num_json_vs_json += 1
                pred_flat = flatten_dict(pred_dict)
                all_keys = set(pred_flat.keys()) | set(gt_flat.keys())

                for key in all_keys:
                    ref_str = gt_flat.get(key, "")
                    hyp_str = pred_flat.get(key, "")

                    if not ref_str and not hyp_str:
                        continue

                    for thresh in OVERLAP_THRESHOLDS:
                        thresh = float(thresh)
                        field_map = map_stats.setdefault(key, {})
                        t = field_map.setdefault(thresh, {"tp": 0, "fp": 0, "fn": 0})

                        if len(ref_str) == 0 and len(hyp_str) > 0:
                            t["fp"] += 1
                        elif len(ref_str) > 0:
                            overlap = 1.0 - cer(ref_str, hyp_str)
                            if overlap >= thresh:
                                t["tp"] += 1
                            else:
                                if len(hyp_str) > 0:
                                    t["fp"] += 1
                                t["fn"] += 1
            else:
                # ---- Path B: pred JSON failed → all non-empty GT fields are FN ----
                num_pred_json_failed += 1
                print(f"Info: line {line_idx} pred JSON unavailable, counting GT fields as FN.")
                for key, ref_str in gt_flat.items():
                    if not ref_str:
                        continue
                    for thresh in OVERLAP_THRESHOLDS:
                        thresh = float(thresh)
                        field_map = map_stats.setdefault(key, {})
                        t = field_map.setdefault(thresh, {"tp": 0, "fp": 0, "fn": 0})
                        t["fn"] += 1

    # ---------- Build compact mAP results ----------
    n_scored = num_json_vs_json + num_pred_json_failed
    json_valid_rate = (f"{num_json_vs_json}/{n_scored} "
                       f"({100 * num_json_vs_json / n_scored:.1f}%)" if n_scored else "N/A")

    map_results = {}
    for thresh in OVERLAP_THRESHOLDS:
        thresh = float(thresh)
        precisions: List[float] = []
        recalls: List[float] = []

        for key, field_map in map_stats.items():
            t = field_map.get(thresh, {"tp": 0, "fp": 0, "fn": 0})
            tp, fp, fn = t["tp"], t["fp"], t["fn"]
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

        mAP_p = sum(precisions) / len(precisions) if precisions else None
        mAP_r = sum(recalls) / len(recalls) if recalls else None

        map_results[str(thresh)] = {
            "mAP_precision": mAP_p,
            "mAP_recall": mAP_r,
            "mAP_f1": _safe_f1(mAP_p, mAP_r),
        }

    results = {
        "input_jsonl": input_path,
        "num_lines_total": num_lines_total,
        "num_scored": n_scored,
        "json_validity_rate": json_valid_rate,
        "num_json_vs_json": num_json_vs_json,
        "num_pred_json_failed_as_fn": num_pred_json_failed,
        "map_cer": map_results,
        "note": (
            "mAP_* are macro-averaged across fields. "
            "When pred JSON is unavailable, all non-empty GT fields count as FN. "
            "json_validity_rate = lines where pred parsed as valid JSON dict."
        ),
    }

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Saved mAP metrics to: {output_path}")
    print(f"json_validity_rate: {json_valid_rate}")
    for thresh, vals in map_results.items():
        print(f"  @{thresh}: P={vals['mAP_precision']:.4f}  R={vals['mAP_recall']:.4f}  F1={vals['mAP_f1']:.4f}")


if __name__ == "__main__":
    compute_all_metrics(INPUT_JSONL_PATH, OUTPUT_PATH)
