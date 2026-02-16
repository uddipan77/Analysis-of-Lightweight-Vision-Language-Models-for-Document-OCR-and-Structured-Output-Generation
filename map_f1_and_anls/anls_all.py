#!/usr/bin/env python3
"""
Compute ANLS* metrics using the official anls_star library.

Reference:
  Peer et al., "ANLS* -- A Universal Document Processing Metric for Generative
  Large Language Models", arXiv:2402.03848, 2024.
  https://github.com/deepopinion/anls_star_metric

Evaluation policy:
  - GT string is always the source of truth.
  - If both pred JSON and GT JSON are available → structured ANLS* (dict vs dict).
  - Otherwise → string-level ANLS (pred string vs GT string).
  - Reports JSON validity rate.
"""
import json
import os
from typing import Dict, Any, Optional, List
from anls_star import anls_score

# ================== USER CONFIG ==================
INPUT_JSONL_PATH = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/inventory_dataset/run_A100_genCER_20260215_095759/test_predictions.jsonl"
OUTPUT_PATH = "/home/vault/iwi5/iwi5298h/models_image_text/gemma/inventory_dataset/run_A100_genCER_20260215_095759/anls_metrics.json"
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


def _resolve_string(obj: Dict[str, Any], string_keys: List[str]) -> Optional[str]:
    """Return the first non-empty string found under *string_keys*."""
    for k in string_keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _string_anls(pred_str: str, gt_str: str, threshold: float = 0.5) -> float:
    """
    String-level ANLS: 1 − (edit_distance / max_len), with 0.5 cutoff.
    """
    if not pred_str and not gt_str:
        return 1.0
    if not pred_str or not gt_str:
        return 0.0
    m, n = len(pred_str), len(gt_str)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred_str[i - 1] == gt_str[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
    sim = 1.0 - dp[m][n] / max(m, n)
    return sim if sim >= threshold else 0.0


# ===================== main =====================

def compute_metrics(input_path: str, output_path: str) -> None:
    all_scores: List[float] = []
    per_field_scores: Dict[str, List[float]] = {}

    num_lines_total = 0
    num_json_vs_json = 0       # pred JSON + GT JSON → structured ANLS*
    num_string_fallback = 0    # pred string vs GT string fallback
    num_failed = 0             # no usable prediction at all → score 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            num_lines_total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: line {line_idx} is not valid JSONL, skipping.")
                continue

            # ---- resolve GT (string is source of truth) ----
            gt_str = _resolve_string(row, GT_STRING_KEYS)
            gt_dict = _resolve_dict(row, GT_DICT_KEYS, GT_STRING_KEYS)

            if gt_str is None and gt_dict is None:
                print(f"Warning: line {line_idx} has no ground truth at all, skipping.")
                continue

            # ---- resolve Pred ----
            pred_dict = _resolve_dict(row, PRED_DICT_KEYS, PRED_STRING_KEYS)
            pred_str = _resolve_string(row, PRED_STRING_KEYS)

            # ---- scoring ----
            # Path A: both dicts available → structured ANLS*
            if pred_dict is not None and gt_dict is not None:
                score, key_scores = anls_score(gt_dict, pred_dict,
                                               return_key_scores=True)
                all_scores.append(score)
                num_json_vs_json += 1

                def _extract(score_dict, prefix=""):
                    for key, node in score_dict.items():
                        fk = f"{prefix}.{key}" if prefix else key
                        per_field_scores.setdefault(fk, []).append(node.anls_score)
                        if hasattr(node, "children") and node.children:
                            _extract(node.children, fk)
                _extract(key_scores)
                continue

            # Path B: fall back to string vs string
            if pred_str and gt_str:
                score = _string_anls(pred_str, gt_str)
                print(f"Info: line {line_idx} using string ANLS fallback: {score:.4f}")
                all_scores.append(score)
                num_string_fallback += 1
                continue

            # Path C: no usable prediction → 0
            print(f"Warning: line {line_idx} no usable prediction, scoring as 0.")
            all_scores.append(0.0)
            num_failed += 1

    # ---- aggregation ----
    n_scored = len(all_scores)
    overall_anls = (sum(all_scores) / n_scored) if n_scored else None
    per_field_means = {k: sum(v) / len(v) for k, v in per_field_scores.items() if v}
    overall_macro = (sum(per_field_means.values()) / len(per_field_means)) if per_field_means else None
    json_valid_rate = (f"{num_json_vs_json}/{n_scored} "
                       f"({100 * num_json_vs_json / n_scored:.1f}%)" if n_scored else "N/A")

    results = {
        "input_jsonl": input_path,
        "library": "anls_star (Peer et al., arXiv:2402.03848)",
        "num_lines_total": num_lines_total,
        "num_scored": n_scored,
        "json_validity_rate": json_valid_rate,
        "num_json_vs_json": num_json_vs_json,
        "num_string_fallback": num_string_fallback,
        "num_failed_as_zero": num_failed,
        "overall_anls_star": overall_anls,
        "overall_anls_star_macro": overall_macro,
        "per_field_anls": per_field_means,
        "note": (
            "Structured ANLS* used when both pred & GT parse as JSON; "
            "otherwise string-level ANLS (Levenshtein, 0.5 cutoff). "
            "json_validity_rate = lines where both pred and GT were valid JSON dicts."
        ),
    }

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(results, out_f, ensure_ascii=False, indent=2)

    print(f"Saved ANLS* metrics to: {output_path}")
    print(f"overall_anls_star       : {overall_anls}")
    print(f"overall_anls_star_macro : {overall_macro}")
    print(f"json_validity_rate      : {json_valid_rate}")


if __name__ == "__main__":
    compute_metrics(INPUT_JSONL_PATH, OUTPUT_PATH)
