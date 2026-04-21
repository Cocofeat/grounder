#!/usr/bin/env python3
"""
Summarize all single-modal QA, Report, and Grounding evaluation results.

Covers: internal, external batch1, external batch2, prospective.
Reports per-modality performance.
Outputs CSVs and prints markdown tables.
"""

import argparse
import csv
import json
import os
import re
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ARCHIVE = BASE_DIR / "evaluation_archive" / "single"
NLTK_DATA_DIR = Path("/home/baiyang/nltk_data")

# ---------------------------------------------------------------------------
# Prospective patient filter (same as multi)
# ---------------------------------------------------------------------------
PROSPECTIVE_PATIENT_IDS = {
    "1", "2", "6", "7", "8", "9", "10", "11", "13", "15", "16", "17",
    "19", "20", "21", "22", "23", "26", "27", "28", "31", "32", "33",
    "35", "37", "39", "40", "41", "43", "44", "45", "46", "48", "50",
    "51", "52", "54", "55", "56", "58", "59", "62", "63", "64", "68",
    "69", "70", "72", "73", "74", "76", "77", "78", "79", "80", "82",
    "83", "84", "86", "88", "92", "93", "95", "99", "100", "103", "104",
    "105", "106", "107", "108", "109", "110", "112", "113", "114", "115",
    "116", "119", "120", "121", "122", "123", "124", "126", "127", "128",
    "129", "130", "131", "132", "133", "134", "135", "136", "137", "138",
    "140", "142", "143", "147",
}

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "Internal",
        "qa": {
            "gt": ARCHIVE / "internal/gt/test_QA_EN.json",
            "pred": ARCHIVE / "internal/pred/internal_test_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "internal/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "internal/pred/internal_test_qa_with_bbox.json",
        },
        "grounding": {
            "gt": ARCHIVE / "internal/gt/test_bbox_v2.json",
            "pred": ARCHIVE / "internal/pred/internal_test_grounding.json",
        },
        "report": {
            "gt": ARCHIVE / "internal/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "internal/pred/internal_test_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "internal/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "internal/pred/internal_test_report_with_bbox.json",
        },
    },
    {
        "name": "Ext1 GXMU",
        "qa": {
            "gt": ARCHIVE / "external1_gxmu_hcc_icc/gt/test_QA_EN_external.json",
            "pred": ARCHIVE / "external1_gxmu_hcc_icc/pred/external1_single_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external1_gxmu_hcc_icc/gt/test_QA_EN_external_with_bbox.json",
            "pred": ARCHIVE / "external1_gxmu_hcc_icc/pred/external1_single_qa_with_bbox.json",
        },
        "grounding": {
            "gt": ARCHIVE / "external1_gxmu_hcc_icc/gt/test_bbox.json",
            "pred": ARCHIVE / "external1_gxmu_hcc_icc/pred/external1_single_grounding.json",
        },
        "report": {
            "gt": ARCHIVE / "external1_gxmu_hcc_icc/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external1_gxmu_hcc_icc/pred/external1_single_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external1_gxmu_hcc_icc/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external1_gxmu_hcc_icc/pred/external1_single_report_with_bbox.json",
        },
    },
    {
        "name": "Ext ENSHI",
        "qa": {
            "gt": ARCHIVE / "external_enshi/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_enshi/pred/external_ENSHI_single_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_enshi/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_enshi/pred/external_ENSHI_single_qa_with_bbox.json",
        },
        "grounding": {
            "gt": ARCHIVE / "external_enshi/gt/test_bbox.json",
            "pred": ARCHIVE / "external_enshi/pred/external_ENSHI_single_grounding.json",
        },
        "report": {
            "gt": ARCHIVE / "external_enshi/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_enshi/pred/external_ENSHI_single_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_enshi/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_enshi/pred/external_ENSHI_single_report_with_bbox.json",
        },
    },
    {
        "name": "Ext GXMU B/N",
        "qa": {
            "gt": ARCHIVE / "external_gxmu_benign_normal/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_gxmu_benign_normal/pred/external_GXMU_bn_single_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_gxmu_benign_normal/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_gxmu_benign_normal/pred/external_GXMU_bn_single_qa_with_bbox.json",
        },
        "grounding": {
            "gt": ARCHIVE / "external_gxmu_benign_normal/gt/test_bbox.json",
            "pred": ARCHIVE / "external_gxmu_benign_normal/pred/external_GXMU_bn_single_grounding.json",
        },
        "report": {
            "gt": ARCHIVE / "external_gxmu_benign_normal/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_gxmu_benign_normal/pred/external_GXMU_bn_single_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_gxmu_benign_normal/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_gxmu_benign_normal/pred/external_GXMU_bn_single_report_with_bbox.json",
        },
    },
    {
        "name": "Ext SanYa",
        "qa": {
            "gt": ARCHIVE / "external_sanya/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_sanya/pred/external_SanYa_single_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_sanya/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_sanya/pred/external_SanYa_single_qa_with_bbox.json",
        },
        "grounding": {
            "gt": ARCHIVE / "external_sanya/gt/test_bbox.json",
            "pred": ARCHIVE / "external_sanya/pred/external_SanYa_single_grounding.json",
        },
        "report": {
            "gt": ARCHIVE / "external_sanya/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_sanya/pred/external_SanYa_single_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_sanya/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_sanya/pred/external_SanYa_single_report_with_bbox.json",
        },
    },
    {
        "name": "Ext Nanning",
        "qa": {
            "gt": ARCHIVE / "external_nanning/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_nanning/pred/external_batch2_liver_MRI_EXTERNAL4_Nanning_ready_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_nanning/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_nanning/pred/external_batch2_liver_MRI_EXTERNAL4_Nanning_ready_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "external_nanning/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_nanning/pred/external_batch2_liver_MRI_EXTERNAL4_Nanning_ready_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_nanning/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_nanning/pred/external_batch2_liver_MRI_EXTERNAL4_Nanning_ready_report_with_bbox.json",
        },
    },
    {
        "name": "Ext Beihai",
        "qa": {
            "gt": ARCHIVE / "external_beihai/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_beihai/pred/external_batch2_liver_MRI_EXTERNAL5_Beihai_First_ready_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_beihai/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_beihai/pred/external_batch2_liver_MRI_EXTERNAL5_Beihai_First_ready_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "external_beihai/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_beihai/pred/external_batch2_liver_MRI_EXTERNAL5_Beihai_First_ready_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_beihai/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_beihai/pred/external_batch2_liver_MRI_EXTERNAL5_Beihai_First_ready_report_with_bbox.json",
        },
    },
    {
        "name": "Ext Guigang",
        "qa": {
            "gt": ARCHIVE / "external_guigang/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_guigang/pred/external_batch2_liver_MRI_EXTERNAL6_Guigang_ready_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_guigang/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_guigang/pred/external_batch2_liver_MRI_EXTERNAL6_Guigang_ready_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "external_guigang/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_guigang/pred/external_batch2_liver_MRI_EXTERNAL6_Guigang_ready_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_guigang/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_guigang/pred/external_batch2_liver_MRI_EXTERNAL6_Guigang_ready_report_with_bbox.json",
        },
    },
    {
        "name": "Ext Guilin",
        "qa": {
            "gt": ARCHIVE / "external_guilin/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_guilin/pred/external_batch2_liver_MRI_EXTERNAL7_Guilin_ready_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_guilin/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_guilin/pred/external_batch2_liver_MRI_EXTERNAL7_Guilin_ready_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "external_guilin/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_guilin/pred/external_batch2_liver_MRI_EXTERNAL7_Guilin_ready_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_guilin/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_guilin/pred/external_batch2_liver_MRI_EXTERNAL7_Guilin_ready_report_with_bbox.json",
        },
    },
    {
        "name": "Ext HeNan",
        "qa": {
            "gt": ARCHIVE / "external_henan/gt/test_QA_EN.json",
            "pred": ARCHIVE / "external_henan/pred/external_batch2_liver_MRI_EXTERNAL8_HeNan_ready_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "external_henan/gt/test_QA_EN_with_bbox.json",
            "pred": ARCHIVE / "external_henan/pred/external_batch2_liver_MRI_EXTERNAL8_HeNan_ready_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "external_henan/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "external_henan/pred/external_batch2_liver_MRI_EXTERNAL8_HeNan_ready_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "external_henan/gt/test_report_generation_EN_with_bbox.json",
            "pred": ARCHIVE / "external_henan/pred/external_batch2_liver_MRI_EXTERNAL8_HeNan_ready_report_with_bbox.json",
        },
    },
    {
        "name": "Prospective",
        "patient_filter": PROSPECTIVE_PATIENT_IDS,
        "qa": {
            "gt": ARCHIVE / "prospective_gxmu/gt/test_QA_EN.json",
            "pred": ARCHIVE / "prospective_gxmu/pred/hcc_icc_mask_64_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "prospective_gxmu/gt/test_QA_EN_with_pred_bbox.json",
            "pred": ARCHIVE / "prospective_gxmu/pred/hcc_icc_mask_64_qa_with_pred_bbox.json",
        },
        "grounding": {
            "gt": ARCHIVE / "prospective_gxmu/gt/test_bbox.json",
            "pred": ARCHIVE / "prospective_gxmu/pred/hcc_icc_mask_64_grounding.json",
        },
        "report": {
            "gt": ARCHIVE / "prospective_gxmu/gt/test_report_generation_EN.json",
            "pred": ARCHIVE / "prospective_gxmu/pred/hcc_icc_mask_64_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "prospective_gxmu/gt/test_report_generation_EN_with_pred_bbox.json",
            "pred": ARCHIVE / "prospective_gxmu/pred/hcc_icc_mask_64_report_with_pred_bbox.json",
        },
    },
]

MERGED_DATASETS = [
    {
        "name": "Ext GXMU (merged)",
        "sources": ["Ext1 GXMU", "Ext GXMU B/N"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def majority_vote(votes):
    if not votes:
        return ""
    return Counter(votes).most_common(1)[0][0]


def compute_f1(preds, gts):
    labels = sorted(set(gts) | set(preds))
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    support = {l: 0 for l in labels}
    for pred, gt in zip(preds, gts):
        support[gt] = support.get(gt, 0) + 1
        if pred == gt:
            tp[gt] = tp.get(gt, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[gt] = fn.get(gt, 0) + 1
    macro_f1_sum = 0.0
    macro_count = 0
    for l in labels:
        p_denom = tp[l] + fp[l]
        r_denom = tp[l] + fn[l]
        precision = tp[l] / p_denom if p_denom > 0 else 0.0
        recall = tp[l] / r_denom if r_denom > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        if support[l] > 0:
            macro_f1_sum += f1
            macro_count += 1
    return macro_f1_sum / macro_count if macro_count > 0 else 0.0


def _extract_patient_id_single(gt_entry):
    """Extract patient_id from a single-modal gt entry using the image field.

    Image path formats:
    - Internal (4 parts): patient_id/modality/images/slice.png -> parts[0]
    - External (5 parts): dataset/patient_id/modality/images/slice.png -> parts[1]
    """
    if "image" in gt_entry:
        parts = gt_entry["image"].split("/")
        if len(parts) >= 5:
            return str(parts[1])  # dataset/patient_id/mod/images/slice
        if len(parts) >= 4:
            return str(parts[0])  # patient_id/mod/images/slice (Internal)
    parts = gt_entry["question_id"].split("/")
    return str(parts[0])


# ---------------------------------------------------------------------------
# QA evaluation (per modality + per QA type, patient-level voting)
# ---------------------------------------------------------------------------

def eval_qa_single(pred_path, gt_path, patient_filter=None):
    """Evaluate single-modal QA with patient-level majority voting.

    Uses positional matching (zip) like the original eval script, because
    single-modal QA reuses the same question_id for different QA types (QA1-QA4)
    on the same slice.

    Returns dict keyed by modality -> {QA1: {acc, f1, n}, ..., Overall: ...}
    Plus an "ALL" key for all modalities combined.
    """
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    if len(preds) != len(gts):
        print(f"    [WARNING] pred({len(preds)}) != gt({len(gts)}) count, using min")

    # Group by (modality, patient_id, qtype) using positional matching
    data_by_mod_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    matched = 0
    filtered = 0
    for pred, gt in zip(preds, gts):
        patient_id = _extract_patient_id_single(gt)
        if patient_filter is not None and patient_id not in patient_filter:
            filtered += 1
            continue
        matched += 1
        modality = gt.get("modality", "unknown")
        qtype = gt["Question_type"]
        data_by_mod_patient_qtype[(modality, patient_id, qtype)]["pred"].append(pred["text"])
        data_by_mod_patient_qtype[(modality, patient_id, qtype)]["gt"].append(gt["label"])

    if patient_filter is not None:
        print(f"    [FILTER] {filtered} entries filtered out, {matched} matched")

    # Select first slice per (modality, patient, qtype) — no voting
    by_mod_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for (modality, patient_id, qtype), data in data_by_mod_patient_qtype.items():
        first_pred = data["pred"][0]
        first_gt = data["gt"][0]
        by_mod_qtype[(modality, qtype)]["pred"].append(first_pred)
        by_mod_qtype[(modality, qtype)]["gt"].append(first_gt)

    # Aggregate results per modality
    modalities = sorted(set(m for (m, _) in by_mod_qtype.keys()))
    results = {}

    for mod in modalities:
        mod_results = {}
        all_p, all_g = [], []
        for qtype in sorted(set(q for (m, q) in by_mod_qtype.keys() if m == mod)):
            data = by_mod_qtype[(mod, qtype)]
            acc = sum(1 for a, b in zip(data["pred"], data["gt"]) if a == b) / len(data["gt"]) if data["gt"] else 0.0
            f1 = compute_f1(data["pred"], data["gt"])
            mod_results[qtype] = {"acc": acc, "f1": f1, "n": len(data["gt"])}
            all_p.extend(data["pred"])
            all_g.extend(data["gt"])
        if all_g:
            acc = sum(1 for a, b in zip(all_p, all_g) if a == b) / len(all_g)
            f1 = compute_f1(all_p, all_g)
            mod_results["Overall"] = {"acc": acc, "f1": f1, "n": len(all_g)}
        results[mod] = mod_results

    # ALL modalities combined
    all_p, all_g = [], []
    all_by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for (mod, qtype), data in by_mod_qtype.items():
        all_by_qtype[qtype]["pred"].extend(data["pred"])
        all_by_qtype[qtype]["gt"].extend(data["gt"])
        all_p.extend(data["pred"])
        all_g.extend(data["gt"])

    all_results = {}
    for qtype, data in all_by_qtype.items():
        acc = sum(1 for a, b in zip(data["pred"], data["gt"]) if a == b) / len(data["gt"]) if data["gt"] else 0.0
        f1 = compute_f1(data["pred"], data["gt"])
        all_results[qtype] = {"acc": acc, "f1": f1, "n": len(data["gt"])}
    if all_g:
        acc = sum(1 for a, b in zip(all_p, all_g) if a == b) / len(all_g)
        f1 = compute_f1(all_p, all_g)
        all_results["Overall"] = {"acc": acc, "f1": f1, "n": len(all_g)}
    results["ALL"] = all_results

    return results


# ---------------------------------------------------------------------------
# Report evaluation (per modality, patient-level)
# ---------------------------------------------------------------------------

def load_report_metrics():
    os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))
    try:
        from tools_coco.eval_report_single_vote import compute_metrics
    except Exception:
        try:
            from eval_report_single_vote import compute_metrics
        except Exception:
            compute_metrics = None
    return compute_metrics


def eval_report_single(pred_path, gt_path, compute_metrics, patient_filter=None):
    """Evaluate single-modal report with patient-level aggregation, per modality."""
    if compute_metrics is None:
        return None
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    gt_mapping = {gt["question_id"]: gt for gt in gts}

    # Group by (modality, patient_id)
    data_by_mod_patient = defaultdict(lambda: {"preds": [], "gts": []})

    for pred in preds:
        qid = pred["question_id"]
        if qid not in gt_mapping:
            continue
        gt = gt_mapping[qid]
        patient_id = _extract_patient_id_single(gt)
        if patient_filter is not None and patient_id not in patient_filter:
            continue
        modality = gt.get("modality", "unknown")
        data_by_mod_patient[(modality, patient_id)]["preds"].append(pred["text"])
        data_by_mod_patient[(modality, patient_id)]["gts"].append(gt["label"])

    # Compute per modality
    modalities = sorted(set(m for (m, _) in data_by_mod_patient.keys()))
    results = {}

    for mod in modalities:
        patient_gt_dict = {}
        patient_pred_dict = {}
        for (m, pid), data in data_by_mod_patient.items():
            if m != mod:
                continue
            patient_gt_dict[pid] = [data["gts"][0]]
            patient_pred_dict[pid] = [data["preds"][0]]
        if patient_pred_dict:
            metrics = compute_metrics(patient_gt_dict, patient_pred_dict)
            results[mod] = {"patient": metrics, "num_patients": len(patient_pred_dict)}

    # ALL modalities combined: one entry per patient (first modality encountered)
    all_patient_gt = {}
    all_patient_pred = {}
    for (m, pid), data in data_by_mod_patient.items():
        if pid not in all_patient_gt:
            all_patient_gt[pid] = [data["gts"][0]]
            all_patient_pred[pid] = [data["preds"][0]]
    if all_patient_gt:
        metrics = compute_metrics(all_patient_gt, all_patient_pred)
        results["ALL"] = {"patient": metrics, "num_patients": len(all_patient_gt)}

    return results


# ---------------------------------------------------------------------------
# Grounding evaluation (reuse logic from eval_grounding.py)
# ---------------------------------------------------------------------------

def parse_bounding_box_single(text):
    if text is None:
        return None
    # Format 1: <box>[[x1, y1, x2, y2]]</box>
    pattern = r'<box>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]</box>'
    match = re.search(pattern, text)
    if match:
        return [int(match.group(1)), int(match.group(2)),
                int(match.group(3)), int(match.group(4))]
    # Format 2: [x1, y1, x2, y2] (e.g. "B. [160, 570, 200, 620]")
    pattern2 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match2 = re.search(pattern2, text)
    if match2:
        return [int(match2.group(1)), int(match2.group(2)),
                int(match2.group(3)), int(match2.group(4))]
    return None


def calculate_iou(box1, box2):
    if box1 is None or box2 is None:
        return None
    if sum(box2) == 0:
        return None
    if sum(box1) == 0:
        return 0.0
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection = 0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union == 0:
        return 0.0
    return intersection / union


def eval_grounding_single(pred_path, gt_path, patient_filter=None):
    """Evaluate single-modal grounding. Returns per-modality and overall metrics."""
    preds_raw = load_jsonl(str(pred_path))
    pred_map = {p["question_id"]: p["text"] for p in preds_raw}

    with open(str(gt_path), "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    iou_thresholds = [0.1, 0.3, 0.5]
    modality_stats = defaultdict(lambda: {"ious": [], "counts": {t: 0 for t in iou_thresholds}})
    all_ious = []
    threshold_counts = {t: 0 for t in iou_thresholds}

    for gt_entry in gt_data:
        qid = gt_entry["question_id"]
        if qid not in pred_map:
            continue
        if patient_filter is not None:
            patient_id = _extract_patient_id_single(gt_entry)
            if patient_id not in patient_filter:
                continue

        pred_box = parse_bounding_box_single(pred_map[qid])
        gt_box = parse_bounding_box_single(gt_entry["label"])
        if pred_box is None or gt_box is None:
            continue
        iou = calculate_iou(pred_box, gt_box)
        if iou is None:
            continue

        mod = gt_entry.get("modality", "unknown")
        modality_stats[mod]["ious"].append(iou)
        all_ious.append(iou)
        for t in iou_thresholds:
            if iou >= t:
                threshold_counts[t] += 1
                modality_stats[mod]["counts"][t] += 1

    results = {}
    for mod, stats in modality_stats.items():
        n = len(stats["ious"])
        if n > 0:
            results[mod] = {
                "mIoU": float(np.mean(stats["ious"])),
                "IoU@0.1": stats["counts"][0.1] / n,
                "IoU@0.3": stats["counts"][0.3] / n,
                "IoU@0.5": stats["counts"][0.5] / n,
                "n": n,
            }
    if all_ious:
        n = len(all_ious)
        results["ALL"] = {
            "mIoU": float(np.mean(all_ious)),
            "IoU@0.1": threshold_counts[0.1] / n,
            "IoU@0.3": threshold_counts[0.3] / n,
            "IoU@0.5": threshold_counts[0.5] / n,
            "n": n,
        }
    return results


# ---------------------------------------------------------------------------
# CSV / printing
# ---------------------------------------------------------------------------

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Merged dataset helpers
# ---------------------------------------------------------------------------

def merge_qa_single(sources, datasets_list, task_key, patient_filter=None):
    """Merge QA data from multiple source datasets and evaluate.
    Uses positional matching (zip) per source dataset."""
    all_pairs = []  # list of (pred, gt) tuples
    for ds in datasets_list:
        if ds["name"] not in sources:
            continue
        cfg = ds.get(task_key)
        if not cfg:
            continue
        src_preds = load_jsonl(str(cfg["pred"]))
        with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
            src_gts = json.load(f)
        for pred, gt in zip(src_preds, src_gts):
            all_pairs.append((pred, gt))

    if not all_pairs:
        return None

    data_by_mod_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for pred, gt in all_pairs:
        patient_id = _extract_patient_id_single(gt)
        if patient_filter is not None and patient_id not in patient_filter:
            continue
        modality = gt.get("modality", "unknown")
        qtype = gt["Question_type"]
        data_by_mod_patient_qtype[(modality, patient_id, qtype)]["pred"].append(pred["text"])
        data_by_mod_patient_qtype[(modality, patient_id, qtype)]["gt"].append(gt["label"])

    by_mod_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for (modality, patient_id, qtype), data in data_by_mod_patient_qtype.items():
        first_pred = data["pred"][0]
        first_gt = data["gt"][0]
        by_mod_qtype[(modality, qtype)]["pred"].append(first_pred)
        by_mod_qtype[(modality, qtype)]["gt"].append(first_gt)

    modalities = sorted(set(m for (m, _) in by_mod_qtype.keys()))
    results = {}
    for mod in modalities:
        mod_results = {}
        all_p, all_g = [], []
        for qtype in sorted(set(q for (m, q) in by_mod_qtype.keys() if m == mod)):
            data = by_mod_qtype[(mod, qtype)]
            acc = sum(1 for a, b in zip(data["pred"], data["gt"]) if a == b) / len(data["gt"]) if data["gt"] else 0.0
            f1 = compute_f1(data["pred"], data["gt"])
            mod_results[qtype] = {"acc": acc, "f1": f1, "n": len(data["gt"])}
            all_p.extend(data["pred"])
            all_g.extend(data["gt"])
        if all_g:
            acc = sum(1 for a, b in zip(all_p, all_g) if a == b) / len(all_g)
            f1 = compute_f1(all_p, all_g)
            mod_results["Overall"] = {"acc": acc, "f1": f1, "n": len(all_g)}
        results[mod] = mod_results

    # ALL
    all_p, all_g = [], []
    all_by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for (mod, qtype), data in by_mod_qtype.items():
        all_by_qtype[qtype]["pred"].extend(data["pred"])
        all_by_qtype[qtype]["gt"].extend(data["gt"])
        all_p.extend(data["pred"])
        all_g.extend(data["gt"])
    all_results = {}
    for qtype, data in all_by_qtype.items():
        acc = sum(1 for a, b in zip(data["pred"], data["gt"]) if a == b) / len(data["gt"]) if data["gt"] else 0.0
        f1 = compute_f1(data["pred"], data["gt"])
        all_results[qtype] = {"acc": acc, "f1": f1, "n": len(data["gt"])}
    if all_g:
        acc = sum(1 for a, b in zip(all_p, all_g) if a == b) / len(all_g)
        f1 = compute_f1(all_p, all_g)
        all_results["Overall"] = {"acc": acc, "f1": f1, "n": len(all_g)}
    results["ALL"] = all_results
    return results


def merge_report_single(sources, datasets_list, task_key, compute_metrics, patient_filter=None):
    """Merge Report data from multiple source datasets and evaluate."""
    if compute_metrics is None:
        return None
    all_preds = []
    all_gts = []
    for ds in datasets_list:
        if ds["name"] not in sources:
            continue
        cfg = ds.get(task_key)
        if not cfg:
            continue
        src_preds = load_jsonl(str(cfg["pred"]))
        with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
            src_gts = json.load(f)
        all_preds.extend(src_preds)
        all_gts.extend(src_gts)

    if not all_preds:
        return None

    gt_mapping = {gt["question_id"]: gt for gt in all_gts}
    data_by_mod_patient = defaultdict(lambda: {"preds": [], "gts": []})
    for pred in all_preds:
        qid = pred["question_id"]
        if qid not in gt_mapping:
            continue
        gt = gt_mapping[qid]
        patient_id = _extract_patient_id_single(gt)
        if patient_filter is not None and patient_id not in patient_filter:
            continue
        modality = gt.get("modality", "unknown")
        data_by_mod_patient[(modality, patient_id)]["preds"].append(pred["text"])
        data_by_mod_patient[(modality, patient_id)]["gts"].append(gt["label"])

    modalities = sorted(set(m for (m, _) in data_by_mod_patient.keys()))
    results = {}
    for mod in modalities:
        patient_gt_dict = {}
        patient_pred_dict = {}
        for (m, pid), data in data_by_mod_patient.items():
            if m != mod:
                continue
            patient_gt_dict[pid] = [data["gts"][0]]
            patient_pred_dict[pid] = [data["preds"][0]]
        if patient_pred_dict:
            metrics = compute_metrics(patient_gt_dict, patient_pred_dict)
            results[mod] = {"patient": metrics, "num_patients": len(patient_pred_dict)}

    all_patient_gt = {}
    all_patient_pred = {}
    for (m, pid), data in data_by_mod_patient.items():
        if pid not in all_patient_gt:
            all_patient_gt[pid] = [data["gts"][0]]
            all_patient_pred[pid] = [data["preds"][0]]
    if all_patient_gt:
        metrics = compute_metrics(all_patient_gt, all_patient_pred)
        results["ALL"] = {"patient": metrics, "num_patients": len(all_patient_gt)}
    return results


def merge_grounding_single(sources, datasets_list, task_key, patient_filter=None):
    """Merge grounding data from multiple source datasets and evaluate."""
    all_preds_raw = []
    all_gt_data = []
    for ds in datasets_list:
        if ds["name"] not in sources:
            continue
        cfg = ds.get(task_key)
        if not cfg:
            continue
        src_preds = load_jsonl(str(cfg["pred"]))
        with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
            src_gts = json.load(f)
        all_preds_raw.extend(src_preds)
        all_gt_data.extend(src_gts)

    if not all_preds_raw:
        return None

    pred_map = {p["question_id"]: p["text"] for p in all_preds_raw}
    iou_thresholds = [0.1, 0.3, 0.5]
    modality_stats = defaultdict(lambda: {"ious": [], "counts": {t: 0 for t in iou_thresholds}})
    all_ious = []
    threshold_counts = {t: 0 for t in iou_thresholds}

    for gt_entry in all_gt_data:
        qid = gt_entry["question_id"]
        if qid not in pred_map:
            continue
        if patient_filter is not None:
            patient_id = _extract_patient_id_single(gt_entry)
            if patient_id not in patient_filter:
                continue
        pred_box = parse_bounding_box_single(pred_map[qid])
        gt_box = parse_bounding_box_single(gt_entry["label"])
        if pred_box is None or gt_box is None:
            continue
        iou = calculate_iou(pred_box, gt_box)
        if iou is None:
            continue
        mod = gt_entry.get("modality", "unknown")
        modality_stats[mod]["ious"].append(iou)
        all_ious.append(iou)
        for t in iou_thresholds:
            if iou >= t:
                threshold_counts[t] += 1
                modality_stats[mod]["counts"][t] += 1

    results = {}
    for mod, stats in modality_stats.items():
        n = len(stats["ious"])
        if n > 0:
            results[mod] = {
                "mIoU": float(np.mean(stats["ious"])),
                "IoU@0.1": stats["counts"][0.1] / n,
                "IoU@0.3": stats["counts"][0.3] / n,
                "IoU@0.5": stats["counts"][0.5] / n,
                "n": n,
            }
    if all_ious:
        n = len(all_ious)
        results["ALL"] = {
            "mIoU": float(np.mean(all_ious)),
            "IoU@0.1": threshold_counts[0.1] / n,
            "IoU@0.3": threshold_counts[0.3] / n,
            "IoU@0.5": threshold_counts[0.5] / n,
            "n": n,
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarize all single-modal evaluation results.")
    parser.add_argument(
        "--output-dir",
        default="evaluation_archive/results_single",
        help="Output directory for CSVs.",
    )
    args = parser.parse_args()

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_metrics = load_report_metrics()

    # -------------------------------------------------------------------
    # Run all evaluations
    # -------------------------------------------------------------------
    qa_all = {}          # name -> {mod -> {QA1: {acc,f1,n}, ..., Overall}}
    qa_bbox_all = {}
    report_all = {}      # name -> {mod -> {patient: {...}, num_patients}}
    report_bbox_all = {}
    grounding_all = {}   # name -> {mod -> {mIoU, IoU@*, n}}

    for ds in DATASETS:
        name = ds["name"]
        pf = ds.get("patient_filter")
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}" + (f" (filtered to {len(pf)} patients)" if pf else ""))
        print(f"{'='*50}")

        for task_key, target_dict, label in [
            ("qa", qa_all, "QA"),
            ("qa_bbox", qa_bbox_all, "QA+bbox"),
        ]:
            cfg = ds.get(task_key)
            if cfg and Path(cfg["gt"]).exists() and Path(cfg["pred"]).exists():
                print(f"  {label} ...")
                target_dict[name] = eval_qa_single(cfg["pred"], cfg["gt"], patient_filter=pf)
            else:
                print(f"  {label}: SKIPPED")

        for task_key, target_dict, label in [
            ("report", report_all, "Report"),
            ("report_bbox", report_bbox_all, "Report+bbox"),
        ]:
            cfg = ds.get(task_key)
            if cfg and Path(cfg["gt"]).exists() and Path(cfg["pred"]).exists():
                print(f"  {label} ...")
                target_dict[name] = eval_report_single(cfg["pred"], cfg["gt"], compute_metrics, patient_filter=pf)
            else:
                print(f"  {label}: SKIPPED")

        grounding_cfg = ds.get("grounding")
        if grounding_cfg and Path(grounding_cfg["gt"]).exists() and Path(grounding_cfg["pred"]).exists():
            print(f"  Grounding ...")
            grounding_all[name] = eval_grounding_single(grounding_cfg["pred"], grounding_cfg["gt"], patient_filter=pf)
        else:
            print(f"  Grounding: SKIPPED")

    # -------------------------------------------------------------------
    # Merged datasets
    # -------------------------------------------------------------------
    for merged in MERGED_DATASETS:
        mname = merged["name"]
        sources = merged["sources"]
        print(f"\n{'='*50}")
        print(f"Merging: {mname} (from {sources})")
        print(f"{'='*50}")

        for task_key, target_dict, label in [("qa", qa_all, "QA"), ("qa_bbox", qa_bbox_all, "QA+bbox")]:
            res = merge_qa_single(sources, DATASETS, task_key)
            if res:
                target_dict[mname] = res
                print(f"  {label}: merged")

        for task_key, target_dict, label in [("report", report_all, "Report"), ("report_bbox", report_bbox_all, "Report+bbox")]:
            res = merge_report_single(sources, DATASETS, task_key, compute_metrics)
            if res:
                target_dict[mname] = res
                print(f"  {label}: merged")

        res = merge_grounding_single(sources, DATASETS, "grounding")
        if res:
            grounding_all[mname] = res
            print(f"  Grounding: merged")

    # -------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------
    all_ds_names = [d["name"] for d in DATASETS] + [m["name"] for m in MERGED_DATASETS]
    modality_order = ["AP", "PVP", "PRE", "T2WI", "DWI", "ALL"]

    def get_modalities(results_dict):
        mods = set()
        for res in results_dict.values():
            if res:
                mods.update(res.keys())
        return [m for m in modality_order if m in mods]

    # === QA summary (Overall per modality) ===
    for task_label, results_dict, csv_name in [
        ("QA", qa_all, "qa_single"),
        ("QA+bbox", qa_bbox_all, "qa_single_with_bbox"),
    ]:
        mods = get_modalities(results_dict)

        # Detail CSV: dataset, modality, qtype, acc, f1, n
        detail_rows = []
        for ds_name in all_ds_names:
            if ds_name not in results_dict:
                continue
            res = results_dict[ds_name]
            for mod in mods:
                if mod not in res:
                    continue
                for qtype in sorted(res[mod].keys(), key=lambda x: (x == "Overall", x)):
                    m = res[mod][qtype]
                    detail_rows.append([ds_name, mod, qtype, f"{m['acc']:.4f}", f"{m['f1']:.4f}", m["n"]])
        write_csv(output_dir / f"{csv_name}_detail.csv",
                  ["dataset", "modality", "question", "acc", "f1", "n"], detail_rows)

        # Summary CSV: dataset, mod1_acc, mod1_f1, mod2_acc, ...
        summary_header = ["dataset"]
        for mod in mods:
            summary_header.extend([f"{mod}_acc", f"{mod}_f1"])
        summary_rows = []
        for ds_name in all_ds_names:
            row = [ds_name]
            if ds_name not in results_dict:
                row.extend(["—", "—"] * len(mods))
            else:
                res = results_dict[ds_name]
                for mod in mods:
                    if mod in res and "Overall" in res[mod]:
                        m = res[mod]["Overall"]
                        row.extend([f"{m['acc']:.4f}", f"{m['f1']:.4f}"])
                    else:
                        row.extend(["—", "—"])
            summary_rows.append(row)
        write_csv(output_dir / f"{csv_name}_summary.csv", summary_header, summary_rows)

        # Print markdown
        print(f"\n\n{'='*80}")
        print(f"{task_label} Single-Modal (patient-level, Overall per modality)")
        print(f"{'='*80}")
        header = "| Dataset |" + " | ".join(f"{m} Acc | {m} F1" for m in mods) + " |"
        sep = "|---|" + "|".join(["---|---"] * len(mods)) + "|"
        print(header)
        print(sep)
        for row in summary_rows:
            print(f"| {' | '.join(row)} |")

    # === Report summary (per modality) ===
    for task_label, results_dict, csv_name in [
        ("Report", report_all, "report_single"),
        ("Report+bbox", report_bbox_all, "report_single_with_bbox"),
    ]:
        mods = get_modalities(results_dict)

        # Detail CSV
        detail_rows = []
        for ds_name in all_ds_names:
            if ds_name not in results_dict or results_dict[ds_name] is None:
                continue
            res = results_dict[ds_name]
            for mod in mods:
                if mod not in res:
                    continue
                m = res[mod]["patient"]
                detail_rows.append([
                    ds_name, mod,
                    f"{m['BLEU-1']:.4f}", f"{m['BLEU-2']:.4f}",
                    f"{m['BLEU-3']:.4f}", f"{m['BLEU-4']:.4f}",
                    f"{m['CIDEr']:.4f}", f"{m['ROUGE_L']:.4f}",
                    f"{m.get('METEOR', 0.0):.4f}",
                    res[mod]["num_patients"],
                ])
        write_csv(output_dir / f"{csv_name}_detail.csv",
                  ["dataset", "modality", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr", "ROUGE_L", "METEOR", "patients"],
                  detail_rows)

        # Summary CSV: dataset, mod_B4, mod_CIDEr, mod_ROUGE, mod_METEOR ...
        summary_header = ["dataset"]
        for mod in mods:
            summary_header.extend([f"{mod}_B4", f"{mod}_CIDEr", f"{mod}_ROUGE", f"{mod}_METEOR"])
        summary_rows = []
        for ds_name in all_ds_names:
            row = [ds_name]
            if ds_name not in results_dict or results_dict[ds_name] is None:
                row.extend(["—", "—", "—", "—"] * len(mods))
            else:
                res = results_dict[ds_name]
                for mod in mods:
                    if mod in res:
                        m = res[mod]["patient"]
                        row.extend([f"{m['BLEU-4']:.4f}", f"{m['CIDEr']:.4f}", f"{m['ROUGE_L']:.4f}", f"{m.get('METEOR', 0.0):.4f}"])
                    else:
                        row.extend(["—", "—", "—", "—"])
            summary_rows.append(row)
        write_csv(output_dir / f"{csv_name}_summary.csv", summary_header, summary_rows)

        # Print markdown
        print(f"\n\n{'='*80}")
        print(f"{task_label} Single-Modal (patient-level, per modality)")
        print(f"{'='*80}")
        header = "| Dataset |" + " | ".join(f"{m} B4 | {m} CIDEr | {m} ROUGE | {m} METEOR" for m in mods) + " |"
        sep = "|---|" + "|".join(["---|---|---|---"] * len(mods)) + "|"
        print(header)
        print(sep)
        for row in summary_rows:
            print(f"| {' | '.join(row)} |")

    # === Grounding summary ===
    grounding_mods = get_modalities(grounding_all)
    if grounding_mods:
        # Detail CSV
        detail_rows = []
        for ds_name in all_ds_names:
            if ds_name not in grounding_all:
                continue
            res = grounding_all[ds_name]
            for mod in grounding_mods:
                if mod not in res:
                    continue
                m = res[mod]
                detail_rows.append([
                    ds_name, mod,
                    f"{m['mIoU']:.4f}", f"{m['IoU@0.1']:.4f}",
                    f"{m['IoU@0.3']:.4f}", f"{m['IoU@0.5']:.4f}", m["n"],
                ])
        write_csv(output_dir / "grounding_single_detail.csv",
                  ["dataset", "modality", "mIoU", "IoU@0.1", "IoU@0.3", "IoU@0.5", "n"],
                  detail_rows)

        # Summary
        summary_header = ["dataset"]
        for mod in grounding_mods:
            summary_header.extend([f"{mod}_mIoU", f"{mod}_IoU@0.5"])
        summary_rows = []
        for ds_name in all_ds_names:
            row = [ds_name]
            if ds_name not in grounding_all:
                row.extend(["—", "—"] * len(grounding_mods))
            else:
                res = grounding_all[ds_name]
                for mod in grounding_mods:
                    if mod in res:
                        m = res[mod]
                        row.extend([f"{m['mIoU']:.4f}", f"{m['IoU@0.5']:.4f}"])
                    else:
                        row.extend(["—", "—"])
            summary_rows.append(row)
        write_csv(output_dir / "grounding_single_summary.csv", summary_header, summary_rows)

        # Print markdown
        print(f"\n\n{'='*80}")
        print("Grounding Single-Modal (per modality)")
        print(f"{'='*80}")
        header = "| Dataset |" + " | ".join(f"{m} mIoU | {m} IoU@0.5" for m in grounding_mods) + " |"
        sep = "|---|" + "|".join(["---|---"] * len(grounding_mods)) + "|"
        print(header)
        print(sep)
        for row in summary_rows:
            print(f"| {' | '.join(row)} |")

    if compute_metrics is None:
        print("\n[WARNING] Report metrics were skipped (pycocoevalcap not available).")

    print(f"\nCSVs written to: {output_dir}")


if __name__ == "__main__":
    main()
