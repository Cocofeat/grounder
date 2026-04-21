#!/usr/bin/env python3
"""
Summarize all multimodal QA and Report evaluation results.

Covers: internal, external batch1, external batch2, prospective.
Outputs CSVs and prints markdown tables.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
ARCHIVE = BASE_DIR / "evaluation_archive"
NLTK_DATA_DIR = Path("/home/baiyang/nltk_data")

# ---------------------------------------------------------------------------
# Dataset definitions: (short_name, gt_dir_in_archive, pred_dir_in_archive,
#                        qa_gt_file, qa_pred_file,
#                        qa_bbox_gt_file, qa_bbox_pred_file)
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

DATASETS = [
    # --- Internal ---
    {
        "name": "Internal",
        "qa": {
            "gt": ARCHIVE / "multi_gating/internal/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/internal/pred/internal_multi_qa_gating.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/internal/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/internal/pred/internal_multi_qa_with_bbox_gating.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/internal/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/internal/pred/internal_multi_report_gating.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/internal/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/internal/pred/internal_multi_report_with_bbox_gating.json",
        },
    },
    # --- External1 GXMU HCC/ICC ---
    {
        "name": "Ext1 GXMU",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/gt/test_QA_EN_external_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/pred/external1_multi_qa_gating.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/gt/test_QA_EN_external_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/pred/external1_multi_qa_with_bbox_gating.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/pred/external1_multi_report_gating.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external1_gxmu_hcc_icc/pred/external1_multi_report_with_bbox_gating.json",
        },
    },
    # --- External ENSHI ---
    {
        "name": "Ext ENSHI",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_enshi/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_enshi/pred/external_ENSHI_multi_qa_gating.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_enshi/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_enshi/pred/external_ENSHI_multi_qa_with_bbox_gating.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_enshi/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_enshi/pred/external_ENSHI_multi_report_gating.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_enshi/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_enshi/pred/external_ENSHI_multi_report_with_bbox_gating.json",
        },
    },
    # --- External GXMU Benign/Normal ---
    {
        "name": "Ext GXMU B/N",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_gxmu_benign_normal/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_gxmu_benign_normal/pred/external_GXMU_bn_multi_qa_gating.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_gxmu_benign_normal/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_gxmu_benign_normal/pred/external_GXMU_bn_multi_qa_with_bbox_gating.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_gxmu_benign_normal/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_gxmu_benign_normal/pred/external_GXMU_bn_multi_report_gating.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_gxmu_benign_normal/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_gxmu_benign_normal/pred/external_GXMU_bn_multi_report_with_bbox_gating.json",
        },
    },
    # --- External SanYa ---
    {
        "name": "Ext SanYa",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_sanya/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_sanya/pred/external_SanYa_multi_qa_gating.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_sanya/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_sanya/pred/external_SanYa_multi_qa_with_bbox_gating.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_sanya/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_sanya/pred/external_SanYa_multi_report_gating.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_sanya/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_sanya/pred/external_SanYa_multi_report_with_bbox_gating.json",
        },
    },
    # --- Batch2: Nanning ---
    {
        "name": "Ext Nanning",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_nanning/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_nanning/pred/external_batch2_multi_liver_MRI_EXTERNAL4_Nanning_ready_multimodal_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_nanning/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_nanning/pred/external_batch2_multi_liver_MRI_EXTERNAL4_Nanning_ready_multimodal_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_nanning/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_nanning/pred/external_batch2_multi_liver_MRI_EXTERNAL4_Nanning_ready_multimodal_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_nanning/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_nanning/pred/external_batch2_multi_liver_MRI_EXTERNAL4_Nanning_ready_multimodal_report_with_bbox.json",
        },
    },
    # --- Batch2: Beihai ---
    {
        "name": "Ext Beihai",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_beihai/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_beihai/pred/external_batch2_multi_liver_MRI_EXTERNAL5_Beihai_First_ready_multimodal_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_beihai/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_beihai/pred/external_batch2_multi_liver_MRI_EXTERNAL5_Beihai_First_ready_multimodal_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_beihai/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_beihai/pred/external_batch2_multi_liver_MRI_EXTERNAL5_Beihai_First_ready_multimodal_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_beihai/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_beihai/pred/external_batch2_multi_liver_MRI_EXTERNAL5_Beihai_First_ready_multimodal_report_with_bbox.json",
        },
    },
    # --- Batch2: Guigang ---
    {
        "name": "Ext Guigang",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_guigang/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_guigang/pred/external_batch2_multi_liver_MRI_EXTERNAL6_Guigang_ready_multimodal_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_guigang/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_guigang/pred/external_batch2_multi_liver_MRI_EXTERNAL6_Guigang_ready_multimodal_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_guigang/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_guigang/pred/external_batch2_multi_liver_MRI_EXTERNAL6_Guigang_ready_multimodal_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_guigang/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_guigang/pred/external_batch2_multi_liver_MRI_EXTERNAL6_Guigang_ready_multimodal_report_with_bbox.json",
        },
    },
    # --- Batch2: Guilin ---
    {
        "name": "Ext Guilin",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_guilin/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_guilin/pred/external_batch2_multi_liver_MRI_EXTERNAL7_Guilin_ready_multimodal_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_guilin/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_guilin/pred/external_batch2_multi_liver_MRI_EXTERNAL7_Guilin_ready_multimodal_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_guilin/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_guilin/pred/external_batch2_multi_liver_MRI_EXTERNAL7_Guilin_ready_multimodal_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_guilin/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_guilin/pred/external_batch2_multi_liver_MRI_EXTERNAL7_Guilin_ready_multimodal_report_with_bbox.json",
        },
    },
    # --- Batch2: HeNan ---
    {
        "name": "Ext HeNan",
        "qa": {
            "gt": ARCHIVE / "multi_gating/external_henan/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/external_henan/pred/external_batch2_multi_liver_MRI_EXTERNAL8_HeNan_ready_multimodal_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/external_henan/gt/test_QA_EN_multimodal_4mod_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_henan/pred/external_batch2_multi_liver_MRI_EXTERNAL8_HeNan_ready_multimodal_qa_with_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/external_henan/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/external_henan/pred/external_batch2_multi_liver_MRI_EXTERNAL8_HeNan_ready_multimodal_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/external_henan/gt/test_report_generation_EN_multimodal_with_bbox.json",
            "pred": ARCHIVE / "multi_gating/external_henan/pred/external_batch2_multi_liver_MRI_EXTERNAL8_HeNan_ready_multimodal_report_with_bbox.json",
        },
    },
    # --- Prospective GXMU (filtered subset) ---
    {
        "name": "Prospective",
        "patient_filter": PROSPECTIVE_PATIENT_IDS,
        "qa": {
            "gt": ARCHIVE / "multi_gating/prospective_gxmu/gt/test_QA_EN_multimodal_4mod.json",
            "pred": ARCHIVE / "multi_gating/prospective_gxmu/pred/hcc_icc_mask_64_multi_qa.json",
        },
        "qa_bbox": {
            "gt": ARCHIVE / "multi_gating/prospective_gxmu/gt/test_QA_EN_multimodal_4mod_with_pred_bbox.json",
            "pred": ARCHIVE / "multi_gating/prospective_gxmu/pred/hcc_icc_mask_64_multi_qa_with_pred_bbox.json",
        },
        "report": {
            "gt": ARCHIVE / "multi_gating/prospective_gxmu/gt/test_report_generation_EN_multimodal.json",
            "pred": ARCHIVE / "multi_gating/prospective_gxmu/pred/hcc_icc_mask_64_multi_report.json",
        },
        "report_bbox": {
            "gt": ARCHIVE / "multi_gating/prospective_gxmu/gt/test_report_generation_EN_multimodal_with_pred_bbox.json",
            "pred": ARCHIVE / "multi_gating/prospective_gxmu/pred/hcc_icc_mask_64_multi_report_with_pred_bbox.json",
        },
    },
]

# ---------------------------------------------------------------------------
# Merged dataset definitions: combine multiple datasets into one evaluation
# ---------------------------------------------------------------------------
MERGED_DATASETS = [
    {
        "name": "Ext GXMU (merged)",
        "sources": ["Ext1 GXMU", "Ext GXMU B/N"],
    },
]


# ---------------------------------------------------------------------------
# QA evaluation helpers (from summarize_multi_internal_results.py)
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


def _extract_patient_id_qa(gt_entry):
    """Extract patient_id from a QA gt entry."""
    patient_id = gt_entry.get("patient_prefix")
    if not patient_id:
        patient_id = gt_entry["question_id"].split("_multimodal_")[0]
    return str(patient_id)


def eval_qa_multi(pred_path, gt_path, patient_filter=None):
    """Evaluate QA with patient-level majority voting. Returns per-question-type and overall.

    Args:
        patient_filter: optional set of patient_id strings to include (others are excluded)
    """
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    # Build lookup by question_id for matching (handles different ordering / extra entries)
    pred_by_qid = {}
    for p in preds:
        pred_by_qid[p["question_id"]] = p

    data_by_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    matched = 0
    filtered = 0
    for gt in gts:
        qid = gt["question_id"]
        patient_id = _extract_patient_id_qa(gt)
        if patient_filter is not None and patient_id not in patient_filter:
            filtered += 1
            continue
        if qid not in pred_by_qid:
            continue
        pred = pred_by_qid[qid]
        matched += 1
        qtype = gt["Question_type"]
        data_by_patient_qtype[(patient_id, qtype)]["pred"].append(pred["text"])
        data_by_patient_qtype[(patient_id, qtype)]["gt"].append(gt["label"])

    if patient_filter is not None:
        print(f"    [FILTER] {filtered} entries filtered out, {matched} matched")
    elif matched < len(gts):
        print(f"    [WARN] Matched {matched}/{len(gts)} GT entries (pred has {len(preds)} entries)")

    by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for (patient_id, qtype), data in data_by_patient_qtype.items():
        voted_pred = majority_vote(data["pred"])
        voted_gt = data["gt"][0]
        by_qtype[qtype]["pred"].append(voted_pred)
        by_qtype[qtype]["gt"].append(voted_gt)

    results = {}
    all_preds = []
    all_gts = []
    for qtype, data in by_qtype.items():
        preds_q = data["pred"]
        gts_q = data["gt"]
        acc = sum(1 for a, b in zip(preds_q, gts_q) if a == b) / len(gts_q) if gts_q else 0.0
        f1 = compute_f1(preds_q, gts_q)
        results[qtype] = {"acc": acc, "f1": f1, "n": len(gts_q)}
        all_preds.extend(preds_q)
        all_gts.extend(gts_q)

    if all_gts:
        acc = sum(1 for a, b in zip(all_preds, all_gts) if a == b) / len(all_gts)
        f1 = compute_f1(all_preds, all_gts)
        results["Overall"] = {"acc": acc, "f1": f1, "n": len(all_gts)}

    return results


# ---------------------------------------------------------------------------
# Report evaluation helpers
# ---------------------------------------------------------------------------

def load_report_metrics():
    os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))
    try:
        from tools_coco.eval_report_multi_vote import compute_metrics
    except Exception:
        try:
            from eval_report_multi_vote import compute_metrics
        except Exception:
            compute_metrics = None
    return compute_metrics


def extract_patient_id_report(qid):
    if "_multimodal_" in qid:
        prefix = qid.split("_multimodal_")[0]
        return prefix.split("/")[-1]
    if "/" in qid:
        return qid.split("/")[0]
    return qid


def eval_report_multi(pred_path, gt_path, compute_metrics, patient_filter=None):
    if compute_metrics is None:
        return None
    preds = load_jsonl(str(pred_path))
    with open(str(gt_path), "r", encoding="utf-8") as f:
        gts = json.load(f)

    gt_mapping = {gt["question_id"]: gt for gt in gts}
    data_by_patient = defaultdict(lambda: {"preds": [], "gts": []})

    for pred in preds:
        qid = pred["question_id"]
        if qid in gt_mapping:
            gt = gt_mapping[qid]
            patient_id = extract_patient_id_report(qid)
            if patient_filter is not None and str(patient_id) not in patient_filter:
                continue
            data_by_patient[patient_id]["preds"].append(pred["text"])
            data_by_patient[patient_id]["gts"].append(gt["label"])

    if patient_filter is not None:
        print(f"    [FILTER] {len(data_by_patient)} patients after filtering")

    patient_gt_dict = {}
    patient_pred_dict = {}
    for patient_id, data in data_by_patient.items():
        patient_gt_dict[patient_id] = [data["gts"][0]]
        patient_pred_dict[patient_id] = [data["preds"][0]]

    patient_metrics = compute_metrics(patient_gt_dict, patient_pred_dict)
    return {
        "patient": patient_metrics,
        "num_patients": len(patient_pred_dict),
    }


# ---------------------------------------------------------------------------
# CSV / printing helpers
# ---------------------------------------------------------------------------

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize all multimodal evaluation results.")
    parser.add_argument(
        "--output-dir",
        default="evaluation_archive/results",
        help="Output directory for CSVs.",
    )
    args = parser.parse_args()

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_metrics = load_report_metrics()

    # -----------------------------------------------------------------------
    # Run all evaluations
    # -----------------------------------------------------------------------
    qa_all = {}       # name -> {QA1: {acc, f1, n}, ..., Overall: ...}
    qa_bbox_all = {}
    report_all = {}   # name -> {patient: {BLEU-1, ...}, num_patients: int}
    report_bbox_all = {}

    for ds in DATASETS:
        name = ds["name"]
        pf = ds.get("patient_filter")
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}" + (f" (filtered to {len(pf)} patients)" if pf else ""))
        print(f"{'='*50}")

        # QA
        qa_cfg = ds.get("qa")
        if qa_cfg and Path(qa_cfg["gt"]).exists() and Path(qa_cfg["pred"]).exists():
            print(f"  QA ...")
            qa_all[name] = eval_qa_multi(qa_cfg["pred"], qa_cfg["gt"], patient_filter=pf)
        else:
            print(f"  QA: SKIPPED (files missing)")

        # QA + bbox
        qa_bbox_cfg = ds.get("qa_bbox")
        if qa_bbox_cfg and Path(qa_bbox_cfg["gt"]).exists() and Path(qa_bbox_cfg["pred"]).exists():
            print(f"  QA+bbox ...")
            qa_bbox_all[name] = eval_qa_multi(qa_bbox_cfg["pred"], qa_bbox_cfg["gt"], patient_filter=pf)
        else:
            print(f"  QA+bbox: SKIPPED (files missing)")

        # Report
        report_cfg = ds.get("report")
        if report_cfg and Path(report_cfg["gt"]).exists() and Path(report_cfg["pred"]).exists():
            print(f"  Report ...")
            report_all[name] = eval_report_multi(report_cfg["pred"], report_cfg["gt"], compute_metrics, patient_filter=pf)
        else:
            print(f"  Report: SKIPPED (files missing)")

        # Report + bbox
        report_bbox_cfg = ds.get("report_bbox")
        if report_bbox_cfg and Path(report_bbox_cfg["gt"]).exists() and Path(report_bbox_cfg["pred"]).exists():
            print(f"  Report+bbox ...")
            report_bbox_all[name] = eval_report_multi(report_bbox_cfg["pred"], report_bbox_cfg["gt"], compute_metrics, patient_filter=pf)
        else:
            print(f"  Report+bbox: SKIPPED (files missing)")

    # -----------------------------------------------------------------------
    # Merged datasets: combine results from multiple sources
    # -----------------------------------------------------------------------
    for merged in MERGED_DATASETS:
        mname = merged["name"]
        sources = merged["sources"]
        print(f"\n{'='*50}")
        print(f"Merging: {mname} (from {sources})")
        print(f"{'='*50}")

        # Merge QA
        for target_dict, task_name in [(qa_all, "qa"), (qa_bbox_all, "qa_bbox")]:
            merged_by_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
            for src in sources:
                if src not in target_dict:
                    continue
                # We need raw patient-level data to merge. Re-evaluate from files.
                pass
            # Simpler approach: re-run evaluation on combined pred/gt files
            # Collect all pred/gt pairs from sources
            all_preds_combined = []
            all_gts_combined = []
            task_key = "qa" if task_name == "qa" else "qa_bbox"
            for src_ds in DATASETS:
                if src_ds["name"] not in sources:
                    continue
                cfg = src_ds.get(task_key)
                if not cfg:
                    continue
                src_preds = load_jsonl(str(cfg["pred"]))
                with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
                    src_gts = json.load(f)
                all_preds_combined.extend(src_preds)
                all_gts_combined.extend(src_gts)

            if all_preds_combined:
                pred_by_qid = {p["question_id"]: p for p in all_preds_combined}
                data_by_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
                for gt in all_gts_combined:
                    qid = gt["question_id"]
                    if qid not in pred_by_qid:
                        continue
                    pred = pred_by_qid[qid]
                    patient_id = _extract_patient_id_qa(gt)
                    qtype = gt["Question_type"]
                    data_by_patient_qtype[(patient_id, qtype)]["pred"].append(pred["text"])
                    data_by_patient_qtype[(patient_id, qtype)]["gt"].append(gt["label"])

                by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
                for (patient_id, qtype), data in data_by_patient_qtype.items():
                    voted_pred = majority_vote(data["pred"])
                    voted_gt = data["gt"][0]
                    by_qtype[qtype]["pred"].append(voted_pred)
                    by_qtype[qtype]["gt"].append(voted_gt)

                results = {}
                all_p, all_g = [], []
                for qtype, data in by_qtype.items():
                    acc = sum(1 for a, b in zip(data["pred"], data["gt"]) if a == b) / len(data["gt"]) if data["gt"] else 0.0
                    f1 = compute_f1(data["pred"], data["gt"])
                    results[qtype] = {"acc": acc, "f1": f1, "n": len(data["gt"])}
                    all_p.extend(data["pred"])
                    all_g.extend(data["gt"])
                if all_g:
                    acc = sum(1 for a, b in zip(all_p, all_g) if a == b) / len(all_g)
                    f1 = compute_f1(all_p, all_g)
                    results["Overall"] = {"acc": acc, "f1": f1, "n": len(all_g)}
                target_dict[mname] = results
                print(f"  {task_name}: {len(data_by_patient_qtype)} patient-qtype pairs")

        # Merge Report
        for target_dict, task_key in [(report_all, "report"), (report_bbox_all, "report_bbox")]:
            if compute_metrics is None:
                continue
            all_preds_combined = []
            all_gts_combined = []
            for src_ds in DATASETS:
                if src_ds["name"] not in sources:
                    continue
                cfg = src_ds.get(task_key)
                if not cfg:
                    continue
                src_preds = load_jsonl(str(cfg["pred"]))
                with open(str(cfg["gt"]), "r", encoding="utf-8") as f:
                    src_gts = json.load(f)
                all_preds_combined.extend(src_preds)
                all_gts_combined.extend(src_gts)

            if all_preds_combined:
                gt_mapping = {gt["question_id"]: gt for gt in all_gts_combined}
                data_by_patient = defaultdict(lambda: {"preds": [], "gts": []})
                for pred in all_preds_combined:
                    qid = pred["question_id"]
                    if qid in gt_mapping:
                        gt = gt_mapping[qid]
                        patient_id = extract_patient_id_report(qid)
                        data_by_patient[patient_id]["preds"].append(pred["text"])
                        data_by_patient[patient_id]["gts"].append(gt["label"])

                patient_gt_dict = {}
                patient_pred_dict = {}
                for patient_id, data in data_by_patient.items():
                    patient_gt_dict[patient_id] = [data["gts"][0]]
                    patient_pred_dict[patient_id] = [data["preds"][0]]

                patient_metrics = compute_metrics(patient_gt_dict, patient_pred_dict)
                target_dict[mname] = {
                    "patient": patient_metrics,
                    "num_patients": len(patient_pred_dict),
                }
                print(f"  {task_key}: {len(patient_pred_dict)} patients")

    # -----------------------------------------------------------------------
    # Output: QA tables
    # -----------------------------------------------------------------------
    all_ds_names = [d["name"] for d in DATASETS] + [m["name"] for m in MERGED_DATASETS]

    def build_qa_rows(results_dict):
        rows = []
        for ds_name in all_ds_names:
            if ds_name not in results_dict:
                continue
            res = results_dict[ds_name]
            for qtype in sorted(res.keys(), key=lambda x: (x == "Overall", x)):
                m = res[qtype]
                rows.append([ds_name, qtype, f"{m['acc']:.4f}", f"{m['f1']:.4f}", m["n"]])
        return rows

    qa_rows = build_qa_rows(qa_all)
    qa_bbox_rows = build_qa_rows(qa_bbox_all)
    write_csv(output_dir / "qa_multi.csv", ["dataset", "question", "acc", "f1", "n"], qa_rows)
    write_csv(output_dir / "qa_multi_with_bbox.csv", ["dataset", "question", "acc", "f1", "n"], qa_bbox_rows)

    # QA summary table (Overall only)
    qa_summary_rows = []
    for ds_name in all_ds_names:
        row = [ds_name]
        for results_dict in [qa_all, qa_bbox_all]:
            if ds_name in results_dict and "Overall" in results_dict[ds_name]:
                m = results_dict[ds_name]["Overall"]
                row.extend([f"{m['acc']:.4f}", f"{m['f1']:.4f}"])
            else:
                row.extend(["—", "—"])
        qa_summary_rows.append(row)
    write_csv(
        output_dir / "qa_summary.csv",
        ["dataset", "QA_acc", "QA_f1", "QA+bbox_acc", "QA+bbox_f1"],
        qa_summary_rows,
    )

    # -----------------------------------------------------------------------
    # Output: Report tables
    # -----------------------------------------------------------------------
    def build_report_rows(results_dict):
        rows = []
        for ds_name in all_ds_names:
            if ds_name not in results_dict or results_dict[ds_name] is None:
                continue
            res = results_dict[ds_name]
            metrics = res["patient"]
            rows.append([
                ds_name,
                f"{metrics['BLEU-1']:.4f}",
                f"{metrics['BLEU-2']:.4f}",
                f"{metrics['BLEU-3']:.4f}",
                f"{metrics['BLEU-4']:.4f}",
                f"{metrics['CIDEr']:.4f}",
                f"{metrics['ROUGE_L']:.4f}",
                res["num_patients"],
            ])
        return rows

    report_header = ["dataset", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr", "ROUGE_L", "patients"]
    report_rows = build_report_rows(report_all)
    report_bbox_rows = build_report_rows(report_bbox_all)
    write_csv(output_dir / "report_multi.csv", report_header, report_rows)
    write_csv(output_dir / "report_multi_with_bbox.csv", report_header, report_bbox_rows)

    # Report summary table
    report_summary_rows = []
    for ds_name in all_ds_names:
        row = [ds_name]
        for results_dict in [report_all, report_bbox_all]:
            if ds_name in results_dict and results_dict[ds_name] is not None:
                m = results_dict[ds_name]["patient"]
                row.extend([f"{m['BLEU-4']:.4f}", f"{m['CIDEr']:.4f}", f"{m['ROUGE_L']:.4f}"])
            else:
                row.extend(["—", "—", "—"])
        report_summary_rows.append(row)
    write_csv(
        output_dir / "report_summary.csv",
        ["dataset", "Rpt_B4", "Rpt_CIDEr", "Rpt_ROUGE", "Rpt+bbox_B4", "Rpt+bbox_CIDEr", "Rpt+bbox_ROUGE"],
        report_summary_rows,
    )

    # -----------------------------------------------------------------------
    # Print markdown tables
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("QA (patient-level majority voting, Overall)")
    print("=" * 80)
    print("| Dataset | QA Acc | QA F1 | QA+bbox Acc | QA+bbox F1 |")
    print("|---|---|---|---|---|")
    for row in qa_summary_rows:
        print(f"| {' | '.join(row)} |")

    print("\n" + "=" * 80)
    print("QA Detail (no bbox)")
    print("=" * 80)
    print("| Dataset | Question | Acc | F1 | N |")
    print("|---|---|---|---|---|")
    for row in qa_rows:
        print(f"| {' | '.join(str(x) for x in row)} |")

    print("\n" + "=" * 80)
    print("QA Detail (with bbox)")
    print("=" * 80)
    print("| Dataset | Question | Acc | F1 | N |")
    print("|---|---|---|---|---|")
    for row in qa_bbox_rows:
        print(f"| {' | '.join(str(x) for x in row)} |")

    print("\n" + "=" * 80)
    print("Report (patient-level)")
    print("=" * 80)
    print("| Dataset | BLEU-4 | CIDEr | ROUGE_L | +bbox BLEU-4 | +bbox CIDEr | +bbox ROUGE_L |")
    print("|---|---|---|---|---|---|---|")
    for row in report_summary_rows:
        print(f"| {' | '.join(row)} |")

    if compute_metrics is None:
        print("\n[WARNING] Report metrics were skipped (pycocoevalcap not available).")

    print(f"\nCSVs written to: {output_dir}")


if __name__ == "__main__":
    main()
