#!/usr/bin/env python3
"""
Summarize multimodal external test results for QA, report, and grounding.

Based on script_coco/test_external_multi_QA3.sh.
Outputs CSVs for reuse.
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
NLTK_DATA_DIR = Path("/home/baiyang/nltk_data")


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


def eval_qa_multi(pred_path, gt_path, dataset_prefix=None):
    preds = load_jsonl(pred_path)
    with open(gt_path, "r", encoding="utf-8") as f:
        gts = json.load(f)

    data_by_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for pred, gt in zip(preds, gts):
        if pred.get("question_id") != gt.get("question_id"):
            continue
        qtype = gt["Question_type"]
        patient_id = gt.get("patient_prefix")
        if not patient_id:
            qid = gt.get("question_id", "")
            patient_id = qid.split("_multimodal_")[0]
        if dataset_prefix:
            patient_id = f"{dataset_prefix}:{patient_id}"
        data_by_patient_qtype[(patient_id, qtype)]["pred"].append(pred["text"])
        data_by_patient_qtype[(patient_id, qtype)]["gt"].append(gt["label"])

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


def load_report_metrics():
    os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))
    try:
        from tools_coco.eval_report_multi_vote import compute_metrics
    except Exception:
        compute_metrics = None
    return compute_metrics


def extract_patient_id_report(gt_entry):
    if "patient_prefix" in gt_entry and gt_entry["patient_prefix"]:
        return gt_entry["patient_prefix"]
    qid = gt_entry.get("question_id", "")
    if "_multimodal_" in qid:
        prefix = qid.split("_multimodal_")[0]
        return prefix.split("/")[-1]
    if "/" in qid:
        return qid.split("/")[-1]
    return qid


def eval_report_multi(pred_path, gt_path, compute_metrics, dataset_prefix=None):
    if compute_metrics is None:
        return None
    preds = load_jsonl(pred_path)
    with open(gt_path, "r", encoding="utf-8") as f:
        gts = json.load(f)

    gt_mapping = {gt["question_id"]: gt for gt in gts}
    data_by_patient = defaultdict(lambda: {"preds": [], "gts": []})

    for pred in preds:
        qid = pred["question_id"]
        if qid in gt_mapping:
            gt = gt_mapping[qid]
            patient_id = extract_patient_id_report(gt)
            if dataset_prefix:
                patient_id = f"{dataset_prefix}:{patient_id}"
            data_by_patient[patient_id]["preds"].append(pred["text"])
            data_by_patient[patient_id]["gts"].append(gt["label"])

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


def eval_grounding_multi(pred_path, gt_path):
    from tools_coco.eval_grounding import evaluate_multi_grounding

    results = evaluate_multi_grounding(pred_path, gt_path)
    if not results:
        return None
    return results["overall"]


def eval_grounding_multi_merged(datasets):
    from tools_coco.eval_grounding import (
        parse_bounding_box_multi,
        parse_bounding_box_single,
        calculate_iou,
        load_ground_truth as load_ground_truth_grounding,
    )

    preds = {}
    gts = {}
    modalities = {}
    patient_labels = {}

    for ds_name, (pred_path, gt_path) in datasets:
        with open(pred_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    preds[f"{ds_name}:{data['question_id']}"] = data["text"]
        gt_data, mods, labels = load_ground_truth_grounding(gt_path)
        for qid, label in gt_data.items():
            key = f"{ds_name}:{qid}"
            gts[key] = label
            modalities[key] = mods.get(qid, ["PRE", "AP", "PVP", "T2WI"])
            patient_labels[key] = labels.get(qid, "unknown")

    common_ids = set(preds.keys()) & set(gts.keys())
    if not common_ids:
        return None

    iou_thresholds = [0.1, 0.3, 0.5]
    all_avg_ious = []
    threshold_counts = {t: 0 for t in iou_thresholds}
    invalid_count = 0

    for qid in common_ids:
        pred_text = preds[qid]
        gt_text = gts[qid]

        pred_boxes = parse_bounding_box_multi(pred_text)
        gt_boxes = parse_bounding_box_multi(gt_text)
        if gt_boxes is None:
            gt_single = parse_bounding_box_single(gt_text)
            if gt_single:
                gt_boxes = [gt_single]

        if pred_boxes is None or gt_boxes is None:
            invalid_count += 1
            continue

        mods = modalities.get(qid, ["PRE", "AP", "PVP", "T2WI"])
        if not isinstance(mods, list):
            mods = ["PRE", "AP", "PVP", "T2WI"]

        sample_ious = []
        for i, mod in enumerate(mods):
            if i < len(pred_boxes) and i < len(gt_boxes):
                iou = calculate_iou(pred_boxes[i], gt_boxes[i])
                if iou is None:
                    continue
                sample_ious.append(iou)
            elif i < len(pred_boxes) and len(gt_boxes) == 1:
                if mod == "AP":
                    iou = calculate_iou(pred_boxes[i], gt_boxes[0])
                    if iou is None:
                        continue
                    sample_ious.append(iou)

        if not sample_ious:
            invalid_count += 1
            continue

        avg_iou = sum(sample_ious) / len(sample_ious)
        all_avg_ious.append(avg_iou)
        for t in iou_thresholds:
            if avg_iou >= t:
                threshold_counts[t] += 1

    valid_count = len(all_avg_ious)
    return {
        "mIoU": sum(all_avg_ious) / valid_count if valid_count else 0.0,
        "IoU@0.1": threshold_counts[0.1] / valid_count if valid_count else 0.0,
        "IoU@0.3": threshold_counts[0.3] / valid_count if valid_count else 0.0,
        "IoU@0.5": threshold_counts[0.5] / valid_count if valid_count else 0.0,
        "valid": valid_count,
        "invalid": invalid_count,
    }


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize multimodal external test results.")
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/multi_external_summary",
        help="Output directory for CSVs.",
    )
    args = parser.parse_args()

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_jobs = {
        "external": ("playground/coco/external1_multi_qa.json",
                     "/mnt/data/by/data/coco_new/labels_2025_11_29/external_multimodal/test_QA_EN_external_multimodal_4mod.json"),
        "external_GXMU_bn": ("playground/coco/external_GXMU_bn_multi_qa.json",
                             "/mnt/data/by/data/coco_new/labels_2025_11_29/external_GXMU_benign_normal_multimodal/test_QA_EN_multimodal_4mod.json"),
        "external_ENSHI": ("playground/coco/external_ENSHI_multi_qa.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_ENSHI_multimodal/test_QA_EN_multimodal_4mod.json"),
        "external_SanYa": ("playground/coco/external_SanYa_multi_qa.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_SanYa_multimodal/test_QA_EN_multimodal_4mod.json"),
    }

    qa_bbox_jobs = {
        "external": ("playground/coco/external1_multi_qa_with_bbox.json",
                     "/mnt/data/by/data/coco_new/labels_2025_11_29/external_multimodal/test_QA_EN_external_multimodal_4mod_with_bbox.json"),
        "external_GXMU_bn": ("playground/coco/external_GXMU_bn_multi_qa_with_bbox.json",
                             "/mnt/data/by/data/coco_new/labels_2025_11_29/external_GXMU_benign_normal_multimodal/test_QA_EN_multimodal_4mod_with_bbox.json"),
        "external_ENSHI": ("playground/coco/external_ENSHI_multi_qa_with_bbox.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_ENSHI_multimodal/test_QA_EN_multimodal_4mod_with_bbox.json"),
        "external_SanYa": ("playground/coco/external_SanYa_multi_qa_with_bbox.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_SanYa_multimodal/test_QA_EN_multimodal_4mod_with_bbox.json"),
    }

    report_jobs = {
        "external": ("playground/coco/external1_multi_report.json",
                     "/mnt/data/by/data/coco_new/labels_2025_11_29/external_multimodal/test_report_generation_EN_multimodal.json"),
        "external_GXMU_bn": ("playground/coco/external_GXMU_bn_multi_report.json",
                             "/mnt/data/by/data/coco_new/labels_2025_11_29/external_GXMU_benign_normal_multimodal/test_report_generation_EN_multimodal.json"),
        "external_ENSHI": ("playground/coco/external_ENSHI_multi_report.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_ENSHI_multimodal/test_report_generation_EN_multimodal.json"),
        "external_SanYa": ("playground/coco/external_SanYa_multi_report.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_SanYa_multimodal/test_report_generation_EN_multimodal.json"),
    }

    report_bbox_jobs = {
        "external": ("playground/coco/external1_multi_report_with_bbox.json",
                     "/mnt/data/by/data/coco_new/labels_2025_11_29/external_multimodal/test_report_generation_EN_multimodal_with_bbox.json"),
        "external_GXMU_bn": ("playground/coco/external_GXMU_bn_multi_report_with_bbox.json",
                             "/mnt/data/by/data/coco_new/labels_2025_11_29/external_GXMU_benign_normal_multimodal/test_report_generation_EN_multimodal_with_bbox.json"),
        "external_ENSHI": ("playground/coco/external_ENSHI_multi_report_with_bbox.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_ENSHI_multimodal/test_report_generation_EN_multimodal_with_bbox.json"),
        "external_SanYa": ("playground/coco/external_SanYa_multi_report_with_bbox.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_SanYa_multimodal/test_report_generation_EN_multimodal_with_bbox.json"),
    }

    grounding_jobs = {
        "external": ("playground/coco/external1_multi_grounding.json",
                     "/mnt/data/by/data/coco_new/labels_2025_11_29/external_multimodal/test_grounding_multimodal_4mod.json"),
        "external_GXMU_bn": ("playground/coco/external_GXMU_bn_multi_grounding.json",
                             "/mnt/data/by/data/coco_new/labels_2025_11_29/external_GXMU_benign_normal_multimodal/test_grounding_multimodal_4mod.json"),
        "external_ENSHI": ("playground/coco/external_ENSHI_multi_grounding.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_ENSHI_multimodal/test_grounding_multimodal_4mod.json"),
        "external_SanYa": ("playground/coco/external_SanYa_multi_grounding.json",
                           "/mnt/data/by/data/coco_new/labels_2025_11_29/external_SanYa_multimodal/test_grounding_multimodal_4mod.json"),
    }

    qa_results = {}
    qa_bbox_results = {}
    for ds, (pred_path, gt_path) in qa_jobs.items():
        pred_full = BASE_DIR / pred_path
        if pred_full.exists() and Path(gt_path).exists():
            qa_results[ds] = eval_qa_multi(str(pred_full), gt_path, dataset_prefix=ds)

    for ds, (pred_path, gt_path) in qa_bbox_jobs.items():
        pred_full = BASE_DIR / pred_path
        if pred_full.exists() and Path(gt_path).exists():
            qa_bbox_results[ds] = eval_qa_multi(str(pred_full), gt_path, dataset_prefix=ds)

    merge_key = "external_merged"
    if "external" in qa_jobs and "external_GXMU_bn" in qa_jobs:
        def eval_qa_multi_merged(datasets):
            data_by_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
            for ds_name, (pred_path, gt_path) in datasets:
                preds = load_jsonl(pred_path)
                with open(gt_path, "r", encoding="utf-8") as f:
                    gts = json.load(f)
                for pred, gt in zip(preds, gts):
                    if pred.get("question_id") != gt.get("question_id"):
                        continue
                    qtype = gt["Question_type"]
                    patient_id = gt.get("patient_prefix") or gt["question_id"].split("_multimodal_")[0]
                    patient_id = f"{ds_name}:{patient_id}"
                    data_by_patient_qtype[(patient_id, qtype)]["pred"].append(pred["text"])
                    data_by_patient_qtype[(patient_id, qtype)]["gt"].append(gt["label"])
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

        qa_results[merge_key] = eval_qa_multi_merged([
            ("external", (str(BASE_DIR / qa_jobs["external"][0]), qa_jobs["external"][1])),
            ("external_GXMU_bn", (str(BASE_DIR / qa_jobs["external_GXMU_bn"][0]), qa_jobs["external_GXMU_bn"][1])),
        ])
        qa_bbox_results[merge_key] = eval_qa_multi_merged([
            ("external", (str(BASE_DIR / qa_bbox_jobs["external"][0]), qa_bbox_jobs["external"][1])),
            ("external_GXMU_bn", (str(BASE_DIR / qa_bbox_jobs["external_GXMU_bn"][0]), qa_bbox_jobs["external_GXMU_bn"][1])),
        ])

    compute_metrics = load_report_metrics()
    report_results = {}
    report_bbox_results = {}
    if compute_metrics is not None:
        for ds, (pred_path, gt_path) in report_jobs.items():
            pred_full = BASE_DIR / pred_path
            if pred_full.exists() and Path(gt_path).exists():
                report_results[ds] = eval_report_multi(
                    str(pred_full), gt_path, compute_metrics, dataset_prefix=ds
                )
        for ds, (pred_path, gt_path) in report_bbox_jobs.items():
            pred_full = BASE_DIR / pred_path
            if pred_full.exists() and Path(gt_path).exists():
                report_bbox_results[ds] = eval_report_multi(
                    str(pred_full), gt_path, compute_metrics, dataset_prefix=ds
                )

        if "external" in report_jobs and "external_GXMU_bn" in report_jobs:
            def eval_report_multi_merged(datasets):
                if compute_metrics is None:
                    return None
                data_by_patient = defaultdict(lambda: {"preds": [], "gts": []})
                for ds_name, (pred_path, gt_path) in datasets:
                    preds = load_jsonl(pred_path)
                    with open(gt_path, "r", encoding="utf-8") as f:
                        gts = json.load(f)
                    gt_mapping = {gt["question_id"]: gt for gt in gts}
                    for pred in preds:
                        qid = pred["question_id"]
                        if qid in gt_mapping:
                            gt = gt_mapping[qid]
                            patient_id = extract_patient_id_report(gt)
                            patient_id = f"{ds_name}:{patient_id}"
                            data_by_patient[patient_id]["preds"].append(pred["text"])
                            data_by_patient[patient_id]["gts"].append(gt["label"])
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

            report_results[merge_key] = eval_report_multi_merged([
                ("external", (str(BASE_DIR / report_jobs["external"][0]), report_jobs["external"][1])),
                ("external_GXMU_bn", (str(BASE_DIR / report_jobs["external_GXMU_bn"][0]), report_jobs["external_GXMU_bn"][1])),
            ])
            report_bbox_results[merge_key] = eval_report_multi_merged([
                ("external", (str(BASE_DIR / report_bbox_jobs["external"][0]), report_bbox_jobs["external"][1])),
                ("external_GXMU_bn", (str(BASE_DIR / report_bbox_jobs["external_GXMU_bn"][0]), report_bbox_jobs["external_GXMU_bn"][1])),
            ])

    grounding_results = {}
    for ds, (pred_path, gt_path) in grounding_jobs.items():
        pred_full = BASE_DIR / pred_path
        if pred_full.exists() and Path(gt_path).exists():
            grounding_results[ds] = eval_grounding_multi(str(pred_full), gt_path)

    if "external" in grounding_jobs and "external_GXMU_bn" in grounding_jobs:
        grounding_results[merge_key] = eval_grounding_multi_merged([
            ("external", (str(BASE_DIR / grounding_jobs["external"][0]), grounding_jobs["external"][1])),
            ("external_GXMU_bn", (str(BASE_DIR / grounding_jobs["external_GXMU_bn"][0]), grounding_jobs["external_GXMU_bn"][1])),
        ])

    qa_rows = []
    for ds, res in qa_results.items():
        for qtype in sorted(res.keys(), key=lambda x: (x == "Overall", x)):
            m = res[qtype]
            qa_rows.append([ds, qtype, f"{m['acc']:.4f}", f"{m['f1']:.4f}", m["n"]])
    write_csv(output_dir / "qa_multi.csv", ["dataset", "question", "acc", "f1", "n"], qa_rows)

    qa_bbox_rows = []
    for ds, res in qa_bbox_results.items():
        for qtype in sorted(res.keys(), key=lambda x: (x == "Overall", x)):
            m = res[qtype]
            qa_bbox_rows.append([ds, qtype, f"{m['acc']:.4f}", f"{m['f1']:.4f}", m["n"]])
    write_csv(output_dir / "qa_multi_with_bbox.csv", ["dataset", "question", "acc", "f1", "n"], qa_bbox_rows)

    if compute_metrics is not None:
        report_rows = []
        for ds, res in report_results.items():
            metrics = res["patient"]
            report_rows.append([
                ds,
                f"{metrics['BLEU-1']:.4f}",
                f"{metrics['BLEU-2']:.4f}",
                f"{metrics['BLEU-3']:.4f}",
                f"{metrics['BLEU-4']:.4f}",
                f"{metrics['CIDEr']:.4f}",
                f"{metrics['ROUGE_L']:.4f}",
                res["num_patients"],
            ])
        write_csv(
            output_dir / "report_multi.csv",
            ["dataset", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr", "ROUGE_L", "patients"],
            report_rows,
        )

        report_bbox_rows = []
        for ds, res in report_bbox_results.items():
            metrics = res["patient"]
            report_bbox_rows.append([
                ds,
                f"{metrics['BLEU-1']:.4f}",
                f"{metrics['BLEU-2']:.4f}",
                f"{metrics['BLEU-3']:.4f}",
                f"{metrics['BLEU-4']:.4f}",
                f"{metrics['CIDEr']:.4f}",
                f"{metrics['ROUGE_L']:.4f}",
                res["num_patients"],
            ])
        write_csv(
            output_dir / "report_multi_with_bbox.csv",
            ["dataset", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr", "ROUGE_L", "patients"],
            report_bbox_rows,
        )

    grounding_rows = []
    for ds, res in grounding_results.items():
        if res is None:
            continue
        grounding_rows.append([
            ds,
            f"{res['mIoU']:.4f}",
            f"{res['IoU@0.1']:.4f}",
            f"{res['IoU@0.3']:.4f}",
            f"{res['IoU@0.5']:.4f}",
            res["valid"],
            res["invalid"],
        ])
    write_csv(
        output_dir / "grounding_multi.csv",
        ["dataset", "mIoU", "IoU@0.1", "IoU@0.3", "IoU@0.5", "valid", "invalid"],
        grounding_rows,
    )

    def print_qa_table(title, results):
        print(title)
        print("| dataset | question | acc | f1 | n |")
        print("|---|---|---|---|---|")
        for ds in sorted(results.keys()):
            res = results[ds]
            for qtype in sorted(res.keys(), key=lambda x: (x == "Overall", x)):
                m = res[qtype]
                print(f"| {ds} | {qtype} | {m['acc']:.4f} | {m['f1']:.4f} | {m['n']} |")
        print()

    def print_report_table(title, results):
        print(title)
        print("| dataset | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | CIDEr | ROUGE_L | patients |")
        print("|---|---|---|---|---|---|---|---|")
        for ds in sorted(results.keys()):
            metrics = results[ds]["patient"]
            print(
                f"| {ds} | {metrics['BLEU-1']:.4f} | {metrics['BLEU-2']:.4f} | "
                f"{metrics['BLEU-3']:.4f} | {metrics['BLEU-4']:.4f} | "
                f"{metrics['CIDEr']:.4f} | {metrics['ROUGE_L']:.4f} | "
                f"{results[ds]['num_patients']} |"
            )
        print()

    def print_grounding_table(title, results):
        print(title)
        print("| dataset | mIoU | IoU@0.1 | IoU@0.3 | IoU@0.5 | valid | invalid |")
        print("|---|---|---|---|---|---|---|")
        for ds in sorted(results.keys()):
            res = results[ds]
            if res is None:
                continue
            print(
                f"| {ds} | {res['mIoU']:.4f} | {res['IoU@0.1']:.4f} | "
                f"{res['IoU@0.3']:.4f} | {res['IoU@0.5']:.4f} | "
                f"{res['valid']} | {res['invalid']} |"
            )
        print()

    print_qa_table("QA (multimodal, external, patient-level)", qa_results)
    print_qa_table("QA_with_bbox (multimodal, external, patient-level)", qa_bbox_results)
    if compute_metrics is None:
        print("Report metrics skipped (compute_metrics unavailable).")
    else:
        print_report_table("Report (multimodal, external, patient-level)", report_results)
        print_report_table("Report_with_bbox (multimodal, external, patient-level)", report_bbox_results)
    print_grounding_table("Grounding (multimodal, external, overall)", grounding_results)

    print(f"Wrote CSVs to {output_dir}")


if __name__ == "__main__":
    main()
