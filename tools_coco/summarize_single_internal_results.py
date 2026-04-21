#!/usr/bin/env python3
"""
Summarize single-model internal test results for QA, report, and grounding.

Based on script_coco/test_internal.sh.
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


def extract_patient_id_single(qid, gt_entry):
    image = gt_entry.get("image")
    if image:
        parts = image.split("/")
        if len(parts) >= 2:
            # Internal paths start with patient_id/modality/...
            # External-like paths start with dataset/patient_id/...
            if parts[1] in {"AP", "PVP", "PRE", "T2WI"}:
                return parts[0]
            return parts[1]
    return qid.split("/")[0]


def eval_qa_single(pred_path, gt_path):
    preds = load_jsonl(pred_path)
    with open(gt_path, "r", encoding="utf-8") as f:
        gts = json.load(f)

    data_by_patient_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for pred, gt in zip(preds, gts):
        if pred.get("question_id") != gt.get("question_id"):
            continue
        qtype = gt["Question_type"]
        patient_id = extract_patient_id_single(gt["question_id"], gt)
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
        from tools_coco.eval_report_single_vote import compute_metrics
    except Exception:
        compute_metrics = None
    return compute_metrics


def eval_report_single(pred_path, gt_path, compute_metrics):
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
            image = gt.get("image", "")
            parts = image.split("/") if image else qid.split("/")
            if len(parts) >= 2 and parts[1] in {"AP", "PVP", "PRE", "T2WI"}:
                patient_id = parts[0]
            else:
                patient_id = parts[1] if len(parts) >= 2 else parts[0]
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


def eval_grounding_single(pred_path, gt_path):
    from tools_coco.eval_grounding import (
        parse_bounding_box_single,
        calculate_iou,
        load_ground_truth as load_ground_truth_grounding,
    )

    preds = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                preds[data["question_id"]] = data["text"]

    ground_truth, _, _ = load_ground_truth_grounding(gt_path)
    common_ids = set(preds.keys()) & set(ground_truth.keys())
    if not common_ids:
        return None

    all_ious = []
    iou_thresholds = [0.1, 0.3, 0.5]
    threshold_counts = {t: 0 for t in iou_thresholds}
    invalid_count = 0
    skipped_zero_gt = 0

    for qid in common_ids:
        pred_box = parse_bounding_box_single(preds[qid])
        gt_box = parse_bounding_box_single(ground_truth[qid])
        if pred_box is None or gt_box is None:
            invalid_count += 1
            continue
        iou = calculate_iou(pred_box, gt_box)
        if iou is None:
            skipped_zero_gt += 1
            continue
        all_ious.append(iou)
        for t in iou_thresholds:
            if iou >= t:
                threshold_counts[t] += 1

    valid_count = len(all_ious)
    return {
        "mIoU": sum(all_ious) / valid_count if valid_count else 0.0,
        "IoU@0.1": threshold_counts[0.1] / valid_count if valid_count else 0.0,
        "IoU@0.3": threshold_counts[0.3] / valid_count if valid_count else 0.0,
        "IoU@0.5": threshold_counts[0.5] / valid_count if valid_count else 0.0,
        "valid": valid_count,
        "invalid": invalid_count,
        "skipped_zero_gt": skipped_zero_gt,
    }


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize single internal test results.")
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/single_internal_summary",
        help="Output directory for CSVs.",
    )
    args = parser.parse_args()

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_pred = BASE_DIR / "playground/coco/internal_test_qa.json"
    qa_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/test/test_QA_EN.json"
    qa_bbox_pred = BASE_DIR / "playground/coco/internal_test_qa_with_bbox.json"
    qa_bbox_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/test/test_QA_EN_with_bbox.json"

    report_pred = BASE_DIR / "playground/coco/internal_test_report.json"
    report_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/test/test_report_generation_EN.json"
    report_bbox_pred = BASE_DIR / "playground/coco/internal_test_report_with_bbox.json"
    report_bbox_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/test/test_report_generation_EN_with_bbox.json"

    grounding_pred = BASE_DIR / "playground/coco/internal_test_grounding.json"
    grounding_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/test/test_bbox_v2.json"

    qa_results = {}
    if qa_pred.exists() and Path(qa_gt).exists():
        qa_results["internal"] = eval_qa_single(str(qa_pred), qa_gt)

    qa_bbox_results = {}
    if qa_bbox_pred.exists() and Path(qa_bbox_gt).exists():
        qa_bbox_results["internal"] = eval_qa_single(str(qa_bbox_pred), qa_bbox_gt)

    compute_metrics = load_report_metrics()
    report_results = {}
    report_bbox_results = {}
    if compute_metrics is not None:
        if report_pred.exists() and Path(report_gt).exists():
            report_results["internal"] = eval_report_single(str(report_pred), report_gt, compute_metrics)
        if report_bbox_pred.exists() and Path(report_bbox_gt).exists():
            report_bbox_results["internal"] = eval_report_single(
                str(report_bbox_pred), report_bbox_gt, compute_metrics
            )

    grounding_results = {}
    if grounding_pred.exists() and Path(grounding_gt).exists():
        grounding_results["internal"] = eval_grounding_single(str(grounding_pred), grounding_gt)

    qa_rows = []
    for ds, res in qa_results.items():
        for qtype in sorted(res.keys(), key=lambda x: (x == "Overall", x)):
            m = res[qtype]
            qa_rows.append([ds, qtype, f"{m['acc']:.4f}", f"{m['f1']:.4f}", m["n"]])
    write_csv(output_dir / "qa_single.csv", ["dataset", "question", "acc", "f1", "n"], qa_rows)

    qa_bbox_rows = []
    for ds, res in qa_bbox_results.items():
        for qtype in sorted(res.keys(), key=lambda x: (x == "Overall", x)):
            m = res[qtype]
            qa_bbox_rows.append([ds, qtype, f"{m['acc']:.4f}", f"{m['f1']:.4f}", m["n"]])
    write_csv(output_dir / "qa_single_with_bbox.csv", ["dataset", "question", "acc", "f1", "n"], qa_bbox_rows)

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
            output_dir / "report_single.csv",
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
            output_dir / "report_single_with_bbox.csv",
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
            res["skipped_zero_gt"],
        ])
    write_csv(
        output_dir / "grounding_single.csv",
        ["dataset", "mIoU", "IoU@0.1", "IoU@0.3", "IoU@0.5", "valid", "skipped_zero_gt"],
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
        print("| dataset | mIoU | IoU@0.1 | IoU@0.3 | IoU@0.5 | valid | skipped_zero_gt |")
        print("|---|---|---|---|---|---|---|")
        for ds in sorted(results.keys()):
            res = results[ds]
            if res is None:
                continue
            print(
                f"| {ds} | {res['mIoU']:.4f} | {res['IoU@0.1']:.4f} | "
                f"{res['IoU@0.3']:.4f} | {res['IoU@0.5']:.4f} | "
                f"{res['valid']} | {res['skipped_zero_gt']} |"
            )
        print()

    print_qa_table("QA (single, internal, patient-level)", qa_results)
    print_qa_table("QA_with_bbox (single, internal, patient-level)", qa_bbox_results)
    if compute_metrics is None:
        print("Report metrics skipped (compute_metrics unavailable).")
    else:
        print_report_table("Report (single, internal, patient-level)", report_results)
        print_report_table("Report_with_bbox (single, internal, patient-level)", report_bbox_results)
    print_grounding_table("Grounding (single, internal, overall)", grounding_results)

    print(f"Wrote CSVs to {output_dir}")


if __name__ == "__main__":
    main()
