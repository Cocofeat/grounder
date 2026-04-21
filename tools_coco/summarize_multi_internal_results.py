#!/usr/bin/env python3
"""
Summarize multimodal internal test results for QA and report.

Based on script_coco/test_internal_multi.sh.
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


def eval_qa_multi(pred_path, gt_path):
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


def extract_patient_id_report(qid):
    if "_multimodal_" in qid:
        prefix = qid.split("_multimodal_")[0]
        return prefix.split("/")[-1]
    if "/" in qid:
        return qid.split("/")[0]
    return qid


def eval_report_multi(pred_path, gt_path, compute_metrics):
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
            patient_id = extract_patient_id_report(qid)
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


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize multimodal internal test results.")
    parser.add_argument(
        "--output-dir",
        default="evaluation_results/multi_internal_summary",
        help="Output directory for CSVs.",
    )
    args = parser.parse_args()

    output_dir = BASE_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    qa_pred = BASE_DIR / "playground/coco/internal_multi_qa.json"
    qa_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/multimodal_test/test_QA_EN_multimodal_4mod.json"
    qa_bbox_pred = BASE_DIR / "playground/coco/internal_multi_qa_with_bbox.json"
    qa_bbox_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/multimodal_test/test_QA_EN_multimodal_4mod_with_bbox.json"

    report_pred = BASE_DIR / "playground/coco/internal_multi_report.json"
    report_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/multimodal_test/test_report_generation_EN_multimodal.json"
    report_bbox_pred = BASE_DIR / "playground/coco/internal_multi_report_with_bbox.json"
    report_bbox_gt = "/mnt/data/by/data/coco_new/labels_2025_11_29/multimodal_test/test_report_generation_EN_multimodal_with_bbox.json"

    qa_results = {}
    if qa_pred.exists() and Path(qa_gt).exists():
        qa_results["internal"] = eval_qa_multi(str(qa_pred), qa_gt)

    qa_bbox_results = {}
    if qa_bbox_pred.exists() and Path(qa_bbox_gt).exists():
        qa_bbox_results["internal"] = eval_qa_multi(str(qa_bbox_pred), qa_bbox_gt)

    compute_metrics = load_report_metrics()
    report_results = {}
    report_bbox_results = {}
    if compute_metrics is not None:
        if report_pred.exists() and Path(report_gt).exists():
            report_results["internal"] = eval_report_multi(str(report_pred), report_gt, compute_metrics)
        if report_bbox_pred.exists() and Path(report_bbox_gt).exists():
            report_bbox_results["internal"] = eval_report_multi(
                str(report_bbox_pred), report_bbox_gt, compute_metrics
            )

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

    print_qa_table("QA (multimodal, internal, patient-level)", qa_results)
    print_qa_table("QA_with_bbox (multimodal, internal, patient-level)", qa_bbox_results)
    if compute_metrics is None:
        print("Report metrics skipped (compute_metrics unavailable).")
    else:
        print_report_table("Report (multimodal, internal, patient-level)", report_results)
        print_report_table("Report_with_bbox (multimodal, internal, patient-level)", report_bbox_results)

    print(f"Wrote CSVs to {output_dir}")


if __name__ == "__main__":
    main()
