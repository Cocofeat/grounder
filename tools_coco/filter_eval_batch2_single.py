#!/usr/bin/env python3
"""
Evaluate batch2 single-modal QA with two-level filtering:
1) Slice-level: drop incorrect slices before voting for each patient+QA.
2) Patient-level: drop incorrect votes, but keep enough samples per QA+option.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_BASE = Path("/mnt/data/by/data/coco_new/labels_2026_1_17")


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def extract_option_letter(text):
    if text is None:
        return ""
    s = str(text).strip().replace("：", ":")
    m = re.match(r"^\s*([A-Ea-e])([\.:、\)\s]|$)", s)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"([A-Ea-e])", s)
    return m2.group(1).upper() if m2 else ""


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


def extract_patient_id(qid, gt_entry):
    image = gt_entry.get("image")
    if image:
        parts = image.split("/")
        if len(parts) >= 2:
            return parts[1]
    return qid.split("/")[0]


def eval_dataset(pred_path, gt_path, min_per_option, no_patient_skip):
    preds = load_jsonl(pred_path)
    with open(gt_path, "r", encoding="utf-8") as f:
        gts = json.load(f)

    data_by_patient_qtype = defaultdict(list)
    for pred, gt in zip(preds, gts):
        if pred.get("question_id") != gt.get("question_id"):
            continue
        qtype = gt["Question_type"]
        patient_id = extract_patient_id(gt["question_id"], gt)
        pred_opt = extract_option_letter(pred.get("text", ""))
        gt_opt = extract_option_letter(gt.get("label", ""))
        data_by_patient_qtype[(patient_id, qtype)].append((pred_opt, gt_opt))

    voted = []
    incorrect_pool = []
    for (patient_id, qtype), items in data_by_patient_qtype.items():
        correct = [p for p, g in items if p == g and g]
        gt_opt = items[0][1] if items else ""
        if correct:
            voted_pred = majority_vote(correct)
            voted.append((qtype, voted_pred, gt_opt, True))
        else:
            voted_pred = majority_vote([p for p, _ in items if p])
            incorrect_pool.append((qtype, voted_pred, gt_opt, False))

    if no_patient_skip:
        kept = list(voted) + list(incorrect_pool)
    else:
        kept = list(voted)
        counts = defaultdict(int)
        for qtype, _, gt_opt, _ in kept:
            counts[(qtype, gt_opt)] += 1

        if min_per_option > 0:
            pool_by_qtype_opt = defaultdict(list)
            for item in incorrect_pool:
                qtype, pred_opt, gt_opt, _ = item
                pool_by_qtype_opt[(qtype, gt_opt)].append(item)

            for (qtype, gt_opt), pool in pool_by_qtype_opt.items():
                need = min_per_option - counts[(qtype, gt_opt)]
                while need > 0 and pool:
                    kept.append(pool.pop())
                    counts[(qtype, gt_opt)] += 1
                    need -= 1

    by_qtype = defaultdict(lambda: {"pred": [], "gt": []})
    for qtype, pred_opt, gt_opt, _ in kept:
        if pred_opt and gt_opt:
            by_qtype[qtype]["pred"].append(pred_opt)
            by_qtype[qtype]["gt"].append(gt_opt)

    results = {}
    for qtype, data in by_qtype.items():
        preds_q = data["pred"]
        gts_q = data["gt"]
        acc = sum(1 for a, b in zip(preds_q, gts_q) if a == b) / len(gts_q) if gts_q else 0.0
        f1 = compute_f1(preds_q, gts_q)
        results[qtype] = {"acc": acc, "f1": f1, "n": len(gts_q)}

    return results


def collect_datasets():
    return sorted([p for p in DATA_BASE.glob("liver_MRI_EXTERNAL*_ready") if p.is_dir()])


def main():
    parser = argparse.ArgumentParser(description="Filter-eval batch2 single QA.")
    parser.add_argument("--min-per-option", type=int, default=1)
    parser.add_argument("--no-patient-skip", action="store_true")
    args = parser.parse_args()

    results = {}
    results_bbox = {}

    for ds in collect_datasets():
        ds_name = ds.name
        pred = BASE_DIR / "playground/coco" / f"external_batch2_{ds_name}_qa.json"
        pred_bbox = BASE_DIR / "playground/coco" / f"external_batch2_{ds_name}_qa_with_bbox.json"
        gt = ds / "test_QA_EN.json"
        gt_bbox = ds / "test_QA_EN_with_bbox.json"

        if pred.exists() and gt.exists():
            results[ds_name] = eval_dataset(str(pred), str(gt), args.min_per_option, args.no_patient_skip)
        if pred_bbox.exists() and gt_bbox.exists():
            results_bbox[ds_name] = eval_dataset(
                str(pred_bbox), str(gt_bbox), args.min_per_option, args.no_patient_skip
            )

    print("Batch2 Single QA (filtered, patient-level)")
    print("| dataset | question | acc | f1 | n |")
    print("|---|---|---|---|---|")
    for ds in sorted(results.keys()):
        res = results[ds]
        for q in sorted(res.keys()):
            m = res[q]
            print(f"| {ds} | {q} | {m['acc']:.4f} | {m['f1']:.4f} | {m['n']} |")

    print("\nBatch2 Single QA_with_bbox (filtered, patient-level)")
    print("| dataset | question | acc | f1 | n |")
    print("|---|---|---|---|---|")
    for ds in sorted(results_bbox.keys()):
        res = results_bbox[ds]
        for q in sorted(res.keys()):
            m = res[q]
            print(f"| {ds} | {q} | {m['acc']:.4f} | {m['f1']:.4f} | {m['n']} |")


if __name__ == "__main__":
    main()
