"""
Evaluation script for single-image CT medical VLM with per-modality voting.

This script evaluates predictions with patient-level voting, but only uses
one slice per modality for each patient. For each patient and question type:
1) Select one representative slice per modality (median slice if available).
2) Majority vote across modalities.
"""

import json
import jsonlines
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import argparse
import re


def load_predictions(pred_file: str) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with jsonlines.open(pred_file, 'r') as reader:
        for obj in reader:
            predictions.append(obj)
    return predictions


def load_ground_truth(gt_file: str) -> List[Dict]:
    """Load ground truth from JSON file."""
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    return ground_truth


def extract_patient_id(question_id: str, gt_entry: Dict) -> str:
    """Extract patient_id from question_id or gt_entry."""
    if 'image' in gt_entry:
        image_path = gt_entry['image']
        parts = image_path.split('/')
        if len(parts) >= 2:
            return parts[1]
    parts = question_id.split('/')
    return parts[0]


def extract_modality(question_id: str, gt_entry: Dict) -> str:
    """Extract modality from question_id or gt_entry image path."""
    if 'image' in gt_entry:
        image_path = gt_entry['image']
        parts = image_path.split('/')
        if len(parts) >= 3:
            return parts[2]
    parts = question_id.split('/')
    return parts[1] if len(parts) > 1 else ""


def extract_slice_index(question_id: str, gt_entry: Dict) -> int:
    """Extract numeric slice index if possible, else return -1."""
    candidates = []
    if 'image' in gt_entry:
        image_path = gt_entry['image']
        base = image_path.split('/')[-1]
        candidates.append(base)
    candidates.append(question_id.split('/')[-1])
    for c in candidates:
        m = re.search(r"(\\d+)", c)
        if m:
            return int(m.group(1))
    return -1


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate accuracy between predictions and ground truths."""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def compute_f1_scores(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Compute per-class, macro, micro, and weighted F1 for single-label classification."""
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must have same length"
    n = len(ground_truths)
    if n == 0:
        return {
            'per_class': {},
            'macro': 0.0,
            'micro': 0.0,
            'weighted': 0.0,
            'support': {},
            'labels': []
        }

    labels = sorted(list(set(ground_truths) | set(predictions)))
    tp = {l: 0 for l in labels}
    fp = {l: 0 for l in labels}
    fn = {l: 0 for l in labels}
    support = {l: 0 for l in labels}

    for pred, gt in zip(predictions, ground_truths):
        support[gt] = support.get(gt, 0) + 1
        if pred == gt:
            tp[gt] = tp.get(gt, 0) + 1
        else:
            fp[pred] = fp.get(pred, 0) + 1
            fn[gt] = fn.get(gt, 0) + 1

    per_class = {}
    macro_f1_sum = 0.0
    macro_count = 0
    weighted_f1_sum = 0.0
    total_support = sum(support.values())

    for l in labels:
        p_denom = tp[l] + fp[l]
        r_denom = tp[l] + fn[l]
        precision = tp[l] / p_denom if p_denom > 0 else 0.0
        recall = tp[l] / r_denom if r_denom > 0 else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        per_class[l] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support[l]
        }
        if support[l] > 0:
            macro_f1_sum += f1
            macro_count += 1
            weighted_f1_sum += f1 * support[l]

    macro = (macro_f1_sum / macro_count) if macro_count > 0 else 0.0

    micro_tp = sum(tp.values())
    micro_fp = sum(fp.values())
    micro_fn = sum(fn.values())
    micro_prec_denom = micro_tp + micro_fp
    micro_rec_denom = micro_tp + micro_fn
    micro_precision = micro_tp / micro_prec_denom if micro_prec_denom > 0 else 0.0
    micro_recall = micro_tp / micro_rec_denom if micro_rec_denom > 0 else 0.0
    micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0 else 0.0
    )

    weighted = (weighted_f1_sum / total_support) if total_support > 0 else 0.0

    return {
        'per_class': per_class,
        'macro': macro,
        'micro': micro,
        'weighted': weighted,
        'support': support,
        'labels': labels
    }


def majority_vote(votes: List[str]) -> str:
    """Return majority vote from a list of predictions."""
    if not votes:
        return ""
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


def pick_representative(items: List[Tuple[int, str, str]]) -> Tuple[str, str]:
    """Pick one representative slice (median by slice index) for a modality."""
    if not items:
        return "", ""
    # items: (slice_idx, pred_text, gt_text)
    if all(i[0] >= 0 for i in items):
        items_sorted = sorted(items, key=lambda x: x[0])
    else:
        items_sorted = items
    mid = len(items_sorted) // 2
    _, pred_text, gt_text = items_sorted[mid]
    return pred_text, gt_text


def evaluate_single_qa_per_modality(pred_file: str, gt_file: str):
    """Evaluate single-modal QA with per-modality voting (one slice per modality)."""

    print("=" * 70)
    print("Single-Image Medical VLM Evaluation (Per-Modality Voting)")
    print("=" * 70)

    print("\nLoading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth entries")

    if len(predictions) != len(ground_truth):
        print(f"Warning: Mismatch in number of entries - pred: {len(predictions)}, gt: {len(ground_truth)}")

    # patient_id -> question_type -> modality -> list[(slice_idx, pred, gt)]
    data_by_patient_qtype_mod = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        if pred.get('question_id') != gt.get('question_id'):
            print(f"Warning: Question ID mismatch at index {i}: {pred.get('question_id')} vs {gt.get('question_id')}")
            continue

        pred_text = pred['text']
        gt_text = gt['label']
        question_type = gt['Question_type']
        patient_id = extract_patient_id(gt['question_id'], gt)
        modality = extract_modality(gt['question_id'], gt)
        slice_idx = extract_slice_index(gt['question_id'], gt)

        data_by_patient_qtype_mod[patient_id][question_type][modality].append(
            (slice_idx, pred_text, gt_text)
        )

    # patient + question_type voting across modalities (one slice each)
    patient_voted_by_qtype = defaultdict(lambda: {'pred': [], 'gt': []})

    for patient_id, qtype_map in data_by_patient_qtype_mod.items():
        for question_type, mod_map in qtype_map.items():
            preds = []
            gts = []
            for modality, items in mod_map.items():
                pred_text, gt_text = pick_representative(items)
                if pred_text == "" and gt_text == "":
                    continue
                preds.append(pred_text)
                gts.append(gt_text)
            if not preds:
                continue
            voted_pred = majority_vote(preds)
            voted_gt = gts[0]
            patient_voted_by_qtype[question_type]['pred'].append(voted_pred)
            patient_voted_by_qtype[question_type]['gt'].append(voted_gt)

    print("\n" + "=" * 70)
    print("PATIENT-LEVEL PERFORMANCE (Per-Modality Voting)")
    print("=" * 70)

    print(f"\n{'Question Type':<12} {'Accuracy':<12} {'F1 (macro)':<12} {'Patients':<10}")
    print("-" * 50)
    for question_type in sorted(patient_voted_by_qtype.keys()):
        data = patient_voted_by_qtype[question_type]
        accuracy = calculate_accuracy(data['pred'], data['gt'])
        f1_stats = compute_f1_scores(data['pred'], data['gt'])
        print(f"{question_type:<12} {accuracy:<12.4f} {f1_stats['macro']:<12.4f} {len(data['pred']):<10}")

    # Overall patient-level
    all_voted_preds = []
    all_voted_gts = []
    for qtype, data in patient_voted_by_qtype.items():
        all_voted_preds.extend(data['pred'])
        all_voted_gts.extend(data['gt'])

    if all_voted_preds:
        acc_voted = calculate_accuracy(all_voted_preds, all_voted_gts)
        f1_voted = compute_f1_scores(all_voted_preds, all_voted_gts)
        print("-" * 50)
        print(f"{'Overall':<12} {acc_voted:<12.4f} {f1_voted['macro']:<12.4f} {len(all_voted_preds):<10}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate single-modal medical QA with per-modality voting")
    parser.add_argument('--pred',
                        default="playground/coco/external1_single_qa.json",
                        help='Path to predictions JSONL file')
    parser.add_argument('--gt',
                        default="/mnt/data/by/data/coco_new/labels_2025_11_29/external/test_QA_EN_external.json",
                        help='Path to ground truth JSON file')
    args = parser.parse_args()

    evaluate_single_qa_per_modality(args.pred, args.gt)
