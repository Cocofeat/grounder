"""
Evaluation script for multi-image (multimodal) CT medical VLM.

This script evaluates predictions with:
1. Sample-level performance (no voting) - average across all samples
2. Patient-level voting - majority vote across slices for each patient

Data format:
- question_id: "{patient_prefix}_multimodal_{Question_type}_{slice_num}"
- Each patient has multiple slices, each slice has 4 question types (QA1-QA4)
"""

import json
import jsonlines
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import re
import argparse


def load_predictions(pred_file: str) -> List[Dict]:
    """Load predictions from JSONL file"""
    predictions = []
    with jsonlines.open(pred_file, 'r') as reader:
        for obj in reader:
            predictions.append(obj)
    return predictions


def load_ground_truth(gt_file: str) -> List[Dict]:
    """Load ground truth from JSON file"""
    with open(gt_file, 'r') as f:
        ground_truth = json.load(f)
    return ground_truth


def parse_question_id(question_id: str) -> Tuple[str, str, str]:
    """Parse question_id to extract patient_id, question_type, and slice_num.

    Format: {patient_prefix}_multimodal_{Question_type}_{slice_num}
    Example: D91A_0000060419_multimodal_QA1_19
    """
    # Find the _multimodal_ marker
    parts = question_id.split('_multimodal_')
    patient_id = parts[0]

    # The rest is {Question_type}_{slice_num}
    rest = parts[1]
    last_underscore = rest.rfind('_')
    question_type = rest[:last_underscore]
    slice_num = rest[last_underscore + 1:]

    return patient_id, question_type, slice_num


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Calculate accuracy between predictions and ground truths"""
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    return correct / len(predictions) if len(predictions) > 0 else 0.0


def extract_option_letter(text: str) -> str:
    """Extract leading option letter (A/B/C/D/E) from text."""
    if text is None:
        return ""
    s = str(text).strip()
    s = s.replace('：', ':')
    m = re.match(r"^\s*([A-Ea-e])([\.:、\)\s]|$)", s)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"([A-Ea-e])", s)
    return m2.group(1).upper() if m2 else ""


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
    """Return majority vote from a list of predictions"""
    if not votes:
        return ""
    vote_counts = Counter(votes)
    return vote_counts.most_common(1)[0][0]


def tally_counts(items: List[str]) -> Dict[str, int]:
    """Count occurrences of option letters A-E"""
    cnt = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    for x in items:
        l = extract_option_letter(x)
        if l in cnt:
            cnt[l] += 1
    return cnt


def evaluate_multimodal_qa(pred_file: str, gt_file: str):
    """Main evaluation function for multimodal QA with patient-level voting"""

    print("=" * 70)
    print("Multi-Image Medical VLM Evaluation")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(gt_file)

    print(f"Loaded {len(predictions)} predictions and {len(ground_truth)} ground truth entries")

    # Verify matching order
    if len(predictions) != len(ground_truth):
        print(f"Warning: Mismatch in number of entries - pred: {len(predictions)}, gt: {len(ground_truth)}")

    # Organize data
    # 1. By question type (for sample-level stats)
    data_by_question_type = defaultdict(lambda: {'pred': [], 'gt': []})

    # 2. By num_modalities (for stats by modality count)
    data_by_num_modalities = defaultdict(lambda: {'pred': [], 'gt': []})

    # 3. By patient + question_type (for patient-level voting)
    data_by_patient_qtype = defaultdict(lambda: {'pred': [], 'gt': []})

    # Process each prediction-ground truth pair
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        if pred['question_id'] != gt['question_id']:
            print(f"Warning: Question ID mismatch at index {i}: {pred['question_id']} vs {gt['question_id']}")
            continue

        pred_text = pred['text']
        gt_text = gt['label']
        question_type = gt['Question_type']
        # Support both old format (patient_prefix) and new format (extract from question_id)
        if 'patient_prefix' in gt:
            patient_id = gt['patient_prefix']
        else:
            # Extract patient_id from question_id: "dataset/patient_id_multimodal_QA*_slice"
            qid = gt['question_id']
            patient_id = qid.split('_multimodal_')[0]
        num_modalities = gt.get('num_modalities', len(gt.get('modalities', [])))

        # Store by question type
        data_by_question_type[question_type]['pred'].append(pred_text)
        data_by_question_type[question_type]['gt'].append(gt_text)

        # Store by num_modalities
        data_by_num_modalities[num_modalities]['pred'].append(pred_text)
        data_by_num_modalities[num_modalities]['gt'].append(gt_text)

        # Store by patient + question_type for voting
        key = f"{patient_id}_{question_type}"
        data_by_patient_qtype[key]['pred'].append(pred_text)
        data_by_patient_qtype[key]['gt'].append(gt_text)

    # =========================================================================
    # Patient-Level Performance (With Voting)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PATIENT-LEVEL PERFORMANCE (With Majority Voting)")
    print("=" * 70)

    # For each patient + question_type, do majority voting across slices
    patient_voted_by_qtype = defaultdict(lambda: {'pred': [], 'gt': []})

    for key, data in data_by_patient_qtype.items():
        # key format: "{patient_id}_{question_type}"
        question_type = key.split('_')[-1]  # Get QA1, QA2, etc.

        # Majority vote for prediction
        voted_pred = majority_vote(data['pred'])
        # Ground truth should be the same for all slices of same patient+qtype
        voted_gt = data['gt'][0]

        patient_voted_by_qtype[question_type]['pred'].append(voted_pred)
        patient_voted_by_qtype[question_type]['gt'].append(voted_gt)

    # Overall patient-level
    all_voted_preds = []
    all_voted_gts = []
    for qtype, data in patient_voted_by_qtype.items():
        all_voted_preds.extend(data['pred'])
        all_voted_gts.extend(data['gt'])

    # Count unique patients
    unique_patients = set()
    for key in data_by_patient_qtype.keys():
        patient_id = '_'.join(key.split('_')[:-1])
        unique_patients.add(patient_id)
    num_patients = len(unique_patients)

    # By Question Type (patient-level)
    print(f"\n{'Question Type':<12} {'Accuracy':<12} {'F1 (macro)':<12} {'Patients':<10}")
    print("-" * 50)
    for question_type in sorted(patient_voted_by_qtype.keys()):
        data = patient_voted_by_qtype[question_type]
        accuracy = calculate_accuracy(data['pred'], data['gt'])
        f1_stats = compute_f1_scores(data['pred'], data['gt'])
        print(f"{question_type:<12} {accuracy:<12.4f} {f1_stats['macro']:<12.4f} {len(data['pred']):<10}")

    # Overall patient-level
    if all_voted_preds:
        acc_voted = calculate_accuracy(all_voted_preds, all_voted_gts)
        f1_voted = compute_f1_scores(all_voted_preds, all_voted_gts)
        print("-" * 50)
        print(f"{'Overall':<12} {acc_voted:<12.4f} {f1_voted['macro']:<12.4f} {num_patients:<10}")


    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate multimodal medical QA with patient-level voting")
    parser.add_argument('--pred',
                        default="playground/coco/total_full_qa_multimodal_4mod.json",
                        help='Path to predictions JSONL file')
    parser.add_argument('--gt',
                        default="/mnt/data/by/data/coco_new/labels/test_QA_EN_multimodal_4mod.json",
                        help='Path to ground truth JSON file')
    args = parser.parse_args()

    evaluate_multimodal_qa(args.pred, args.gt)
